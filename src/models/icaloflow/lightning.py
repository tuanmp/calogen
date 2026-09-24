"""iCaloFlow Lightning module (arXiv:2305.11934).

The Inductive CaloFlow (iCaloFlow) model generates calorimeter showers
layer by layer using three chained conditional Masked Autoregressive Flows
(MAFs) built from rational-quadratic-spline (RQS) MADE blocks on top of
`normflows`.  The paper's distilled IAF "students" (Flow-2/Flow-3) are also
implemented: ``student_mode`` trains them by probability density distillation
(PDD) from a frozen teacher, and generation samples through the fast parallel
students when they are loaded.

Flow roles (paper Table 1; DS2/LEMURS geometry, 45 layers x 144 cells):

  Flow-1  p(E_i | E_inc)              features 45,  cond 1
          per-layer deposited energies given the incident energy
  Flow-2  p(I_0 | E_inc, E_0)         features 144, cond 2
          unit-normalized voxels of layer 0
  Flow-3  p(I_i | E_inc, E_i, E_{i-1}, I_{i-1}, i)   features 144, cond 191
          unit-normalized voxels of layer i, conditioned on the previous
          layer's shape (the "inductive" step)

Training (paper Table 3 DS2): Adam + OneCycleLR, batch 1000 —
  Flow-1 (base 1e-5, max 1e-4, 500 ep), Flow-2 (2e-5/1e-3, 200 ep),
  Flow-3 (2e-5/1e-3, 60 ep).  All flows run in float64.

A ``fit`` run trains exactly one flow, selected by ``which_flow``:
1 = Flow-1, 2 = Flow-2, 4 = Flow-3.  A generation run (``which_flow=7``)
loads the three per-flow checkpoints and samples showers inductively
(Flow-1 -> E~_i proxy energies, Flow-2 -> layer 0, Flow-3 -> layers 1..44),
returning ``(incident_energies_MeV, showers_MeV, t)`` for ``PredictionWriter``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import lightning as pl
import torch
from calogen.models.base import BaseGenerativeModule
from calogen.models.icaloflow.datamodule import (
    log10_energy,
    one_hot,
    preprocess_cond_flow2,
    preprocess_cond_flow3,
)
from calogen.models.icaloflow.model import (
    add_noise,
    build_flow,
    build_student_flow,
    chi2_loss,
    guided_cascade,
    inverse_logit,
    logabsdet_of_base,
    logit_trafo,
)

# Paper Table 3 (DS2) OneCycleLR values per flow: (base_lr, max_lr).
_DEFAULT_LR = {
    1: (1e-5, 1e-4),
    2: (2e-5, 1e-3),
    4: (2e-5, 1e-3),
}

# Student distillation LRs (paper Table 3): flow-3 max is reduced.
_STUDENT_DEFAULT_LR = {
    2: (2e-5, 1e-3),
    4: (2e-5, 5e-4),
}


class ICaloFlowLightningModule(BaseGenerativeModule):
    """Trains one iCaloFlow flow or runs the 3-flow generation pipeline.

    The class always knows how to build all three flows but only
    constructs the ones its run needs (``which_flow``), which keeps
    training memory low.  Generation (``which_flow=7``) builds all three
    and populates them from per-flow checkpoints in ``setup``.

    With ``student_mode=True`` a Flow-2/Flow-3 run additionally builds the
    IAF student distilled (PDD) from the frozen teacher checkpoint; a
    generation run can then decode through the fast parallel students.
    """

    plugin_name = "icaloflow"

    def __init__(
        self,
        which_flow: int,
        flow1_path: Optional[str] = None,
        flow2_path: Optional[str] = None,
        flow3_path: Optional[str] = None,
        hidden_size: int = 256,
        n_blocks: int = 8,
        n_bins: int = 8,
        tail_bound: float = 14.0,
        num_layers_flow1: int = 1,
        num_layers_flow2: int = 2,
        num_layers_flow3: int = 2,
        optimizer_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        noise_level: float = 5e-3,
        threshold_cut: float = 1.5e-2,
        layer_boundaries: Optional[List[int]] = None,
        normalization_constant: Optional[float] = None,
        max_lr_scale: Optional[float] = None,
        num_layers: int = 45,
        cells_per_layer: int = 144,
        energy_ref_mev: float = 10**4.5,
        student_mode: bool = False,
        student_hidden_size: int = 256,
        student_width: float = 1.0,
        beta: float = 0.5,
        flow2_student_path: Optional[str] = None,
        flow3_student_path: Optional[str] = None,
        model_dtype="float64",
        **kwargs,
    ):
        super().__init__()
        if which_flow not in (1, 2, 4, 7):
            raise ValueError(f"which_flow must be 1, 2, 4 or 7; got {which_flow}")
        self.which_flow = which_flow
        self.flow1_path = flow1_path
        self.flow2_path = flow2_path
        self.flow3_path = flow3_path
        self.student_mode = student_mode
        self.student_hidden_size = student_hidden_size
        self.student_width = student_width
        self.beta = beta
        self.flow2_student_path = flow2_student_path
        self.flow3_student_path = flow3_student_path
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.num_layers_flow1 = num_layers_flow1
        self.num_layers_flow2 = num_layers_flow2
        self.num_layers_flow3 = num_layers_flow3
        self.optimizer_params = dict(optimizer_params or {})
        self.scheduler_params = dict(scheduler_params or {})
        self.noise_level = noise_level
        self.threshold_cut = threshold_cut
        self.layer_boundaries = layer_boundaries
        self.normalization_constant = normalization_constant
        self.max_lr_scale = max_lr_scale
        self.num_layers = num_layers
        self.cells_per_layer = cells_per_layer
        self.energy_ref_mev = energy_ref_mev
        self.num_train_samples = 0
        self.steps_per_epoch = 1
        self.save_hyperparameters()

        # Flow (features, context_features) dims, derived from the grid
        # geometry so the module is not tied to the LEMURS FCCeeALLEGRO shape.
        flow_dims = {
            1: (num_layers, 1),
            2: (cells_per_layer, 2),
            4: (cells_per_layer, 1 + 1 + 1 + cells_per_layer + (num_layers - 1)),
        }

        # Build only the flows this run needs (all three for generation).
        self.flow1 = self.flow2 = self.flow3 = None
        if which_flow in (1, 7):
            features, context = flow_dims[1]
            self.flow1 = build_flow(
                features, context, num_layers_flow1, hidden_size, n_blocks, n_bins, tail_bound
            )
        if which_flow in (2, 7):
            features, context = flow_dims[2]
            self.flow2 = build_flow(
                features, context, num_layers_flow2, hidden_size, n_blocks, n_bins, tail_bound
            )
        if which_flow in (4, 7):
            features, context = flow_dims[4]
            self.flow3 = build_flow(
                features, context, num_layers_flow3, hidden_size, n_blocks, n_bins, tail_bound
            )

        # Student IAF flows (PDD distillation); only built in student_mode.
        self.flow2_student = self.flow3_student = None
        if student_mode:
            if which_flow in (2, 7):
                features, context = flow_dims[2]
                self.flow2_student = build_student_flow(
                    self.flow2, features, context, num_layers_flow2,
                    student_hidden_size, n_blocks, n_bins, tail_bound,
                )
            if which_flow in (4, 7):
                features, context = flow_dims[4]
                self.flow3_student = build_student_flow(
                    self.flow3, features, context, num_layers_flow3,
                    student_hidden_size, n_blocks, n_bins, tail_bound,
                )

        match model_dtype:
            case "float32":
                self.model_dtype = torch.float32
            case "float64":
                self.model_dtype = torch.float64
            case _:
                raise ValueError(f"model_dtype must be 'float32' or 'float64'; got {model_dtype}")

    @property
    def active_flow(self):
        """The single flow trained by this run; generation runs have none."""
        if self.which_flow == 1:
            return self.flow1
        if self.which_flow == 2:
            return self.flow2
        if self.which_flow == 4:
            return self.flow3
        raise ValueError(
            f"which_flow={self.which_flow} is a generation run; no single active flow"
        )

    @property
    def active_student(self):
        """The student IAF distilled by this run; generation runs have none."""
        if not self.student_mode:
            raise ValueError("active_student requires student_mode=True")
        if self.which_flow == 2:
            return self.flow2_student
        if self.which_flow == 4:
            return self.flow3_student
        raise ValueError(
            f"which_flow={self.which_flow} is a generation run; no active student"
        )

    @property
    def num_dim(self) -> int:
        """Dimensionality of the joint latent space.

        Concatenation of Flow-1 (layer energies), Flow-2 (layer-0 shape) and
        Flow-3 (layers 1..num_layers-1): ``num_layers + cells_per_layer +
        (num_layers - 1) * cells_per_layer = num_layers * (cells_per_layer + 1)``.
        For LEMURS FCCeeALLEGRO this is 45 * 145 = 6525, matching the CINN.
        """
        return self.num_layers * (self.cells_per_layer + 1)

    def _get_trainer(self):
        """Trainer reference or None (Lightning's ``trainer`` property raises)."""
        return getattr(self, "_trainer", None)

    def _freeze_teacher(self):
        """Freeze the teacher flows (eval, no grad) for PDD distillation."""
        for flow in (self.flow1, self.flow2, self.flow3):
            if flow is not None:
                flow.eval()
                for p in flow.parameters():
                    p.requires_grad_(False)

    def setup(self, stage: Optional[str] = None):
        trainer = self._get_trainer()
        dm = trainer.datamodule if trainer is not None else None
        if stage == "fit" and dm is not None and hasattr(dm, "num_train_samples"):
            self.num_train_samples = dm.num_train_samples
            self.steps_per_epoch = max(
                1, math.ceil(self.num_train_samples / dm.batch_size)
            )
        else:
            self.steps_per_epoch = max(
                1, int(self.scheduler_params.get("steps_per_epoch", 1))
            )
        if self.normalization_constant is None and dm is not None:
            self.normalization_constant = getattr(dm, "normalization_constant", None)
        if self.layer_boundaries is None and dm is not None:
            self.layer_boundaries = getattr(dm, "layer_boundaries", None)
        # The reference runs all flows in float64; cast the active flows here.
        for flow in (self.flow1, self.flow2, self.flow3):
            if flow is not None:
                flow.to(self.model_dtype)
        if self.student_mode and stage == "fit":
            # PDD: load the pretrained teacher, then freeze it; students stay trainable.
            for flow, path, prefix in (
                (self.flow2, self.flow2_path, "flow2."),
                (self.flow3, self.flow3_path, "flow3."),
            ):
                if flow is None:
                    continue
                if path is None:
                    raise ValueError(
                        f"Student distillation requires the {prefix.rstrip('.')} "
                        "teacher checkpoint; got None"
                    )
                self._load_flow_submodule(flow, path, prefix)
                flow.to(self.model_dtype)
            self._freeze_teacher()
            for student in (self.flow2_student, self.flow3_student):
                if student is not None:
                    student.to(self.model_dtype)
        if stage in ("predict", "test") and self.which_flow == 7:
            print("Loading per-flow checkpoints for generation...")
            self.load_flows_for_generation(
                self.flow1_path, self.flow2_path, self.flow3_path,
                flow2_student_path=self.flow2_student_path,
                flow3_student_path=self.flow3_student_path,
            )

    def configure_optimizers(self):
        if self.student_mode:
            flow = self.active_student  # raises for generation runs
            base_lr, default_max_lr = _STUDENT_DEFAULT_LR[self.which_flow]
        else:
            flow = self.active_flow  # raises for generation runs
            base_lr, default_max_lr = _DEFAULT_LR[self.which_flow]
        lr = float(
            self.scheduler_params.get("lr", self.optimizer_params.get("lr", base_lr))
        )
        if "max_lr" in self.scheduler_params:
            max_lr = float(self.scheduler_params["max_lr"])
        elif self.max_lr_scale is not None:
            max_lr = lr * float(self.max_lr_scale)
        else:
            max_lr = float(default_max_lr)
        epochs = max(1, int(self.scheduler_params.get("epochs", 1)))
        steps_per_epoch = max(
            1, int(self.scheduler_params.get("steps_per_epoch", self.steps_per_epoch))
        )
        optimizer = torch.optim.Adam(
            flow.parameters(),
            lr=lr,
            betas=tuple(self.optimizer_params.get("betas", (0.9, 0.999))),
            eps=float(self.optimizer_params.get("eps", 1e-8)),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=epochs * steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _prepare_flow_batch(self, batch):
        """Split a flow-specific batch into (x, cond) cast to the flow dtype."""
        flow = self.active_flow
        dtype = next(flow.parameters()).dtype
        if self.which_flow == 1:
            x = batch["energy_dep"]
            c = batch["energy_inc"]
        elif self.which_flow == 2:
            x = batch["layer"]
            c = preprocess_cond_flow2(batch)
        else:
            x = batch["layer"]
            # Pass the full layer count so the one-hot always has
            # num_layers - 1 classes (a batch may not contain every layer).
            c = preprocess_cond_flow3(batch, num_layers=self.num_layers)
        return x.to(dtype=dtype), c.to(dtype=dtype)

    def _prepare_student_batch(self, batch):
        """Split a flow-2/flow-3 batch into (x, cond) cast to the student dtype."""
        dtype = next(self.active_student.parameters()).dtype
        x = batch["layer"]
        if self.which_flow == 2:
            c = preprocess_cond_flow2(batch)
        else:
            c = preprocess_cond_flow3(batch, num_layers=self.num_layers)
        return x.to(dtype=dtype), c.to(dtype=dtype)

    def _pdd_losses(self, x, c):
        """PDD distillation losses for one batch: (chi_x, chi_z, KL, combined).

        Mirrors the reference ``train_flow_2_student``: the x-pass distills the
        teacher's data->latent map, the z-pass the student's latent->data map,
        and the KL (teacher term detached) is monitored, not optimized.
        """
        teacher = self.active_flow
        student = self.active_student
        features = teacher.q0.d
        width = self.student_width
        beta = self.beta

        # x-pass (data space): teacher encodes x, student must reproduce it.
        teacher_noise, _, t_steps, t_p = guided_cascade(
            teacher.flows, x, c, "inverse", return_steps=True, return_p=True
        )
        student_data, _, s_steps, s_p = guided_cascade(
            student.flows, teacher_noise, c, "forward", return_steps=True, return_p=True
        )
        loss_chi_x = chi2_loss(student_data, x)
        for i, step in enumerate(s_steps):
            loss_chi_x = loss_chi_x + chi2_loss(step, t_steps[-i - 1])
        for i, p in enumerate(s_p):
            loss_chi_x = loss_chi_x + chi2_loss(p, t_p[-i - 1])

        # z-pass (latent space): student samples, teacher must map back to noise.
        noise = torch.randn(x.shape[0], features, dtype=x.dtype, device=x.device) * width
        pts, log_student_pre, s_steps_z, s_p_z = guided_cascade(
            student.flows, noise, c, "forward", return_steps=True, return_p=True
        )
        z_teacher, log_teacher_pre, t_steps_z, t_p_z = guided_cascade(
            teacher.flows, pts, c, "inverse", return_steps=True, return_p=True
        )
        loss_chi_z = chi2_loss(z_teacher, noise)
        for i, step in enumerate(s_steps_z):
            loss_chi_z = loss_chi_z + chi2_loss(step, t_steps_z[-i - 1])
        for i, p in enumerate(s_p_z):
            loss_chi_z = loss_chi_z + chi2_loss(p, t_p_z[-i - 1])

        # KL (monitored only): the teacher term is detached, never optimized.
        log_student = logabsdet_of_base(noise, width=width) - log_student_pre
        log_teacher = logabsdet_of_base(z_teacher, width=width) + log_teacher_pre
        kl = (log_student - log_teacher.detach()).mean()

        loss = (1.0 - beta) * loss_chi_z + beta * loss_chi_x
        return loss_chi_x, loss_chi_z, kl, loss

    def training_step(self, batch, batch_idx):
        if self.student_mode:
            x, c = self._prepare_student_batch(batch)
            loss_chi_x, loss_chi_z, kl, loss = self._pdd_losses(x, c)
            self.log(
                "train_loss", loss.detach(),
                on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0],
            )
            self.log(
                "train_loss_chi_x", loss_chi_x.detach(),
                on_step=True, on_epoch=True, batch_size=x.shape[0],
            )
            self.log(
                "train_loss_chi_z", loss_chi_z.detach(),
                on_step=True, on_epoch=True, batch_size=x.shape[0],
            )
            self.log(
                "train_KL", kl.detach(),
                on_step=True, on_epoch=True, batch_size=x.shape[0],
            )
            return {"loss": loss, "batch_size": x.shape[0]}
        x, c = self._prepare_flow_batch(batch)
        loss = -self.active_flow.log_prob(x, c).mean()
        return {"loss": loss, "batch_size": x.shape[0]}

    def on_train_batch_end(self, outputs: Dict[str, torch.Tensor], *args, **kwargs) -> None:
        if self.student_mode:
            return  # PDD metrics are logged from training_step
        loss = outputs["loss"]
        flow_dim = self.active_flow.q0.d
        self.log(
            "train_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=outputs["batch_size"],
        )
        self.log(
            "train_logprob_per_dim",
            loss / flow_dim,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=outputs["batch_size"]
        )

    def validation_step(self, batch, batch_idx):
        if self.student_mode:
            self.active_student.eval()
            x, c = self._prepare_student_batch(batch)
            with torch.no_grad():
                _, _, kl, loss = self._pdd_losses(x, c)
            self.log(
                "val_loss", loss,
                on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0],
            )
            self.log(
                "val_KL", kl,
                on_step=False, on_epoch=True, batch_size=x.shape[0],
            )
            return {"loss": loss, "batch_size": x.shape[0]}
        x, c = self._prepare_flow_batch(batch)
        loss = -self.active_flow.log_prob(x, c).mean()

        # Encode to the latent (base-space) variable so PlotLatentCallback can
        # histogram a few dimensions against the standard normal: for a well
        # trained flow the base-space z should be ~ N(0, 1).  `inverse` is the
        # parallel (density) direction, so this is cheap.  We return the FULL
        # latent; the callback selects the columns to plot.
        z = self.active_flow.inverse(x, c).detach()

        return {"loss": loss.detach(), "batch_size": x.shape[0], "z": z}

    def on_validation_batch_end(self, outputs: Dict[str, torch.Tensor], *args, **kwargs) -> None:
        if self.student_mode:
            return  # PDD metrics are logged from validation_step
        loss = outputs["loss"]
        flow_dim = self.active_flow.q0.d
        self.log(
            "val_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=outputs["batch_size"],
        )
        self.log(
            "val_logprob_per_dim",
            loss / flow_dim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=outputs["batch_size"]
        )
        

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _load_flow_submodule(self, submodule, ckpt_path, prefix):
        """Load one flow's weights from a per-flow Lightning training checkpoint.

        Per-flow checkpoints hold the whole module state dict; keep only the
        keys under ``prefix`` (e.g. ``flow1.``) and strip it. Strict so a
        wrong checkpoint/architecture fails loudly instead of loading nothing.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        sub_state = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        submodule.load_state_dict(sub_state, strict=True)

    def load_flows_for_generation(
        self,
        flow1_path=None,
        flow2_path=None,
        flow3_path=None,
        flow2_student_path=None,
        flow3_student_path=None,
    ):
        """Apply each per-flow checkpoint to the matching submodule.

        Raises if any required checkpoint is missing, so a generation run
        never silently proceeds with a randomly-initialized flow.  In
        student_mode the IAF student checkpoints (``flow2_student.`` /
        ``flow3_student.`` prefixes) are loaded too, enabling fast parallel
        sampling through the students.

        NOT inference-mode: loading state into the flows must keep their
        parameters writable so a later ``load_state_dict`` (e.g. from
        ``LightningModule.load_from_checkpoint`` on a wrapper that embeds
        this module) can still copy into them.
        """
        for flow, path, prefix in (
            (self.flow1, flow1_path, "flow1."),
            (self.flow2, flow2_path, "flow2."),
            (self.flow3, flow3_path, "flow3."),
        ):
            if path is None:
                raise ValueError(
                    f"Generation requires a checkpoint for the "
                    f"{prefix.rstrip('.')} flow; got None"
                )
            if flow is None:
                raise ValueError(
                    f"Cannot load {path}: {prefix.rstrip('.')} flow is not built"
                )
            self._load_flow_submodule(flow, path, prefix)
            flow.to(self.model_dtype)
        if self.student_mode:
            for flow, path, prefix in (
                (self.flow2_student, flow2_student_path, "flow2_student."),
                (self.flow3_student, flow3_student_path, "flow3_student."),
            ):
                if path is None:
                    raise ValueError(
                        f"Generation requires a checkpoint for the "
                        f"{prefix.rstrip('.')} flow; got None"
                    )
                if flow is None:
                    raise ValueError(
                        f"Cannot load {path}: {prefix.rstrip('.')} flow is not built"
                    )
                self._load_flow_submodule(flow, path, prefix)
                flow.to(self.model_dtype)

    # ------------------------------------------------------------------
    # Joint latent encode / decode (HMC groundwork)
    # ------------------------------------------------------------------

    def _layer_energies(self, shower_mev: torch.Tensor) -> torch.Tensor:
        """Per-layer deposited energy (B, num_layers) MeV from layer-major flat showers."""
        lb = [int(v) for v in self.layer_boundaries]
        return torch.stack(
            [shower_mev[:, lo:hi].sum(dim=1) for lo, hi in zip(lb[:-1], lb[1:])],
            dim=1,
        )

    def _layer_energy_slices(self):
        lb = [int(v) for v in self.layer_boundaries]
        return list(zip(lb[:-1], lb[1:]))

    @staticmethod
    def _voxel_logit(layer: torch.Tensor, noise_level: float = 0.0) -> torch.Tensor:
        """Unit-normalize a layer's voxels then logit-transform (with optional noise)."""
        if noise_level > 0:
            layer = layer + torch.rand_like(layer) * noise_level
        return logit_trafo(layer / layer.sum(dim=1, keepdim=True))

    @staticmethod
    def _select_flow(use_student, student, teacher):
        """Return the student flow when requested and loaded, else the teacher."""
        return student if (use_student and student is not None) else teacher

    def _cond_inc(self, energy_mev: torch.Tensor) -> torch.Tensor:
        """log10(E_inc / energy_ref_mev) condition, cast to the flow dtype."""
        dtype = next(self.flow1.parameters()).dtype
        return log10_energy(energy_mev.to(dtype=dtype), self.energy_ref_mev)

    # ------------------------------------------------------------------
    # Per-flow encode / decode (individual flows can be used directly)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_flow1(
        self,
        layer_energy_mev: torch.Tensor,
        energy_mev: torch.Tensor,
        noise_level: float = 0.0,
    ) -> torch.Tensor:
        """Flow-1: layer energies (B, num_layers) MeV -> z1 latent (B, num_layers)."""
        dtype = next(self.flow1.parameters()).dtype
        norm = float(self.normalization_constant)
        c_inc = self._cond_inc(energy_mev)
        E = layer_energy_mev.to(dtype=dtype)
        if noise_level > 0:
            E = E + torch.rand_like(E) * noise_level
        return self.flow1.inverse(logit_trafo(E / norm), c_inc)

    @torch.inference_mode()
    def decode_flow1(
        self, z1: torch.Tensor, energy_mev: torch.Tensor
    ) -> torch.Tensor:
        """Flow-1: z1 latent (B, num_layers) -> layer energies (B, num_layers) MeV."""
        norm = float(self.normalization_constant)
        thr = float(self.threshold_cut)
        c_inc = self._cond_inc(energy_mev)
        e_dep_proxy = inverse_logit(self.flow1.forward(z1, c_inc)) * norm
        if thr > 0:
            e_dep_proxy = torch.where(
                e_dep_proxy < thr, torch.zeros_like(e_dep_proxy), e_dep_proxy
            )
        return e_dep_proxy

    @torch.inference_mode()
    def encode_flow2(
        self,
        layer0_mev: torch.Tensor,
        E0_mev: torch.Tensor,
        energy_mev: torch.Tensor,
        noise_level: float = 0.0,
        use_student: bool = False,
    ) -> torch.Tensor:
        """Flow-2: layer-0 voxels (B, cells) MeV -> z2 latent (B, cells)."""
        flow2 = self._select_flow(use_student, self.flow2_student, self.flow2)
        dtype = next(self.flow1.parameters()).dtype
        norm = float(self.normalization_constant)
        c_inc = self._cond_inc(energy_mev)
        E0 = E0_mev.to(dtype=dtype)
        if noise_level > 0:
            E0 = E0 + torch.rand_like(E0) * noise_level
        cond2 = torch.cat([c_inc, logit_trafo(E0 / norm) / 4.0], dim=1)
        return flow2.inverse(
            self._voxel_logit(layer0_mev.to(dtype=dtype), noise_level), cond2
        )

    @torch.inference_mode()
    def decode_flow2(
        self,
        z2: torch.Tensor,
        E0_mev: torch.Tensor,
        energy_mev: torch.Tensor,
        use_student: bool = False,
    ) -> torch.Tensor:
        """Flow-2: z2 latent (B, cells) -> layer-0 voxels (B, cells) MeV."""
        flow2 = self._select_flow(use_student, self.flow2_student, self.flow2)
        dtype = next(self.flow1.parameters()).dtype
        norm = float(self.normalization_constant)
        thr = float(self.threshold_cut)
        c_inc = self._cond_inc(energy_mev)
        E0 = E0_mev.to(dtype=dtype)
        cond2 = torch.cat([c_inc, logit_trafo(E0 / norm) / 4.0], dim=1)
        layer0 = inverse_logit(flow2.forward(z2, cond2)) * E0
        if thr > 0:
            layer0 = torch.where(layer0 < thr, torch.zeros_like(layer0), layer0)
        return layer0

    @torch.inference_mode()
    def encode_flow3(
        self,
        layer_i_mev: torch.Tensor,
        layer_prev_mev: torch.Tensor,
        E_i_mev: torch.Tensor,
        E_prev_mev: torch.Tensor,
        i: int,
        energy_mev: torch.Tensor,
        noise_level: float = 0.0,
        use_student: bool = False,
    ) -> torch.Tensor:
        """Flow-3: layer-i voxels (B, cells) MeV -> z3_i latent (B, cells).

        ``i`` is the 1-based layer number (1..num_layers-1).  ``layer_prev_mev`` /
        ``E_prev_mev`` are the previous layer's voxels and energy.
        """
        flow3 = self._select_flow(use_student, self.flow3_student, self.flow3)
        dtype = next(self.flow1.parameters()).dtype
        norm = float(self.normalization_constant)
        b = layer_i_mev.shape[0]
        device = layer_i_mev.device
        c_inc = self._cond_inc(energy_mev)
        E_i = E_i_mev.to(dtype=dtype)
        E_prev = E_prev_mev.to(dtype=dtype)
        cond3 = torch.cat(
            [
                c_inc,
                logit_trafo(E_i / norm),
                logit_trafo(E_prev / norm),
                self._voxel_logit(layer_prev_mev.to(dtype=dtype), noise_level),
                one_hot(
                    torch.full((b,), i, dtype=torch.long, device=device),
                    self.num_layers - 1,
                ),
            ],
            dim=1,
        )
        return flow3.inverse(
            self._voxel_logit(layer_i_mev.to(dtype=dtype), noise_level), cond3
        )

    @torch.inference_mode()
    def decode_flow3(
        self,
        z3_i: torch.Tensor,
        E_i_mev: torch.Tensor,
        E_prev_mev: torch.Tensor,
        layer_prev_mev: torch.Tensor,
        i: int,
        energy_mev: torch.Tensor,
        use_student: bool = False,
    ) -> torch.Tensor:
        """Flow-3: z3_i latent (B, cells) -> layer-i voxels (B, cells) MeV.

        ``i`` is the 1-based layer number (1..num_layers-1).  The condition's
        previous-layer term is noise-augmented (``self.noise_level``) like the
        reference generation, with an all-zero guard.
        """
        flow3 = self._select_flow(use_student, self.flow3_student, self.flow3)
        dtype = next(self.flow1.parameters()).dtype
        norm = float(self.normalization_constant)
        thr = float(self.threshold_cut)
        b = z3_i.shape[0]
        c_inc = self._cond_inc(energy_mev)
        E_i = E_i_mev.to(dtype=dtype)
        E_prev = E_prev_mev.to(dtype=dtype)
        prev = layer_prev_mev.to(dtype=dtype)
        noisy = add_noise(prev, self.noise_level)
        noisy_sum = noisy.sum(dim=1, keepdim=True)
        noisy_sum = torch.where(
            noisy_sum <= 0, torch.ones_like(noisy_sum), noisy_sum
        )
        cond3 = torch.cat(
            [
                c_inc,
                logit_trafo(E_i / norm),
                logit_trafo(E_prev / norm),
                logit_trafo(noisy / noisy_sum),
                one_hot(
                    torch.full((b,), i, dtype=torch.long, device=z3_i.device),
                    self.num_layers - 1,
                ),
            ],
            dim=1,
        )
        layer = inverse_logit(flow3.forward(z3_i, cond3)) * E_i
        if thr > 0:
            layer = torch.where(layer < thr, torch.zeros_like(layer), layer)
        return layer

    @torch.inference_mode()
    def encode_joint(
        self,
        shower_mev: torch.Tensor,
        energy_mev: torch.Tensor,
        noise_level: float = 0.0,
        use_student: bool = False,
    ) -> torch.Tensor:
        """Joint-encode full showers into the concatenated latent.

        ``shower_mev`` is (B, num_layers * cells_per_layer) layer-major flat MeV;
        ``energy_mev`` is (B, 1) MeV.  Returns (B, num_dim) float32 ``z`` =
        ``[z1 | z2 | z3_1 ... z3_{num_layers-1}]``.

        If ``noise_level > 0``, uniform ``U[0, noise_level]`` MeV noise is added to
        the layer energies and voxels *before* preprocessing, matching the reference
        training dataloader.  Empirically this puts the encoded truth latents inside
        the flow's calibrated regime and collapses the heavy empty-layer latent tail
        (see the HMC design plan); ``decode_joint`` is always noise-free so it stays
        the deterministic inverse of the generative map.

        ``use_student=True`` encodes through the IAF students (Flow-2/Flow-3) when
        loaded, enabling a self-consistent ``decode_joint(z, ..., use_student=True)
        o encode_joint(..., use_student=True)`` round-trip for quantifying the
        student's own invertibility.  Student encoding runs the sequential (slow)
        density direction, so the teacher (``use_student=False``, default) is used
        for fast encoding.  Flow-1 always encodes with the teacher (no student).

        This is the per-flow composition of ``encode_flow1``/``encode_flow2``/
        ``encode_flow3``.
        """
        if self.flow1 is None or self.flow2 is None or self.flow3 is None:
            raise ValueError("encode_joint requires all three flows (which_flow=7)")
        if self.layer_boundaries is None:
            raise ValueError("encode_joint requires layer_boundaries")
        if self.normalization_constant is None:
            raise ValueError("encode_joint requires normalization_constant")
        dtype = next(self.flow1.parameters()).dtype
        b = shower_mev.shape[0]
        s = shower_mev.to(dtype=dtype)
        E = self._layer_energies(s)  # (B, num_layers) MeV

        z1 = self.encode_flow1(E, energy_mev, noise_level)

        lo0, hi0 = self._layer_energy_slices()[0]
        z2 = self.encode_flow2(s[:, lo0:hi0], E[:, 0:1], energy_mev,
                               noise_level, use_student)

        z3_blocks = []
        slices = self._layer_energy_slices()
        for i in range(1, self.num_layers):
            loi, hii = slices[i]
            lop, hip = slices[i - 1]
            z3_blocks.append(self.encode_flow3(
                s[:, loi:hii], s[:, lop:hip], E[:, i:i + 1], E[:, i - 1:i],
                i, energy_mev, noise_level, use_student,
            ))
        z3 = torch.cat(z3_blocks, dim=1)

        z = torch.cat([z1, z2, z3], dim=1)
        return z.to(dtype=torch.float32)

    @torch.inference_mode()
    def _sample_latent(self, b: int, c_inc: torch.Tensor) -> torch.Tensor:
        """Draw a joint latent from the base distributions (B, num_dim) float64."""
        dtype = next(self.flow1.parameters()).dtype
        device = c_inc.device
        n2 = self.cells_per_layer
        z1 = torch.randn(b, self.num_layers, dtype=dtype, device=device)
        z2 = torch.randn(b, n2, dtype=dtype, device=device)
        z3 = torch.randn(b, (self.num_layers - 1) * n2, dtype=dtype, device=device)
        return torch.cat([z1, z2, z3], dim=1)

    @torch.inference_mode()
    def decode_joint(
        self, z: torch.Tensor, energy_mev: torch.Tensor, use_student: bool = False
    ) -> torch.Tensor:
        """Decode a joint latent (B, num_dim) into a layer-major flat MeV shower.

        Deterministic generative map: splits ``z`` into the per-flow blocks and
        runs Flow-1 -> E~_i, Flow-2 -> layer 0, Flow-3 -> layers 1..num_layers-1,
        mirroring ``predict_step``.  Returns (B, num_layers * cells_per_layer) MeV
        float32.  This is the inverse of ``encode_joint`` (for noise-free encode).
        ``use_student=True`` decodes through the IAF students (Flow-2/Flow-3) when
        loaded.  This is the per-flow composition of ``decode_flow1``/
        ``decode_flow2``/``decode_flow3``.
        """
        if self.flow1 is None or self.flow2 is None or self.flow3 is None:
            raise ValueError("decode_joint requires all three flows (which_flow=7)")
        if self.normalization_constant is None:
            raise ValueError("decode_joint requires normalization_constant")
        dtype = next(self.flow1.parameters()).dtype
        b = z.shape[0]
        z = z.to(dtype=dtype)

        n1 = self.num_layers
        n2 = self.cells_per_layer
        z1 = z[:, :n1]
        z2 = z[:, n1:n1 + n2]
        z3 = z[:, n1 + n2:]

        # Flow-1: per-layer deposited energies E~_i (MeV).
        e_dep_proxy = self.decode_flow1(z1, energy_mev)

        # Flow-2: layer-0 voxels.
        layers = [self.decode_flow2(z2, e_dep_proxy[:, 0:1], energy_mev, use_student)]

        # Flow-3: layers 1..num_layers-1, conditioned on the previous generated layer.
        for i in range(1, self.num_layers):
            layers.append(self.decode_flow3(
                z3[:, (i - 1) * n2:i * n2],
                e_dep_proxy[:, i:i + 1], e_dep_proxy[:, i - 1:i],
                layers[-1], i, energy_mev, use_student,
            ))

        # Layer-major flat (B, num_layers * cells_per_layer) MeV.
        shower_mev = torch.cat(layers, dim=1)
        return shower_mev.to(dtype=torch.float32)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.flow1 is None or self.flow2 is None or self.flow3 is None:
            raise ValueError("Generation requires all three flows (which_flow=7)")
        if self.normalization_constant is None:
            raise ValueError("normalization_constant is required for generation")
        if "energy_inc" not in batch:
            raise KeyError(
                "Generation expects the Flow-1 data view (batch key 'energy_inc'); "
                "configure the predict datamodule with which_flow=1"
            )
        t0 = time.time()
        dtype = next(self.flow1.parameters()).dtype

        c_inc = batch["energy_inc"].to(dtype=dtype)  # (B,1) log10(E_inc / energy_ref)
        b = c_inc.shape[0]
        e_inc_mev = self.energy_ref_mev * 10 ** c_inc  # (B,1) MeV

        # Sample the joint latent from the base distributions and decode.
        z = self._sample_latent(b, c_inc)
        # In student_mode, decode through the fast IAF students (Flow-2/3).
        shower_mev = self.decode_joint(z, e_inc_mev, use_student=self.student_mode)

        elapsed = time.time() - t0
        return (
            e_inc_mev.detach().cpu().numpy().astype("float32"),
            shower_mev.detach().cpu().numpy().astype("float32"),
            elapsed / b,  # per-sample, matching CaloINN's predict_step
        )


def decode_joint_chunked(
    generator: ICaloFlowLightningModule,
    z: torch.Tensor,
    energy_mev: torch.Tensor,
    batch_size: Optional[int],
    device: str | torch.device,
):
    """Decode latents to flat MeV showers, chunked to bound peak GPU memory.

    ``decode_joint`` runs the three float64 flows, so decoding a large latent
    batch in one call can exhaust GPU memory.  This helper slices the batch
    into ``batch_size``-row chunks and calls ``decode_joint`` per chunk,
    concatenating the resulting layer-major flat MeV showers.

    Parameters
    ----------
    generator : ICaloFlowLightningModule
        The 3-flow generation module exposing ``decode_joint``.
    z : torch.Tensor  shape (n, num_dim)
        Joint latents (CPU or device).  Chunks are moved to ``device``.
    energy_mev : torch.Tensor  shape (n, 1)
        Incident energies in MeV.
    batch_size : int or None
        Rows per ``decode_joint`` call; ``None`` decodes the whole batch at
        once (single call).
    device : str or torch.device
        Device on which the flows run.

    Returns
    -------
    np.ndarray  shape (n, num_layers * cells_per_layer) float32 MeV.
    """
    import numpy as np

    n = z.shape[0]
    if batch_size is None or batch_size >= n:
        with torch.inference_mode():
            return (
                generator.decode_joint(z.to(device), energy_mev.to(device))
                .cpu()
                .numpy()
            )

    parts = []
    with torch.inference_mode():
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            parts.append(
                generator.decode_joint(
                    z[start:stop].to(device), energy_mev[start:stop].to(device)
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(parts, axis=0)
