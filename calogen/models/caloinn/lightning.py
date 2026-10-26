import os
import time
from collections.abc import Mapping
from functools import partial
from typing import Any, Optional, Union

import h5py
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import brier_score_loss
from torchmetrics import AUROC, MetricCollection

from calogen.core import data_util
from calogen.core import postprocess as torch_postprocess
from calogen.core.diagnostics import TrainingDiagnostics
from calogen.core.plotting import plot_latent_histo
from calogen.models.base import BaseGenerativeModule
from calogen.models.caloinn.cinn import CINN


class _SkipLastTwoScheduler:
    """Wrapper that skips the last 2 scheduler.step() calls, matching legacy.

    Trainer.py lines 150-153:
        if i < self.scheduler.total_steps-2:
            self.scheduler.step()
        else:
            pass

    This prevents OneCycleLR from reaching its terminal annealing, freezing
    the LR at the penultimate value for the final 2 optimizer updates.
    """
    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._total = getattr(scheduler, "total_steps", 0)
        self._step_count = 0

    def step(self):
        if self._step_count < self._total - 2:
            self._scheduler.step()
        self._step_count += 1

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, sd):
        self._scheduler.load_state_dict(sd)

    def get_last_lr(self):
        return self._scheduler.get_last_lr()

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


class CaloINNLightningModule(BaseGenerativeModule):
    """Lightning module that preserves the CINN training logic."""

    plugin_name = "caloinn"

    def __init__(
        self,
        setup_data_sample_path=None,
        enable_diagnostics=False,
        max_init_samples=10000,
        shuffle_init_batch=True,
        cinn_params={},
        width_noise=1e-7,
        optimizer_params=None,
        scheduler_params=None,
        # Phase 2: accept preprocessed data arrays directly, bypassing file read
        init_data_x=None,
        init_data_c=None,
        init_layer_boundaries=None,
        max_samples=None,
        actnorm_calibration_samples=None,
        subtract_predict_noise: bool=True,
        # XML configuration (needed for data loading and HLF)
        xml_path=None,
        xml_ptype=None,
        # Data preprocessing params (used by load_init_tensors)
        dataset_params=None,
        custom_noise=False,
        inspect_dimensions: list[int] = [0, 1, 2],
        num_noise_doses: int = 1,
        **kwargs
    ):
        super().__init__()

        if max_samples is not None:
            max_init_samples = max_samples
        if actnorm_calibration_samples is None:
            actnorm_calibration_samples = max_init_samples

        self.cinn_params = cinn_params
        self.optimizer_params = dict(optimizer_params or {})
        self.scheduler_params = dict(scheduler_params or {})
        self.width_noise = width_noise
        self.batch_number = 0
        self._run_batch_diagnostics = False
        self._diag_batch_idx = -1
        self.diagnostics = TrainingDiagnostics(enabled=enable_diagnostics)
        self.xml_path = xml_path
        self.xml_ptype = xml_ptype
        self.setup_data_sample_path = setup_data_sample_path
        self.custom_noise = custom_noise
        self.inspect_dimensions = inspect_dimensions
        self.save_hyperparameters()
        self.subtract_predict_noise = subtract_predict_noise

        # Phase 2: Support direct data arrays from DataModule, bypassing
        # redundant HDF5 file reads. Legacy path (via setup_data_sample_path)
        # is kept for backwards compatibility.
        if init_data_x is not None and init_data_c is not None:
            sample_x = init_data_x
            sample_c = init_data_c
            self.layer_boundaries = init_layer_boundaries
        else:
            assert setup_data_sample_path is not None, (
                "Must provide either setup_data_sample_path or init_data_x/init_data_c"
            )
            assert setup_data_sample_path.endswith((".hdf5", ".h5")), (
                "setup_data_sample_path must be an .hdf5 or .h5 file path"
            )
            sample_x, sample_c, self.layer_boundaries = self.load_init_tensors()
        self.num_dim = int(sample_x.shape[1])

        # max_init_samples — randomly subsample init data to match
        # legacy behavior where only max_samples are used for CINN
        # initialization (trainer.py lines 62-64).  When load_init_tensors
        # already returns exactly max_init_samples rows (LEMURS path), this
        # block is a no-op.  It remains active for CaloChallenge paths that
        # still load the full dataset.
        n_full_init = int(sample_x.shape[0])
        if max_init_samples is not None and max_init_samples > 0 and max_init_samples < n_full_init:
            torch.manual_seed(42)  # deterministic subsample (legacy uses randperm)
            rand_idx = torch.randperm(n_full_init)[: int(max_init_samples)]
            sample_x = sample_x[rand_idx]
            sample_c = sample_c[rand_idx]

        n_calib = min(int(actnorm_calibration_samples), int(sample_x.shape[0]))
        self._actnorm_calib_x = sample_x[:n_calib].detach()
        self._actnorm_calib_c = sample_c[:n_calib].detach()

        self.num_train_samples = max(1, n_full_init)

        if self.custom_noise:
            q = self.eval_quantiles(torch.clone(sample_x))
        else:
            q = torch.tensor(self.width_noise, dtype=sample_x.dtype)

        self.register_buffer("q", q)

        rank_zero_info(f"Initializing CINN with \nx={sample_x}, \nc={sample_c}")

        self.model = CINN(self.cinn_params, sample_x, sample_c)

    def load_init_tensors(self):

        path = self.setup_data_sample_path
        max_init = self.hparams.get("max_init_samples", 10000)
        shuffle = self.hparams.get("shuffle_init_batch", True)
        rank_zero_info(f"Loading sample data from {path} to initialize model parameters...")

        with h5py.File(path, "r") as f:
            is_lemurs = "incident_energy" in f

        if is_lemurs:
            from calogen.core.xml_handler import XMLHandler
            from calogen.data.data_util import _build_data_dict
            from calogen.data.sources import HDF5Source

            xml_handler = XMLHandler(
                particle_name=self.xml_ptype,
                filename=self.xml_path,
            )
            layer_boundaries = np.unique(xml_handler.GetBinEdges())

            source = HDF5Source(path)
            n = len(source)
            n_sample = min(max_init, n)
            if shuffle:
                rng = np.random.RandomState(42)
                idx = rng.choice(n, size=n_sample, replace=False)
            else:
                idx = np.arange(n_sample)
            showers, energies = source.read_rows(idx)
            showers = source.truth_format.flatten_showers(showers)
            energies = source.truth_format.format_energy(energies)
            
            source.close()

            sample_data = _build_data_dict(showers, energies, layer_boundaries)
        else:
            sample_data, layer_boundaries = data_util.load_data(
                path,
                self.xml_ptype,
                self.xml_path,
            )

        ds_params = self.hparams.get("dataset_params") or {}
        x, c = data_util.preprocess(
            sample_data,
            layer_boundaries,
            ds_params.get("eps", 1.0e-10),
            u0up_cut=ds_params.get("u0up_cut", 7.0),
            u0low_cut=ds_params.get("u0low_cut", 0.0),
            rew=ds_params.get("pt_rew", 1.0),
            dep_cut=ds_params.get("dep_cut", 1.0e10),
        )

        for _ in range(self.hparams.get("num_noise_doses", 1)):
            x = data_util.add_noise(x, self.width_noise)

        dtype = torch.get_default_dtype()
        x = torch.tensor(x, dtype=dtype)
        c = torch.tensor(c, dtype=dtype)
        
        return x, c, layer_boundaries

    def setup(self, stage: str):
        # Phase 2: num_train_samples is set during __init__ from data size.
        # If a Lightning datamodule is attached, prefer its count (may differ
        # if DataModule uses different preprocessing filters).
        if stage == "fit":
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
                and hasattr(self.trainer.datamodule, "num_train_samples")
            ):
                self.num_train_samples = self.trainer.datamodule.num_train_samples
                self.steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                # Fallback: keep __init__ value; steps_per_epoch computed elsewhere
                self.steps_per_epoch = max(1, self.num_train_samples // 512)
            rank_zero_info(
                f"Number of training samples: {self.num_train_samples}"
            )

    def configure_optimizers(self):
        """Build optimizer + LR scheduler, matching legacy Trainer.set_optimizer.

        Phase 4: supports all five legacy scheduler types:
        - one_cycle_lr  (used by pions_odd_discrete.yaml)
        - step
        - reduce_on_plateau
        - cycle_lr
        - multi_step_lr
        """
        opt_cfg = self.optimizer_params
        sched_cfg = self.scheduler_params
        lr_sched_mode = sched_cfg.get("lr_scheduler", "one_cycle_lr")

        # ---- Optimiser (identical to legacy) ----
        optimizer = torch.optim.AdamW(
            self.model.params_trainable,
            lr=float(opt_cfg.get("lr", 0.0002)),   # legacy default: 0.0002
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-6)),    # legacy default: 1e-6
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )

        steps_per_epoch = int(
            sched_cfg.get(
                "steps_per_epoch",
                getattr(self, "steps_per_epoch", 1),
            )
        )

        # ---- LR Scheduler (dispatched by type) ----
        if lr_sched_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.get("lr_decay_epochs", 30),
                gamma=sched_cfg.get("lr_decay_factor", 0.1),
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}

        elif lr_sched_mode == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.get("factor", 0.8),
                patience=sched_cfg.get("patience", 50),
                cooldown=sched_cfg.get("cooldown", 100),
                threshold=sched_cfg.get("threshold", 5e-5),
                threshold_mode=sched_cfg.get("threshold_mode", "rel"),
                verbose=True,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }

        elif lr_sched_mode == "one_cycle_lr":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(
                    sched_cfg.get("max_lr", opt_cfg.get("lr", 1e-5) * 10.0)
                ),
                epochs=int(
                    sched_cfg.get("epochs", 1)
                    if "epochs" in sched_cfg
                    else sched_cfg.get("cycle_epochs", 1)
                ),
                steps_per_epoch=max(1, steps_per_epoch),
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif lr_sched_mode == "cycle_lr":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=float(sched_cfg.get("lr", opt_cfg.get("lr", 1e-5))),
                max_lr=float(sched_cfg.get("max_lr", opt_cfg.get("lr", 1e-5) * 10)),
                step_size_up=int(sched_cfg.get("step_size_up", 2000)),
                mode=sched_cfg.get("cycle_mode", "triangular"),
                cycle_momentum=False,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif lr_sched_mode == "multi_step_lr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_cfg.get(
                    "milestones", [2730, 8190, 13650, 27300]
                ),
                gamma=sched_cfg.get("gamma", 0.5),
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}

        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_sched_mode}")

        # Phase 3c: wrap scheduler to skip last 2 steps (matching legacy trainer.py:150-153)
        scheduler_config["scheduler"] = _SkipLastTwoScheduler(scheduler_config["scheduler"])
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def eval_quantiles(self, data):
        cp = torch.clone(data)
        cp[cp == 0] = torch.nan
        quantiles = torch.nanquantile(cp, q=0.01, dim=0).reshape(1, -1)
        return quantiles

    def get_actnorm_calibration_batch(self, max_samples=None):
        if not hasattr(self, "_actnorm_calib_x") or not hasattr(self, "_actnorm_calib_c"):
            return None
        n = self._actnorm_calib_x.shape[0]
        if max_samples is not None:
            n = min(int(max_samples), int(n))
        return self._actnorm_calib_x[:n], self._actnorm_calib_c[:n]


    def _compute_losses(self, x, c, run_diagnostics=False):
        # Diagnostics before log_prob
        log_probs = self.model.log_prob(x, c)
        if run_diagnostics:
            self.diagnostics.log_loss_diagnostics(self.model, x, c, log_probs, self.batch_number)
        
        inn_loss = -torch.mean(log_probs)
        if self.model.bayesian:
            kl_loss = self.model.get_kl() / self.num_train_samples
            loss = inn_loss + kl_loss
        else:
            kl_loss = None
            loss = inn_loss
        return loss, inn_loss, kl_loss

    def _apply_input_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add uniform noise [0, width_noise) to input tensor.

        NOTE (Phase 3): Legacy MyDataLoader already adds noise per-batch in
        __next__, so the streaming dataset (PreprocessedStreamingDataset)
        does the same in __iter__.  The training step therefore does NOT call
        this method — adding noise here would apply it *twice*.

        This method is kept for backwards-compatibility with dataloaders
        that do not add noise (e.g., a vanilla TensorDataset).
        """
        if self.width_noise <= 0:
            return x
        return x + torch.rand_like(x) * self.width_noise

    def on_fit_start(self):
        if self.model.bayesian:
            self.model.enable_map()

        rank_zero_info("Starting training...")

    def on_after_backward(self):
        # Restrict gradient diagnostics to the first batch of the first epoch.
        if self._run_batch_diagnostics:
            self.diagnostics.log_gradient_diagnostics(
                self,
                tag=f"after_backward_epoch{self.current_epoch}_batch{self._diag_batch_idx}",
            )

    def on_before_optimizer_step(self, optimizer):
        # Explicit one-time check before the very first optimizer step.
        self.diagnostics.maybe_log_before_first_optimizer_step(self, self._run_batch_diagnostics)

    # Phase 3c: _SkipLastTwoScheduler wrapper needs explicit lr_scheduler_step
    # because Lightning's _validate_scheduler_api doesn't recognise it.
    def lr_scheduler_step(self, scheduler, metric=None):
        scheduler.step()

    @staticmethod
    def _alert_non_finite_loss(loss, x, c, batch_idx, phase):
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite {phase} loss at batch {batch_idx}: "
                f"loss={loss.detach().cpu().item()}, x_min={x.min().detach().cpu().item()}, "
                f"x_max={x.max().detach().cpu().item()}, "
                f"c_min={c.min().detach().cpu().item()}, c_max={c.max().detach().cpu().item()}"
            )

    def training_step(self, batch, batch_idx):
        x, c = batch
        self.batch_number += 1
        self._run_batch_diagnostics = self.diagnostics.should_run_batch(self.current_epoch, batch_idx)
        self._diag_batch_idx = int(batch_idx)
        if self._run_batch_diagnostics:
            self.diagnostics.log_first_block_inside_ratio(self.model, x, c, tag=f"epoch{self.current_epoch}_batch{batch_idx}")
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=self._run_batch_diagnostics)

        self._alert_non_finite_loss(loss, x, c, batch_idx, phase="training")

        return {
            "loss": loss
        }

    def on_training_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        loss = outputs["loss"]
        x, c = batch
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("train_loss_per_dim", loss.detach() / x.shape[1], on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_start(self) -> None:
        self.validation_inspect_latent = []

    def validation_step(self, batch, batch_idx):
        x, c = batch
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=False)

        z = self.model(x, c)[0][..., self.inspect_dimensions].half()

        self._alert_non_finite_loss(loss, x, c, batch_idx, phase="validation")

        return {
            "loss": loss.detach(),
            "z": z.detach(),
        }

    def on_validation_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:

        loss = outputs["loss"]
        x, c = batch
        z = outputs["z"]
        self.validation_inspect_latent.append(z)
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        self.log("val_loss_per_dim", loss.detach() / x.shape[1], on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch
        if len(self.validation_inspect_latent) > 0:
            self.validation_inspect_latent = torch.cat(self.validation_inspect_latent, dim=0).cpu().numpy()
            if self.trainer is not None and getattr(self.trainer, "log_dir") is not None:
                save_path = os.path.join(self.trainer.log_dir, f"latent_hist_epoch{current_epoch}.png")
            else:
                save_path = f"./latent_hist_epoch{current_epoch}.png"
            plot_latent_histo(
                self.validation_inspect_latent,
                save_path=save_path,
                latent_variables=self.inspect_dimensions,
                bins=100,
                xrange=(-4, 4),
            )
            rank_zero_info(f"📊 Saved latent histogram to {save_path}")

        self.validation_inspect_latent = []

    def on_test_epoch_start(self) -> None:
        self.test_inspect_latent = []
        
    def test_step(self, batch, batch_idx):
        x, c = batch
        # Phase 3: noise is in dataloader, not here
        loss, inn_loss, kl_loss = self._compute_losses(x, c, run_diagnostics=False)
        z = self.model(x, c)[0][..., self.inspect_dimensions].half()

        self._alert_non_finite_loss(loss, x, c, batch_idx, phase="test")
        return {"loss": loss, "z": z.detach()}

    def on_test_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss = outputs["loss"]
        x, c = batch
        z = outputs["z"]
        self.test_inspect_latent.append(z)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("test_loss_per_dim", loss / x.shape[1], on_step=False, on_epoch=True, batch_size=x.shape[0])

    def on_test_epoch_end(self) -> None:
        if len(self.test_inspect_latent) > 0:
            self.test_inspect_latent = torch.cat(self.test_inspect_latent, dim=0).cpu().numpy()
            if self.trainer is not None and getattr(self.trainer, "log_dir") is not None:
                save_path = os.path.join(self.trainer.log_dir, f"latent_hist_test_epoch{self.current_epoch}.png")
            else:
                save_path = f"./latent_hist_test_epoch{self.current_epoch}.png"
            plot_latent_histo(
                self.test_inspect_latent,
                save_path=save_path,
                latent_variables=self.inspect_dimensions,
                bins=100,
                xrange=(-4, 4),
            )
            rank_zero_info(f"📊 Saved test latent histogram to {save_path}")

        self.test_inspect_latent = []

    def on_predict_start(self):
        rank_zero_info("Starting prediction...")
        # seed with current time for variability across runs; legacy code does not set a seed at all before generation.
        torch.manual_seed(int(time.time()))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Generate once per predict run; Lightning still requires a predict dataloader.
        x, c = batch

        samples, t = self.conditional_generate(c, measure_gen_time=True)
        # self.log("predict_gen_time", t / c.shape[0], on_step=False, on_epoch=True, prog_bar=True, batch_size=c.shape[0])
        if self.subtract_predict_noise:
            samples = samples - self.width_noise * self.hparams.get("num_noise_doses", 1)  # Remove noise added during preprocessing
        samples = samples[:, 0, ...]
        data = torch_postprocess.postprocess(
            samples,
            c,
            layer_boundaries=self.layer_boundaries,
            quantiles=self.q,
        )

        number_of_layers = len(self.layer_boundaries) - 1
        incident_energies = data["energy"]
        layers = [data[f"layer_{i}"] for i in range(number_of_layers)]
        assert torch.allclose(incident_energies, c, rtol=0, atol=1e-3), "Mismatch between input energies and postprocessed energies"
        incident_energies = incident_energies * 1.e3

        shower = torch.cat(layers, dim=1) * 1.e3

        return incident_energies.detach().cpu().numpy().astype(np.float32), shower.detach().cpu().numpy().astype(np.float32), t / c.shape[0]


    @torch.inference_mode()
    def conditional_generate(
        self,
        incident_energies: torch.Tensor,
        measure_gen_time=False,
    ):
        self.model.eval()
        device = self.device

        t1 = time.time()
        samples = self.model.sample(1, incident_energies.to(device))
        t_diff = time.time() - t1

        if measure_gen_time:
            return samples, t_diff
        return samples

    # ------------------------------------------------------------------
    #  Standalone generation / encoding / decoding methods
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_single_energy(self, energy, num_samples, batch_size=1000, output_file=None):
        """Generate samples at a single incident energy (ds1-style).

        Parameters
        ----------
        energy : float
            log2 energy value (e.g., 16 → 2^16 MeV ≈ 65.5 GeV).
        num_samples : int
            Number of samples to generate.
        batch_size : int
            Generation batch size (controls peak GPU memory).
        output_file : str, optional
            If given, save generated showers to this HDF5 path.

        Returns
        -------
        dict
            Postprocessed data with ``"energy"`` and ``"layer_*"`` keys.
        """
        self.model.eval()
        device = self.device

        energies = torch.full((num_samples, 1), 2 ** energy / 1e3, device=device)

        samples = torch.zeros((num_samples, 1, self.num_dim), device=device)
        for batch in range((num_samples + batch_size - 1) // batch_size):
            start = batch_size * batch
            stop = min(batch_size * (batch + 1), num_samples)
            samples[start:stop] = self.model.sample(1, energies[start:stop])

        samples = samples[:, 0, ...]
        samples = samples - self.width_noise * self.hparams.get("num_noise_doses", 1)  # Remove noise added during preprocessing

        data = torch_postprocess.postprocess(
            samples, energies,
            layer_boundaries=self.layer_boundaries,
            quantiles=self.q,
        )

        # Preserve the legacy NumPy dict return type for downstream callers.
        data_np = {k: v.detach().cpu().numpy() for k, v in data.items()}

        if output_file is not None:
            truth_format = getattr(self.trainer.datamodule, "truth_format", None)
            if truth_format is not None:
                data_util.save_data_with_format(data_np, output_file, truth_format)
            else:
                data_util.save_data(data_np, filename=output_file)

        return data_np

    @torch.inference_mode()
    def mcmc_sample(
        self,
        energy,
        n_chains=100,
        n_steps=500,
        burn_in=10,
        thin=1,
        classifier_ckpt=None,
        calibrator_path=None,
        log_transform=False,
        voxel_energy_cutoff=None,
        seed=None,
        output_file=None,
    ):
        """MCMC-corrected sampling via classifier density ratio.

        Runs Independent Metropolis-Hastings chains that use a trained
        classifier to reweight CINN proposals toward the true Geant4
        distribution.  Requires a trained classifier checkpoint and a
        fitted temperature calibrator.

        Parameters
        ----------
        energy : float
            log2 energy value (e.g., 16 → 2^16 MeV ≈ 65.5 GeV).
        n_chains : int
            Number of independent Markov chains.
        n_steps : int
            Total MH steps per chain (including burn-in).
        burn_in : int
            Initial steps to discard (default 10; IMH needs little burn-in).
        thin : int
            Keep every ``thin``-th post-burn-in sample (default 1 = no thinning).
        classifier_ckpt : str
            Path to the trained ``MLPClassifier`` Lightning checkpoint (``.ckpt``).
        calibrator_path : str
            Path to the fitted temperature calibrator (``.json``).
        log_transform : bool
            If True, apply ``log1p`` to energy-normalised cells before
            feeding to the classifier.  Must match the classifier's
            training config.
        voxel_energy_cutoff : float, optional
            Zero out cells below this MeV threshold before HLF computation.
            Must match the classifier's training config.
        seed : int, optional
            Random seed for reproducibility.
        output_file : str, optional
            If given, save MCMC-corrected showers to this HDF5 path.

        Returns
        -------
        dict
            Postprocessed data with ``"energy"`` and ``"layer_*"`` keys,
            same format as ``generate_single_energy``.
        """
        if classifier_ckpt is None:
            raise ValueError("classifier_ckpt is required for MCMC sampling.")
        if calibrator_path is None:
            raise ValueError("calibrator_path is required for MCMC sampling.")

        from functools import partial

        from mcmc.calibration import TemperatureCalibrator
        from mcmc.classifier import ClassifierWrapper
        from mcmc.convert import cinn_sample_to_classifier_input
        from mcmc.sampler import IMHSampler

        device = self.device
        energy_gev = 2 ** energy / 1e3  # log2 energy → GeV

        # -- Load classifier ------------------------------------------------
        rank_zero_info(f"Loading classifier from {classifier_ckpt}")
        classifier = ClassifierWrapper.load_from_checkpoint(
            classifier_ckpt, device=device
        )

        # -- Load calibrator ------------------------------------------------
        rank_zero_info(f"Loading calibrator from {calibrator_path}")
        calibrator = TemperatureCalibrator.load(calibrator_path)

        # -- Build conversion closure (binds module-level params) -----------
        convert_fn = partial(
            cinn_sample_to_classifier_input,
            layer_boundaries=self.layer_boundaries,
            q=self.q,
            width_noise=self.width_noise * self.hparams.get("num_noise_doses", 1),
            xml_path=self.hparams.xml_path,
            particle=self.hparams.xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff,
        )

        # -- Run MCMC -------------------------------------------------------
        rank_zero_info(
            f"MCMC: {n_chains} chains × {n_steps} steps, "
            f"burn_in={burn_in}, thin={thin}, energy={energy_gev:.2f} GeV"
        )
        sampler = IMHSampler(
            model=self.model,
            classifier=classifier,
            calibrator=calibrator,
            conversion_fn=convert_fn,
            device=device,
        )

        result = sampler.sample(
            energy_gev=energy_gev,
            n_chains=n_chains,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )

        # -- Postprocess ----------------------------------------------------
        samples_np = result["samples"]           # (n_kept, 730)
        energies_np = np.full(
            (samples_np.shape[0], 1), energy_gev, dtype=np.float32
        )

        # Subtract width noise (matches generate_single_energy)
        samples = torch.from_numpy(samples_np - self.width_noise * self.hparams.get("num_noise_doses", 1)).to(device)
        energies = torch.from_numpy(energies_np).to(device)

        data = torch_postprocess.postprocess(
            samples,
            energies,
            layer_boundaries=self.layer_boundaries,
            quantiles=self.q,
        )

        # Preserve the legacy NumPy dict return type for downstream callers.
        data_np = {k: v.detach().cpu().numpy() for k, v in data.items()}

        rank_zero_info(
            f"MCMC complete: {samples.shape[0]} samples, "
            f"acceptance_rate={result['acceptance_rate']:.3f}"
        )

        if output_file is not None:
            truth_format = getattr(self.trainer.datamodule, "truth_format", None)
            if truth_format is not None:
                data_util.save_data_with_format(data_np, output_file, truth_format)
            else:
                data_util.save_data(data_np, filename=output_file)

        return data_np

    @torch.inference_mode()
    def generate_latent(self, val_data_path, output_file=None, num_samples=None,
                        batch_size=1000):
        """Encode validation showers into latent-space features and save.

        Creates a MyDataLoader from *val_data_path* (all events, no split),
        encodes each batch through the model, and saves
        ``latent_features`` + ``incident_energies`` (MeV) to HDF5.

        Parameters
        ----------
        val_data_path : str
            Path to validation HDF5 file.
        output_file : str, optional
            Path for the output HDF5 file (default: ``latent_features.hdf5``
            in the trainer's default root dir).
        num_samples : int, optional
            Cap on number of events to encode.
        batch_size : int
            Encoding batch size.

        Returns
        -------
        latent_features : np.ndarray  shape (N, latent_dim)
        incident_energies : np.ndarray  shape (N, 1), in MeV
        """
        # The streaming backend was not transferred in calogen Phase 1
        # (sharded + LEMURS only). Encode through the sharded datamodule instead.
        raise NotImplementedError(
            "generate_latent depends on the streaming backend, which is not "
            "available in calogen Phase 1."
        )

    @torch.inference_mode()
    def decode_latent(self, z, c):
        """Decode latent features to showers.

        Parameters
        ----------
        z : torch.Tensor
            Latent features.
        c : torch.Tensor
            Conditions in GeV.

        Returns
        -------
        torch.Tensor
            Decoded showers.
        """
        shower, log_prob = self.model(z, c, rev=True)
        return shower

    def post_process_decoded(self, decoded, energies):
        """Postprocess decoded showers.

        Parameters
        ----------
        decoded : torch.Tensor
            Decoded showers.
        energies : torch.Tensor
            Incident energies in GeV.

        Returns
        -------
        dict
            Postprocessed data with ``"energy"`` and ``"layer_*"`` keys.
        """
        samples = decoded - self.width_noise * self.hparams.get("num_noise_doses", 1)
        data = torch_postprocess.postprocess(
            samples, energies,
            layer_boundaries=self.layer_boundaries,
            quantiles=self.q,
        )
        return data

    @torch.inference_mode()
    def generate_from_latent(self, latent_input_path, output_file=None,
                             num_samples=None, batch_size=1000):
        """Decode latent features back to showers.

        Reads an HDF5 file containing ``latent_features`` and
        ``incident_energies`` (MeV), decodes them through the model in
        reverse, postprocesses, and optionally saves the generated
        showers.

        Parameters
        ----------
        latent_input_path : str
            HDF5 file with ``latent_features`` and ``incident_energies``.
        output_file : str, optional
            Path for the generated showers HDF5.
        num_samples : int, optional
            Cap on number of events to decode.
        batch_size : int
            Decoding batch size.

        Returns
        -------
        dict
            Postprocessed data with ``"energy"`` and ``"layer_*"`` keys.
        """
        import h5py

        self.model.eval()
        device = self.device

        with h5py.File(latent_input_path, "r") as f:
            if "latent_features" not in f or "incident_energies" not in f:
                raise ValueError(
                    "Input file must contain 'latent_features' and "
                    "'incident_energies' datasets."
                )
            latent_np = f["latent_features"][:]
            energies_np = f["incident_energies"][:]

        if latent_np.ndim != 2 or latent_np.shape[1] != self.num_dim:
            raise ValueError(
                f"Expected latent dim {self.num_dim}, got {latent_np.shape}"
            )

        if num_samples is not None and num_samples < len(latent_np):
            idx = np.random.choice(
                len(latent_np), size=num_samples, replace=False
            )
            latent_np = latent_np[idx]
            energies_np = energies_np[idx]

        energies_np = energies_np / 1e3  # MeV → GeV

        dtype = torch.get_default_dtype()
        latent = torch.tensor(latent_np, dtype=dtype)
        energies = torch.tensor(energies_np, dtype=dtype)
        n_total = latent.shape[0]

        samples = torch.zeros((n_total, 1, self.num_dim), dtype=dtype)
        for batch in range((n_total + batch_size - 1) // batch_size):
            start = batch_size * batch
            stop = min(batch_size * (batch + 1), n_total)
            z_l = latent[start:stop].to(device)
            c_l = energies[start:stop].to(device)
            decoded, _ = self.model(z_l, c_l, rev=True)
            samples[start:stop, 0] = decoded.cpu()

        samples = samples[:, 0, ...].cpu().numpy()
        energies_np = energies.cpu().numpy()
        samples -= self.width_noise * self.hparams.get("num_noise_doses", 1)  # Remove noise added during preprocessing

        data = data_util.postprocess(
            samples, energies_np,
            layer_boundaries=self.layer_boundaries,
            threshold=self.width_noise * self.hparams.get("num_noise_doses", 1),
            quantiles=self.q.detach().cpu().numpy(),
        )

        if output_file is not None:
            data_util.save_data(data, filename=output_file)

        return data