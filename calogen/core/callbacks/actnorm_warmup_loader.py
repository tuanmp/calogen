"""Loader-based ActNorm warmup callback.

Like :class:`ActNormWarmupCallback` but pulls the calibration batch from the
train dataloader at fit start instead of a separately cached module tensor.
This enables deterministic data-dependent ActNorm initialization WITHOUT
loading/preprocessing any init data at model creation.

Runs once at ``on_fit_start`` (before the first epoch). Seeds the dataloader
draw so the warmup batch (and the resulting ActNorm statistics) is
reproducible.
"""

from typing import Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ActNormLoaderWarmupCallback(L.Callback):
    """One-shot ActNorm calibration from the train dataloader at fit start."""

    def __init__(
        self,
        enabled: bool = True,
        num_samples: int = 4096,
        deterministic_seed: Optional[int] = 2026,
        disable_batchnorm_updates: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.num_samples = int(num_samples)
        self.deterministic_seed = deterministic_seed
        self.disable_batchnorm_updates = bool(disable_batchnorm_updates)
        self._done = False

    @staticmethod
    def _count_pending_actnorm(cinn_model) -> int:
        n_pending = 0
        for mod in cinn_model.model.modules():
            if mod.__class__.__name__ == "ActNorm" and bool(getattr(mod, "init_on_next_batch", False)):
                n_pending += 1
        return n_pending

    def _get_calibration_batch(self, trainer: L.Trainer) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Pull up to ``num_samples`` rows from the train dataloader.

        Must be called inside a seeded RNG context (see ``on_fit_start``) so
        the batch (and thus the ActNorm initialization) is reproducible.
        """
        datamodule = trainer.datamodule
        if datamodule is None:
            return None

        xs, cs = [], []
        n = 0
        for x, c in datamodule.train_dataloader():
            xs.append(x)
            cs.append(c)
            n += x.shape[0]
            if n >= self.num_samples:
                break
        if not xs:
            return None
        x = torch.cat(xs, dim=0)[: self.num_samples]
        c = torch.cat(cs, dim=0)[: self.num_samples]
        return x, c

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.enabled or self._done:
            return

        n_pending = self._count_pending_actnorm(pl_module.model)
        if n_pending == 0:
            rank_zero_info("[ActNormLoaderWarmup] No pending ActNorm layers; skipping warmup.")
            self._done = True
            return

        # Seed BEFORE the dataloader draw so the warmup batch is reproducible.
        rng_ctx = torch.random.fork_rng(enabled=self.deterministic_seed is not None)
        with rng_ctx:
            if self.deterministic_seed is not None:
                torch.manual_seed(int(self.deterministic_seed))
            batch = self._get_calibration_batch(trainer)
        if batch is None:
            rank_zero_info("[ActNormLoaderWarmup] No calibration batch available; skipping warmup.")
            self._done = True
            return

        x, c = batch
        x = x.to(pl_module.device)
        c = c.to(pl_module.device)

        rank_zero_info(
            f"[ActNormLoaderWarmup] Running warmup with {n_pending} pending ActNorm layers, "
            f"x={x}\nc={c}\n"
        )

        cinn = pl_module.model
        was_training = cinn.training

        bn_modules = []
        bn_prev_states = []
        if self.disable_batchnorm_updates:
            for mod in cinn.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    bn_modules.append(mod)
                    bn_prev_states.append(mod.training)
                    mod.eval()

        cinn.train()
        try:
            rng_ctx = torch.random.fork_rng(enabled=self.deterministic_seed is not None)
            with rng_ctx:
                if self.deterministic_seed is not None:
                    torch.manual_seed(int(self.deterministic_seed))
                with torch.no_grad():
                    _ = cinn.forward(x, c, rev=False)
        finally:
            for mod, state in zip(bn_modules, bn_prev_states):
                mod.train(state)
            cinn.train(was_training)

        n_pending_after = self._count_pending_actnorm(pl_module.model)
        rank_zero_info(
            f"[ActNormLoaderWarmup] Warmup completed: pending ActNorm {n_pending} -> {n_pending_after}, "
            f"batch_size={int(x.shape[0])}, seed={self.deterministic_seed}"
        )
        self._done = True

        if hasattr(pl_module, "_actnorm_calib_x"):
            pl_module._actnorm_calib_x = None
            pl_module._actnorm_calib_c = None
