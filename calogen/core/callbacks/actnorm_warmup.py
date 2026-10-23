from typing import Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ActNormWarmupCallback(L.Callback):
    """Run a one-shot forward pass at fit start to calibrate ActNorm deterministically."""

    def __init__(
        self,
        enabled: bool = True,
        num_samples: int = 1024,
        deterministic_seed: Optional[int] = 2026,
        disable_batchnorm_updates: bool = True,
        calibration_source: str = "train_loader",
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.num_samples = int(num_samples)
        self.deterministic_seed = deterministic_seed
        self.disable_batchnorm_updates = bool(disable_batchnorm_updates)
        self.calibration_source = str(calibration_source)
        self._done = False

    @staticmethod
    def _count_pending_actnorm(cinn_model) -> int:
        n_pending = 0
        for mod in cinn_model.model.modules():
            if mod.__class__.__name__ == "ActNorm" and bool(getattr(mod, "init_on_next_batch", False)):
                n_pending += 1
        return n_pending

    def _get_calibration_batch(self, trainer: L.Trainer, pl_module: L.LightningModule) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.calibration_source == "module_cache":
            if hasattr(pl_module, "get_actnorm_calibration_batch"):
                batch = pl_module.get_actnorm_calibration_batch(max_samples=self.num_samples)
                if batch is not None:
                    return batch

        if self.calibration_source not in ("train_loader", "module_cache"):
            raise ValueError(
                "ActNormWarmupCallback.calibration_source must be 'train_loader' or 'module_cache'"
            )

        datamodule = trainer.datamodule
        if datamodule is None:
            return None

        loader = datamodule.train_dataloader()
        batch = next(iter(loader))
        x, c = batch
        return x[: self.num_samples], c[: self.num_samples]

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.enabled or self._done:
            return

        n_pending = self._count_pending_actnorm(pl_module.model)
        if n_pending == 0:
            rank_zero_info("[ActNormWarmup] No pending ActNorm layers; skipping warmup.")
            self._done = True
            return

        batch = self._get_calibration_batch(trainer, pl_module)
        if batch is None:
            rank_zero_info("[ActNormWarmup] No calibration batch available; skipping warmup.")
            self._done = True
            return

        x, c = batch
        x = x.to(pl_module.device)
        c = c.to(pl_module.device)

        rank_zero_info(f"[ActNormWarmup] Running warmup with {n_pending} pending ActNorm layers, "
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
            f"[ActNormWarmup] Warmup completed: pending ActNorm {n_pending} -> {n_pending_after}, "
            f"batch_size={int(x.shape[0])}, seed={self.deterministic_seed}"
        )
        self._done = True

        if hasattr(pl_module, "_actnorm_calib_x"):
            pl_module._actnorm_calib_x = None
            pl_module._actnorm_calib_c = None