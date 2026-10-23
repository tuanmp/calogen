import os
from typing import Any, Dict, List, Optional

import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning_fabric.utilities import rank_zero_info
from calogen.core.plotting import plot_latent_histo


class PlotLatentCallback(Callback):

    def __init__(
        self,
        bins: int = 100,
        xrange: tuple = (-4.0, 4.0),
        latent_variables: Optional[List[int]] = None,
    ):
        super().__init__()
        self.bins = bins
        self.xrange = xrange
        # Plot the first three latent dimensions by default; plotting every
        # dimension of a high-dimensional flow is too costly.
        self.latent_variables = [0, 1, 2] if latent_variables is None else latent_variables
        self._z = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._z = []

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if outputs is None:
            return
        if not isinstance(outputs, dict):
            raise ValueError("Expected outputs to be a dictionary.")
        if "z" not in outputs:
            raise ValueError("Expected outputs to contain 'z' key.")
        # The module must return the FULL latent (B, latent_dim); the callback
        # selects the columns to histogram.
        z = outputs["z"]
        self._z.append(z[:, self.latent_variables].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        if not trainer.is_global_zero:
            return

        if not self._z:
            rank_zero_info("PlotLatentCallback: no latent variables accumulated, skipping.")
            return

        self._z = np.concatenate(self._z, axis=0)

        save_dir = getattr(trainer, "log_dir", None)
        if save_dir is None:
            save_dir = trainer.default_root_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"val_latent_hist_epoch{trainer.current_epoch}.png"
        )
        plot_latent_histo(
            self._z,
            save_path=save_path,
            latent_variables=self.latent_variables,
            bins=self.bins,
            xrange=self.xrange,
        )
        rank_zero_info(f"📊 Saved latent histogram to {save_path} ")

        self._z = []  # Clear the stored latent variables for the next epoch
