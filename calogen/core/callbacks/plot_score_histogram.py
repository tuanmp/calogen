import os

import matplotlib.pyplot as plt
import numpy as np
from lightning import Callback, LightningModule, Trainer
from lightning_fabric.utilities import rank_zero_info
from sklearn.metrics import roc_auc_score


class ScoreHistogramCallback(Callback):

    def __init__(self, bins: int = 50, xlim: tuple = (0.0, 1.0)):
        super().__init__()
        self.bins = bins
        self.xlim = xlim
        self._y_hat = []
        self._y = []

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Clear the accumulated validation scores at the start of each epoch
        self._y_hat = []
        self._y = []

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: dict | None, *args, **kwargs) -> None:
        if outputs is None:
            return
        if not isinstance(outputs, dict):
            raise ValueError("Expected outputs to be a dictionary.")
        if "y_hat" not in outputs or "y" not in outputs:
            raise ValueError("Expected outputs to contain 'y_hat' and 'y' keys.")

        y_hat = outputs["y_hat"]
        y = outputs["y"]

        self._y_hat.append(y_hat.float().detach().cpu().numpy().flatten())
        self._y.append(y.float().detach().cpu().numpy().flatten())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        if not self._y_hat or not self._y:
            rank_zero_info("ValScoreHistogramCallback: no validation scores accumulated, skipping.")
            return

        self._y_hat = np.concatenate(self._y_hat)
        self._y = np.concatenate(self._y)

        if self._y_hat.size == 0 or self._y.size == 0:
            rank_zero_info("ValScoreHistogramCallback: no validation scores accumulated, skipping.")
            return

        score_truth = self._y_hat[self._y == 1]
        score_gen = self._y_hat[self._y == 0]

        noise_multiplier = float(getattr(pl_module, "noise_amplifier", 1))
        if noise_multiplier <= 0:
            noise_multiplier = 1.0
        sample_weight = np.ones_like(self._y, dtype=float)
        sample_weight[self._y == 0] = 1.0 / noise_multiplier
        auc = roc_auc_score(self._y, self._y_hat, sample_weight=sample_weight)

        gen_label = "generated (y=0)"
        title = f"Epoch {trainer.current_epoch} — AUROC {auc:.4f}"
        if noise_multiplier != 1.0:
            gen_label = f"generated (y=0) / {noise_multiplier:.0f}"
            title += f" — M={noise_multiplier:.0f}"

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_weighted = axes[0]
        ax_weighted.hist(score_truth, bins=self.bins, range=self.xlim, alpha=0.5,
                         color="blue", label="truth (y=1)")
        ax_weighted.hist(score_gen, bins=self.bins, range=self.xlim, alpha=0.5,
                         color="red", label=gen_label,
                         weights=np.full_like(score_gen, 1.0 / noise_multiplier))
        ax_weighted.set_xlabel("score (sigmoid of logit)")
        ax_weighted.set_ylabel("weighted count")
        ax_weighted.set_xlim(self.xlim)
        ax_weighted.set_title("Counts (negatives scaled by 1/M)")
        ax_weighted.legend()

        ax_density = axes[1]
        ax_density.hist(score_truth, bins=self.bins, range=self.xlim, density=True, alpha=0.5,
                        color="blue", label="truth (y=1)")
        ax_density.hist(score_gen, bins=self.bins, range=self.xlim, density=True, alpha=0.5,
                        color="red", label="generated (y=0)")
        ax_density.set_xlabel("score (sigmoid of logit)")
        ax_density.set_ylabel("density")
        ax_density.set_xlim(self.xlim)
        ax_density.set_title("Density")
        ax_density.legend()

        fig.suptitle(title)

        save_dir = getattr(trainer, "log_dir", None)
        if save_dir is None:
            save_dir = trainer.default_root_dir
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"val_score_hist_epoch{trainer.current_epoch:03d}_auc{auc:.4f}.png"
        )
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        rank_zero_info(f"📊 Saved validation score histogram to {save_path} ")

        self._y_hat = []  # Clear the stored validation scores for the next epoch
        self._y = []  # Clear the stored validation labels for the next epoch
