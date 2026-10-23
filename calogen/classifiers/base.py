"""Model-agnostic NCE classifier base.

``NCEClassifier`` holds the generic Lightning training/validation/test loop,
metrics and optimizer setup.  ``NoiseContrastiveDensityRatioEstimator`` adds
the latent noise-contrastive density-ratio math against a base distribution
``q0`` (``p(z|c)/q(z) = M * exp(logit)``).

Model plugins supply ``_get_latent`` (and may override
``_get_input_from_latent_and_condition`` / ``get_input_from_batch``).  This
module deliberately imports no concrete generative model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss
from torchmetrics import AUROC, CalibrationError, MetricCollection


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        activate_input_projector: bool = False,
        dropout_input_layer: float = 0.0,
    ):
        super().__init__()

        assert not (batch_norm and layer_norm), "Batch normalization and layer normalization cannot be used together."

        activation_layers = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activation_layers:
            raise ValueError(f"Unsupported activation '{activation}'. Available: {sorted(activation_layers.keys())}")

        modules = []

        input_projector = nn.LazyLinear(hidden_dim)
        modules.append(input_projector)
        if activate_input_projector:
            modules.append(activation_layers[activation]())
        if dropout_input_layer > 0:
            modules.append(nn.Dropout(dropout_input_layer))

        for _ in range(num_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                modules.append(nn.LayerNorm(hidden_dim))
            modules.append(activation_layers[activation]())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*modules)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.output_layer(x)
        return x


class Classifier(pl.LightningModule):
    """Generic binary classifier Lightning loop + calibration metrics.

    Subclasses implement :meth:`get_input_from_batch`, which returns
    ``(X_input, y, auxiliary)`` where ``y=1`` marks truth and ``y=0`` marks
    generated/noise samples.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
    ):
        super().__init__()
        self.net = MLP(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
        )

        self.metrics = MetricCollection({
            "auroc": AUROC(task="binary"),
            "expected_calibration_error": CalibrationError(n_bins=50, norm="l1", task="binary"),
        })

        # Accumulators for epoch-level Brier / ECE (not batch-level)
        self._val_y_hat: list = []
        self._val_y: list = []
        self._test_y_hat: list = []
        self._test_y: list = []
        self._val_z: list = []

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.amsgrad = amsgrad

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            amsgrad=self.amsgrad,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def get_input_from_batch(self, batch):
        raise NotImplementedError("subclasses must implement get_input_from_batch")

    def criterion(self, logits, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    def training_step(self, batch, batch_idx):
        X_input, y, _ = self.get_input_from_batch(batch)
        logits = self(X_input)
        loss = self.criterion(logits, y)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.log('train_loss', outputs["loss"], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        X_input, y, z_truth = self.get_input_from_batch(batch)
        logits = self(X_input)
        loss = self.criterion(logits, y)
        return {
            "loss": loss,
            "y_hat": torch.sigmoid(logits),
            "y": y,
            "z": z_truth,
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('val_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.update(y_hat, y)
        # Accumulate for epoch-level Brier/ECE (not computed per batch)
        self._val_y_hat.append(y_hat.float().detach().cpu().numpy().flatten())
        self._val_y.append(y.float().detach().cpu().numpy().flatten())

    def on_validation_epoch_end(self) -> None:
        computed = self.metrics.compute()
        self.log("val_aucroc", computed["auroc"], on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ece', computed["expected_calibration_error"], on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        # Brier / ECE — computed once over all batches
        y_hat_all = np.concatenate(self._val_y_hat)
        y_all = np.concatenate(self._val_y)
        self.log('val_brier', brier_score_loss(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self._val_y_hat.clear()
        self._val_y.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_batch_end(self, outputs: torch.Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        y_hat = outputs["y_hat"]
        y = outputs["y"]
        self.log('test_loss', outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics.update(y_hat, y)
        # Accumulate for epoch-level Brier/ECE (not computed per batch)
        self._test_y_hat.append(y_hat.detach().cpu().numpy().flatten())
        self._test_y.append(y.detach().cpu().numpy().flatten())

    def on_test_epoch_end(self) -> None:
        computed = self.metrics.compute()
        self.log("test_aucroc", computed["auroc"], on_epoch=True, prog_bar=True, logger=True)
        self.log("test_ece", computed["expected_calibration_error"], on_epoch=True, prog_bar=True, logger=True)
        self.metrics.reset()
        y_hat_all = np.concatenate(self._test_y_hat)
        y_all = np.concatenate(self._test_y)
        self.log('test_brier', brier_score_loss(y_all, y_hat_all), on_epoch=True, prog_bar=True, logger=True)
        self._test_y_hat.clear()
        self._test_y.clear()



class NoiseContrastiveDensityRatioEstimator(Classifier):
    """Latent NCE density-ratio estimator.

    Truth latents ``z_truth = _get_latent(x, c)`` are labelled 1; ``M =
    noise_amplifier`` fresh samples from ``q0`` are labelled 0.  The trained
    classifier yields ``r(z,c) = p(z|c)/q(z) = M * exp(logit)``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.0,
        noise_amplifier: int = 1,
        activation: str = "relu",
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        **kwargs,
    ):
        self.noise_amplifier = noise_amplifier
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            amsgrad=amsgrad,
        )

    @property
    def q0(self):
        std = getattr(self, 'sigma', 1.0)
        return torch.distributions.Normal(0, std)

    @torch.inference_mode()
    def _get_latent(self, x, c):
        raise NotImplementedError("subclasses must implement _get_latent")

    @staticmethod
    def _split_batch(batch):
        """Return ``(z, c)`` from a dict or tuple batch.

        The precomputed latent store yields ``{"z": ..., "c": ...}`` batches,
        while model-specific datamodules yield ``(z, c)`` tuples.
        """
        if isinstance(batch, Mapping):
            return batch["z"], batch["c"]
        x_truth, c = batch
        return x_truth, c

    def get_input_from_batch(self, batch):
        x_truth, c = self._split_batch(batch)

        z_truth = self._get_latent(x_truth, c)
        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)

        # amplify the number of noise samples by self.noise_amplifier
        z_gen = self.q0.sample((z_truth.shape[0] * self.noise_amplifier, *z_truth.shape[1:])).to(z_truth.device)
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        X = torch.cat([z_truth, z_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(1 + self.noise_amplifier, 1)

        X_input = torch.cat([X, c], dim=1)

        return X_input, y, z_truth

    def estimate_density_ratio(self, z, c):
        """Estimate the density ratio p(z|c) / q(z) using the trained classifier."""

        logits = self(torch.cat([z, c], dim=1))
        return float(self.noise_amplifier) * torch.exp(logits)  # r(z,c) = p(z|c)/q(z) = M * exp(logit)

    def estimate_log_density(self, z, c):
        """Estimate the log density p_theta(z) = q0 * (p_theta / q0) using the trained classifier."""
        logits = self(torch.cat([z, c], dim=1))
        log_noise_amplifier = z.new_tensor(float(self.noise_amplifier)).log()
        return self.q0.log_prob(z).sum(dim=-1) + log_noise_amplifier + logits.squeeze()

    def _turn_off_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def estimate_log_density_gradient(self, z, c):
        """Estimate the gradient of the log density p_theta(z) = q0 * (p_theta / q0)."""
        z.requires_grad_(True)
        self._turn_off_grad()
        log_density = self.estimate_log_density(z, c)
        log_density.sum().backward()
        return z.grad


class PrecomputedLatentNCE(NoiseContrastiveDensityRatioEstimator):
    """Latent NCE on precomputed latents with an isotropic ``N(0, I)`` base.

    The memmap store already holds generator latents ``z``, so ``_get_latent``
    is the identity and the shared isotropic-NCE math is reused unchanged.
    """

    @torch.inference_mode()
    def _get_latent(self, z, c):
        return z

