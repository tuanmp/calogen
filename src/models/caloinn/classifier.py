"""CaloINN (CINN) classifier plugins.

- :class:`CaloINNCLF` — shower-space binary classifier fed by the
  high-level-feature converter (:mod:`calogen.models.caloinn.features.convert`).
- :class:`CaloINNLatentNCE` — latent-space NCE density-ratio estimator over
  the CINN latent ``z = generator.model(x, c, rev=False)``.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from calogen.classifiers import NoiseContrastiveDensityRatioEstimator
from calogen.models.caloinn.features.convert import (
    cinn_sample_to_classifier_input_torch,
)
from calogen.models.caloinn.lightning import CaloINNLightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class CaloINNCLF(NoiseContrastiveDensityRatioEstimator):
    """CaloINN shower-space classifier for density-ratio estimation.

    Loads a pre-trained CaloINN generator and trains an MLP to separate truth
    showers (label 1) from generator samples (label 0) in high-level-feature
    space.
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
        generator_ckpt_path: str = None,
        voxel_energy_cutoff: float = None,
        log_transform: bool = False,
        representation: str = "physical",
        blocks: tuple = ("layer_frac", "cog", "width"),
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        kw_overrides: Optional[dict] = {},
        **kwargs
    ):
        if generator_ckpt_path is None:
            raise ValueError("generator_ckpt_path must be provided for CaloINNCLF.")
        rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
        # Build the generator BEFORE the classifier net so the RNG order matches
        # the original implementation (generator init, then MLP init).
        generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
        rank_zero_info(f"   ✅ Generator loaded. Model has {sum(p.numel() for p in generator.parameters())} parameters.")

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
        self.generator = generator

        self.convert_func = partial(
            cinn_sample_to_classifier_input_torch,
            layer_boundaries=self.generator.layer_boundaries,
            q=self.generator.q,
            width_noise=self.generator.width_noise,
            xml_path=self.generator.xml_path,
            particle=self.generator.xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff,
            representation=representation,
            blocks=blocks,
        )

        self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z"])

    def get_input_from_batch(self, batch):

        x_truth, c = batch
        x_gen = self.generator.conditional_generate(c, measure_gen_time=False).squeeze(1)

        y_truth = torch.ones(x_truth.shape[0], 1, device=x_truth.device)
        y_gen = torch.zeros(x_gen.shape[0], 1, device=x_gen.device)

        X = torch.cat([x_truth, x_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(2, 1)

        X_input = self.convert_func(X, c)

        return X_input, y, None

    def estimate_density_ratio(self, x, c):
        """Estimate the density ratio p(x|c) / q(x|c) using the trained classifier.

        Parameters
        ----------
        x : torch.Tensor shape (N, D)
            Samples from the proposal distribution q(x|c).
        c : torch.Tensor shape (N, C)
            Conditioning variables corresponding to each sample x.
        """
        x_clf = self.convert_func(x, c)
        logits = self.forward(x_clf).squeeze(-1)  # shape (N,)
        return torch.exp(logits)  # r(x,c) = p(x|c)/q(x|c) = exp(logit)


class CaloINNLatentNCE(NoiseContrastiveDensityRatioEstimator):
    """Latent-space NCE density-ratio estimator over the CaloINN latent."""

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
        generator_ckpt_path: str = None,
        voxel_energy_cutoff: float = None,
        log_transform: bool = False,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        kw_overrides: Optional[dict] = {},
        **kwargs
    ):
        if generator_ckpt_path is None:
            raise ValueError("generator_ckpt_path must be provided for CaloINNLatentNCE.")
        rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
        # Build the generator BEFORE the classifier net so the RNG order matches
        # the original implementation (generator init, then MLP init).
        generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
        rank_zero_info(f"   ✅ Generator loaded. Model has {sum(p.numel() for p in generator.parameters())} parameters.")

        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            noise_amplifier=noise_amplifier,
            activation=activation,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            amsgrad=amsgrad,
        )
        self.generator = generator

        self.convert_func = partial(
            cinn_sample_to_classifier_input_torch,
            layer_boundaries=self.generator.layer_boundaries,
            q=self.generator.q,
            width_noise=self.generator.width_noise,
            xml_path=self.generator.xml_path,
            particle=self.generator.xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff
        )

        self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z"])

    @torch.inference_mode()
    def _get_latent(self, x, c):
        z = self.generator.model(x, c, rev=False)[0]
        return z
