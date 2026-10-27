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
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from calogen.models.caloinn.features.convert import showers_to_classifier_input_torch

from ...models.caloinn.features.representation import internal_to_classifier_input
from ...models.caloinn.lightning import CaloINNLightningModule
from ..base import GeneratorMixin, NoiseContrastiveDensityRatioEstimator


class InternalShowerNCE(NoiseContrastiveDensityRatioEstimator, GeneratorMixin):
    """CaloINN shower-space classifier for density-ratio estimation.

    Loads a pre-trained CaloINN generator and trains an MLP to separate truth
    showers (label 1) from generator samples (label 0) in high-level-feature
    space.

    Uses calogen.data.datasets.caloinn.CaloINNDataset to load raw shower and preprocess. Return shower (x) and conditions (c).
    Generate training samples using the same conditions with CaloINN generator. 
    Compute high-level features for both truth and gen samples. These include layer fractions, center of gravity, and shower width.
    Train a binary classifier to distinguish between truth and gen samples in the high-level feature space.
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
        log_transform: bool = False,
        blocks: tuple = ("layer_frac", "cog", "width"),
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        kw_overrides: Optional[dict] = {}
    ):
        if generator_ckpt_path is None:
            raise ValueError("generator_ckpt_path must be provided for CaloINNCLF.")
        rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
        # Build the generator BEFORE the classifier net so the RNG order matches
        # the original implementation (generator init, then MLP init).
        # generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
        generator = self.load_generator(CaloINNLightningModule, generator_ckpt_path, kw_overrides)
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
        self.generator.eval()  # Ensure generator is in eval mode during classifier training

        self.convert_func = partial(
            internal_to_classifier_input,
            layer_boundaries=self.generator.layer_boundaries,
            blocks=blocks,
            log_transform=log_transform
        )

        self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z"])

    def get_input_from_batch(self, batch):

        x_truth, c = batch
        x_gen = self.generate(c).squeeze(1)

        y_truth = torch.ones(x_truth.shape[0], 1, device=x_truth.device)
        y_gen = torch.zeros(x_gen.shape[0], 1, device=x_gen.device)

        X = torch.cat([x_truth, x_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(2, 1)

        X_input = torch.concat(self.convert_func(X, c), dim=1)

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
        logits = self.forward(torch.concat(self.convert_func(x, c), dim=1)).squeeze(-1)  # shape (N,)
        return torch.exp(logits)  # r(x,c) = p(x|c)/q(x|c) = exp(logit)

class HLFShowerNCE(InternalShowerNCE):
    """CaloINN shower-space classifier for density-ratio estimation.

    Loads a pre-trained CaloINN generator and trains an MLP to separate truth
    showers (label 1) from generator samples (label 0) in high-level-feature
    space.

    Uses calogen.data.datasets.caloinn.CaloINNDataset to load raw shower and preprocess. Return shower (x) and conditions (c).
    Generate training samples using the same conditions with CaloINN generator. 
    Compute high-level features for both truth and gen samples. These include layer fractions, center of gravity, and shower width.
    Train a binary classifier to distinguish between truth and gen samples in the high-level feature space.
    """

    def get_input_from_batch(self, batch):
    
        x_truth, c = batch

        x_gen = self.generate(c).squeeze(1)

        y_truth = torch.ones(x_truth.shape[0], 1, device=x_truth.device)
        y_gen = torch.zeros(x_gen.shape[0], 1, device=x_gen.device)

        X = torch.cat([x_truth, x_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(2, 1)

        cells, c, hlf = self.convert_func(X, c)

        X_input = torch.concat([hlf, c], dim=1)

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
        cells, c, hlf = self.convert_func(x, c)
        logits = self.forward(torch.concat([hlf, c], dim=1)).squeeze(-1)  # shape (N,)
        return torch.exp(logits)  # r(x,c) = p(x|c)/q(x|c) = exp(logit)

class LemursPhysicalShowerNCE(NoiseContrastiveDensityRatioEstimator, GeneratorMixin):

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
        generator_ckpt_path: str | None = None,
        log_transform: bool = False,
        voxel_energy_cutoff: float | None = None,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        kw_overrides: Optional[dict] = {}
    ):
        if generator_ckpt_path is None:
            raise ValueError("generator_ckpt_path must be provided for CaloINNCLF.")
        rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
        # Build the generator BEFORE the classifier net so the RNG order matches
        # the original implementation (generator init, then MLP init).
        # generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
        generator = self.load_generator(CaloINNLightningModule, generator_ckpt_path, kw_overrides)
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
        self.generator.eval()  # Ensure generator is in eval mode during classifier training
    
        self.convert_func = partial(
            showers_to_classifier_input_torch,
            xml_path=self.generator.xml_path,
            particle=self.generator.xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff
        )
    
        self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z"])

        self.strict_loading = False

    def get_input_from_batch(self, batch):
        # Assume unprocessed showers and conditions
        # both showers and system in MeV

        x_truth, c = batch
        c_gev = c.reshape(-1, 1) / 1e3

        x_gen = self.generate(c_gev)
        # convert caloinn shower representation to physical showers in mev, similar to x_truth
        data = self.generator.post_process_decoded(x_gen, c_gev)
        num_layers = len(self.generator.layer_boundaries) - 1
        x_gen = torch.cat([data[f"layer_{i}"] for i in range(num_layers)], dim=1) # this is in GeV
        x_gen = x_gen * 1000.0  # Convert generated showers back to MeV 

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
            Samples from the proposal distribution q(x|c). Assumed un-postprocessed showers from caloinn (immediately after conditional_generate)
        c : torch.Tensor shape (N, C)
            Conditioning variables corresponding to each sample x.
        """

        # convert caloinn shower representation to physical showers in mev, similar to x_truth
        data = self.generator.post_process_decoded(x, c)
        num_layers = len(self.generator.layer_boundaries) - 1
        x_gen = torch.cat([data[f"layer_{i}"] for i in range(num_layers)], dim=1) # this is in GeV
        x_gen = x_gen * 1000.0  # Convert generated showers back to MeV 

        X_input = self.convert_func(x_gen, c * 1000) 

        logits = self.forward(X_input).squeeze(-1)  # shape (N,)
        return torch.exp(logits)  # r(x,c) = p(x|c)/q(x|c) = exp(logit)
    

# class CaloINNLatentNCE(NoiseContrastiveDensityRatioEstimator):
#     """Latent-space NCE density-ratio estimator over the CaloINN latent."""

#     def __init__(
#         self,
#         hidden_dim: int = 256,
#         num_layers: int = 4,
#         batch_norm: bool = False,
#         layer_norm: bool = True,
#         output_dim: int = 1,
#         dropout: float = 0.0,
#         noise_amplifier: int = 1,
#         activation: str = "relu",
#         generator_ckpt_path: str = None,
#         voxel_energy_cutoff: float = None,
#         log_transform: bool = False,
#         lr: float = 1e-3,
#         step_size: int = 10,
#         gamma: float = 0.95,
#         amsgrad: bool = True,
#         kw_overrides: Optional[dict] = {},
#         **kwargs
#     ):
#         if generator_ckpt_path is None:
#             raise ValueError("generator_ckpt_path must be provided for CaloINNLatentNCE.")
#         rank_zero_info(f"🔄 Loading generator from {generator_ckpt_path}")
#         # Build the generator BEFORE the classifier net so the RNG order matches
#         # the original implementation (generator init, then MLP init).
#         generator = CaloINNLightningModule.load_from_checkpoint(generator_ckpt_path, **kw_overrides)
#         rank_zero_info(f"   ✅ Generator loaded. Model has {sum(p.numel() for p in generator.parameters())} parameters.")

#         super().__init__(
#             hidden_dim=hidden_dim,
#             num_layers=num_layers,
#             batch_norm=batch_norm,
#             layer_norm=layer_norm,
#             output_dim=output_dim,
#             dropout=dropout,
#             noise_amplifier=noise_amplifier,
#             activation=activation,
#             lr=lr,
#             step_size=step_size,
#             gamma=gamma,
#             amsgrad=amsgrad,
#         )
#         self.generator = generator

#         self.convert_func = partial(
#             cinn_sample_to_classifier_input_torch,
#             layer_boundaries=self.generator.layer_boundaries,
#             q=self.generator.q,
#             width_noise=self.generator.width_noise,
#             xml_path=self.generator.xml_path,
#             particle=self.generator.xml_ptype,
#             log_transform=log_transform,
#             voxel_energy_cutoff=voxel_energy_cutoff
#         )

#         self.save_hyperparameters(ignore=["generator", "net", "convert_func", "metrics", "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z"])

#     @torch.inference_mode()
#     def _get_latent(self, x, c):
#         z = self.generator.model(x, c, rev=False)[0]
#         return z
