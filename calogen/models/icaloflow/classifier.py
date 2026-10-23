"""iCaloFlow latent-space density-ratio (NCE) classifier for HMC.

The classifier distinguishes teacher-forced joint-encodes of truth showers
(label 1) from N(0, I) base draws (label 0) on ``[z, c]``, where ``z`` is
the 6525-dimensional iCaloFlow joint latent and ``c`` the incident energy
(in GeV, matching the CaloINN HMC conventions).  HMC then targets
``p(z|c) propto N(0,I) * M * exp(logit(z, c))``.

Reuses the NCE machinery from ``NoiseContrastiveDensityRatioEstimator``
(``get_input_from_batch``, ``estimate_log_density``, gradients) but swaps
the CINN generator for the iCaloFlow 3-flow module and the CINN ``rev=True``
latent encoder for the teacher-forced ``encode_joint``.
"""

from __future__ import annotations

import os

import torch
from calogen.classifiers.base import NoiseContrastiveDensityRatioEstimator
from calogen.models.icaloflow.lightning import ICaloFlowLightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

FCCEEALLEGRO_SAMPLES = 899368
FCCEEALLEGRO_ENERGY_NORM = 100000.

class ICaloFlowNCE(NoiseContrastiveDensityRatioEstimator):
    """NCE density-ratio estimator over the iCaloFlow joint latent.

    ``flow1_path/flow2_path/flow3_path`` point at the per-flow iCaloFlow
    training checkpoints.  ``normalization_constant`` and
    ``layer_boundaries`` must match the ones used to train the flows (and
    to preprocess the truth showers), since ``encode_joint`` needs them.

    Truth latents are encoded with the reference training noise
    (``noise_level``, default 5e-3 MeV) so they sit in the flows'
    calibrated regime and the heavy empty-layer latent tail collapses.
    The condition ``c`` is the incident energy in GeV.
    """

    def __init__(
        self,
        flow1_path: str,
        flow2_path: str,
        flow3_path: str,
        normalization_constant: float,
        layer_boundaries,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
        noise_amplifier: int = 50,
        noise_level: float = 5e-3,
        num_layers_grid: int = 45,
        cells_per_layer: int = 144,
        energy_ref_mev: float = 10 ** 4.5,
        hidden_size: int = 256,
        n_blocks: int = 8,
        n_bins: int = 8,
        tail_bound: float = 14.0,
        num_layers_flow1: int = 1,
        num_layers_flow2: int = 2,
        num_layers_flow3: int = 2,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        sigma:float=1.0,
        **kwargs,
    ):
        rank_zero_info("🔄 Loading iCaloFlow generator (3 flows) for NCE classifier")
        # Build the generator BEFORE the classifier net so the RNG order matches
        # the original implementation (generator init, then MLP init).
        generator = ICaloFlowLightningModule(
            which_flow=7,
            num_layers=num_layers_grid,
            cells_per_layer=cells_per_layer,
            energy_ref_mev=energy_ref_mev,
            layer_boundaries=layer_boundaries,
            normalization_constant=normalization_constant,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            n_bins=n_bins,
            tail_bound=tail_bound,
            num_layers_flow1=num_layers_flow1,
            num_layers_flow2=num_layers_flow2,
            num_layers_flow3=num_layers_flow3,
        )
        generator.load_flows_for_generation(flow1_path, flow2_path, flow3_path)

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
        self.noise_level = float(noise_level)
        self.sigma = sigma
        self.generator = generator
        rank_zero_info(
            f" -> ✅ Generator loaded. Joint latent dim = {self.generator.num_dim}, norm={self.generator.normalization_constant:.3e} MeV"
        )

        self.save_hyperparameters(
            ignore=[
                "generator", "net", "metrics",
                "_val_y_hat", "_val_y", "_test_y_hat", "_test_y", "_val_z",
            ]
        )

    # ------------------------------------------------------------------
    # Latent encoding (the iCaloFlow seam)
    # ------------------------------------------------------------------

    def setup(self, stage=None):
        """Adopt grid constants from the datamodule if not supplied.

        ``encode_joint`` needs ``layer_boundaries`` and
        ``normalization_constant``; these are normally computed by the
        data module at setup, so prefer the datamodule's values over the
        constructor's (which may be None).
        """
        super().setup(stage)
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None) if trainer is not None else None
        if dm is not None:
            gen = self.generator
            if gen.layer_boundaries is None:
                gen.layer_boundaries = getattr(dm, "layer_boundaries", None)
            if gen.normalization_constant is None:
                gen.normalization_constant = getattr(
                    dm, "normalization_constant", None
                )
            rank_zero_info(
                f"🔄 Adopted normalization constant = {gen.normalization_constant:.3e} MeV"
            )

    @torch.inference_mode()
    def _get_latent(self, x, c):
        """Teacher-forced joint-encode of raw showers into the joint latent.

        ``x`` : (B, num_layers * cells_per_layer) float32 raw shower MeV.
        ``c`` : (B, 1) float32 incident energy in GeV.

        Noise is added at encode (``noise_level`` MeV) to match the reference
        training dataloader; this puts the encoded latents in the flows'
        calibrated regime and collapses the empty-layer latent tail.
        """
        z = self.generator.encode_joint(
            x, c * 1e3, noise_level=self.noise_level  # GeV -> MeV
        )
        return z
    
    def _get_condition_from_batch(self, batch):
        return batch["energy_inc"]

    # ------------------------------------------------------------------
    # Training data (inherit NCE density-ratio math; adjust condition units)
    # ------------------------------------------------------------------

    def _get_input_from_latent_and_condition(self, z, c):
        return torch.cat([z, c], dim=1)

    def get_input_from_batch(self, batch):
        """Build [z, c] classifier input: encoded truth vs N(0,I) noise.

        The raw-shower datamodule yields ``c`` in MeV; the classifier and
        density-ratio conventions use GeV, so convert here.  The inherited
        ``estimate_log_density``/``estimate_log_density_gradient`` operate on
        this same ``[z, c_gev]`` representation.
        """
        x_truth, c = batch
        c_gev = c / 1e3

        z_truth = self._get_latent(x_truth, c_gev)
        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)

        z_gen = self.q0.sample(
            (z_truth.shape[0] * self.noise_amplifier, *z_truth.shape[1:])
        ).to(z_truth.device)
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        X = torch.cat([z_truth, z_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c_gev.repeat(1 + self.noise_amplifier, 1)

        X_input = self._get_input_from_latent_and_condition(X, c)
        return X_input, y, z_truth

class ICaloFlowNCEMultivariateNormal(ICaloFlowNCE):
    """NCE density-ratio estimator over the iCaloFlow joint latent.

    This subclass uses a multivariate normal base distribution for the
    NCE noise, rather than the default standard normal.  The covariance
    is diagonal with variance ``sigma**2``.
    """

    def __init__(
        self,
        flow1_path: str,
        flow2_path: str,
        flow3_path: str,
        q0_mean: str | None,
        q0_cov: str | None,
        normalization_constant: float,
        layer_boundaries,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
        noise_amplifier: int = 50,
        noise_level: float = 5e-3,
        num_layers_grid: int = 45,
        cells_per_layer: int = 144,
        energy_ref_mev: float = 10 ** 4.5,
        hidden_size: int = 256,
        n_blocks: int = 8,
        n_bins: int = 8,
        tail_bound: float = 14.0,
        num_layers_flow1: int = 1,
        num_layers_flow2: int = 2,
        num_layers_flow3: int = 2,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        **kwargs,
    ):

        ICaloFlowNCE.__init__(self, 
            flow1_path=flow1_path,
            flow2_path=flow2_path,
            flow3_path=flow3_path,
            normalization_constant=normalization_constant,
            layer_boundaries=layer_boundaries,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            noise_amplifier=noise_amplifier,
            noise_level=noise_level,
            num_layers_grid=num_layers_grid,
            cells_per_layer=cells_per_layer,
            energy_ref_mev=energy_ref_mev,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            n_bins=n_bins,
            tail_bound=tail_bound,
            num_layers_flow1=num_layers_flow1,
            num_layers_flow2=num_layers_flow2,
            num_layers_flow3=num_layers_flow3,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            amsgrad=amsgrad,
            sigma=1.0
        )
        self._setup_q0(q0_mean, q0_cov)
    
    def _setup_q0(self, q0_mean, q0_cov):
        z_dim = self.generator.flow1.q0.d + self.generator.flow2.q0.d + self.generator.flow3.q0.d
        self.q0_mean = torch.zeros(z_dim, dtype=torch.float32)
        self.q0_cov = torch.eye(z_dim, dtype=torch.float32) 
        if q0_mean is not None:
            self.q0_mean = torch.load(q0_mean, map_location="cpu").to(torch.float32)
        if q0_cov is not None:
            self.q0_cov = torch.load(q0_cov, map_location="cpu").to(torch.float32)

        self._q0 = torch.distributions.MultivariateNormal(
            loc=self.q0_mean, covariance_matrix=self.q0_cov
        )

    @property
    def q0(self):
        if self.q0_mean.device != self.device or self.q0_cov.device != self.device:
            self.q0_mean = self.q0_mean.to(self.device)
            self.q0_cov = self.q0_cov.to(self.device)
            self._q0 = torch.distributions.MultivariateNormal(
                loc=self.q0_mean.to(self.device),
                covariance_matrix=self.q0_cov.to(self.device)
            )
        return self._q0

    def get_input_from_batch(self, batch):
        """Build [z, c] classifier input: encoded truth vs N(0,I) noise.

        The raw-shower datamodule yields ``c`` in MeV; the classifier and
        density-ratio conventions use GeV, so convert here.  The inherited
        ``estimate_log_density``/``estimate_log_density_gradient`` operate on
        this same ``[z, c_gev]`` representation.
        """
        x_truth, c = batch
        c_gev = c / 1e3

        z_truth = self._get_latent(x_truth, c_gev)
        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)

        z_gen = self.q0.sample(
            (z_truth.shape[0] * self.noise_amplifier,)
        ).to(z_truth.device)
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        X = torch.cat([z_truth, z_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c_gev.repeat(1 + self.noise_amplifier, 1)

        X_input = self._get_input_from_latent_and_condition(X, c)
        return X_input, y, z_truth

    def estimate_log_density(self, z, c):
        """Estimate log p(z|c) = q0(z) * (p/q0) with the multivariate-normal base.

        ``MultivariateNormal.log_prob`` already returns one value per row, so
        the inherited ``.sum(dim=-1)`` (correct for the per-element
        ``Normal.log_prob`` of the standard-normal base) would collapse the
        batch to a scalar and corrupt batched-HMC acceptance.
        """
        logits = self._get_input_from_latent_and_condition(z, c)
        log_noise_amplifier = z.new_tensor(float(self.noise_amplifier)).log()
        return self.q0.log_prob(z) + log_noise_amplifier + logits.squeeze()

class ICaloFlowNCEMultivariateNormalFlow1V3(ICaloFlowNCEMultivariateNormal):

    def _setup_q0(self, q0_mean, q0_cov):
        """Set up the base distribution q0."""
        z_dim = self.generator.flow1.q0.d
        self.q0_mean = torch.zeros(z_dim, dtype=torch.float32)
        self.q0_cov = torch.eye(z_dim, dtype=torch.float32) 
        if q0_mean is not None:
            self.q0_mean = torch.load(q0_mean, map_location="cpu").to(torch.float32)
        if q0_cov is not None:
            self.q0_cov = torch.load(q0_cov, map_location="cpu").to(torch.float32)

        self._q0 = torch.distributions.MultivariateNormal(
            loc=self.q0_mean, covariance_matrix=self.q0_cov
        )

    """ this class is used with the datamodule that sets flow 1"""
    def setup(self, stage=None):
        super().setup(stage)
        self.generator.which_flow = 1
    
    def _get_condition_from_batch(self, batch):
        return batch["energy_mev_scaled"]

    def get_input_from_batch(self, batch):
    
        x_truth, c = self.generator._prepare_flow_batch(batch)

        with torch.inference_mode():
            z_truth = self.generator.flow1.inverse(x_truth, c).to(next(self.net.parameters()).dtype)

        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)
        
        z_gen = self.q0.sample(
            (z_truth.shape[0] * self.noise_amplifier,)
        ).to(z_truth.device)[:, :z_truth.shape[1]].to(z_truth.dtype)
        assert z_gen.shape[0] == z_truth.shape[0] * self.noise_amplifier, f"z_gen.shape[0] = {z_gen.shape[0]}, z_truth.shape[0] * self.noise_amplifier = {z_truth.shape[0] * self.noise_amplifier}"
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        # use linear scale energy for the condition
        c_mev = batch["energy_mev_scaled"].reshape(-1, 1).repeat(1 + self.noise_amplifier, 1).to(z_truth.dtype) 

        X = torch.cat([z_truth, z_gen], dim=0)
        X = self._get_input_from_latent_and_condition(X, c_mev)
        y = torch.cat([y_truth, y_gen], dim=0)

        return X, y, z_truth

