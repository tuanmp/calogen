
import torch
from calogen.classifiers.base import NoiseContrastiveDensityRatioEstimator
from calogen.core.latents import load_latent_q0


class PrecomputedLatentWhitenedNCE(NoiseContrastiveDensityRatioEstimator):
    """Latent NCE on precomputed latents with a data-derived whitened base.

    The base ``q0 = N(mu, Sigma)`` is applied as an eigendecomposition whitening
    ``w = diag(1/sqrt(lambda)) U^T (z - mu)`` so the effective base is ``N(0, I)``
    and the shared isotropic-NCE math (sampling, ratio, log-density) is reused
    unchanged.  ``q0_dir`` holds ``mean.pt`` / ``eigvecs.pt`` / ``eigvals.pt`` /
    ``meta.json`` produced by ``scripts/fit_latent_mvn.py``.

    ``mean``/``eigvecs``/``eigvals`` are registered as non-persistent buffers, so
    they follow ``.to(device)``/dtype but are not written into checkpoints.
    """

    def __init__(
        self,
        q0_dir: str,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        batch_norm: bool = False,
        layer_norm: bool = True,
        output_dim: int = 1,
        dropout: float = 0.5,
        activation: str = "relu",
        noise_amplifier: int = 1,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.95,
        amsgrad: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            noise_amplifier=noise_amplifier,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            amsgrad=amsgrad,
        )
        mean, eigvecs, eigvals, meta = load_latent_q0(q0_dir)
        d = int(mean.shape[0])
        if tuple(mean.shape) != (d,) or tuple(eigvecs.shape) != (d, d) or tuple(eigvals.shape) != (d,):
            raise ValueError(
                f"q0 stats in {q0_dir} have inconsistent shapes: "
                f"mean={tuple(mean.shape)} eigvecs={tuple(eigvecs.shape)} eigvals={tuple(eigvals.shape)}"
            )
        if bool((eigvals <= 0).any()):
            raise ValueError(f"q0 eigenvalues in {q0_dir} must be positive")
        self.q0_dir = q0_dir
        self.q0_meta = meta
        self._q0_logdet = float(torch.log(eigvals.double()).sum().item())
        self.register_buffer("_q0_mean", mean, persistent=False)
        self.register_buffer("_q0_eigvecs", eigvecs, persistent=False)
        self.register_buffer("_q0_inv_std", 1.0 / torch.sqrt(eigvals), persistent=False)
        self.register_buffer("_q0_sqrt_eigvals", torch.sqrt(eigvals), persistent=False)

    @property
    def q0(self):
        """Return the whitened base distribution ``q0 = N(mu, Sigma)``."""
        return torch.distributions.Normal(
            loc=torch.zeros_like(self._q0_mean), scale=torch.ones_like(self._q0_mean)
        )

    def get_input_from_batch(self, batch):
        x_truth, c = batch["z"], batch["c"]

        z_truth = self._get_latent(x_truth, c)
        y_truth = torch.ones(z_truth.shape[0], 1, device=z_truth.device)

        # amplify the number of noise samples by self.noise_amplifier
        z_gen = self.q0.sample((z_truth.shape[0] * self.noise_amplifier, )).to(z_truth.device)
        y_gen = torch.zeros(z_gen.shape[0], 1, device=z_gen.device)

        X = torch.cat([z_truth, z_gen], dim=0)
        y = torch.cat([y_truth, y_gen], dim=0)
        c = c.repeat(1 + self.noise_amplifier, 1)

        X_input = torch.cat([X, c], dim=1)

        return X_input, y, z_truth

    def whiten(self, z: torch.Tensor) -> torch.Tensor:
        """``w = diag(1/sqrt(lambda)) U^T (z - mu)`` (row vectors)."""
        mean = self._q0_mean.to(device=z.device, dtype=z.dtype)
        eigvecs = self._q0_eigvecs.to(device=z.device, dtype=z.dtype)
        inv_std = self._q0_inv_std.to(device=z.device, dtype=z.dtype)
        return ((z - mean) @ eigvecs) * inv_std

    def unwhiten(self, w: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`whiten`."""
        mean = self._q0_mean.to(device=w.device, dtype=w.dtype)
        eigvecs = self._q0_eigvecs.to(device=w.device, dtype=w.dtype)
        sqrt_eigvals = self._q0_sqrt_eigvals.to(device=w.device, dtype=w.dtype)
        return (w * sqrt_eigvals) @ eigvecs.T + mean

    @torch.inference_mode()
    def _get_latent(self, z, c):
        return self.whiten(z)

    def estimate_log_density(self, z, c):
        """``log p(z) ~= log q0(z) + log(M) + logit_theta(w, c)``.

        ``q0`` is the whitened base, so its log-density is ``N(0, I)`` at ``w``
        minus the change-of-variables constant ``0.5 * logdet(Sigma)``.
        """
        w = self.whiten(z)
        logits = self(torch.cat([w, c], dim=1))
        log_noise_amplifier = w.new_tensor(float(self.noise_amplifier)).log()
        log_base = self.q0.log_prob(w).sum(dim=-1)
        return log_base - 0.5 * self._q0_logdet + log_noise_amplifier + logits.squeeze()
