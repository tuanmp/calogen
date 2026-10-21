import torch 
from torch import nn
import normflows as nf

class ConditionalBaseDistribution(nf.distributions.BaseDistribution):
    """Base class for conditional base distributions.

    Subclasses must implement `forward()` and `log_prob()`, which take
    a `context` argument. The context is passed through the model to
    compute the conditional distribution.
    """

    def forward(self, num_samples: int, context: torch.Tensor):
        """Sample from the conditional base distribution given context.

        Args:
            num_samples: Number of samples to draw.
            context:     Context vectors, shape (batch_size, context_dim).
        Returns:
            z:      Samples, shape (num_samples, d).
            log_p:  Log probability for each sample, shape (num_samples,).
        """
        raise NotImplementedError

    def log_prob(self, z: torch.Tensor, context: torch.Tensor):
        """Compute log probability of given z under the conditional base distribution.

        Args:
            z:       Latent codes, shape (batch_size, d).
            context: Context vectors, shape (batch_size, context_dim).

        Returns:
            log_p:   Log probability, shape (batch_size,).
        """
        raise NotImplementedError

    def sample(self, num_samples=1, context: torch.Tensor | None = None, **kwargs):
        raise NotImplementedError

class MinimalResampledGaussian(ConditionalBaseDistribution):
    """Minimal resampled Gaussian base distribution — no groups, no flows, no affine.

    A stripped-down alternative to ContinuousContextResampledGaussian.
    Proposes z ~ N(0, I_d) and accepts each sample with probability
    a([z, context]) in [0, 1], retrying up to T times per sample.

    Designed to be used directly as the q0 base of a ConditionalNormalizingFlow —
    the `context=` kwarg matches what ConditionalNormalizingFlow passes.

    Z (the normalizing constant) is a single global scalar, tracked via
    exponential moving average (EMA) during training.

    Key simplifications vs ContinuousContextResampledGaussian:
      - `d` (int) instead of `shape` (tuple) — flat latent space, no reshaping.
      - No group factorization — every sample is a single rejection-sampled draw.
      - No built-in flows or affine transforms — the outer model handles all that.
      - `context=` kwarg directly (no y= wrapper needed).
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(self, d: int, a: nn.Module, T: int, eps: float,
                 context_dim: int,
                 use_context_for_Z: bool = True,
                 Z_context_range: tuple = (1.0, 2.0)):
        """
        Args:
            d:                   Number of latent dimensions (scalar).
            a:                   Acceptance network. Input: [z, context] of
                                 shape (batch, d + context_dim). Output: scalar
                                 acceptance probability in [0, 1] per sample.
            T:                   Maximum rejection sampling attempts per sample.
            eps:                 EMA decay rate for the Z estimate, 0 < eps < 1.
                                 Smaller = smoother, larger = faster adaptation.
            context_dim:         Dimensionality of the conditioning vector.
            use_context_for_Z:   If True, draw contexts from Z_context_range
                                 during Z estimation. If False, use zero context.
            Z_context_range:     (low, high) — context sampling range for Z
                                 estimation when use_context_for_Z=True.
        """
        super().__init__()
        self.d = d
        self.a = a
        self.T = T
        self.eps = eps
        self.context_dim = context_dim
        self.use_context_for_Z = use_context_for_Z
        self.Z_context_range = Z_context_range
        self.q0 = nf.distributions.DiagGaussian((d,), trainable=False)

        print(f"    Initialized MinimalResampledGaussian with d={d}, T={T}, eps={eps}, context_dim={context_dim}")
        print(f"    In this implementation, Z is estimated per context during inference, and is not tracked as a global EMA.")

    def sample_and_base_log_prob(self, num_samples: int, context: torch.Tensor):
        """Sample from the resampled Gaussian given context.
        A context must be provided

        Args:
            num_samples: Number of samples to draw.
            context:     Context vectors, shape (num_samples, context_dim).
                         If None, uses zero context.
        Returns:
            z:      Accepted samples, shape (num_samples, d).
        """
        num_ctx = len(context) 

        sample = torch.zeros(num_samples * num_ctx, self.d, dtype=torch.float32, device=context.device)
        log_prob_base = torch.zeros(num_samples * num_ctx, dtype=torch.float32, device=context.device)
        context = context.repeat(num_samples, 1)  # shape (num_samples * num_context, context_dim)
        
        for step in range(self.T):
            # find all entries that still need to be sampled (i.e. are still zero)
            need_sampling = sample.eq(0).all(dim=-1)
            # sample proposals for all unsampled entries
            eps, log_prob = self.q0(num_samples=need_sampling.sum().item())
            _in = eps
            if context is not None:
                ctx = context[need_sampling]
                _in = torch.cat([eps, ctx], dim=-1)
            
            # compute acceptance probabilities for the proposals
            acc = self.a(_in).squeeze(-1)
            # sample a Bernoulli decision for each proposal
            dec = torch.rand_like(acc) < acc
            # update where we need to fill in the accepted samples

            if step == self.T - 1:
                # on the last step, fill in any remaining unsampled entries with the proposals
                dec = torch.ones_like(dec, dtype=torch.bool)
            sample[need_sampling][dec] = eps[dec]
            log_prob_base[need_sampling][dec] = log_prob[dec]
            if torch.all(sample.ne(0).all(dim=-1)):
                break

        # sample = sample.view(num_samples, num_ctx, self.d)
        # log_prob_base = log_prob_base.view(num_samples, num_ctx)

        return sample, log_prob_base

    # ------------------------------------------------------------------
    #  Shared helpers (eliminate duplicated code in forward / log_prob)
    # ------------------------------------------------------------------

    def _compute_acceptance(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute a([z, context]) via the acceptance network.

        Concatenates z and context along the feature dim, runs through
        the acceptance network, and squeezes to shape (N,).
        """
        combined = torch.cat([z, context], dim=-1)
        if not self.training:
            with torch.no_grad():
                return self.a(combined).squeeze(-1)
        return self.a(combined).squeeze(-1)

    def _log_acceptance_term(self, acc: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Compute log of the acceptance contribution to log p(z|context).

        From the T-step resampling formula:
            p_a(z) = (1 - α) * a(z, ctx) / Z + α
        where α = (1 - Z)^{T-1} accounts for survival probability to the
        T-th (final) rejection sampling attempt.
        """
        alpha = (1 - Z) ** (self.T - 1)
        return torch.log((1 - alpha) * acc / Z + alpha)

    def _estimate_conditional_Z(self, context: torch.Tensor, num_samples: int = 1000):
        """Estimate Z for a specific context by averaging acceptance.
        Only use in density estimation mode on inference

        Args:
            context:     Context vectors, shape (batch_size, context_dim).
            num_samples: Number of proposals to draw for the estimate.
        """

        dtype = torch.float32
        device = context.device
        N = context.shape[0]

        # eps = torch.randn(num_samples * N, self.d, dtype=dtype, device=device)
        eps = self.q0.sample(num_samples * N)  # shape (num_samples * N, d)
        ctx = context.repeat(1, num_samples)  \
            .view(num_samples * N, self.context_dim) # each context gets N samples
        acc = self._compute_acceptance(eps, ctx)
        Z_estimate = acc.view(N, num_samples).mean(dim=1)
        return Z_estimate
    
    def estimate_conditional_Z(self, context: torch.Tensor, num_samples: int = 1000):
        """Estimate Z for a specific context by averaging acceptance.
        Only use in density estimation mode on inference

        Args:
            context:     Context vectors, shape (batch_size, context_dim).
            num_samples: Number of proposals to draw for the estimate.
        """
        if self.training:
            return self._estimate_conditional_Z(context=context, num_samples=num_samples)
        with torch.no_grad():
            return self._estimate_conditional_Z(context=context, num_samples=num_samples)

    # ------------------------------------------------------------------
    #  forward(): sample z ~ resampled Gaussian and compute log p(z|ctx)
    # ------------------------------------------------------------------
    def forward(self, num_samples: int, context: torch.Tensor):
        """Sample from the resampled Gaussian given context.
        In this class, context is always needed (no default zero context).

        Returns:
            z:      Accepted samples, shape (num_samples, d).
            log_p:  Log probability for each sample, shape (num_samples,).
        """
        dtype = torch.float32
        device = context.device

        n_context = len(context)

        # sample proposals for each context (shape: num_samples x n_context x d)
        # sample = self.sample(num_samples=num_samples, context=context)  # shape (num_samples, num_context, d)
        # sample = sample.view(num_samples * n_context, self.d)  # flatten to (num_samples * num_context, d) in blocks of context
        sample, log_p_gauss = self.sample_and_base_log_prob(num_samples=num_samples, context=context)  # shape (num_samples, num_context, d)
        # sample = sample.view(num_samples * n_context, self.d)  # flatten to (num_samples * num_context, d) in blocks of context
        # log_p_gauss = log_p_gauss.view(num_samples * n_context)  # flatten to (num_samples * num_context, ) in blocks of context

        # repeat context for each sample
        ctx = context.repeat(num_samples, 1)  # shape (num_samples * num_context, context_dim)
        ctx = ctx.view(num_samples * n_context, self.context_dim)

        # ===== Z estimation =====
        Z = self.estimate_conditional_Z(context=context, num_samples=10000) # shape (num_context, )
        Z = Z.repeat(num_samples) # shape (num_samples * num_context, )

        # ===== Log-probability of the resampled Gaussian =====
        # log p(z | context) = log N(z; 0, I_d) + log p_a(z, context)

        # (2) Acceptance term (recompute acceptance for FINAL samples with grad)
        acc = self._compute_acceptance(sample, ctx) # shape (num_samples * num_context, )
        log_p_a = self._log_acceptance_term(acc, Z) 

        log_p = log_p_gauss + log_p_a

        # sample = sample.view(num_samples, n_context, self.d)  # reshape back to (num_samples, num_context, d)
        # log_p = log_p.view(num_samples, n_context)  # reshape back to (num_samples, num_context)

        return sample, log_p

    # ------------------------------------------------------------------
    #  log_prob(): compute log p(z | context) for GIVEN z
    # ------------------------------------------------------------------
    def log_prob(self, z: torch.Tensor, context: torch.Tensor):
        """Compute log probability of given z under the resampled Gaussian.

        No inverse flows — the outer model handles those. This only computes
        the resampled Gaussian density itself. Also updates the running Z
        estimate during training.

        Args:
            z:       Latent codes, shape (batch_size, d).
            context: Context vectors, shape (batch_size, context_dim).

        Returns:
            log_p:   Log probability, shape (batch_size,).
        """
        batch_size = z.size(0)
        dtype = z.dtype
        device = z.device
        sample = z.to(device=device, dtype=dtype)

        assert sample.shape[0] == context.shape[0], "Batch size of z and context must match."

        # ===== Z estimation =====
        Z = self.estimate_conditional_Z(context=context, num_samples=10000) # shape (B, )

        # (1) Gaussian base log-density
        log_p_gauss = self.q0.log_prob(sample)  # shape (B , )

        # (2) Acceptance term (recompute acceptance for FINAL samples with grad)
        acc = self._compute_acceptance(sample, context) # shape (num_samples * num_context, )
        log_p_a = self._log_acceptance_term(acc, Z) 

        log_p = log_p_gauss + log_p_a

        return log_p

    def sample(self, num_samples, context: torch.Tensor, **kwargs):
        """Sample from the resampled Gaussian given context.

        Args:
            num_samples: Number of samples to draw.
            context:     Context vectors, shape (batch_size, context_dim).
                         If None, uses zero context.
        Returns:
            z:      Accepted samples, shape (num_samples, d).
        """

        sample, _ = self.sample_and_base_log_prob(num_samples=num_samples, context=context)

        return sample

