"""iCaloFlow model construction: preprocessing helpers and a conditional MAF-RQS flow builder.

Mirrors the inductive-caloflow reference (arXiv:2305.11934) on top of normflows
instead of nflows. Each flow is a conditional Masked Autoregressive Flow (MAF)
built from rational-quadratic-spline MADE blocks, wrapped so that
`ConditionalNormalizingFlow.sample` uses the sequential (autoregressive)
direction and `log_prob` uses the parallel (density) direction.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from normflows import ConditionalNormalizingFlow
from normflows.distributions import DiagGaussian
from normflows.flows.base import Reverse
from normflows.flows.mixing import Permute
from normflows.flows.neural_spline.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressive,
)
from normflows.nets.made import MADE, MaskedFeedforwardBlock, MaskedLinear

# Reference spline/made hyperparameters (inductive-caloflow run.py)
_MIN_BIN_WIDTH = 1e-6
_MIN_BIN_HEIGHT = 1e-6
_MIN_DERIVATIVE = 1e-6

# Logit stabilization constant (paper Appendix A)
ALPHA = 1e-6


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) / (torch.exp(x) + 1.0)


def logit_trafo(x: torch.Tensor) -> torch.Tensor:
    local_x = ALPHA + (1.0 - 2.0 * ALPHA) * x
    return logit(local_x)


def inverse_logit(
    x: torch.Tensor, clamp_low: float = 0.0, clamp_high: float = 1.0
) -> torch.Tensor:
    return ((sigmoid(x) - ALPHA) / (1.0 - 2.0 * ALPHA)).clamp(clamp_low, clamp_high)


def add_noise(input_array: torch.Tensor, noise_level: float = 1e-4) -> torch.Tensor:
    noise = torch.rand(
        size=input_array.size(), dtype=input_array.dtype, device=input_array.device
    ) * noise_level
    return input_array + noise


def compute_normalization_constant(
    showers_flat: Union[torch.Tensor, np.ndarray],
    layer_boundaries: Union[List[int], np.ndarray],
    safety_factor: float = 1.5,
) -> float:
    """Return safety_factor x max per-layer summed energy over the sample.

    `showers_flat` is (N, total_cells) in MeV with cells grouped layer-major,
    and `layer_boundaries` gives the cumulative cell offset of each layer
    (length num_layers + 1). Guards against an all-zero sample by returning
    the safety factor alone rather than 0, which would break downstream
    normalization divisions.
    """
    showers = torch.as_tensor(showers_flat)
    boundaries = np.asarray(layer_boundaries, dtype=int)
    if boundaries.ndim != 1 or len(boundaries) < 2:
        raise ValueError("layer_boundaries must be a 1-D array of length num_layers + 1")
    if boundaries[0] != 0 or boundaries[-1] != showers.shape[1]:
        raise ValueError(
            "layer_boundaries must start at 0 and end at the total number of cells"
        )
    if not (np.diff(boundaries) > 0).all():
        raise ValueError("layer_boundaries must be strictly increasing")
    layer_sums = torch.stack(
        [
            showers[:, lo:hi].sum(dim=1)
            for lo, hi in zip(boundaries[:-1], boundaries[1:])
        ],
        dim=1,
    )
    max_layer_energy = float(layer_sums.max())
    if max_layer_energy <= 0.0:
        return float(safety_factor)
    return safety_factor * max_layer_energy


# ---------------------------------------------------------------------------
# normflows adapters
# ---------------------------------------------------------------------------


class ConditionalReverse(Reverse):
    """Reverse that forwards the context to the wrapped flow.

    normflows' `Reverse.forward(z)` / `inverse(z)` drop the context argument,
    which breaks conditioning inside `ConditionalNormalizingFlow`.
    """

    def forward(self, z: torch.Tensor, context: torch.Tensor = None):
        return self.flow.inverse(z, context=context)

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None):
        return self.flow.forward(z, context=context)


class IAFRQS(MaskedPiecewiseRationalQuadraticAutoregressive):
    """IAF version of MAF-RQS: forward = parallel inverse-spline (fast sampling).

    Swaps the elementwise directions of `MaskedPiecewiseRationalQuadratic
    Autoregressive`, so `.forward` runs the MADE in parallel and applies the
    *inverse* spline (fast generation), while `.inverse` is the sequential
    forward-spline (slow density estimation). Used as a plain block (no
    `ConditionalReverse`) in the student flow so that
    `ConditionalNormalizingFlow.sample`/`.forward` hits the fast direction.
    """

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)


class _ContextFeedforwardMADE(MADE):
    """MADE with reference feedforward blocks and a single context injection.

    normflows' MADE only supports context with residual blocks; the reference
    uses feedforward blocks and injects the context once, after the initial
    layer, via `+ activation(context_layer(context))`. This reimplements the
    reference construction and forward pass using normflows'
    `MaskedFeedforwardBlock` (which ignores context, matching the reference).
    """

    def __init__(
        self,
        features: int,
        hidden_features: int,
        context_features: Optional[int] = None,
        num_blocks: int = 2,
        output_multiplier: int = 1,
        random_mask: bool = False,
        activation=nn.functional.relu,
        dropout_probability: float = 0.0,
        use_batch_norm: bool = False,
        init_identity: bool = True,
    ):
        # Initialize nn.Module directly: MADE.__init__ would raise
        # NotImplementedError for context + feedforward blocks.
        nn.Module.__init__(self)
        self.initial_layer = MaskedLinear(
            in_degrees=torch.arange(1, features + 1),
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, hidden_features)
        self.activation = activation
        prev_out_degrees = self.initial_layer.degrees
        blocks = []
        for _ in range(num_blocks):
            # Context is injected once via context_layer in forward; the
            # reference feedforward blocks ignore context, so pass None.
            blocks.append(
                MaskedFeedforwardBlock(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    context_features=None,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
        )
        if init_identity:
            torch.nn.init.constant_(self.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                self.final_layer.bias,
                float(np.log(np.exp(1 - _MIN_DERIVATIVE) - 1)),
            )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = None):
        temps = self.initial_layer(inputs)
        if context is not None:
            temps = temps + self.activation(self.context_layer(context))
        temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps)
        return self.final_layer(temps)


# ---------------------------------------------------------------------------
# Flow builder
# ---------------------------------------------------------------------------


def build_flow(
    features: int,
    context_features: int,
    num_layers: int,
    hidden_size: int = 256,
    n_blocks: int = 8,
    n_bins: int = 8,
    tail_bound: float = 14.0,
) -> ConditionalNormalizingFlow:
    """Build a conditional MAF-RQS flow mirroring the inductive-caloflow reference.

    `num_layers` is the number of MADE hidden layers (1 for Flow-1, 2 for
    Flow-2/Flow-3). Composes `n_blocks` RQS-MADE blocks wrapped in
    `ConditionalReverse` (sampling direction = forward) interleaved with
    shuffling permutations, dropping the final permutation like the reference.
    """
    base = DiagGaussian([features], trainable=False)
    flows = []
    for i in range(n_blocks):
        # context_features=None here because normflows' feedforward MADE cannot
        # be constructed with context; the net is replaced below.
        mprqa = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=features,
            hidden_features=hidden_size,
            context_features=None,
            num_bins=n_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_layers,
            use_residual_blocks=False,
            random_mask=False,
            activation=nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            init_identity=True,
            min_bin_width=_MIN_BIN_WIDTH,
            min_bin_height=_MIN_BIN_HEIGHT,
            min_derivative=_MIN_DERIVATIVE,
        )
        mprqa.autoregressive_net = _ContextFeedforwardMADE(
            features=features,
            hidden_features=hidden_size,
            context_features=context_features,
            num_blocks=num_layers,
            output_multiplier=mprqa._output_dim_multiplier(),
            random_mask=False,
            activation=nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            init_identity=True,
        )
        flows.append(ConditionalReverse(mprqa))
        if i < n_blocks - 1:
            flows.append(Permute(features, mode="shuffle"))
    return ConditionalNormalizingFlow(base, flows)


# ---------------------------------------------------------------------------
# Student IAF builder
# ---------------------------------------------------------------------------


def build_student_flow(
    teacher_flow: ConditionalNormalizingFlow,
    features: int,
    context_features: int,
    num_layers: int,
    hidden_size: int = 256,
    n_blocks: int = 8,
    n_bins: int = 8,
    tail_bound: float = 14.0,
) -> ConditionalNormalizingFlow:
    """Build an IAF student flow mirroring a trained MAF teacher.

    The student uses plain `IAFRQS` blocks (no `ConditionalReverse`) so
    `ConditionalNormalizingFlow.sample`/`.forward` hits the fast parallel
    direction.  The teacher's `Permute` buffers are copied so both flows
    share the same permutation structure, which PDD's per-block MSE relies
    on.  `num_layers` is the number of MADE hidden layers (2 for Flow-2/3).
    """
    teacher_perms = [
        (p.perm.detach().clone(), p.inv_perm.detach().clone())
        for p in teacher_flow.flows
        if isinstance(p, Permute)
    ]
    if len(teacher_perms) != n_blocks - 1:
        raise ValueError(
            f"Teacher has {len(teacher_perms)} permutations but n_blocks={n_blocks} "
            "requires n_blocks - 1; PDD block alignment would be broken"
        )
    base = DiagGaussian([features], trainable=False)
    flows = []
    for i in range(n_blocks):
        iafrqs = IAFRQS(
            features=features,
            hidden_features=hidden_size,
            context_features=None,
            num_bins=n_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_layers,
            use_residual_blocks=False,
            random_mask=False,
            activation=nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            init_identity=True,
            min_bin_width=_MIN_BIN_WIDTH,
            min_bin_height=_MIN_BIN_HEIGHT,
            min_derivative=_MIN_DERIVATIVE,
        )
        iafrqs.autoregressive_net = _ContextFeedforwardMADE(
            features=features,
            hidden_features=hidden_size,
            context_features=context_features,
            num_blocks=num_layers,
            output_multiplier=iafrqs._output_dim_multiplier(),
            random_mask=False,
            activation=nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            init_identity=True,
        )
        flows.append(iafrqs)
        if i < n_blocks - 1:
            perm = Permute(features, mode="shuffle")
            t_perm, t_inv = teacher_perms[i]
            perm.perm.copy_(t_perm)
            perm.inv_perm.copy_(t_inv)
            flows.append(perm)
    return ConditionalNormalizingFlow(base, flows)


# ---------------------------------------------------------------------------
# Guided cascade + PDD loss helpers
# ---------------------------------------------------------------------------


def chi2_loss(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE summed over features, averaged over the batch."""
    return ((input1 - input2) ** 2).sum(dim=1).mean()


def logabsdet_of_base(noise: torch.Tensor, width: float = 1.0) -> torch.Tensor:
    """Reference KL base term, per row.

    Matches the inductive-caloflow reference exactly: ``-0.5 * sum((noise /
    width)**2) - 0.5 * d * log(2*pi)``.  This is ``log N(z | 0, I)`` with
    ``width`` acting as a pre-rescaling of ``z``; it is exactly the base
    log-density only for ``width == 1.0`` (the reference omits the
    ``-d * log(width)`` normalization, and ``student_width`` defaults to 1).
    """
    shape = noise.size()[1]
    ret = -0.5 * ((noise / width) ** 2).sum(dim=1)
    log_z = 0.5 * shape * np.log(2 * np.pi)
    return ret - log_z


def _block_p_source(block, direction):
    """Return ``(autoregressive_net, p_before)`` for a block in a guided walk.

    Mirrors the reference ``GuidedCompositeTransform._cascade`` timing:
      - teacher MAF block (`ConditionalReverse`): p BEFORE on the density
        walk (``direction="inverse"``), AFTER on the sampling walk.
      - student IAF block (`IAFRQS`): p BEFORE on the sampling walk
        (``direction="forward"``), AFTER on the density walk.
    `Permute` blocks have no net (``(None, False)``).
    """
    if isinstance(block, IAFRQS):
        return block.autoregressive_net, (direction == "forward")
    if isinstance(block, ConditionalReverse):
        return block.flow.autoregressive_net, (direction == "inverse")
    return None, False


def guided_cascade(
    blocks,
    inputs: torch.Tensor,
    context,
    direction: str,
    return_steps: bool = False,
    return_p: bool = False,
):
    """Walk a block list applying ``.forward``/``.inverse``, collecting steps/p.

    ``direction="forward"`` applies each block's ``.forward`` in order (the
    sampling direction of our composition); ``direction="inverse"`` applies
    each block's ``.inverse`` in reverse order (the density direction).  With
    ``return_steps``/``return_p`` the intermediate outputs (steps, one per
    block plus the input) and the RQS blocks' MADE outputs (``p``) are
    returned, mirroring the reference ``GuidedCompositeTransform`` used by PDD.

    NOTE: for ``direction="inverse"`` the returned ``steps``/``p`` lists are in
    *reverse* block order (the blocks are walked as ``flows[::-1]``), while
    ``direction="forward"`` returns them in forward block order.  PDD pairs
    teacher/student entries across the two directions via ``[-i-1]``.

    Returns ``(outputs, total_logabsdet)``, plus ``steps`` and/or ``p`` as
    requested.
    """
    if direction == "forward":
        ordered = list(blocks)

        def apply(block, x):
            return block.forward(x, context)

    elif direction == "inverse":
        ordered = list(reversed(blocks))

        def apply(block, x):
            return block.inverse(x, context)

    else:
        raise ValueError(f"direction must be 'forward' or 'inverse'; got {direction}")

    steps = [inputs]
    outputs = inputs
    total_logabsdet = inputs.new_zeros(inputs.shape[0])
    ret_p = []
    for block in ordered:
        p_net, p_before = _block_p_source(block, direction)
        if return_p and p_net is not None and p_before:
            ret_p.append(p_net(outputs, context))
        outputs, logabsdet = apply(block, outputs)
        if return_p and p_net is not None and not p_before:
            ret_p.append(p_net(outputs, context))
        steps.append(outputs)
        total_logabsdet = total_logabsdet + logabsdet

    if return_steps and return_p:
        return outputs, total_logabsdet, steps, ret_p
    if return_steps:
        return outputs, total_logabsdet, steps
    if return_p:
        return outputs, total_logabsdet, ret_p
    return outputs, total_logabsdet
