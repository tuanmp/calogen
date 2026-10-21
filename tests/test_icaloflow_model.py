"""Unit tests for the iCaloFlow model builder and preprocessing helpers."""

import numpy as np
import pytest
import torch

from calogen.models.icaloflow.model import (
    add_noise,
    build_flow,
    compute_normalization_constant,
    inverse_logit,
    logit_trafo,
)

# Small flow config to keep tests fast
_N_BLOCKS = 2
_HIDDEN_SIZE = 32
_TAIL_BOUND = 3.0


def _build_test_flow(features, context_features, num_layers):
    return build_flow(
        features,
        context_features,
        num_layers,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=_N_BLOCKS,
        tail_bound=_TAIL_BOUND,
    ).double()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def test_logit_trafo_inverse_logit_round_trip():
    torch.manual_seed(0)
    x = torch.rand(1000) * 0.98 + 0.01  # in [0.01, 0.99]
    x_round = inverse_logit(logit_trafo(x))
    assert torch.allclose(x_round, x, atol=1e-6)


def test_inverse_logit_clamps_to_unit_interval():
    out = inverse_logit(torch.tensor([50.0, -50.0]))
    assert torch.allclose(out, torch.tensor([1.0, 0.0]))


def test_add_noise_shape_identity_and_bounds():
    torch.manual_seed(0)
    x = torch.rand(4, 8)
    assert torch.equal(add_noise(x, 0.0), x)
    x_noisy = add_noise(x, 0.1)
    assert x_noisy.shape == x.shape
    deltas = x_noisy - x
    assert (deltas >= 0).all() and (deltas < 0.1).all()


def test_compute_normalization_constant():
    # 2 samples, 3 layers of 5 cells each; per-layer sums:
    #   row 0: [15, 40, 65]  row 1: [50, 5, 2.5]  -> global max = 65
    showers = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    boundaries = [0, 5, 10, 15]
    expected = 1.5 * 65.0
    assert compute_normalization_constant(showers, boundaries) == pytest.approx(
        expected
    )


def test_compute_normalization_constant_all_zeros_guard():
    showers = torch.zeros(2, 15)
    const = compute_normalization_constant(showers, [0, 5, 10, 15])
    assert const > 0.0
    assert np.isfinite(const)


# ---------------------------------------------------------------------------
# Flow builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features,context_features,num_layers", [(45, 1, 1), (144, 191, 2)]
)
def test_build_flow_sample_shape_and_dtype(features, context_features, num_layers):
    torch.manual_seed(0)
    flow = _build_test_flow(features, context_features, num_layers)
    context = torch.randn(4, context_features, dtype=torch.float64)
    x, _ = flow.sample(4, context)
    assert x.shape == (4, features)
    assert x.dtype == torch.float64


@pytest.mark.parametrize(
    "features,context_features,num_layers", [(45, 1, 1), (144, 191, 2)]
)
def test_build_flow_log_prob_shape_and_finite(
    features, context_features, num_layers
):
    torch.manual_seed(0)
    flow = _build_test_flow(features, context_features, num_layers)
    context = torch.randn(4, context_features, dtype=torch.float64)
    x = torch.randn(4, features, dtype=torch.float64)
    lp = flow.log_prob(x, context)
    assert lp.shape == (4,)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize(
    "features,context_features,num_layers", [(45, 1, 1), (144, 191, 2)]
)
def test_build_flow_invertible(features, context_features, num_layers):
    torch.manual_seed(0)
    flow = _build_test_flow(features, context_features, num_layers)
    context = torch.randn(4, context_features, dtype=torch.float64)
    z = torch.randn(4, features, dtype=torch.float64)
    x = flow.forward(z, context)
    z_round = flow.inverse(x, context)
    assert z_round.shape == z.shape
    assert torch.allclose(z_round, z, atol=1e-4)


@pytest.mark.parametrize(
    "features,context_features,num_layers", [(45, 1, 1), (144, 191, 2)]
)
def test_build_flow_conditions_on_context(features, context_features, num_layers):
    torch.manual_seed(0)
    flow = _build_test_flow(features, context_features, num_layers)
    c = context_features
    x_zeros, _ = flow.sample(8, torch.zeros(8, c, dtype=torch.float64))
    x_ones, _ = flow.sample(8, torch.ones(8, c, dtype=torch.float64))
    assert not torch.allclose(x_zeros, x_ones)
