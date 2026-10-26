"""Tests for the model-agnostic HLF blocks and the CaloINN representation decoder.

Kept independent of any checkpoint or binning XML: the native LEMURS features
need neither.  ``calogen.eval.observables`` is used as the reference
implementation for the (numpy, MeV-based) centre-of-energy and widths.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.core import postprocess as torch_postprocess
from calogen.data.hlf import (
    CELLS_PER_LAYER,
    N_LAYERS,
    N_PHI,
    X_V,
    Y_V,
    active_fraction,
    cell_coordinates,
    center_of_energy,
    compute_hlf,
    hlf_feature_names,
    layer_energy,
    layer_energy_frac,
    width,
)
from calogen.eval import observables as obs
from calogen.models.caloinn.features.convert import cinn_sample_to_classifier_input
from calogen.models.caloinn.features.representation import (
    internal_to_classifier_input,
    layer_fractions,
    split_internal,
)

BOUNDARIES = np.arange(0, N_LAYERS * CELLS_PER_LAYER + 1, CELLS_PER_LAYER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_cells(n=6, seed=0, per_layer_normalised=False):
    rng = np.random.RandomState(seed)
    cells = rng.uniform(0.0, 1.0, size=(n, N_LAYERS, CELLS_PER_LAYER)).astype(np.float32)
    if per_layer_normalised:
        cells /= cells.sum(axis=2, keepdims=True)
    return cells.reshape(n, -1)


def _internal_batch(n=6, seed=0):
    """Cell fractions + telescoped extra dims, as the generator produces them."""
    rng = np.random.RandomState(seed)
    cells = _random_cells(n, seed, per_layer_normalised=True)
    f = rng.uniform(0.0, 0.6, size=(n, N_LAYERS - 1)).astype(np.float32)
    e_tot = rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float32)
    extra = np.concatenate([e_tot, f], axis=1)
    return torch.from_numpy(np.concatenate([cells, extra], axis=1))


# ---------------------------------------------------------------------------
# Layer deposits
# ---------------------------------------------------------------------------

def test_layer_fractions_match_unnormalize_layers():
    """The closed form must agree with the authoritative 45-step loop."""
    x = _internal_batch(n=8, seed=1)
    n_layers = N_LAYERS
    cells, extra = split_internal(x, n_layers)

    # Uniform cells make the unnormalised layer sum equal to the layer energy.
    uniform = torch.full_like(cells, 1.0 / CELLS_PER_LAYER)
    e_inc = torch.full((x.shape[0], 1), 70.0)
    unnorm = torch_postprocess.unnormalize_layers(
        torch.cat([uniform, extra], dim=1), e_inc, BOUNDARIES
    )
    reference = unnorm.reshape(-1, n_layers, CELLS_PER_LAYER).sum(dim=2) / e_inc

    got = layer_fractions(extra)
    assert got.shape == (x.shape[0], n_layers)
    np.testing.assert_allclose(got.numpy(), reference.numpy(), atol=1e-5)


def test_layer_fractions_are_deposits_not_fractions_of_remaining():
    """Sum of lambda equals E_tot / E_inc = extra_dims[:, 0]."""
    x = _internal_batch(n=5, seed=2)
    _, extra = split_internal(x, N_LAYERS)
    lam = layer_fractions(extra)
    np.testing.assert_allclose(
        lam.sum(dim=1).numpy(), extra[:, 0].numpy(), rtol=1e-5, atol=1e-6
    )
    assert (lam >= 0).all()


def test_layer_energy_frac_needs_energies_or_deposit():
    cells = torch.from_numpy(_random_cells())
    with pytest.raises(ValueError, match="layer_energy_frac needs"):
        layer_energy_frac(cells, BOUNDARIES)


def test_layer_energy_frac_from_energies_and_from_deposit():
    cells = torch.from_numpy(_random_cells())
    e_inc = torch.full((cells.shape[0], 1), 4.0)
    from_e = layer_energy_frac(cells, BOUNDARIES, energies=e_inc)
    np.testing.assert_allclose(
        from_e.numpy(), (layer_energy(cells, BOUNDARIES) / 4.0).numpy(), rtol=1e-6
    )
    supplied = torch.rand_like(from_e)
    assert torch.equal(
        layer_energy_frac(cells, BOUNDARIES, layer_frac=supplied), supplied
    )


# ---------------------------------------------------------------------------
# Cell-derived blocks
# ---------------------------------------------------------------------------

def test_coordinates_are_lemurs_r_major():
    coords = cell_coordinates()
    # r-major: index = r * N_PHI + phi, so index 1 is (r_mid[0], phi_mid[1])
    assert coords["x"].shape == (CELLS_PER_LAYER,)
    np.testing.assert_allclose(
        float(coords["x"][0]), obs.R_MID[0] * np.cos(obs.PHI_MID[0]), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(coords["y"][1]), obs.R_MID[0] * np.sin(obs.PHI_MID[1]), rtol=1e-6
    )
    np.testing.assert_allclose(
        float(coords["y"][N_PHI]), obs.R_MID[1] * np.sin(obs.PHI_MID[0]), rtol=1e-6
    )
    np.testing.assert_allclose(coords["x"].numpy(), X_V, rtol=1e-6)
    np.testing.assert_allclose(coords["y"].numpy(), Y_V, rtol=1e-6)


def test_blocks_match_eval_observables():
    cells = _random_cells(seed=3)
    out = compute_hlf(
        torch.from_numpy(cells), BOUNDARIES, blocks=("cog", "width")
    ).numpy()
    x, y, wx, wy = obs.native_center_of_energy(cells)
    np.testing.assert_allclose(out[:, :N_LAYERS], x, atol=1e-4)
    np.testing.assert_allclose(out[:, N_LAYERS:2 * N_LAYERS], y, atol=1e-4)
    np.testing.assert_allclose(out[:, 2 * N_LAYERS:3 * N_LAYERS], wx, atol=1e-3)
    np.testing.assert_allclose(out[:, 3 * N_LAYERS:], wy, atol=1e-3)


def test_hlf_are_scale_invariant():
    """A common per-layer (or per-event) energy scale must not change cog/width."""
    cells = torch.from_numpy(_random_cells(seed=4))
    a = compute_hlf(cells, BOUNDARIES, blocks=("cog", "width"))
    b = compute_hlf(cells * 1e3, BOUNDARIES, blocks=("cog", "width"))
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_empty_layer_gives_zeros_not_nan():
    cells = torch.from_numpy(_random_cells(seed=5))
    cells = cells.clone()
    cells[:, :CELLS_PER_LAYER] = 0.0  # whole first layer empty
    out = compute_hlf(cells, BOUNDARIES, blocks=("cog", "width"))
    assert torch.isfinite(out).all()
    assert float(out[:, 0].abs().max()) == 0.0  # cog_x of layer 0
    assert float(out[:, 2 * N_LAYERS].abs().max()) == 0.0  # sigma_x of layer 0


def test_active_fraction_counts_nonzero_cells():
    cells = torch.zeros(2, N_LAYERS * CELLS_PER_LAYER)
    cells[0, : CELLS_PER_LAYER // 2] = 1.0
    frac = active_fraction(cells, BOUNDARIES)
    assert float(frac[0, 0]) == pytest.approx(0.5)
    assert float(frac[0, 1]) == 0.0
    assert float(frac[1].abs().max()) == 0.0


def test_non_uniform_layers_rejected():
    cells = torch.zeros(1, 100)
    with pytest.raises(ValueError, match="equally sized layers"):
        layer_energy(cells, [0, 40, 100])


def test_compute_hlf_shape_and_names():
    cells = torch.from_numpy(_random_cells(seed=6))
    out = compute_hlf(cells, BOUNDARIES, layer_frac=torch.ones(cells.shape[0], N_LAYERS))
    names = hlf_feature_names(BOUNDARIES)
    assert out.shape == (cells.shape[0], 5 * N_LAYERS)
    assert len(names) == out.shape[1]
    assert names[0] == "layer_frac_00"
    assert names[N_LAYERS] == "cog_x_00"
    assert names[2 * N_LAYERS] == "cog_y_00"
    assert names[3 * N_LAYERS] == "width_x_00"
    assert names[4 * N_LAYERS] == "width_y_00"


# ---------------------------------------------------------------------------
# Internal representation -> classifier input
# ---------------------------------------------------------------------------

def test_internal_input_passes_cells_through_untouched():
    x = _internal_batch(n=4, seed=7)
    c = torch.linspace(1.0, 4.0, 4).reshape(-1, 1)
    cells_out, c_out, hlf = internal_to_classifier_input(
        x, c, BOUNDARIES, blocks=("layer_frac", "cog", "width")
    )

    n_cells = N_LAYERS * CELLS_PER_LAYER
    assert cells_out.shape == (4, n_cells)
    assert torch.equal(cells_out, x[:, :n_cells])  # no transformation
    assert c_out.shape == (4, 1)
    assert torch.equal(c_out, c)
    assert hlf.shape == (4, 5 * N_LAYERS)

    cells, extra = split_internal(x, N_LAYERS)
    expected = compute_hlf(
        cells, BOUNDARIES, layer_frac=layer_fractions(extra)
    )
    assert torch.allclose(hlf, expected)


def test_internal_input_without_deposit_block_is_shorter():
    x = _internal_batch(n=3, seed=8)
    c = torch.full((3, 1), 5.0)
    _, _, hlf = internal_to_classifier_input(x, c, BOUNDARIES, blocks=("cog", "width"))
    assert hlf.shape[1] == 4 * N_LAYERS


def test_internal_input_log_transform_stays_finite_with_negative_cells():
    x = _internal_batch(n=2, seed=9)
    x[:, 0] = -5.0  # pathological generator output
    c = torch.full((2, 1), 5.0)
    cells_out, _, _ = internal_to_classifier_input(x, c, BOUNDARIES, log_transform=True)
    assert torch.isfinite(cells_out).all()


def test_convert_representation_switch():
    x = _internal_batch(n=2, seed=10)
    c = np.full((2, 1), 5.0, dtype=np.float32)
    boundaries = BOUNDARIES.tolist()
    kwargs = dict(q=torch.zeros(x.shape[1]), width_noise=0.0, xml_path="unused.xml", particle="photon")

    internal = cinn_sample_to_classifier_input(
        x.numpy(), c, boundaries, representation="internal", **kwargs
    )
    assert internal.shape == (2, N_LAYERS * CELLS_PER_LAYER + 1 + 5 * N_LAYERS)
    assert internal.dtype == np.float32

    with pytest.raises(ValueError, match="representation must be"):
        cinn_sample_to_classifier_input(
            x.numpy(), c, boundaries, representation="nonsense", **kwargs
        )
