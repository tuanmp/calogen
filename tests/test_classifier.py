"""Unit tests for the model-agnostic NCE classifier base and CaloINN features.

No checkpoints required: a mock generator supplies `_get_latent` so the shared
noise-contrastive math can be exercised directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.classifiers.base import MLP, Classifier, NoiseContrastiveDensityRatioEstimator
from calogen.models.caloinn.features.convert import (
    _apply_voxel_cutoff,
    _compute_hlf,
    _scale_shower,
    cinn_sample_to_classifier_input,
)

_XML = os.path.join(
    os.path.dirname(__file__), "..", "configs", "binning_pion_odd_coarse_hcal.xml"
)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_output_shape_and_grad():
    net = MLP(hidden_dim=16, num_layers=2, layer_norm=True, dropout=0.0)
    x = torch.randn(8, 5, requires_grad=True)
    out = net(x)
    assert out.shape == (8, 1)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Shared NCE math (mock generator)
# ---------------------------------------------------------------------------

class _MockNCE(NoiseContrastiveDensityRatioEstimator):
    """NCE with a trivial latent map, to test the shared base in isolation."""

    def __init__(self, latent_dim=4, cond_dim=1, **kw):
        super().__init__(hidden_dim=8, num_layers=1, dropout=0.0, **kw)
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

    @torch.inference_mode()
    def _get_latent(self, x, c):
        return x[:, : self.latent_dim]


def _batch(n=6, latent_dim=4, cond_dim=1):
    x = torch.randn(n, latent_dim)
    c = torch.rand(n, cond_dim)
    return (x, c)


def test_nce_batch_shapes_with_noise_amplifier():
    m = _MockNCE(noise_amplifier=3)
    X, y, z = m.get_input_from_batch(_batch())
    assert z.shape == (6, 4)
    assert X.shape == (6 + 18, 4 + 1)   # truth + M*noise, latent + cond
    assert y.shape == (24, 1)
    assert torch.all(y[:6] == 1) and torch.all(y[6:] == 0)


def test_nce_training_step_and_density_ratio():
    m = _MockNCE(noise_amplifier=2)
    out = m.training_step(_batch(), 0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    z = torch.randn(5, 4)
    c = torch.rand(5, 1)
    ratio = m.estimate_density_ratio(z, c)
    assert ratio.shape == (5, 1)
    log_density = m.estimate_log_density(z, c)
    assert log_density.shape == (5,)
    grad = m.estimate_log_density_gradient(z.clone(), c)
    assert grad.shape == (5, 4)


def test_nce_base_is_generator_free():
    """The shared base must not import a concrete generative model."""
    import calogen.classifiers.base as base
    src = open(base.__file__).read()
    assert "calogen.models" not in src
    assert "CaloINNLightningModule" not in src
    assert "ICaloFlowLightningModule" not in src


def test_nce_classifier_requires_get_input():
    with pytest.raises(NotImplementedError):
        Classifier(hidden_dim=8, num_layers=1).get_input_from_batch((torch.randn(2, 3),))


# ---------------------------------------------------------------------------
# CaloINN shower -> classifier-input conversion
# ---------------------------------------------------------------------------

class TestConversion:
    def test_scale_shower(self):
        X_mev = np.array([[1000.0, 2000.0], [500.0, 1000.0]], dtype=np.float32)
        Einc_mev = np.array([[2000.0], [1000.0]], dtype=np.float32)
        X_proc, cond_proc = _scale_shower(X_mev, Einc_mev)
        np.testing.assert_allclose(X_proc[0], [0.5, 1.0], atol=1e-5)
        np.testing.assert_allclose(cond_proc[0], [2.0], atol=1e-5)

    def test_voxel_cutoff(self):
        X = np.array([[0.5, 2.0, 0.1], [3.0, 0.8, 4.0]], dtype=np.float32)
        result = _apply_voxel_cutoff(X, cutoff_mev=1.0)
        expected = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_voxel_cutoff_none(self):
        X = np.random.randn(4, 10).astype(np.float32)
        result = _apply_voxel_cutoff(X, cutoff_mev=None)
        np.testing.assert_array_equal(result, X)

    def test_hlf_shape(self):
        rng = np.random.RandomState(42)
        X_mev = rng.uniform(0, 1000, size=(4, 720)).astype(np.float32)
        Einc_mev = np.full((4, 1), 65536.0, dtype=np.float32)
        hlf = _compute_hlf(X_mev, Einc_mev, _XML, "pion")
        assert hlf.shape == (4, 51), f"Expected (4, 51), got {hlf.shape}"

    def test_full_conversion_shape(self):
        rng = np.random.RandomState(42)
        x = rng.uniform(0.01, 0.5, size=(2, 730)).astype(np.float32)
        c = np.array([[60.0], [70.0]], dtype=np.float32)
        q = torch.tensor(5e-6)
        layer_boundaries = [0, 48, 96, 144, 192, 240, 336, 432, 528, 624, 720]

        result = cinn_sample_to_classifier_input(
            x, c, layer_boundaries, q=q, width_noise=5e-6,
            xml_path=_XML, particle="pion",
            log_transform=False, voxel_energy_cutoff=None,
        )
        assert result.shape == (2, 772)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_conversion_with_log_transform(self):
        rng = np.random.RandomState(42)
        x = rng.uniform(0.01, 0.5, size=(2, 730)).astype(np.float32)
        c = np.array([[60.0], [70.0]], dtype=np.float32)
        q = torch.tensor(5e-6)
        layer_boundaries = [0, 48, 96, 144, 192, 240, 336, 432, 528, 624, 720]

        result = cinn_sample_to_classifier_input(
            x, c, layer_boundaries, q=q, width_noise=5e-6,
            xml_path=_XML, particle="pion",
            log_transform=True, voxel_energy_cutoff=1.0,
        )
        assert np.all(result[:, :720] >= 0)
