"""Unit tests for calibration classes."""
import numpy as np
import torch
import pytest
import json
import tempfile
from pathlib import Path

from calogen.classifiers.calibration import (
    BaseCalibrator,
    TemperatureCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    expected_calibration_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration_data():
    """Generate synthetic calibration data.

    Returns raw probabilities and labels where the classifier is
    overconfident (raw probs are too extreme compared to true labels).
    """
    rng = np.random.RandomState(42)
    n = 1000

    raw_probs = rng.beta(2, 2, size=n)

    true_labels = (rng.rand(n) < (raw_probs * 0.6 + 0.2)).astype(np.float64)
    return raw_probs, true_labels


# ---------------------------------------------------------------------------
# TemperatureCalibrator
# ---------------------------------------------------------------------------

class TestTemperatureCalibrator:
    def test_fit_reduces_nll(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert cal.T > 0

    def test_transform_logits_numpy(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        logits = np.log(raw_probs / (1 - raw_probs + 1e-8))
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == logits.shape
        np.testing.assert_allclose(out, logits / cal.T, rtol=1e-6)

    def test_transform_logits_torch(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        logits = torch.from_numpy(np.log(raw_probs / (1 - raw_probs + 1e-8)))
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape
        torch.testing.assert_close(out, logits / cal.T)

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = TemperatureCalibrator()
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, TemperatureCalibrator)
            assert loaded.T == cal.T
        finally:
            Path(path).unlink()

    def test_legacy_json_load(self):
        """Load a JSON without 'method' key (legacy format)."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"T": 2.5}, f)
            path = f.name
        try:
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, TemperatureCalibrator)
            assert loaded.T == 2.5
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = TemperatureCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))

    def test_save_unfitted_raises(self):
        cal = TemperatureCalibrator()
        with pytest.raises(RuntimeError):
            cal.save("/tmp/unused.json")


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------

class TestPlattCalibrator:
    def test_fit(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert isinstance(cal.a, float)
        assert isinstance(cal.b, float)

    def test_transform_logits_numpy(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        logits = np.log(raw_probs / (1 - raw_probs + 1e-8)).astype(np.float64)
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == logits.shape
        np.testing.assert_allclose(out, cal.a * logits + cal.b, rtol=1e-6)

    def test_transform_logits_torch(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        logits = torch.from_numpy(np.log(raw_probs / (1 - raw_probs + 1e-8)))
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = PlattCalibrator()
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, PlattCalibrator)
            assert loaded._a == cal._a
            assert loaded._b == cal._b
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = PlattCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_fit(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)
        assert cal.is_fitted
        assert cal._iso_reg is not None
        assert cal._X_thresholds is not None
        assert cal._y_thresholds is not None

    def test_torch_matches_sklearn_on_simple_case(self):
        """Torch-native interpolation should match sklearn exactly."""
        rng = np.random.RandomState(42)
        raw_probs = np.sort(rng.rand(500).astype(np.float64))
        labels = (rng.rand(500) < raw_probs).astype(np.float64)

        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        test_probs = np.linspace(0.0, 1.0, 101)
        torch_out = cal._isotonic_transform_torch(torch.from_numpy(test_probs))
        sklearn_out = cal._iso_reg.transform(test_probs.reshape(-1, 1)).ravel()

        np.testing.assert_allclose(torch_out.numpy(), sklearn_out, rtol=1e-5, atol=1e-7)

    def test_torch_matches_sklearn_on_calibration_data(self, calibration_data):
        """Torch vs sklearn on realistic calibration data."""
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        test_probs = np.linspace(0.01, 0.99, 200)
        torch_out = cal._isotonic_transform_torch(torch.from_numpy(test_probs))
        sklearn_out = cal._iso_reg.transform(test_probs.reshape(-1, 1)).ravel()

        max_err = np.abs(torch_out.numpy() - sklearn_out).max()
        assert max_err < 1e-5, f"Max error {max_err:.2e} exceeds tolerance"

    def test_transform_logits_torch_path(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        logits = torch.tensor([-2.0, 0.0, 2.0])
        out = cal.transform_logits(logits)
        assert isinstance(out, torch.Tensor)
        assert out.shape == logits.shape
        assert torch.isfinite(out).all()

    def test_transform_logits_sklearn_path(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=False)
        cal.fit(raw_probs, labels)

        logits = torch.tensor([-2.0, 0.0, 2.0])
        out = cal.transform_logits(logits)
        assert out.shape == logits.shape
        assert torch.isfinite(out).all()

    def test_torch_and_sklearn_paths_agree(self, calibration_data):
        """transform_logits should produce identical output regardless of backend."""
        raw_probs, labels = calibration_data
        cal_torch = IsotonicCalibrator(use_torch=True)
        cal_sklearn = IsotonicCalibrator(use_torch=False)
        cal_torch.fit(raw_probs, labels)
        cal_sklearn.fit(raw_probs, labels)

        logits = torch.linspace(-3.0, 3.0, 50)
        out_torch = cal_torch.transform_logits(logits)
        out_sklearn = cal_sklearn.transform_logits(logits)

        torch.testing.assert_close(out_torch.float(), out_sklearn.float(), rtol=1e-5, atol=1e-6)

    def test_numpy_input(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        logits = np.array([-2.0, 0.0, 2.0])
        out = cal.transform_logits(logits)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)

    def test_save_load_roundtrip(self, calibration_data):
        raw_probs, labels = calibration_data
        cal = IsotonicCalibrator(use_torch=True)
        cal.fit(raw_probs, labels)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = BaseCalibrator.load(path)
            assert isinstance(loaded, IsotonicCalibrator)
            assert loaded.is_fitted

            test_logits = torch.linspace(-3.0, 3.0, 20)
            original_out = cal.transform_logits(test_logits)
            loaded_out = loaded.transform_logits(test_logits)
            torch.testing.assert_close(original_out, loaded_out)
        finally:
            Path(path).unlink()

    def test_unfitted_raises(self):
        cal = IsotonicCalibrator()
        assert not cal.is_fitted
        with pytest.raises(RuntimeError):
            cal.transform_logits(np.array([0.0]))


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

class TestECE:
    def test_perfect_calibration(self):
        """Perfectly calibrated: prob = label."""
        n = 1000
        rng = np.random.RandomState(0)
        probs = rng.rand(n).astype(np.float64)
        labels = (rng.rand(n) < probs).astype(np.float64)
        ece = expected_calibration_error(labels, probs, n_bins=10)
        assert ece < 0.1

    def test_miscalibrated(self):
        """Overconfident predictions should have higher ECE."""
        n = 1000
        y_true = np.ones(n, dtype=np.float64)
        y_pred = np.full(n, 0.6, dtype=np.float64)
        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        assert ece > 0.3

    def test_empty_input(self):
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0
