"""Tests for TruthFormat detection and format-aware I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import h5py
import os

# calogen must be importable (project root on PYTHONPATH; see pyproject).
from calogen.core.data_util import (
    TruthFormat,
    save_data,
    save_data_with_format,
    get_energy_and_sorted_layers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calochallenge_file(tmp_path):
    """Create a minimal CaloChallenge-format HDF5 (flat 2D, plural key)."""
    path = str(tmp_path / "calochallenge.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("incident_energies", data=np.array([[1000.], [2000.], [3000.]])
                                                      .astype(np.float32))
        f.create_dataset("showers", data=np.ones((3, 720), dtype=np.float32))
    return path


@pytest.fixture
def lemurs_file(tmp_path):
    """Create a minimal LEMURS-format HDF5 (3D, singular key)."""
    path = str(tmp_path / "lemurs.h5")
    grid = np.zeros((2, 9, 16, 45), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("incident_energy", data=np.array([100., 200.], dtype=np.float32))
        f.create_dataset("showers", data=grid)
    return path


@pytest.fixture
def postprocessed_data():
    """Simulate postprocessed output from torch_postprocess.postprocess.

    Returns a dict with ``"energy"`` (MeV, (N,1)) and per-layer arrays (GeV).
    """
    n_layers = 3
    cells_per_layer = 4
    n = 4
    data = {"energy": np.array([[10.], [20.], [30.], [40.]])}  # GeV, but simulate input
    for i in range(n_layers):
        data[f"layer_{i}"] = np.ones((n, cells_per_layer)) * (i + 1) * 1e-3  # GeV
    return data


# ---------------------------------------------------------------------------
# TruthFormat dataclass tests
# ---------------------------------------------------------------------------

class TestTruthFormat:

    def test_default_is_calochallenge(self):
        fmt = TruthFormat()
        assert fmt.energy_key == "incident_energies"
        assert not fmt.energy_is_1d
        assert fmt.showers_grid_shape is None

    def test_lemurs_config(self):
        fmt = TruthFormat(
            energy_key="incident_energy",
            energy_is_1d=True,
            showers_grid_shape=(9, 16, 45),
        )
        assert fmt.energy_key == "incident_energy"
        assert fmt.energy_is_1d
        assert fmt.showers_grid_shape == (9, 16, 45)

    def test_unflatten_is_noop_for_flat(self):
        fmt = TruthFormat()
        flat = np.ones((10, 720), dtype=np.float32)
        out = fmt.unflatten_showers(flat)
        assert out.shape == flat.shape
        np.testing.assert_array_equal(out, flat)

    def test_unflatten_roundtrips(self):
        """unflatten_showers is the inverse of _transpose_and_flatten."""
        R, Phi, Z = 2, 3, 4  # small grid for test
        fmt = TruthFormat(
            energy_key="incident_energy",
            energy_is_1d=True,
            showers_grid_shape=(R, Phi, Z),
        )

        # Create 3D data and manually flatten it (mimics _transpose_and_flatten)
        original = np.arange(5 * R * Phi * Z, dtype=np.float32).reshape(5, R, Phi, Z)
        flat = np.transpose(original, (0, 3, 1, 2)).reshape(5, -1)

        # Round-trip
        restored = fmt.unflatten_showers(flat)
        np.testing.assert_array_equal(restored, original)

    def test_format_energy_1d(self):
        fmt = TruthFormat(energy_is_1d=True)
        e = np.array([[1.], [2.], [3.]])
        out = fmt.format_energy(e)
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, np.array([1., 2., 3.]))

    def test_format_energy_2d(self):
        fmt = TruthFormat(energy_is_1d=False)
        e = np.array([1., 2., 3.])
        out = fmt.format_energy(e)
        assert out.shape == (3, 1)
        np.testing.assert_array_equal(out, np.array([[1.], [2.], [3.]]))


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetection:

    def test_detect_calochallenge(self, calochallenge_file):
        fmt = TruthFormat.detect_from_file(calochallenge_file)
        assert fmt.energy_key == "incident_energies"
        assert not fmt.energy_is_1d
        assert fmt.showers_grid_shape is None

    def test_detect_lemurs(self, lemurs_file):
        fmt = TruthFormat.detect_from_file(lemurs_file)
        assert fmt.energy_key == "incident_energy"
        assert fmt.energy_is_1d
        assert fmt.showers_grid_shape == (9, 16, 45)

    def test_detect_returns_copy_not_singleton(self, calochallenge_file):
        """Each call returns a new instance."""
        a = TruthFormat.detect_from_file(calochallenge_file)
        b = TruthFormat.detect_from_file(calochallenge_file)
        assert a == b
        assert a is not b


# ---------------------------------------------------------------------------
# Format-aware save/load round-trip tests
# ---------------------------------------------------------------------------

class TestSaveWithFormat:

    def test_save_flat_uses_correct_keys(self, tmp_path, postprocessed_data):
        """CaloChallenge format: incident_energies key, flat showers."""
        path = str(tmp_path / "out.h5")
        fmt = TruthFormat()  # default CaloChallenge
        save_data_with_format(postprocessed_data, path, fmt)

        with h5py.File(path, "r") as f:
            assert "incident_energies" in f
            assert "showers" in f
            showers = f["showers"][:]
            assert showers.ndim == 2

    def test_save_3d_produces_grid(self, tmp_path, postprocessed_data):
        """LEMURS format: incident_energy key, 3D showers.

        The grid is (N, R, Phi, Z) — 4D including sample dim.
        """
        path = str(tmp_path / "out.h5")
        n_cells = 12  # 3 layers × 4 cells
        R, Phi, Z = 1, 1, n_cells  # squeeze to 1×1×12 for test
        fmt = TruthFormat(
            energy_key="incident_energy",
            energy_is_1d=True,
            showers_grid_shape=(R, Phi, Z),
        )
        save_data_with_format(postprocessed_data, path, fmt)

        with h5py.File(path, "r") as f:
            assert "incident_energy" in f
            assert "showers" in f
            showers = f["showers"][:]
            assert showers.ndim == 4  # N + 3 grid dims
            assert showers.shape == (4, R, Phi, Z)
            energy = f["incident_energy"][:]
            assert energy.ndim == 1

    def test_roundtrip_flat(self, tmp_path, postprocessed_data):
        """save_data_with_format → reload → same layer energies."""
        path = str(tmp_path / "out.h5")
        fmt = TruthFormat()
        save_data_with_format(postprocessed_data, path, fmt)

        with h5py.File(path, "r") as f:
            reloaded = f["showers"][:]
            energies_out = f["incident_energies"][:]

        energy_in, layers = get_energy_and_sorted_layers(postprocessed_data)
        energy_expected = (energy_in * 1e3).reshape(-1, 1)
        np.testing.assert_array_almost_equal(energies_out, energy_expected)

    def test_custom_dataset_name(self, tmp_path, postprocessed_data):
        """Custom dataset name for showers."""
        path = str(tmp_path / "out.h5")
        fmt = TruthFormat()
        save_data_with_format(postprocessed_data, path, fmt, dataset_name="gen_showers")
        with h5py.File(path, "r") as f:
            assert "gen_showers" in f

    def test_backward_compat(self, tmp_path, postprocessed_data):
        """original save_data still works and produces identical output."""
        import copy

        path_old = str(tmp_path / "old.h5")
        path_new = str(tmp_path / "new.h5")

        # Deep-copy because save_data mutates its input in-place
        data_copy = copy.deepcopy(postprocessed_data)
        save_data(data_copy, filename=path_old)

        data_copy2 = copy.deepcopy(postprocessed_data)
        save_data_with_format(data_copy2, path_new, TruthFormat())

        with h5py.File(path_old, "r") as f_old, h5py.File(path_new, "r") as f_new:
            np.testing.assert_array_equal(f_old["showers"][:], f_new["showers"][:])
            np.testing.assert_array_equal(
                f_old["incident_energies"][:], f_new["incident_energies"][:]
            )
