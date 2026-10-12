import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pytest

from calogen.eval.io import ShowerSet, load_showers

R, PHI, Z = 9, 16, 45


def _write_lemurs(path, n=8, seed=0):
    rng = np.random.RandomState(seed)
    grid = rng.random((n, R, PHI, Z)).astype(np.float32)
    energy = np.linspace(1e3, 1e5, n).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("showers", data=grid)
        f.create_dataset("incident_energy", data=energy)
    return grid, energy


def test_load_showers_shape_and_dtype(tmp_path):
    path = str(tmp_path / "t.h5")
    _write_lemurs(path)
    s = load_showers(path, "truth")
    assert isinstance(s, ShowerSet)
    assert s.showers.shape == (8, 6480)
    assert s.showers.dtype == np.float32
    assert s.energies.shape == (8,)
    assert s.grid_shape == (R, PHI, Z)


def test_layer_order_matches_source(tmp_path):
    path = str(tmp_path / "t.h5")
    grid, _ = _write_lemurs(path)
    s = load_showers(path, "truth")
    # layer 0 block reshaped (R,Phi) must equal raw grid[:, :, :, 0]
    block = s.showers[0, 0:144].reshape(R, PHI)
    assert np.allclose(block, grid[0, :, :, 0])


def test_max_events_deterministic(tmp_path):
    path = str(tmp_path / "t.h5")
    _write_lemurs(path)
    a = load_showers(path, "t", max_events=3, seed=7)
    b = load_showers(path, "t", max_events=3, seed=7)
    c = load_showers(path, "t", max_events=3, seed=8)
    assert np.array_equal(a.showers, b.showers)
    assert not np.array_equal(a.showers, c.showers)


def test_threshold(tmp_path):
    path = str(tmp_path / "t.h5")
    _write_lemurs(path)
    s = load_showers(path, "t", threshold_mev=0.5)
    assert (s.showers == 0).any()
    assert s.showers.max() > 0.5


def test_energy_range(tmp_path):
    path = str(tmp_path / "t.h5")
    _write_lemurs(path)
    s = load_showers(path, "t", energy_range=(1e4, 1e5))
    assert s.energies.min() >= 1e4
