"""Tests for the consolidated ``calogen.data`` CaloINN sources/format.

Covers the LEMURS 3-D -> flat conversion (TruthFormat.flatten_showers), its
agreement with the legacy ``core.lemurs._transpose_and_flatten``, and
worker-pickle safety of the HDF5 / memmap sources.
"""

from __future__ import annotations

import json
import os
import pickle

import h5py
import numpy as np
import pytest

from calogen.core.lemurs import _transpose_and_flatten
from calogen.data.data_util import TruthFormat
from calogen.data.datasets.caloinn import CaloINNDataset
from calogen.data.datamodules.caloinn import CaloINNDataModule
from calogen.data.sources import HDF5Source, MemmapSource

_LEMURS_GRID = (9, 16, 45)
_LEMURS_CELLS = 9 * 16 * 45  # 6480
_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "binning_lemurs_fcceeallegro.xml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_hdf5(path, n=4, grid=None, seed=0):
    """Write a minimal truth HDF5 in either flat or LEMURS-grid layout."""
    rng = np.random.RandomState(seed)
    if grid is None:
        showers = (rng.rand(n, _LEMURS_CELLS) * 10.0).astype(np.float32)
        ekey, energies = "incident_energies", (rng.rand(n, 1) * 1000 + 1000).astype(np.float32)
    else:
        showers = (rng.rand(n, *grid) * 10.0).astype(np.float32)
        ekey, energies = "incident_energy", (rng.rand(n) * 1000 + 1000).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("showers", data=showers)
        f.create_dataset(ekey, data=energies)
    return showers, energies


def _write_memmap(meta_path, n=4, dim=6, seed=0):
    """Write a raw memmap store (x, c) + manifest for MemmapSource."""
    rng = np.random.RandomState(seed)
    x = rng.rand(n, dim).astype(np.float32)
    c = rng.rand(n, 1).astype(np.float32)
    xm = np.memmap(str(meta_path).replace("meta.json", "x.dat"), mode="w+", dtype="float32", shape=(n, dim))
    cm = np.memmap(str(meta_path).replace("meta.json", "c.dat"), mode="w+", dtype="float32", shape=(n, 1))
    xm[:] = x; xm.flush()
    cm[:] = c; cm.flush()
    meta = {"fields": {
        "x": {"file": "x.dat", "shape": [n, dim], "dtype": "float32"},
        "c": {"file": "c.dat", "shape": [n, 1], "dtype": "float32"},
    }}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return x, c


# ---------------------------------------------------------------------------
# TruthFormat flatten/unflatten
# ---------------------------------------------------------------------------

def test_flatten_unflatten_roundtrip():
    tf = TruthFormat(energy_key="incident_energy", energy_is_1d=True, showers_grid_shape=_LEMURS_GRID)
    grid = np.arange(2 * _LEMURS_CELLS, dtype=np.float32).reshape(2, *_LEMURS_GRID)
    flat = tf.flatten_showers(grid)
    assert flat.shape == (2, _LEMURS_CELLS)
    np.testing.assert_array_equal(tf.unflatten_showers(flat), grid)


def test_flatten_noop_for_flat_and_2d():
    flat_arr = np.random.rand(3, _LEMURS_CELLS).astype(np.float32)
    # flat format (no grid) -> unchanged (and same object)
    assert TruthFormat().flatten_showers(flat_arr) is flat_arr
    # LEMURS tag but already-2-D input -> unchanged
    assert TruthFormat(showers_grid_shape=_LEMURS_GRID).flatten_showers(flat_arr) is flat_arr


def test_flatten_matches_core_lemurs_transpose():
    grid = np.random.rand(4, *_LEMURS_GRID).astype(np.float32)
    tf = TruthFormat(showers_grid_shape=_LEMURS_GRID)
    np.testing.assert_array_equal(tf.flatten_showers(grid), _transpose_and_flatten(grid))


# ---------------------------------------------------------------------------
# TruthFormat detection
# ---------------------------------------------------------------------------

def test_detect_from_file_lemurs(tmp_path):
    p = str(tmp_path / "lemurs.h5")
    _write_hdf5(p, n=2, grid=_LEMURS_GRID)
    tf = TruthFormat.detect_from_file(p)
    assert tf.energy_key == "incident_energy"
    assert tf.energy_is_1d
    assert tf.showers_grid_shape == _LEMURS_GRID


def test_detect_from_file_calochallenge(tmp_path):
    p = str(tmp_path / "calo.h5")
    _write_hdf5(p, n=2)
    tf = TruthFormat.detect_from_file(p)
    assert tf.energy_key == "incident_energies"
    assert not tf.energy_is_1d
    assert tf.showers_grid_shape is None


# ---------------------------------------------------------------------------
# Sources: read order + pickle round-trip
# ---------------------------------------------------------------------------

def test_hdf5source_reads_in_requested_order(tmp_path):
    p = str(tmp_path / "lemurs.h5")
    raw, _ = _write_hdf5(p, n=5, grid=_LEMURS_GRID)
    src = HDF5Source(p)
    showers, _ = src.read_rows(np.array([4, 0, 2]))
    np.testing.assert_array_equal(showers[0], raw[4])
    np.testing.assert_array_equal(showers[1], raw[0])
    np.testing.assert_array_equal(showers[2], raw[2])


def test_hdf5source_pickle_roundtrip(tmp_path):
    p = str(tmp_path / "calo.h5")
    raw, _ = _write_hdf5(p, n=3)
    src = HDF5Source(p)
    src.read_rows(np.array([0, 1]))  # open the HDF5 handle first
    src2 = pickle.loads(pickle.dumps(src))
    s, _ = src2.read_rows(np.array([2, 1]))
    np.testing.assert_array_equal(s[0], raw[2])
    np.testing.assert_array_equal(s[1], raw[1])


def test_memmap_source_pickle_roundtrip(tmp_path):
    x, c = _write_memmap(tmp_path / "meta.json")
    src = MemmapSource(str(tmp_path / "meta.json"), TruthFormat())
    src2 = pickle.loads(pickle.dumps(src))  # _arrays/_lock dropped then rebuilt
    xs, cs = src2.read_rows(np.array([3, 1]))
    np.testing.assert_array_equal(xs[0], x[3])
    np.testing.assert_array_equal(cs[1], c[1])


# ---------------------------------------------------------------------------
# Dataset integrates the flatten (regression for 3-D LEMURS)
# ---------------------------------------------------------------------------

class _StubSource:
    """Minimal source returning pre-set rows (not an HDF5Source → local indexing)."""

    def __init__(self, showers, energies, truth_format):
        self._s, self._e, self.truth_format = showers, energies, truth_format

    def read_rows(self, idx):
        return self._s[idx], self._e[idx]

    def close(self):  # pragma: no cover
        pass


def test_dataset_flattens_3d_lemurs_before_preprocess():
    grid = np.random.rand(3, *_LEMURS_GRID).astype(np.float32) + 1.0
    energies = np.full((3,), 5000.0, dtype=np.float32)
    tf = TruthFormat(energy_key="incident_energy", energy_is_1d=True, showers_grid_shape=_LEMURS_GRID)
    source = _StubSource(grid, energies, tf)
    layer_boundaries = np.arange(0, _LEMURS_CELLS + 1, 16 * 9)  # 45 layers x 144 cells

    ds = CaloINNDataset(
        source=source,
        indices=np.arange(3),
        layer_boundaries=layer_boundaries,
        xml_filename="dummy.xml",
        particle_type="photon",
        eps=1e-10,
        u0up_cut=1e9,
        u0low_cut=0.0,
        rew=1.0,
        dep_cut=1e10,
        width_noise=0.0,
    )
    x, c = ds._get_numpy_batch(np.array([0, 1, 2]))
    # Without flattening, _build_data_dict would slice the Z axis (45) and fail.
    n_extra = len(layer_boundaries) - 1  # extra dims appended by preprocess
    assert x.shape == (3, _LEMURS_CELLS + n_extra)
    assert c.shape == (3, 1)
    assert np.isfinite(x).all()


# ---------------------------------------------------------------------------
# Memmap cache key depends on split + file identity, not preprocessing params
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(_XML), reason="binning XML not present")
def test_cache_key_depends_on_split_and_file(tmp_path):
    p = str(tmp_path / "lemurs.h5")
    _write_hdf5(p, n=4, grid=_LEMURS_GRID)

    def mk(val_frac, shuffle=True):
        return CaloINNDataModule(
            data_path=p, val_data_path=p, batch_size=2, xml_path=_XML,
            xml_ptype="photon", val_frac=val_frac, shuffle=shuffle,
            cache_mode="memmap", cache_dir=str(tmp_path / "cache"),
        )

    a = mk(0.1)
    b = mk(0.1)
    c = mk(0.5)
    d = mk(0.1, shuffle=False)

    assert a._cache_path("train") == b._cache_path("train")   # same params -> same
    assert a._cache_path("train") != a._cache_path("val")     # split name
    assert a._cache_path("train") != c._cache_path("train")   # val_frac differs
    assert a._cache_path("train") != d._cache_path("train")   # shuffle differs

    # In-place file change (mtime) invalidates the cache. Capture the key
    # *before* touching the file (the key is computed lazily at call time).
    key_before = a._cache_path("train")
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert mk(0.1)._cache_path("train") != key_before
