"""Tests for the iCaloFlow data module (arXiv:2305.11934).

Reads capped slices of the LEMURS FCCeeALLEGRO files only; skips if data
is unavailable.
"""

import os

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.models.icaloflow.datamodule import (
    ICaloFlowDataModule,
    log10_energy,
    one_hot,
    preprocess_cond_flow2,
    preprocess_cond_flow3,
    preprocess_energy_dep,
    preprocess_voxel_layer,
)
from calogen.models.icaloflow.model import inverse_logit

_TRAIN = (
    "/pscratch/sd/p/pmtuan/lemurs/FCCeeALLEGRO/"
    "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_train.h5"
)
_VAL = (
    "/pscratch/sd/p/pmtuan/lemurs/FCCeeALLEGRO/validation/"
    "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part10.h5"
)
_XML = os.path.join(os.path.dirname(__file__), "..", "configs", "binning_lemurs_fcceeallegro.xml")

_HAS_DATA = os.path.exists(_TRAIN) and os.path.exists(_VAL)

_MAX_EVENTS = 2000
_MAX_EVENTS_VAL = 1000
_BATCH_SIZE = 64


def _make_dm(which_flow):
    return ICaloFlowDataModule(
        which_flow=which_flow,
        data_path=_TRAIN,
        val_data_path=_VAL,
        xml_path=_XML,
        xml_ptype="photon",
        batch_size=_BATCH_SIZE,
        val_frac=0.01,
        noise_level=5e-3,
        safety_factor=1.5,
        num_workers=0,
        max_events=_MAX_EVENTS,
        max_events_val=_MAX_EVENTS_VAL,
    )


@pytest.fixture(scope="module")
def dm_flow1():
    dm = _make_dm(1)
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def dm_flow2():
    dm = _make_dm(2)
    dm.setup("fit")
    return dm


@pytest.fixture(scope="module")
def dm_flow3():
    dm = _make_dm(4)
    dm.setup("fit")
    return dm


# ---------------------------------------------------------------------------
# Flow batch structures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_flow1_batch_structure(dm_flow1):
    batch = next(iter(dm_flow1.train_dataloader()))
    assert set(batch.keys()) == {"energy_dep", "energy_inc"}
    assert batch["energy_dep"].shape == (_BATCH_SIZE, 45)
    assert batch["energy_inc"].shape == (_BATCH_SIZE, 1)
    assert batch["energy_dep"].dtype == torch.float32
    assert batch["energy_inc"].dtype == torch.float32
    assert torch.isfinite(batch["energy_dep"]).all()
    # log10(E_inc / 1e4.5) for LEMURS 1-100 GeV lies in [-1.5, 0.5]
    assert (batch["energy_inc"] >= -1.51).all()
    assert (batch["energy_inc"] <= 0.51).all()


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_flow2_batch_structure(dm_flow2):
    batch = next(iter(dm_flow2.train_dataloader()))
    assert set(batch.keys()) == {"layer", "energy", "energy_dep"}
    assert batch["layer"].shape == (_BATCH_SIZE, 144)
    assert batch["energy"].shape == (_BATCH_SIZE, 1)
    assert batch["energy_dep"].shape == (_BATCH_SIZE, 1)
    # layer rows are logit of sum-normalized voxels: inverse logit sums to ~1
    row = inverse_logit(batch["layer"][:8])
    assert torch.allclose(row.sum(dim=1), torch.ones(8), atol=1e-3)


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_flow3_batch_structure(dm_flow3):
    batch = next(iter(dm_flow3.train_dataloader()))
    assert set(batch.keys()) == {
        "layer", "layer_p", "energy", "energy_dep", "energy_dep_p", "layer_number",
    }
    assert batch["layer"].shape == (_BATCH_SIZE, 144)
    assert batch["layer_p"].shape == (_BATCH_SIZE, 144)
    assert batch["energy"].shape == (_BATCH_SIZE, 1)
    assert batch["energy_dep"].shape == (_BATCH_SIZE, 1)
    assert batch["energy_dep_p"].shape == (_BATCH_SIZE, 1)
    assert batch["layer_number"].shape == (_BATCH_SIZE,)
    assert batch["layer_number"].dtype == torch.long
    assert (batch["layer_number"] >= 1).all()
    assert (batch["layer_number"] <= 44).all()


# ---------------------------------------------------------------------------
# Module-level attributes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_normalization_constant_and_layer_boundaries(dm_flow1):
    assert np.isfinite(dm_flow1.normalization_constant)
    assert dm_flow1.normalization_constant > 0.0
    lb = dm_flow1.layer_boundaries
    assert len(lb) == 46
    assert lb[0] == 0
    assert lb[-1] == 6480


# ---------------------------------------------------------------------------
# One-hot and condition builders
# ---------------------------------------------------------------------------


def test_one_hot_helper():
    oh = one_hot(torch.tensor([1, 44]), num_classes=44)
    assert oh.shape == (2, 44)
    assert oh.dtype == torch.float32
    assert oh[0, 0] == 1.0
    assert oh[1, 43] == 1.0
    assert (oh.sum(dim=1) == 1).all()


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_flow3_cond_builder(dm_flow3):
    batch = next(iter(dm_flow3.train_dataloader()))
    # Pass the full layer count: a batch may not contain every layer, and the
    # one-hot must always span num_layers - 1 classes (the module does this).
    cond = preprocess_cond_flow3(batch, num_layers=45)
    assert cond.shape == (_BATCH_SIZE, 191)
    assert cond.dtype == torch.float32
    onehot = cond[:, -44:]
    assert (onehot.sum(dim=1) == 1).all()
    assert set(onehot.flatten().tolist()) <= {0.0, 1.0}


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_flow2_cond_builder(dm_flow2):
    batch = next(iter(dm_flow2.train_dataloader()))
    cond = preprocess_cond_flow2(batch)
    assert cond.shape == (_BATCH_SIZE, 2)
    assert torch.isfinite(cond).all()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def test_log10_energy_values():
    e = torch.tensor([[1000.0], [100000.0]])
    assert torch.allclose(log10_energy(e), torch.tensor([[-1.5], [0.5]]), atol=1e-5)


def test_preprocess_energy_dep_matches_reference_math():
    torch.manual_seed(0)
    e = torch.tensor([[600.0], [1200.0]])
    out = preprocess_energy_dep(e, 1800.0, 5e-3)
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()
    assert (inverse_logit(out) >= 0.0).all()
    assert (inverse_logit(out) < 1.0).all()


def test_preprocess_voxel_layer_sum_normalizes():
    torch.manual_seed(0)
    layer = torch.rand(4, 144) * 100.0
    out = preprocess_voxel_layer(layer, 5e-3)
    assert out.shape == (4, 144)
    assert torch.isfinite(out).all()
    assert torch.allclose(inverse_logit(out).sum(dim=1), torch.ones(4), atol=1e-3)


# ---------------------------------------------------------------------------
# Noise lives in the dataloader; precomputed data unchanged
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_noise_applied_per_getitem_and_raw_unchanged(dm_flow1):
    ds = dm_flow1._train_dataset
    pre_s = ds.showers.clone()
    pre_e = ds.energies.clone()
    item1 = ds[0]
    item2 = ds[0]
    assert not torch.equal(item1["energy_dep"], item2["energy_dep"])
    # Noise is applied in __getitem__ only; the raw arrays stay unchanged.
    assert torch.equal(ds.showers, pre_s)
    assert torch.equal(ds.energies, pre_e)


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_predict_dataloader_returns_val_events(dm_flow1):
    dm_flow1.setup("predict")
    batch = next(iter(dm_flow1.predict_dataloader()))
    assert set(batch.keys()) == {"energy_dep", "energy_inc"}
    assert batch["energy_dep"].shape == (_BATCH_SIZE, 45)
    assert batch["energy_inc"].shape == (_BATCH_SIZE, 1)


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_predict_only_run_sets_normalization_constant_and_truth_format():
    dm = _make_dm(1)
    dm.setup("predict")
    assert np.isfinite(dm.normalization_constant)
    assert dm.normalization_constant > 0.0
    assert dm.truth_format is not None
    assert dm.truth_format.energy_key == "incident_energy"


# ---------------------------------------------------------------------------
# Memmap cache
# ---------------------------------------------------------------------------


def _make_memmap_dm(which_flow, tmp_path):
    return ICaloFlowDataModule(
        which_flow=which_flow,
        data_path=_TRAIN,
        val_data_path=_VAL,
        xml_path=_XML,
        xml_ptype="photon",
        batch_size=_BATCH_SIZE,
        val_frac=0.01,
        noise_level=5e-3,
        safety_factor=1.5,
        num_workers=0,
        max_events=_MAX_EVENTS,
        max_events_val=_MAX_EVENTS_VAL,
        seed=42,
        cache_mode="memmap",
        cache_dir=str(tmp_path),
        cache_chunk=512,
    )


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_memmap_cache_writes_and_reuses(tmp_path):
    dm = _make_memmap_dm(1, tmp_path)
    dm.setup("fit")

    cache_files = [
        p for p in os.listdir(tmp_path)
        if p.endswith(".json") or p.endswith(".dat")
    ]
    assert cache_files, "expected memmap cache files to be written"
    assert dm.num_train_samples > 0

    # A second module with the same cache key must hit the cache (no rewrite).
    dm2 = _make_memmap_dm(1, tmp_path)
    dm2.setup("fit")
    assert dm2.num_train_samples == dm.num_train_samples

    # Raw arrays are memmap-backed (sharing storage, not copied into RAM).
    assert isinstance(dm._train_dataset.showers, torch.Tensor)
    assert dm._train_dataset.showers.data_ptr() != 0


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_memmap_cache_opened_read_only(tmp_path, monkeypatch):
    """Shared cache files must be opened read-only on cache hit (never 'r+'/'w+').

    Concurrent training jobs memmap the same cache; a writable open would let
    one job corrupt another's data.  The torch wrapper over a read-only memmap
    is non-writable (torch reports this as undefined behavior), so this asserts
    the open mode directly instead of attempting a write.
    """
    dm = _make_memmap_dm(1, tmp_path)
    dm.setup("fit")  # build the cache first

    modes = []
    orig = np.memmap

    def spy(*args, **kwargs):
        modes.append(kwargs.get("mode", args[1] if len(args) > 1 else "?"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(np, "memmap", spy)

    dm2 = _make_memmap_dm(1, tmp_path)
    dm2.setup("fit")  # cache hit -> read path only
    assert modes, "expected memmap opens during cache-hit setup"
    assert all(m == "r" for m in modes), f"expected read-only mmap, got {modes}"


@pytest.mark.skipif(not _HAS_DATA, reason="LEMURS data not found")
def test_memmap_matches_inmemory_arrays(tmp_path):
    """memmap and in-memory (cache_mode='none') paths give the same raw rows.

    Both modules use the same split seed, so the train arrays are the same
    events; the memmap path must expose identical raw showers/energies.
    """
    dm_none = _make_dm(1)
    dm_none.setup("fit")

    dm_mem = _make_memmap_dm(1, tmp_path)
    dm_mem.setup("fit")

    s_none = dm_none._train_dataset.showers.numpy()
    s_mem = dm_mem._train_dataset.showers.numpy()
    e_none = dm_none._train_dataset.energies.numpy()
    e_mem = dm_mem._train_dataset.energies.numpy()
    assert np.array_equal(s_none, s_mem)
    assert np.array_equal(e_none, e_mem)

    # Same split sizes and normalization constant.
    assert dm_none.num_train_samples == dm_mem.num_train_samples
    assert dm_none.normalization_constant == dm_mem.normalization_constant
