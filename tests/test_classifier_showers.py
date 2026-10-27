"""Raw-shower → classifier-input pipeline (the ``caloxtreme_clf`` contract).

The caloxtreme classifier was trained on the *physical* representation built
from raw showers: ``[log1p(X / E_inc) | E_inc[GeV] | XML high-level features]``.
These tests pin that layout and the deterministic (``shuffle=False``) split so
the same checkpoints can be evaluated/sampled from calogen.
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.data.datamodules.classifier import ShowerClassifierDataModule
from calogen.data.datasets.classifier import ShowerClassifierDataset
from calogen.data.sources import HDF5Source
from calogen.models.caloinn.features.convert import (
    _apply_voxel_cutoff,
    _compute_hlf,
    _scale_shower,
    showers_to_classifier_input,
    showers_to_classifier_input_torch,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_XML = os.path.join(_HERE, "..", "configs", "binning_lemurs_fcceeallegro.xml")
_CELLS = 9 * 16 * 45          # 6480 LEMURS cells
_HLF = 1 + 45 + 4 * 45        # E_tot + E_layers + EC_eta/phi + width_eta/phi
_DIM = _CELLS + 1 + _HLF      # 6707: caloxtreme checkpoint input width

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_XML), reason="LEMURS binning XML not available"
)

_CALOXTREME = "/global/cfs/cdirs/m3443/usr/pmtuan/caloxtreme_clf"


def _write_lemurs_h5(path: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    showers = rng.gamma(shape=0.5, scale=10.0, size=(n, 9, 16, 45)).astype(np.float32)
    energies = rng.uniform(1_000.0, 100_000.0, size=n).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("showers", data=showers, chunks=(min(n, 8), 9, 16, 45))
        f.create_dataset("incident_energy", data=energies, chunks=(min(n, 8),))
    return showers, energies


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_transform,cutoff", [(False, None), (True, 1.0)])
def test_showers_to_classifier_input_matches_step_by_step_pipeline(log_transform, cutoff):
    rng = np.random.RandomState(0)
    x_mev = rng.gamma(0.5, 10.0, size=(4, _CELLS)).astype(np.float32)
    e_mev = np.array([10_000.0, 20_000.0, 40_000.0, 80_000.0], dtype=np.float32)

    out = showers_to_classifier_input(
        x_mev, e_mev, xml_path=_XML, particle="photon",
        log_transform=log_transform, voxel_energy_cutoff=cutoff,
    )

    cut = _apply_voxel_cutoff(x_mev, cutoff)
    hlf = _compute_hlf(cut, e_mev, _XML, "photon")
    x_proc, cond = _scale_shower(cut, e_mev)
    if log_transform:
        x_proc = np.log1p(x_proc)
    ref = np.concatenate([x_proc, cond, hlf], axis=1).astype(np.float32)

    assert out.shape == (4, _DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=0, atol=0)
    # condition column is the incident energy in GeV
    np.testing.assert_allclose(out[:, _CELLS], e_mev / 1000.0, rtol=1e-6)


@pytest.mark.parametrize("log_transform,cutoff", [(False, None), (True, 1.0)])
def test_torch_converter_matches_numpy_reference(log_transform, cutoff):
    rng = np.random.RandomState(3)
    x_mev = rng.gamma(0.5, 10.0, size=(4, _CELLS)).astype(np.float32)
    e_mev = np.array([10_000.0, 20_000.0, 40_000.0, 80_000.0], dtype=np.float32)

    reference = showers_to_classifier_input(
        x_mev, e_mev, xml_path=_XML, particle="photon",
        log_transform=log_transform, voxel_energy_cutoff=cutoff,
    )
    on_device = showers_to_classifier_input_torch(
        torch.from_numpy(x_mev), torch.from_numpy(e_mev),
        xml_path=_XML, particle="photon",
        log_transform=log_transform, voxel_energy_cutoff=cutoff,
    )

    np.testing.assert_allclose(
        on_device.numpy(), reference, rtol=1e-5, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_yields_one_classifier_input_per_row(tmp_path):
    path = str(tmp_path / "gen.h5")
    _write_lemurs_h5(path, n=6, seed=1)
    source = HDF5Source(path)
    ds = ShowerClassifierDataset(
        [(source, np.array([1, 4]), 0.0)],
        xml_path=_XML, particle="photon",
        log_transform=True, voxel_energy_cutoff=1.0,
    )

    assert len(ds) == 2
    x, c, y = ds[0]
    assert x.shape == (_DIM,)
    assert c.shape == (1,)
    assert float(y) == 0.0

    batch = ds.__getitems__([0, 1])
    assert len(batch) == 2
    stacked = torch.stack([item[0] for item in batch])
    assert stacked.shape == (2, _DIM)


# ---------------------------------------------------------------------------
# DataModule: deterministic split + no-shuffle ordering
# ---------------------------------------------------------------------------

def test_datamodule_shuffle_false_takes_tail_of_each_file(tmp_path):
    gen_path = str(tmp_path / "gen.h5")
    truth_path = str(tmp_path / "truth.h5")
    test_path = str(tmp_path / "test.h5")
    _write_lemurs_h5(gen_path, n=20, seed=2)
    _write_lemurs_h5(truth_path, n=20, seed=3)
    _write_lemurs_h5(test_path, n=10, seed=4)

    dm = ShowerClassifierDataModule(
        gen_data_path=gen_path, truth_data_path=truth_path, test_data_path=test_path,
        xml_path=_XML, particle="photon",
        val_fraction=0.25, batch_size=4, shuffle=False, num_workers=0,
        log_transform=True, voxel_energy_cutoff=1.0,
    )
    dm.setup()

    (_, gen_indices, _), (_, truth_indices, _) = dm._val_dataset.groups
    # sklearn's train_test_split(shuffle=False) holds out the tail
    np.testing.assert_array_equal(gen_indices, np.arange(15, 20))
    np.testing.assert_array_equal(truth_indices, np.arange(15, 20))
    assert len(dm._val_dataset) == 10

    batches = list(dm.val_dataloader())
    labels = torch.cat([b[2] for b in batches])
    np.testing.assert_array_equal(labels.numpy(), [0.0] * 5 + [1.0] * 5)
    assert batches[0][0].shape[1] == _DIM


def test_train_dataloader_interleaves_classes_when_shuffle_false(tmp_path):
    # caloxtreme's LEMURS configs set shuffle=False (files are pre-shuffled) and
    # rely on a batch sampler to mix generator and truth rows.
    gen_path = str(tmp_path / "gen.h5")
    truth_path = str(tmp_path / "truth.h5")
    test_path = str(tmp_path / "test.h5")
    _write_lemurs_h5(gen_path, n=20, seed=8)
    _write_lemurs_h5(truth_path, n=20, seed=9)
    _write_lemurs_h5(test_path, n=10, seed=10)

    dm = ShowerClassifierDataModule(
        gen_data_path=gen_path, truth_data_path=truth_path, test_data_path=test_path,
        xml_path=_XML, particle="photon", val_fraction=0.25, batch_size=6,
        shuffle=False, num_workers=0,
        log_transform=True, voxel_energy_cutoff=1.0,
    )
    dm.setup()

    full_batches = list(dm.train_dataloader())[:-1]  # drop the ragged last batch
    assert len(full_batches) > 0
    for batch in full_batches:
        assert set(batch[2].unique().tolist()) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# Cross-repo classifier weights
# ---------------------------------------------------------------------------

def test_external_checkpoint_loads_into_lazy_mlp(tmp_path):
    from calogen.classifiers.base import MLP, load_state_dict_from_checkpoint

    torch.manual_seed(0)
    source = MLP(hidden_dim=16, num_layers=2, layer_norm=True, dropout=0.5)
    x = torch.randn(4, _DIM)
    source(x)  # materialise the lazy input projector
    ckpt_path = str(tmp_path / "clf.ckpt")
    torch.save(
        {
            # caloxtreme-style metadata: no calogen class path, instantiator
            # that jsonargparse would reject if trusted.
            "hyper_parameters": {
                "_class_path": "module.lightning.MLPClassifier",
                "_instantiator": "lightning.pytorch.cli.instantiate_module",
            },
            "state_dict": source.state_dict(),
        },
        ckpt_path,
    )

    target = MLP(hidden_dim=16, num_layers=2, layer_norm=True, dropout=0.5)
    load_state_dict_from_checkpoint(target, ckpt_path)
    target.eval()
    source.eval()
    torch.testing.assert_close(target(x), source(x))

    # The head can also be extracted from a module that wraps the MLP under
    # ``net.`` (as CaloINNCLF does next to its generator).
    class _Wrapper(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.net = mlp

    wrapped_path = str(tmp_path / "wrapped.ckpt")
    torch.save({"state_dict": _Wrapper(source).state_dict()}, wrapped_path)
    wrapped_target = _Wrapper(
        MLP(hidden_dim=16, num_layers=2, layer_norm=True, dropout=0.5)
    )
    load_state_dict_from_checkpoint(wrapped_target.net, wrapped_path, prefix="net.")
    wrapped_target.eval()
    torch.testing.assert_close(wrapped_target.net(x), source(x))

    with pytest.raises(RuntimeError):
        load_state_dict_from_checkpoint(
            MLP(hidden_dim=8, num_layers=1), ckpt_path
        )


# ---------------------------------------------------------------------------
# CaloINNCLF batch contract
# ---------------------------------------------------------------------------

def test_caloinnclf_accepts_precomputed_batches():
    from calogen.models.caloinn.classifier import CaloINNCLF

    X = torch.zeros(3, _DIM)
    c = torch.ones(3, 1)
    y = torch.tensor([0.0, 1.0, 0.0])

    # The precomputed branch does not touch the generator, so an instance
    # without __init__ is enough to exercise it.
    module = object.__new__(CaloINNCLF)
    X_input, y_out, c_out = module.get_input_from_batch((X, c, y))

    assert X_input is X
    assert c_out is c
    assert y_out.shape == (3, 1)


# ---------------------------------------------------------------------------
# Cross-repo parity: our batches == caloxtreme's batches
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isdir(_CALOXTREME), reason="caloxtreme_clf repo not available"
)
def test_datamodule_batches_match_caloxtreme(tmp_path):
    if _CALOXTREME not in sys.path:
        sys.path.append(_CALOXTREME)
    from data.lemurs_datamodule import LEMURSHDF5MLPDataModule  # noqa: E402

    gen_path = str(tmp_path / "gen.h5")
    truth_path = str(tmp_path / "truth.h5")
    test_path = str(tmp_path / "test.h5")
    _write_lemurs_h5(gen_path, n=24, seed=5)
    _write_lemurs_h5(truth_path, n=24, seed=6)
    _write_lemurs_h5(test_path, n=8, seed=7)

    kwargs = dict(
        gen_data_path=gen_path, truth_data_path=truth_path, test_data_path=test_path,
        xml_path=_XML, particle="photon", val_fraction=0.25, batch_size=6,
        min_energy=None, split_seed=42, num_workers=0, prefetch_factor=2,
        pin_memory=False, voxel_energy_cutoff=1.0, log_transform=True,
        shuffle=False, cache_mode="none",
    )
    ours = ShowerClassifierDataModule(**kwargs)
    ours.setup()
    ref = LEMURSHDF5MLPDataModule(**kwargs)
    ref.setup()

    for (x_ours, c_ours, y_ours), ref_batch in zip(
        ours.val_dataloader(), ref.val_dataloader()
    ):
        x_ref, c_ref, hlf_ref, y_ref = ref_batch
        np.testing.assert_allclose(
            x_ours.numpy(), np.concatenate([x_ref, c_ref, hlf_ref], axis=1),
            rtol=1e-5, atol=1e-5,
        )
        np.testing.assert_allclose(c_ours.numpy(), c_ref.numpy(), rtol=1e-6)
        np.testing.assert_array_equal(y_ours.numpy(), y_ref.numpy())
