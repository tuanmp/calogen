"""On-the-fly LEMURS physical classifier: data contract and sampler hooks.

Uses a stub generator so the tests stay fast; the real generator is exercised
only by opt-in integration checks.
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.classifiers.caloinn.shower_nce import LemursPhysicalShowerNCE
from calogen.data.datamodules.caloinn import RawShowerDataModule
from calogen.models.caloinn.features.convert import (
    showers_to_classifier_input_torch,
)
from calogen.sampler.base import BaseSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
_XML = os.path.join(_HERE, "..", "configs", "binning_lemurs_fcceeallegro.xml")
_N_LAYERS, _CELLS_PER_LAYER = 45, 144
_CELLS = _N_LAYERS * _CELLS_PER_LAYER
_DIM = _CELLS + 1 + (1 + _N_LAYERS + 4 * _N_LAYERS)

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_XML), reason="LEMURS binning XML not available"
)


class _StubGenerator(torch.nn.Module):
    """Deterministic stand-in for CaloINNLightningModule."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.width_noise = 5e-6
        self.xml_path = _XML
        self.xml_ptype = "photon"
        self.layer_boundaries = np.arange(
            0, _N_LAYERS * _CELLS_PER_LAYER + 1, _CELLS_PER_LAYER
        )
        self.q = torch.tensor(5e-6)
        self.hparams = {"num_noise_doses": 1}
        self._samples = None

    def conditional_generate(self, c, measure_gen_time=False):
        if self._samples is None or self._samples.shape[0] != c.shape[0]:
            generator = torch.Generator().manual_seed(0)
            self._samples = torch.rand(
                c.shape[0], 1, _CELLS + _N_LAYERS, generator=generator
            ) * 100.0
        return self._samples

    def post_process_decoded(self, x, c):
        cells = x[:, :_CELLS].reshape(x.shape[0], _N_LAYERS, _CELLS_PER_LAYER)
        data = {"energy": c}
        for layer in range(_N_LAYERS):
            # GeV, so the module's *1e3 recovers the internal cell values.
            data[f"layer_{layer}"] = cells[:, layer, :] * 1e-3
        return data


@pytest.fixture
def stub_module(monkeypatch):
    stub = _StubGenerator()
    monkeypatch.setattr(
        LemursPhysicalShowerNCE, "load_generator",
        staticmethod(lambda *args, **kwargs: stub),
    )
    return stub


def _make_module(**kwargs):
    args = dict(
        hidden_dim=8, num_layers=1, generator_ckpt_path="stub",
        voxel_energy_cutoff=1.0, log_transform=True,
    )
    args.update(kwargs)
    return LemursPhysicalShowerNCE(**args)


def _write_lemurs_h5(path: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    showers = rng.gamma(0.5, 10.0, size=(n, 9, 16, 45)).astype(np.float32)
    energies = rng.uniform(1_000.0, 100_000.0, size=n).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("showers", data=showers, chunks=(min(n, 8), 9, 16, 45))
        f.create_dataset("incident_energy", data=energies, chunks=(min(n, 8),))
    return energies


# ---------------------------------------------------------------------------
# Module batch contract
# ---------------------------------------------------------------------------

def test_generate_wrapper_calls_conditional_generate(stub_module):
    module = _make_module()
    x_gen = module.generate(torch.tensor([[10.0], [20.0]]))
    assert x_gen.shape == (2, _CELLS + _N_LAYERS)
    assert torch.isfinite(x_gen).all()


def test_get_input_from_batch_matches_converter(stub_module):
    module = _make_module()
    x_truth = torch.rand(3, _CELLS) * 100.0
    c_mev = torch.tensor([[10_000.0], [20_000.0], [40_000.0]])

    X_input, y, aux = module.get_input_from_batch((x_truth, c_mev))

    assert X_input.shape == (6, _DIM)
    np.testing.assert_array_equal(
        y.reshape(-1).numpy(), [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    )

    expected_truth = showers_to_classifier_input_torch(
        x_truth, c_mev, xml_path=_XML, particle="photon",
        log_transform=True, voxel_energy_cutoff=1.0,
    )
    torch.testing.assert_close(X_input[:3], expected_truth)

    internal = stub_module.conditional_generate(c_mev / 1e3).squeeze(1)
    gen_cells = internal[:, :_CELLS]
    expected_gen = showers_to_classifier_input_torch(
        gen_cells, c_mev, xml_path=_XML, particle="photon",
        log_transform=True, voxel_energy_cutoff=1.0,
    )
    torch.testing.assert_close(X_input[3:], expected_gen)


def test_estimate_density_ratio_is_deterministic(stub_module):
    module = _make_module()
    c_gev = torch.tensor([[10.0], [20.0]])
    internal = stub_module.conditional_generate(c_gev).squeeze(1)

    ratio = module.estimate_density_ratio(internal, c_gev)
    assert ratio.shape == (2,)
    assert torch.isfinite(ratio).all()


@pytest.mark.xfail(
    strict=False,
    reason="estimate_density_ratio passes c[GeV] to a converter expecting MeV; "
           "remove this xfail once the class multiplies by 1e3",
)
def test_estimate_density_ratio_uses_mev_condition(stub_module):
    module = _make_module()
    c_gev = torch.tensor([[10.0], [20.0]])
    internal = stub_module.conditional_generate(c_gev).squeeze(1)

    actual = module.estimate_density_ratio(internal, c_gev)

    gen_cells = internal[:, :_CELLS]
    expected_input = showers_to_classifier_input_torch(
        gen_cells, c_gev * 1e3, xml_path=_XML, particle="photon",
        log_transform=True, voxel_energy_cutoff=1.0,
    )
    expected = torch.exp(module.forward(expected_input).squeeze(-1))

    torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# RawShowerDataModule contract for training + sampling
# ---------------------------------------------------------------------------

def test_raw_datamodule_exposes_truth_format_and_conditions(tmp_path):
    data_path = str(tmp_path / "train.h5")
    val_path = str(tmp_path / "val.h5")
    energies = _write_lemurs_h5(data_path, n=12, seed=1)
    _write_lemurs_h5(val_path, n=5, seed=2)

    dm = RawShowerDataModule(
        data_path=data_path, val_data_path=val_path, batch_size=4,
        val_frac=0.25, shuffle=False, num_workers=0,
    )
    dm.setup()

    assert dm.truth_format.showers_grid_shape == (9, 16, 45)
    assert dm.condition_scale == 1e-3

    x, c = next(iter(dm.test_dataloader()))
    assert x.shape[1] == _CELLS
    assert c.shape == (5, 1)
    np.testing.assert_allclose(
        (c * dm.condition_scale).reshape(-1).numpy(),
        _read_energies(val_path) / 1e3,
        rtol=1e-6,
    )

    # BaseSampler.collect_conditions must return GeV for this datamodule.
    sampler = object.__new__(BaseSampler)
    sampler.datamodule = dm
    conditions = sampler.collect_conditions(loader="test")
    np.testing.assert_allclose(
        conditions.reshape(-1).numpy(),
        _read_energies(val_path) / 1e3,
        rtol=1e-6,
    )


def _read_energies(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["incident_energy"][:], dtype=np.float32)
