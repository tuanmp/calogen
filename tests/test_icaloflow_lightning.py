"""Unit tests for the iCaloFlow Lightning module (arXiv:2305.11934).

CPU-only and fast: tiny flows, tiny batches. No dataset/GPU required.
"""

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.models.icaloflow.lightning import ICaloFlowLightningModule

_N_BLOCKS = 2
_N_BLOCKS_PREDICT = 1  # sequential MAF sampling is slow; 1 block keeps predict fast
_HIDDEN_SIZE = 16
_TAIL_BOUND = 3.0
_NUM_LAYERS = 45
_CELLS_PER_LAYER = 144
_TOTAL_CELLS = _NUM_LAYERS * _CELLS_PER_LAYER  # 6480


def _make_module(which_flow, **kwargs):
    """Small module for fast tests; data constants set to dummy values."""
    return ICaloFlowLightningModule(
        which_flow=which_flow,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=kwargs.pop("n_blocks", _N_BLOCKS),
        tail_bound=_TAIL_BOUND,
        normalization_constant=1500.0,
        layer_boundaries=[0, 144, 288, 432],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


def test_configure_optimizers_returns_optimizer_and_step_scheduler():
    mod = _make_module(which_flow=1)
    cfg = mod.configure_optimizers()
    assert set(cfg.keys()) == {"optimizer", "lr_scheduler"}
    optimizer = cfg["optimizer"]
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert len(optimizer.param_groups[0]["params"]) > 0
    sched_cfg = cfg["lr_scheduler"]
    assert sched_cfg["interval"] == "step"
    assert hasattr(sched_cfg["scheduler"], "step")


# ---------------------------------------------------------------------------
# training_step
# ---------------------------------------------------------------------------


def test_training_step_flow1_finite_loss():
    torch.manual_seed(0)
    mod = _make_module(which_flow=1).double()
    batch = {
        "energy_dep": torch.randn(8, 45),
        "energy_inc": torch.randn(8, 1),
    }
    out = mod.training_step(batch, 0)
    assert set(out.keys()) == {"loss", "batch_size"}
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["batch_size"] == 8


def test_training_step_flow3_cond_dims_finite_loss():
    torch.manual_seed(0)
    mod = _make_module(which_flow=4).double()
    batch = {
        "layer": torch.randn(8, 144),
        "layer_p": torch.randn(8, 144),
        "energy": torch.randn(8, 1),
        "energy_dep": torch.randn(8, 1),
        "energy_dep_p": torch.randn(8, 1),
        "layer_number": torch.randint(1, 45, (8,)),
    }
    out = mod.training_step(batch, 0)
    assert set(out.keys()) == {"loss", "batch_size"}
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["batch_size"] == 8


# ---------------------------------------------------------------------------
# Checkpoint loading (per-flow training ckpt -> generation module)
# ---------------------------------------------------------------------------


def test_load_flows_for_generation_matches_weights(tmp_path):
    torch.manual_seed(0)
    paths = {}
    flow1_train = _make_module(which_flow=1)
    paths["flow1"] = str(tmp_path / "flow1.ckpt")
    torch.save({"state_dict": flow1_train.state_dict()}, paths["flow1"])
    for which_flow, name in [(2, "flow2"), (4, "flow3")]:
        train_mod = _make_module(which_flow=which_flow)
        paths[name] = str(tmp_path / f"{name}.ckpt")
        torch.save({"state_dict": train_mod.state_dict()}, paths[name])

    gen_mod = _make_module(which_flow=7)
    gen_mod.load_flows_for_generation(
        flow1_path=paths["flow1"],
        flow2_path=paths["flow2"],
        flow3_path=paths["flow3"],
    )

    expected = {
        k[len("flow1."):]: v
        for k, v in flow1_train.state_dict().items()
        if k.startswith("flow1.")
    }
    loaded = gen_mod.flow1.state_dict()
    assert set(loaded.keys()) == set(expected.keys())
    for key in loaded:
        assert torch.equal(loaded[key], expected[key])


def test_load_flows_for_generation_requires_all_checkpoints(tmp_path):
    torch.manual_seed(0)
    train_mod = _make_module(which_flow=1)
    ckpt_path = str(tmp_path / "flow1.ckpt")
    torch.save({"state_dict": train_mod.state_dict()}, ckpt_path)

    gen_mod = _make_module(which_flow=7)
    with pytest.raises(ValueError, match="flow2"):
        gen_mod.load_flows_for_generation(flow1_path=ckpt_path)


# ---------------------------------------------------------------------------
# Generation (predict_step)
# ---------------------------------------------------------------------------


def test_predict_step_shapes_and_finite():
    torch.manual_seed(0)
    # Randomly-initialized flows: we only check shapes/finiteness, so no ckpt
    # loading. Single-block flows keep the 44-step sequential MAF sampling
    # fast (2-block/float64 runs take minutes on CPU).
    gen_mod = _make_module(which_flow=7, n_blocks=_N_BLOCKS_PREDICT)

    batch = {"energy_inc": torch.tensor([[-1.5], [0.5]])}
    energies, showers, t = gen_mod.predict_step(batch, 0)
    assert energies.shape == (2, 1)
    assert showers.shape == (2, _TOTAL_CELLS)
    assert isinstance(t, float)
    assert np.isfinite(energies).all()
    assert np.isfinite(showers).all()
    assert (energies > 0).all()
    assert (showers >= 0).all()


# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------


def test_active_flow_params_float64_after_setup():
    mod = _make_module(which_flow=1)
    mod.setup("fit")
    params = list(mod.flow1.parameters())
    assert params
    assert all(p.dtype == torch.float64 for p in params)


def test_validation_step_returns_latent_z():
    torch.manual_seed(0)
    mod = _make_module(which_flow=1).double()
    batch = {
        "energy_dep": torch.randn(8, 45),
        "energy_inc": torch.randn(8, 1),
    }
    out = mod.validation_step(batch, 0)
    assert set(out.keys()) == {"loss", "batch_size", "z"}
    assert out["z"].shape == (8, 45)
    assert torch.isfinite(out["z"]).all()
