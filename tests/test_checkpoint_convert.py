"""Standalone caloxtreme → calogen checkpoint conversion."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.classifiers.base import MLP
from calogen.classifiers.caloinn.shower_nce import LemursPhysicalShowerNCE

_HERE = os.path.dirname(os.path.abspath(__file__))
_XML = os.path.join(_HERE, "..", "configs", "binning_lemurs_fcceeallegro.xml")
_SCRIPT = os.path.join(_HERE, "..", "scripts", "caloinn-nce", "convert_caloxtreme_ckpt.py")
_DIM = 6480 + 1 + 226

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_XML), reason="LEMURS binning XML not available"
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_caloxtreme_ckpt", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Head(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = MLP(hidden_dim=hidden_dim, num_layers=1, layer_norm=True)

    def forward(self, x):
        return self.net(x)


def _old_checkpoint(path: str) -> tuple[dict, torch.Tensor, torch.Tensor]:
    head = _Head(hidden_dim=8)
    x = torch.randn(3, _DIM)
    ref = head(x)
    state = dict(head.state_dict())
    ckpt = {
        "state_dict": state,
        "hyper_parameters": {
            "_class_path": "module.lightning.MLPClassifier",
            "_instantiator": "lightning.pytorch.cli.instantiate_module",
            "hidden_dim": 8,
            "num_layers": 1,
            "batch_norm": False,
            "layer_norm": True,
            "output_dim": 1,
            "dropout": 0.0,
            "activation": "relu",
        },
        "epoch": 7,
        "global_step": 42,
        "pytorch-lightning_version": "2.6.1",
    }
    torch.save(ckpt, path)
    return ckpt, x, ref


def test_converter_preserves_weights_and_metadata(tmp_path):
    converter = _load_converter()
    src = str(tmp_path / "old.ckpt")
    dst = str(tmp_path / "new.ckpt")
    _, x, ref = _old_checkpoint(src)

    converter.convert_checkpoint(
        src, dst,
        generator_ckpt="/tmp/gen.ckpt",
        xml_path=_XML,
        voxel_energy_cutoff=1.0,
        log_transform=True,
    )

    out = torch.load(dst, map_location="cpu", weights_only=False)
    assert out["epoch"] == 7
    assert out["global_step"] == 42
    assert out["hparams_name"] == "kwargs"
    hp = out["hyper_parameters"]
    assert hp["_class_path"].endswith("LemursPhysicalShowerNCE")
    assert "_instantiator" not in hp
    assert hp["generator_ckpt_path"] == "/tmp/gen.ckpt"
    assert hp["kw_overrides"] == {"xml_path": _XML}
    assert hp["hidden_dim"] == 8 and hp["num_layers"] == 1
    assert all(key.startswith("net.") for key in out["state_dict"])

    head = _Head(hidden_dim=8)
    head.load_state_dict(out["state_dict"])
    head.eval()
    torch.testing.assert_close(head(x), ref)


def test_converted_checkpoint_loads_natively(tmp_path, monkeypatch):
    converter = _load_converter()
    src = str(tmp_path / "old.ckpt")
    dst = str(tmp_path / "new.ckpt")
    _old_checkpoint(src)
    converter.convert_checkpoint(
        src, dst, generator_ckpt="/tmp/gen.ckpt", xml_path=_XML,
    )

    from test_physical_shower_nce import _StubGenerator

    monkeypatch.setattr(
        LemursPhysicalShowerNCE, "load_generator",
        staticmethod(lambda *args, **kwargs: _StubGenerator()),
    )
    model = LemursPhysicalShowerNCE.load_from_checkpoint(dst)
    assert model.strict_loading is False
    assert tuple(model.net.net[0].weight.shape) == (8, _DIM)
    np.testing.assert_array_equal(
        model.hparams.kw_overrides, {"xml_path": _XML}
    )
