import pytest

from calogen.eval.config import EvalConfig


def test_from_yaml_roundtrip(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text(
        "truth: /t.h5\n"
        "generators:\n"
        "  - {label: CaloINN, path: /g.h5}\n"
        "binning_xml: configs/binning_lemurs_fcceeallegro.xml\n"
        "max_events: 1000\n"
    )
    cfg = EvalConfig.from_yaml(str(y))
    assert cfg.truth == "/t.h5"
    assert cfg.particle == "photon"
    assert cfg.max_events == 1000
    assert cfg.generators == [{"label": "CaloINN", "path": "/g.h5"}]
    assert cfg.to_dict()["out_dir"] == "eval_out"


def test_unknown_key_raises(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("truth: /t.h5\ngenerators: []\nbinning_xml: x.xml\nbogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        EvalConfig.from_yaml(str(y))


def test_empty_generators_raises(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("truth: /t.h5\ngenerators: []\nbinning_xml: x.xml\n")
    with pytest.raises(ValueError, match="generators"):
        EvalConfig.from_yaml(str(y))
