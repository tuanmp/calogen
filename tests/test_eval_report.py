import os, json
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")

from calogen.eval.config import EvalConfig
from calogen.eval.report import evaluate_lemurs

R, PHI, Z = 9, 16, 45
XML = "configs/binning_lemurs_fcceeallegro.xml"


def _write(path, n=30, seed=0):
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("showers", data=rng.random((n, R, PHI, Z)).astype(np.float32))
        f.create_dataset("incident_energy", data=np.linspace(1e3, 1e5, n).astype(np.float32))


def test_evaluate_lemurs_end_to_end(tmp_path):
    truth = str(tmp_path / "truth.h5")
    gen = str(tmp_path / "gen.h5")
    _write(truth, seed=0)
    _write(gen, seed=1)
    cfg = EvalConfig(truth=truth, generators=[{"label": "Gen", "path": gen}],
                     binning_xml=XML, out_dir=str(tmp_path / "out"),
                     observables=["e_tot_frac", "active_fraction"])
    res = evaluate_lemurs(cfg)
    assert os.path.exists(os.path.join(res.out_dir, "metrics.json"))
    assert os.path.exists(os.path.join(res.out_dir, "summary.txt"))
    assert any(p.endswith(".png") for p in res.artifacts)
    with open(os.path.join(res.out_dir, "metrics.json")) as f:
        assert isinstance(json.load(f), list)


def test_evaluate_lemurs_multiple_generators(tmp_path):
    truth = str(tmp_path / "truth.h5")
    g1 = str(tmp_path / "g1.h5")
    g2 = str(tmp_path / "g2.h5")
    _write(truth, seed=0)
    _write(g1, seed=1)
    _write(g2, seed=2)
    cfg = EvalConfig(
        truth=truth,
        generators=[{"label": "A", "path": g1}, {"label": "B", "path": g2}],
        binning_xml=XML, out_dir=str(tmp_path / "out"),
        observables=["e_tot_frac"], make_pdf=False)
    res = evaluate_lemurs(cfg)
    gens = {r["generator"] for r in res.metrics}
    assert {"A", "B"}.issubset(gens), f"metrics must be tagged per generator: {gens}"
    a = [r for r in res.metrics if r["generator"] == "A" and r["metric"] == "wasserstein"]
    b = [r for r in res.metrics if r["generator"] == "B" and r["metric"] == "wasserstein"]
    assert len(a) == 1 and len(b) == 1
    # both generators appear in the single overlay plot's summary line
    line = [ln for ln in res.summary.splitlines() if ln.startswith("e_tot_frac")][0]
    assert "A:" in line and "B:" in line
