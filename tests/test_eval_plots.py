import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import numpy as np

from calogen.eval.plots import set_style, hist_with_ratio, save_fig


def test_hist_with_ratio_and_save(tmp_path):
    set_style()
    rng = np.random.RandomState(0)
    ref = rng.normal(size=1000)
    gens = {"CaloINN": rng.normal(loc=0.1, size=1000)}
    fig = hist_with_ratio(ref, gens, bins=np.linspace(-4, 4, 25), xlabel="x")
    assert len(fig.axes) == 2  # main + ratio
    out = tmp_path / "x.png"
    save_fig(fig, str(out))
    assert out.exists() and out.stat().st_size > 0
