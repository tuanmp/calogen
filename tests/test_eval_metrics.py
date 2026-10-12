import numpy as np

from calogen.eval.metrics import (
    MetricRecord, distribution_metrics, separation_power, corr_matrix_diff,
)


def test_separation_power_identical_is_zero():
    h = np.array([0.2, 0.3, 0.5])
    assert separation_power(h, h) == 0.0


def test_separation_power_disjoint_is_large():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    assert separation_power(a, b) > 0.9


def test_distribution_metrics_identical():
    rng = np.random.RandomState(0)
    ref = rng.normal(size=(500, 3))
    recs = distribution_metrics(ref, ref, "obs", bins=np.linspace(-4, 4, 20),
                                per_layer=True, n_layers=3)
    for r in recs:
        assert isinstance(r, MetricRecord)
        assert r.value == 0.0 or r.metric in ("separation_power", "ks")


def test_distribution_metrics_detects_shift():
    rng = np.random.RandomState(0)
    ref = rng.normal(size=(500,))
    gen = rng.normal(loc=1.0, size=(500,))
    recs = distribution_metrics(ref, gen, "obs", bins=np.linspace(-4, 6, 20),
                                per_layer=False, n_layers=1)
    by = {r.metric: r.value for r in recs}
    assert by["wasserstein"] > 0.5
    assert by["ks"] > 0.3


def test_corr_matrix_diff():
    a = np.eye(3)
    assert corr_matrix_diff(a, a) == 0.0
    assert corr_matrix_diff(a, np.zeros((3, 3))) > 0
