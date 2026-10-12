"""Distribution-distance metrics for LEMURS evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


@dataclass
class MetricRecord:
    observable: str
    layer: int
    metric: str
    value: float


def separation_power(ref_norm: np.ndarray, gen_norm: np.ndarray) -> float:
    """Triangular discrimination (eq. 15 of 2009.03796) on normalized hists."""
    a = np.asarray(ref_norm, dtype=np.float64)
    b = np.asarray(gen_norm, dtype=np.float64)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-16)))


def corr_matrix_diff(ref_corr: np.ndarray, gen_corr: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(ref_corr) - np.asarray(gen_corr)))


def _hist_norm(values, bins):
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(np.float64)


def distribution_metrics(ref, gen, name, bins, per_layer, n_layers) -> List[MetricRecord]:
    ref = np.asarray(ref, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    out: List[MetricRecord] = []

    if not per_layer:
        ref = ref.reshape(-1)
        gen = gen.reshape(-1)
        out.append(MetricRecord(name, -1, "wasserstein", float(wasserstein_distance(ref, gen))))
        out.append(MetricRecord(name, -1, "ks", float(ks_2samp(ref, gen).statistic)))
        out.append(MetricRecord(name, -1, "separation_power",
                                separation_power(_hist_norm(ref, bins), _hist_norm(gen, bins))))
        return out

    for layer in range(n_layers):
        r, g = ref[:, layer], gen[:, layer]
        out.append(MetricRecord(name, layer, "wasserstein", float(wasserstein_distance(r, g))))
        out.append(MetricRecord(name, layer, "ks", float(ks_2samp(r, g).statistic)))
        out.append(MetricRecord(name, layer, "separation_power",
                                separation_power(_hist_norm(r, bins), _hist_norm(g, bins))))
    return out
