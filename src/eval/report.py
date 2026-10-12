"""Orchestrate a LEMURS evaluation run and write artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from calogen.eval.config import EvalConfig
from calogen.eval.io import load_showers
from calogen.eval.metrics import MetricRecord, corr_matrix_diff, distribution_metrics
from calogen.eval.observables import N_LAYERS, OBSERVABLES, make_context


@dataclass
class EvalResult:
    out_dir: str
    metrics: List[dict]
    artifacts: List[str]
    summary: str


def _bins_for(obs, ref_vals, gen_vals, n_bins=60):
    flat = np.concatenate([np.asarray(ref_vals).reshape(-1),
                           *[np.asarray(g).reshape(-1) for g in gen_vals]])
    flat = flat[np.isfinite(flat)]
    if obs.xscale == "log":
        pos = flat[flat > 0]
        lo = np.log10(pos.min()) if pos.size else -6.0
        hi = np.log10(pos.max()) if pos.size else 0.0
        if hi <= lo:
            hi = lo + 1.0
        return np.logspace(lo, hi, n_bins)
    lo, hi = flat.min(), flat.max()
    if hi <= lo:
        pad = max(abs(lo), 1.0) * 1e-6
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins)


def evaluate_lemurs(cfg: EvalConfig) -> EvalResult:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from calogen.eval.plots import hist_with_ratio, save_fig, set_style

    set_style()
    os.makedirs(cfg.out_dir, exist_ok=True)

    names = cfg.observables or list(OBSERVABLES)
    for n in names:
        if n not in OBSERVABLES:
            raise ValueError(f"Unknown observable {n!r}; valid: {sorted(OBSERVABLES)}")

    truth = load_showers(cfg.truth, "truth", max_events=cfg.max_events,
                         threshold_mev=cfg.threshold_mev,
                         energy_range=cfg.energy_range, seed=cfg.seed)
    gens = [load_showers(g["path"], g["label"], max_events=cfg.max_events,
                         threshold_mev=cfg.threshold_mev,
                         energy_range=cfg.energy_range, seed=cfg.seed)
            for g in cfg.generators]

    ctx_truth = make_context(cfg.binning_xml, cfg.particle, truth.showers, truth.energies)
    ctx_gens = [make_context(cfg.binning_xml, cfg.particle, g.showers, g.energies) for g in gens]

    records: List[MetricRecord] = []
    artifacts: List[str] = []
    lines = [f"truth: {cfg.truth} ({len(truth.showers)} events)"]

    pdf = PdfPages(os.path.join(cfg.out_dir, "summary.pdf")) if cfg.make_pdf else None

    def emit(fig, path):
        save_fig(fig, path, close=False)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)
        artifacts.append(path)

    for name in names:
        obs = OBSERVABLES[name]
        ref = obs.compute(ctx_truth)
        gen_vals = {g.label: obs.compute(ctx) for g, ctx in zip(gens, ctx_gens)}

        if name == "layer_energy_corr":
            diffs = []
            for g, ctx in zip(gens, ctx_gens):
                diff = corr_matrix_diff(ref, obs.compute(ctx))
                diffs.append(round(diff, 4))
                records.append(MetricRecord(name, -1, "corr_frobenius", diff))
            lines.append(f"{name}: frobenius vs truth = {diffs}")
            continue

        bins = _bins_for(obs, ref, list(gen_vals.values()))
        for g in gens:
            records += distribution_metrics(ref, gen_vals[g.label], name, bins,
                                            obs.per_layer, N_LAYERS)

        if obs.per_layer:
            for layer in range(N_LAYERS):
                fig = hist_with_ratio(ref[:, layer], {k: v[:, layer] for k, v in gen_vals.items()},
                                      bins=bins, xlabel=obs.xlabel, xscale=obs.xscale)
                emit(fig, os.path.join(cfg.out_dir, "plots", f"{name}_layer{layer + 1:02d}.png"))
        else:
            fig = hist_with_ratio(ref, gen_vals, bins=bins, xlabel=obs.xlabel, xscale=obs.xscale)
            emit(fig, os.path.join(cfg.out_dir, "plots", f"{name}.png"))

        if obs.per_layer:
            w = [r.value for r in records if r.observable == name and r.metric == "wasserstein"]
            lines.append(f"{name}: mean wasserstein={np.mean(w):.4g} over {len(w)} layers")
        else:
            w = [r.value for r in records if r.observable == name and r.metric == "wasserstein"]
            lines.append(f"{name}: wasserstein={w[0]:.4g}" if w else f"{name}: n/a")

    if pdf is not None:
        pdf.close()
        artifacts.append(os.path.join(cfg.out_dir, "summary.pdf"))

    metrics = [asdict(r) for r in records]
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    summary = "\n".join(lines)
    with open(os.path.join(cfg.out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)
    return EvalResult(out_dir=cfg.out_dir, metrics=metrics, artifacts=artifacts, summary=summary)
