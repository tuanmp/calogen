"""Clean hist-with-ratio renderers (no globals, no LaTeX requirement)."""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from matplotlib import pyplot as plt

_DUP = lambda a: np.append(a, a[-1])


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": "medium",
        "mathtext.default": "rm",
        "text.usetex": False,
    })


def hist_with_ratio(ref, gens: Dict[str, np.ndarray], bins, xlabel,
                    xscale: str = "linear", ratio_ylim=(0.5, 1.5),
                    title: Optional[str] = None):
    bins = np.asarray(bins, dtype=np.float64)
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(5, 4.5), sharex=True,
        gridspec_kw={"height_ratios": (4, 1), "hspace": 0.0})

    counts_ref, _ = np.histogram(ref, bins=bins)
    total_ref = counts_ref.sum()
    ref_norm = counts_ref / total_ref if total_ref else counts_ref.astype(float)
    ref_err = ref_norm / np.sqrt(np.maximum(counts_ref, 1.0))
    ax.step(bins, _DUP(ref_norm), where="post", color="k", lw=1.0, label="truth")
    ax.fill_between(bins, _DUP(ref_norm + ref_err), _DUP(np.maximum(ref_norm - ref_err, 0)),
                    step="post", color="k", alpha=0.2)
    rax.fill_between(bins, _DUP(1 - ref_err / np.maximum(ref_norm, 1e-12)),
                     _DUP(1 + ref_err / np.maximum(ref_norm, 1e-12)),
                     step="post", color="k", alpha=0.2)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, gen) in enumerate(gens.items()):
        counts, _ = np.histogram(gen, bins=bins)
        total = counts.sum()
        norm = counts / total if total else counts.astype(float)
        err = norm / np.sqrt(np.maximum(counts, 1.0))
        c = colors[i % len(colors)]
        ax.step(bins, _DUP(norm), where="post", lw=1.2, color=c, label=label)
        ax.fill_between(bins, _DUP(norm + err), _DUP(np.maximum(norm - err, 0)),
                        step="post", color=c, alpha=0.2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(counts_ref > 0, counts / counts_ref, np.nan)
        rax.step(bins, _DUP(ratio), where="post", lw=1.0, color=c)

    rax.axhline(1.0, color="k", lw=1.0)
    rax.axhline(ratio_ylim[0], color="k", ls="--", lw=0.5)
    rax.axhline(ratio_ylim[1], color="k", ls="--", lw=0.5)
    rax.set_ylim(*ratio_ylim)
    rax.set_xlabel(xlabel)
    rax.set_ylabel("model/truth")
    ax.set_ylabel("a.u.")
    ax.set_xscale(xscale)
    rax.set_xscale(xscale)
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(frameon=False, fontsize=10)
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.2)
    return fig


def save_fig(fig, path: str, close: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
