"""Clean hist-with-ratio renderers (no globals, no LaTeX requirement)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib import pyplot as plt

_DUP = lambda a: np.append(a, a[-1])

default_single_fig_size = (8, 6)

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



def save_fig(fig, path: str | Path, close: bool = True) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)

def plot_score_histo(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 50,
    title: str = "Score Distribution",
    xlabel: str = "Score",
    ylabel: str = "Frequency",
    log_scale: bool = False,
    save_path: Path | str | None = None
):
    """
    Plots a histogram of scores with different colors for each label.

    Args:
        scores (np.ndarray): Array of scores.
        labels (np.ndarray): Array of labels corresponding to the scores.
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
        save_path (Path | str | None): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=default_single_fig_size)
    unique_labels = np.unique(labels, sorted=True)

    for label in unique_labels:
        ax.hist(scores[labels == label], bins=bins, range=(0,1), alpha=0.5, label=f'y = {label}', density=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)

def plot_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 50,
    title: str = "Calibration Plot",
    xlabel: str = "Predicted Probability",
    ylabel: str = "Observed Frequency",
    save_path: Path | str | None = None
):
    """
    Plots a calibration curve comparing predicted probabilities to observed frequencies.

    Args:
        scores (np.ndarray): Array of predicted probabilities.
        labels (np.ndarray): Array of true labels (0 or 1).
        bins (int): Number of bins for the calibration plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    score_histo, bin_edges = np.histogram(scores, bins=bins, range=(0, 1))
    true_histo, _ = np.histogram(scores[labels == 1], bins=bin_edges)

    predicted_prob = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed_freq = np.divide(true_histo, score_histo, out=np.zeros_like(true_histo, dtype=float), where=score_histo != 0)

    fig, ax = plt.subplots(figsize=default_single_fig_size)
    ax.plot(predicted_prob, observed_freq, marker='o', linestyle='-', label='Calibration Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)

