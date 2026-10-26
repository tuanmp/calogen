from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calogen.eval.plots import set_style

set_style()

def plot_score_histo(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 50,
    title: str = "Score Distribution",
    xlabel: str = "Score",
    ylabel: str = "Frequency",
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
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels, sorted=True)

    for label in unique_labels:
        ax.hist(scores[labels == label], bins=bins, range=(0,1), alpha=0.5, label=f'y = {label}', density=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)
        plt.savefig(save_path)

    plt.close()

def plot_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    bins: int = 10,
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

    fig, ax = plt.subplots(figsize=(8, 6))
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
        if isinstance(save_path, str):
            save_path = Path(save_path)
        plt.savefig(save_path)

    plt.close()
