#!/usr/bin/env python
"""Plot the latent mean and covariance matrix of a CaloINN latent store.

The mean ``mu`` and covariance ``Sigma`` can either be loaded from the fitted
base assets produced by ``scripts/fit_latent_mvn.py`` (``--q0-dir``) or
recomputed from the raw memmap store (``--latents-dir`` / ``--split``).

Rendered figures (written to ``--out-dir``):

  * ``latent_mean.png``         - mean vector over latent dimensions
  * ``latent_covariance.png``   - covariance matrix heatmap (strided)
  * ``latent_correlation.png``  - correlation matrix heatmap (strided)
  * ``latent_variance.png``     - per-dimension variance (diagonal of Sigma)
  * ``latent_eigenvalues.png``  - eigenvalue spectrum (only from ``--q0-dir``)

Usage::

    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/plot-latent.py \\
        --q0-dir /pscratch/sd/p/pmtuan/caloinn_latents/fcceeallegro/epoch146/q0 \\
        --out-dir plots/latent

    # or recompute from the store
    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/plot-latent.py \\
        --latents-dir /pscratch/sd/p/pmtuan/caloinn_latents/fcceeallegro/epoch146 \\
        --split train --out-dir plots/latent
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float32)

from calogen.core.latents import load_latent_q0, read_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--q0-dir", default=None,
                     help="Directory with mean.pt/eigvecs.pt/eigvals.pt (preferred).")
    src.add_argument("--latents-dir", default=None,
                     help="Precomputed latent store; mean/cov are recomputed.")
    p.add_argument("--split", default="train", help="Store split to read (default train).")
    p.add_argument("--out-dir", default="plots/latent", help="Where to write figures.")
    p.add_argument("--max-dim", type=int, default=1500,
                   help="Max displayed matrix side; larger d is strided (default 1500).")
    p.add_argument("--chunk-size", type=int, default=16384,
                   help="Rows per chunk when recomputing from the store.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--format", default="png", choices=["png", "pdf"])
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading / computing the mean and covariance
# ---------------------------------------------------------------------------

def compute_mean_cov(latents_dir: str, split: str, chunk_size: int, device: str):
    """Streaming mean + covariance from a precomputed latent store (fp64)."""
    manifest = read_manifest(latents_dir)
    field = manifest["splits"][split]["z"]
    z_mm = np.memmap(
        os.path.join(latents_dir, field["file"]),
        mode="r", dtype=field["dtype"], shape=tuple(field["shape"]),
    )
    n, d = int(z_mm.shape[0]), int(z_mm.shape[1])
    dev = torch.device(device)
    print(f"[plot-latent] computing {split}: N={n:,} d={d} on {dev} (fp64)")

    t0 = time.time()
    sum_vec = torch.zeros(d, dtype=torch.float64, device=dev)
    gram = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = torch.from_numpy(np.asarray(z_mm[start:end])).to(device=dev, dtype=torch.float64)
        sum_vec += chunk.sum(dim=0)
        gram += chunk.T @ chunk
        del chunk
    mean = sum_vec / n
    cov = gram / n - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    print(f"[plot-latent] mean/cov in {time.time() - t0:.1f}s, trace={float(torch.trace(cov)):.3e}")
    return mean.cpu().numpy(), cov.cpu().numpy(), None


def load_from_q0(q0_dir: str):
    """Load the fitted mean and reconstruct the (regularized) covariance."""
    mean, eigvecs, eigvals, meta = load_latent_q0(q0_dir)
    mean = mean.double()
    eigvecs = eigvecs.double()
    eigvals = eigvals.double()
    cov = ((eigvecs * eigvals) @ eigvecs.T).numpy()
    print(f"[plot-latent] loaded q0: d={mean.shape[0]} shrinkage={meta.get('shrinkage_delta')} "
          f"trace={meta.get('trace')}")
    return mean.numpy(), cov, eigvals.numpy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _stride(d: int, max_dim: int) -> int:
    return max(1, int(np.ceil(d / max_dim)))


def plot_mean(mean: np.ndarray, out_path: str, dpi: int):
    d = mean.shape[0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(d), mean, lw=0.6, color="C0")
    ax.axhline(0.0, color="k", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel(r"mean $\mu_i$")
    ax.set_title(f"Latent mean (d={d}, |mu|={np.linalg.norm(mean):.3f})")
    ax.set_xlim(0, d - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot-latent] wrote {out_path}")


def plot_matrix(mat: np.ndarray, out_path: str, title: str, cmap: str,
                diverging: bool, max_dim: int, dpi: int, cbar_label: str,
                exclude_diagonal: bool = False):
    d = mat.shape[0]
    s = _stride(d, max_dim)
    sub = mat[::s, ::s]
    scale_src = sub
    if exclude_diagonal:
        # The variance on the diagonal dwarfs the off-diagonal structure;
        # exclude it so the range reveals the (co)variance bands.
        scale_src = sub.copy()
        np.fill_diagonal(scale_src, 0.0)
    if diverging or exclude_diagonal:
        m = np.nanpercentile(np.abs(scale_src), 99)
        vmin, vmax = -m, m
    else:
        vmin = np.nanmin(sub)
        vmax = np.nanpercentile(sub, 99.5)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
                   aspect="equal", origin="upper")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("latent dimension")
    ax.set_title(f"{title}\n(d={d}, shown every {s} dims -> {sub.shape[0]}x{sub.shape[1]})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot-latent] wrote {out_path}")


def plot_variance(cov: np.ndarray, out_path: str, dpi: int):
    var = np.diag(cov)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(var.shape[0]), var, lw=0.6, color="C1")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel(r"variance $\Sigma_{ii}$")
    ax.set_title(f"Per-dimension variance (mean={var.mean():.3g}, max={var.max():.3g})")
    ax.set_xlim(0, var.shape[0] - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot-latent] wrote {out_path}")


def plot_eigenvalues(eigvals: np.ndarray, out_path: str, dpi: int):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(np.arange(eigvals.shape[0]), np.clip(eigvals, 1e-12, None), lw=1.0)
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.set_title(f"Latent covariance spectrum (d={eigvals.shape[0]})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[plot-latent] wrote {out_path}")


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.q0_dir is not None:
        mean, cov, eigvals = load_from_q0(args.q0_dir)
        source = args.q0_dir
    else:
        mean, cov, eigvals = compute_mean_cov(
            args.latents_dir, args.split, args.chunk_size, args.device
        )
        source = f"{args.latents_dir}:{args.split}"

    diag = np.diag(cov)
    std = np.sqrt(np.clip(diag, 1e-30, None))
    corr = cov / np.outer(std, std)

    ext = args.format
    plot_mean(mean, os.path.join(args.out_dir, f"latent_mean.{ext}"), args.dpi)
    plot_matrix(cov, os.path.join(args.out_dir, f"latent_covariance.{ext}"),
                "Latent covariance", "coolwarm", True, args.max_dim, args.dpi,
                r"$\Sigma_{ij}$", exclude_diagonal=True)
    plot_matrix(corr, os.path.join(args.out_dir, f"latent_correlation.{ext}"),
                "Latent correlation", "coolwarm", True, args.max_dim, args.dpi,
                r"$\rho_{ij}$", exclude_diagonal=False)
    plot_variance(cov, os.path.join(args.out_dir, f"latent_variance.{ext}"), args.dpi)
    if eigvals is not None:
        plot_eigenvalues(eigvals, os.path.join(args.out_dir, f"latent_eigenvalues.{ext}"), args.dpi)

    stats = {
        "source": source,
        "D": int(mean.shape[0]),
        "mean_norm": float(np.linalg.norm(mean)),
        "mean_abs_max": float(np.abs(mean).max()),
        "trace": float(np.trace(cov)),
        "var_mean": float(diag.mean()),
        "var_min": float(diag.min()),
        "var_max": float(diag.max()),
    }
    stats_path = os.path.join(args.out_dir, "latent_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[plot-latent] wrote {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
