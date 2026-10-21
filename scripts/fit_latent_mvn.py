#!/usr/bin/env python
"""Fit a data-derived Gaussian base for the precomputed latent NCE.

Reads the training latents from a precomputed store, estimates the mean ``mu``
and covariance ``Sigma`` (float64, streamed/chunked), regularizes it (optional
shrinkage toward the diagonal + eigenvalue floor), and saves the
eigendecomposition so ``PrecomputedLatentWhitenedNCE`` can whiten
``w = diag(1/sqrt(lambda)) U^T (z - mu)``.

Outputs ``mean.pt``, ``eigvecs.pt``, ``eigvals.pt``, ``meta.json`` in ``--out-dir``.

Usage::

    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/fit_latent_mvn.py \
        --latents-dir /pscratch/sd/p/pmtuan/caloinn_latents/fcceeallegro/epoch146
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float32)

from calogen.core.latents import read_manifest, save_latent_q0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--out-dir", default=None, help="default: <latents-dir>/q0")
    p.add_argument("--chunk-size", type=int, default=16384)
    p.add_argument("--shrinkage", default="auto", help="'auto' or a float in [0,1] toward diag(Sigma)")
    p.add_argument("--val-rows", type=int, default=4096, help="rows used to score shrinkage candidates")
    p.add_argument("--eigenvalue-floor", type=float, default=1e-6, help="floor relative to max eigenvalue")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _split_field(latents_dir: str, split: str, name: str):
    manifest = read_manifest(latents_dir)
    return manifest, manifest["splits"][split][name]


def _open_mmap(latents_dir: str, field) -> np.memmap:
    return np.memmap(
        os.path.join(latents_dir, field["file"]),
        mode="r",
        dtype=field["dtype"],
        shape=tuple(field["shape"]),
    )


def _cholesky(cov: torch.Tensor) -> torch.Tensor:
    """Cholesky with auto-escalating relative jitter."""
    eps = 1e-10 * float(torch.diagonal(cov).mean())
    for _ in range(30):
        try:
            chol = torch.linalg.cholesky(cov)
            return chol
        except Exception:
            cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
            eps *= 10.0
    raise RuntimeError("Cholesky failed even after aggressive jitter")


def _mvn_nll(cov: torch.Tensor, centered_val: torch.Tensor) -> float:
    """Mean negative log-likelihood of centered_val under N(0, cov)."""
    chol = _cholesky(cov)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    sol = torch.linalg.solve_triangular(chol, centered_val.T, upper=False)
    quad = (sol ** 2).sum(dim=0).mean()
    d = cov.shape[0]
    return float(0.5 * (logdet + quad + d * math.log(2 * math.pi)))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.latents_dir, "q0")
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not args.overwrite:
        print(f"[fit-q0] {out_dir}/meta.json already exists (use --overwrite)")
        return

    manifest, z_field = _split_field(args.latents_dir, args.split, "z")
    z_mm = _open_mmap(args.latents_dir, z_field)
    n, d = int(z_mm.shape[0]), int(z_mm.shape[1])
    dev = torch.device(args.device)
    print(f"[fit-q0] {args.split}: N={n:,} d={d} on {dev} (fp64)")

    t0 = time.time()
    sum_vec = torch.zeros(d, dtype=torch.float64, device=dev)
    gram = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk = torch.from_numpy(np.asarray(z_mm[start:end])).to(device=dev, dtype=torch.float64)
        sum_vec += chunk.sum(dim=0)
        gram += chunk.T @ chunk
        del chunk
    mean = sum_vec / n
    cov = gram / n - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    mean = mean.cpu()
    cov = cov.cpu()
    print(f"[fit-q0] Gram in {time.time() - t0:.1f}s, trace={float(torch.trace(cov)):.3e}")

    diag_cov = torch.diagonal(cov).clone()

    val_field = None
    try:
        _, val_field = _split_field(args.latents_dir, args.val_split, "z")
    except KeyError:
        print(f"[fit-q0] warning: no '{args.val_split}' split; scoring shrinkage on train subset")
    if val_field is not None:
        v_mm = _open_mmap(args.latents_dir, val_field)
        nv = min(args.val_rows, int(v_mm.shape[0]))
        val = torch.from_numpy(np.asarray(v_mm[:nv])).double()
    else:
        nv = min(args.val_rows, n)
        val = torch.from_numpy(np.asarray(z_mm[:nv])).double()
    val_centered = val - mean

    if isinstance(args.shrinkage, str) and args.shrinkage == "auto":
        grid = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    else:
        grid = [float(args.shrinkage)]
    scores = []
    for delta in grid:
        cand = (1.0 - delta) * cov + delta * torch.diag(diag_cov)
        nll = _mvn_nll(cand, val_centered)
        scores.append((delta, nll))
        print(f"[fit-q0] shrinkage delta={delta:<6g} val NLL={nll:.6e}")
    delta = min(scores, key=lambda s: s[1])[0]
    print(f"[fit-q0] selected shrinkage delta={delta:g}")

    cov = (1.0 - delta) * cov + delta * torch.diag(diag_cov)
    cov = 0.5 * (cov + cov.T)

    t0 = time.time()
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    print(f"[fit-q0] eigh in {time.time() - t0:.1f}s")

    floor_abs = max(args.eigenvalue_floor * float(eigvals[0]), torch.finfo(torch.float64).tiny)
    n_floored = int((eigvals < floor_abs).sum())
    eigvals = eigvals.clamp_min(floor_abs)
    logdet = float(torch.log(eigvals).sum())

    val_nll = _mvn_nll(
        (eigvecs * eigvals) @ eigvecs.T, val_centered
    )

    meta = {
        "source_manifest_key": manifest.get("key"),
        "split": args.split,
        "N": n,
        "D": d,
        "whitening": "eigen",
        "shrinkage_delta": delta,
        "shrinkage_scores": [{"delta": s, "val_nll": nll} for s, nll in scores],
        "eigenvalue_floor_rel": args.eigenvalue_floor,
        "eigenvalue_floor_abs": floor_abs,
        "n_eigenvalues_floored": n_floored,
        "trace": float(torch.trace(cov)),
        "logdet": logdet,
        "eig_min": float(eigvals[-1]),
        "eig_max": float(eigvals[0]),
        "val_nll": val_nll,
    }
    save_latent_q0(
        out_dir,
        mean.detach().float().cpu(),
        eigvecs.detach().float().cpu(),
        eigvals.detach().float().cpu(),
        meta,
    )
    print(f"[fit-q0] saved {out_dir} (logdet={logdet:.3e}, floored={n_floored}, val_nll={val_nll:.6e})")
    with open(os.path.join(out_dir, "meta.json")) as f:
        print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
