#!/usr/bin/env python
"""Convert a legacy *preprocessed* CaloINN memmap cache back to the raw
LEMURS (N, R, Phi, Z) MeV layout used by ``calogen.data``.

Why
---
``calogen.core.sharded._write_memmap_cache`` (legacy) iterates the **noised**
dataset, so its cache holds ``preprocess(raw) + 1 frozen noise dose`` in the
flat ``(N, 6525)`` layout.  ``calogen.data.datamodules.caloinn.CaloINNDataModule``
instead stores **raw** rows and re-runs ``preprocess`` + noise at load time.
To replay the legacy "1 frozen + 1 fresh" (= 2 dose) regime through the new
pipeline we therefore have to invert ``preprocess``.

Inversion
---------
``preprocess`` per layer z::

    cells = raw / (sum(raw) + eps)          (normalise_layers, applied twice)
    u0    = sum_z(LE_z) / c                 (extra_dims[:, 0])
    r_k   = LE_{k-1} / (S_{k-1} + eps)      (extra_dims[:, k], k = 1..n_z-1)

with ``S_k = sum_{j>=k} LE_j``.  Because ``S_0 = u0 * c`` is *known*, the layer
energies are recovered with a **forward** recursion (a contraction -- no
amplification, unlike the backward form)::

    LE_{k-1} = r_k * (S_{k-1} + eps);   S_k = S_{k-1} - LE_{k-1}

To make ``preprocess`` reproduce the cached cells ``t`` we need
``raw = t * g`` with ``g = eps / (1 - sum(t))``.  Two regimes:

* ``sum(t)`` far from 1 (layer empty: the cached cells are *pure frozen noise*,
  ``sum(t) ~ 3.6e-4``) -> that exact ``g`` is well conditioned and the layer,
  frozen noise included, is reproduced **exactly**.
* ``sum(t) = 1 + delta`` (layer populated, ``delta ~ 3.4e-4`` = sum of 144
  noises) -> ``g`` would be *negative*, i.e. unphysical.  Any positive ``g``
  works (the estimator is scale invariant), but ``preprocess`` then divides by
  the noise-inflated sum, giving an irreducible ``(1 - 1/sum(t)) ~ 3.4e-4``
  relative shrinkage.  That is ~2.4e-6 for a typical cell (1/144) -- about half
  a noise quantum -- and at most 3.4e-4 on the brightest cell of a layer.

Run first with ``--verify-only``: it converts a chunk in memory and checks the
``raw -> preprocess`` round trip before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

LEGACY_KEY = "cc8b1875"  # key of the copied flat cache (informational only)


# ─────────────────────────────────────────────────────────────────────
#  helpers
# ─────────────────────────────────────────────────────────────────────
def new_cache_key(data_path: str, val_frac: float, shuffle: bool) -> str:
    """Reproduce ``calogen.data.datamodules.memmap_mixin.MemmapMixin._cache_key``."""
    try:
        st = os.stat(data_path)
        file_id = f"{os.path.realpath(data_path)}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        file_id = os.path.realpath(data_path)
    material = f"{file_id}|val_frac={val_frac}|shuffle={shuffle}"
    return hashlib.md5(material.encode()).hexdigest()[:16]


def load_flat(meta_path: str) -> dict:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    arrays = {}
    for name, info in meta["fields"].items():
        if len(info["shape"]) != 2:
            raise SystemExit(
                f"{meta_path}: field '{name}' has shape {info['shape']}; expected the "
                f"legacy flat (N, features) layout. Refusing to guess."
            )
        arrays[name] = np.memmap(
            os.path.join(os.path.dirname(meta_path), info["file"]),
            mode="r", dtype=info["dtype"], shape=tuple(info["shape"]),
        )
    return arrays


def layer_sums(cells: np.ndarray, n_z: int, c_cells: int) -> np.ndarray:
    return cells.reshape(len(cells), n_z, c_cells // n_z).sum(axis=2)


def invert_chunk(x0: np.ndarray, c: np.ndarray, n_z: int, c_cells: int, eps: float):
    """flat preprocessed row -> (raw GeV cells (M, n_z, cells), LE (M, n_z))."""
    cells = np.asarray(x0[:, :c_cells], dtype=np.float64)
    extra = np.asarray(x0[:, c_cells:], dtype=np.float64)
    cc = np.asarray(c, dtype=np.float64).reshape(len(x0))

    # forward recursion for the layer energies (contraction -> stable)
    S = extra[:, 0] * cc                       # S_0 = u0 * c = total layer energy
    LE = np.empty((len(x0), n_z))
    for k in range(1, n_z):
        le = extra[:, k] * (S + eps)
        LE[:, k - 1] = le
        S = S - le
    LE[:, -1] = S

    # scale factor g = eps / (1 - sum(t)) when the layer is eps-dominated,
    # otherwise any positive value (the cells estimator is scale invariant).
    t_sum = layer_sums(cells, n_z, c_cells)
    degenerate = (1.0 - t_sum) > 1e-4
    g = np.where(degenerate, eps / np.maximum(1.0 - t_sum, 1e-300), np.maximum(LE, 0.0) + eps)

    raw = cells.reshape(len(x0), n_z, -1) * g[:, :, None]        # GeV
    return raw, LE, degenerate, t_sum


def to_grid(raw: np.ndarray, r_dim: int, phi_dim: int, n_z: int) -> np.ndarray:
    """(M, n_z, cells) layer-major -> (M, R, Phi, Z) truth layout."""
    return raw.reshape(len(raw), n_z, r_dim, phi_dim).transpose(0, 2, 3, 1)


# ─────────────────────────────────────────────────────────────────────
#  verification
# ─────────────────────────────────────────────────────────────────────
def verify(src: str, c_cells: int, n_z: int, grid_shape, eps: float,
           cuts: dict, limit: int) -> None:
    from calogen.core.xml_handler import XMLHandler
    from calogen.data import data_util as du

    lb = np.unique(XMLHandler(particle_name=cuts["xml_ptype"],
                              filename=cuts["xml_path"]).GetBinEdges())
    if lb[-1] != c_cells or len(lb) - 1 != n_z:
        raise SystemExit(f"xml layer boundaries {lb[:3]}... disagree with the cache layout")

    r_dim, phi_dim, z_dim = grid_shape
    if z_dim != n_z or r_dim * phi_dim * z_dim != c_cells:
        raise SystemExit(f"grid {grid_shape} inconsistent with {c_cells} cells / {n_z} layers")

    for split in ("val", "train"):
        meta = os.path.join(src, f"{split}_{LEGACY_KEY}.json")
        if not os.path.exists(meta):
            print(f"[verify] {split}: no flat meta, skipped")
            continue
        arr = load_flat(meta)
        M = min(limit, len(arr["x"]))
        x0 = np.array(arr["x"][:M])
        c = np.array(arr["c"][:M])

        cells = x0[:, :c_cells]
        print(f"\n[verify] {split}: {M} events")
        print(f"  already noised? exact zeros {100 * (cells == 0).mean():.3f}% "
              f"(raw data ~57%), values in (0,5e-6) {100 * ((cells > 0) & (cells < 5e-6)).mean():.2f}%")

        raw, LE, degenerate, t_sum = invert_chunk(x0, c, n_z, c_cells, eps)
        print(f"  empty (eps-dominated) layers: {100 * degenerate.mean():.1f}% of (event,layer)"
              f"   LE<0 clipped: {int((LE < 0).sum())}")

        grid = to_grid(raw, r_dim, phi_dim, n_z)
        flat = np.transpose(grid, (0, 3, 1, 2)).reshape(M, -1).astype(np.float32)
        flat *= np.float32(1e3)                                    # MeV
        data = du._build_data_dict(flat, (np.asarray(c).reshape(M) * 1e3).astype(np.float32), lb)
        x2, _ = du.preprocess(data, lb, eps, u0up_cut=cuts["u0up_cut"],
                              u0low_cut=cuts["u0low_cut"], rew=cuts["rew"],
                              dep_cut=cuts["dep_cut"])
        if len(x2) != M:
            print(f"  !! preprocess dropped {M - len(x2)} events (cut instability)")
        n = min(len(x2), M)
        d = np.abs(x2[:n, :c_cells] - x0[:n, :c_cells])
        de = np.abs(x2[:n, c_cells:] - x0[:n, c_cells:])
        print(f"  round trip cells  max|d| {d.max():.3e}  p99.99 {np.quantile(d, 0.9999):.3e}"
              f"   (noise quantum 5e-6, expected bias 3.4e-4)")
        print(f"  round trip extras max|d| {de.max():.3e}")
        print(f"  raw MeV range {flat.min():.3e} .. {flat.max():.3e}")


# ─────────────────────────────────────────────────────────────────────
#  conversion
# ─────────────────────────────────────────────────────────────────────
def convert(src: str, dst: str, c_cells: int, n_z: int, grid_shape,
            eps: float, chunk: int, key: str, provenance: dict) -> None:
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SystemExit("src and dst must differ")
    os.makedirs(dst, exist_ok=True)
    r_dim, phi_dim, z_dim = grid_shape

    import tqdm

    for split in ("train", "val"):
        smeta = os.path.join(src, f"{split}_{LEGACY_KEY}.json")
        if not os.path.exists(smeta):
            print(f"[convert] {split}: no flat meta, skipped")
            continue
        src_arr = load_flat(smeta)
        n = len(src_arr["x"])
        x_path = os.path.join(dst, f"{split}_x.dat")
        c_path = os.path.join(dst, f"{split}_c.dat")
        if os.path.exists(x_path) and os.path.getsize(x_path) != n * c_cells * 4:
            raise SystemExit(f"{x_path} exists with an unexpected size; refusing to overwrite")

        x_out = np.memmap(x_path, mode="w+", dtype="float32", shape=(n, r_dim, phi_dim, z_dim))
        c_out = np.memmap(c_path, mode="w+", dtype="float32", shape=(n,))

        with tqdm.tqdm(total=n, desc=f"convert '{split}'", unit="evt") as pbar:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                x0 = np.array(src_arr["x"][start:end])
                c = np.array(src_arr["c"][start:end])
                raw, _, _, _ = invert_chunk(x0, c, n_z, c_cells, eps)
                grid = to_grid(raw, r_dim, phi_dim, n_z)
                x_out[start:end] = (grid * 1e3).astype(np.float32)
                c_out[start:end] = (np.asarray(c).reshape(-1) * 1e3).astype(np.float32)
                pbar.update(end - start)
        x_out.flush(); c_out.flush()
        del x_out, c_out

        meta = {
            "fields": {
                "x": {"file": f"{split}_x.dat", "shape": [n, r_dim, phi_dim, z_dim], "dtype": "float32"},
                "c": {"file": f"{split}_c.dat", "shape": [n], "dtype": "float32"},
            }
        }
        with open(os.path.join(dst, f"{split}_{key}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[convert] {split}: {n} events -> {dst}")

    with open(os.path.join(dst, "_conversion.json"), "w") as f:
        json.dump(provenance, f, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="legacy frozen-noise cache dir")
    p.add_argument("--dst", default="", help="destination dir for the raw cache")
    p.add_argument("--data-path", required=True, help="training HDF5 (for the cache key)")
    p.add_argument("--xml-path", default="configs/binning_lemurs_fcceeallegro.xml")
    p.add_argument("--xml-ptype", default="photon")
    p.add_argument("--grid", default="9,16,45", help="R,Phi,Z of the truth showers")
    p.add_argument("--eps", type=float, default=1e-10)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--u0up-cut", type=float, default=0.20)
    p.add_argument("--u0low-cut", type=float, default=0.0)
    p.add_argument("--rew", type=float, default=1.0)
    p.add_argument("--dep-cut", type=float, default=600.0)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--verify-limit", type=int, default=4096)
    a = p.parse_args(argv)

    from calogen.core.xml_handler import XMLHandler
    lb = np.unique(XMLHandler(particle_name=a.xml_ptype, filename=a.xml_path).GetBinEdges())
    c_cells, n_z = int(lb[-1]), len(lb) - 1
    grid_shape = tuple(int(v) for v in a.grid.split(","))
    cuts = dict(xml_path=a.xml_path, xml_ptype=a.xml_ptype, u0up_cut=a.u0up_cut,
                u0low_cut=a.u0low_cut, rew=a.rew, dep_cut=a.dep_cut)

    key = new_cache_key(a.data_path, a.val_frac, a.shuffle)
    print(f"cache key (calogen.data): {key}   cells={c_cells} layers={n_z} grid={grid_shape}")

    if a.verify_only:
        verify(a.src, c_cells, n_z, grid_shape, a.eps, cuts, a.verify_limit)
        return 0
    if not a.dst:
        raise SystemExit("--dst is required unless --verify-only")

    provenance = {
        "source": os.path.realpath(a.src),
        "legacy_key": LEGACY_KEY,
        "data_path": os.path.realpath(a.data_path),
        "new_cache_key": key,
        "cells": c_cells, "layers": n_z, "grid": grid_shape,
        "eps": a.eps, "cuts": cuts, "val_frac": a.val_frac, "shuffle": a.shuffle,
        "units": "x in MeV (R,Phi,Z); c in MeV (N,)",
        "note": ("inverse of legacy preprocess; layer sums are renormalised by the "
                 "noise-inflated sum(t) -> ~3.4e-4 relative shrinkage on populated layers, "
                 "empty layers (pure frozen noise) reproduced exactly"),
    }
    convert(a.src, a.dst, c_cells, n_z, grid_shape, a.eps, a.chunk, key, provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
