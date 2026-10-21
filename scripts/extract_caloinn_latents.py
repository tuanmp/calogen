#!/usr/bin/env python
"""Extract CaloINN latents to a memmap store for NCE classifier training.

Runs a frozen generator over the LEMURS data and writes the latents ``z`` (and
their condition ``c``) to memmap ``.dat`` files plus a ``manifest.json``
consumable by :class:`calogen.core.latents.PrecomputedLatentDataModule`.

Splits:
  train / val : the training HDF5, using the classifier's train/val split
  test        : the independent validation HDF5 (``val_data_path``), filtered
                with the same cuts; encoded with the same ``width_noise``

Re-running on an existing store only extracts the *missing* splits (unless
``--overwrite``), so adding ``test`` does not recompute train/val.

Usage::

    # add the test split to an existing store
    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/extract_caloinn_latents.py \
        --config configs/caloinn/clfbase_lemurs_fcceeallegro_M1.yaml \
        --generator-ckpt /global/.../fcceeallegro-caloinn-epoch=146-val_loss=7574.05.ckpt \
        --out-dir /pscratch/sd/p/pmtuan/caloinn_latents/fcceeallegro/epoch146 \
        --splits test
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time

import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import calogen.compat  # noqa: F401  (legacy checkpoint module aliases)
from calogen.core.latents import manifest_path, read_manifest, write_manifest
from calogen.core.lemurs import LEMURSHDF5Source, LEMURSDataModule
from calogen.core.sharded import _CaloINNDataset
from calogen.models.caloinn.lightning import CaloINNLightningModule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="classifier config whose data.init_args define the split")
    p.add_argument("--generator-ckpt", required=True, help="CaloINN generator checkpoint to encode with")
    p.add_argument("--out-dir", required=True, help="latent store directory")
    p.add_argument("--splits", default="train,val,test", help="comma-separated splits to ensure")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--xml-path", default=None, help="generator xml_path override (default: config data xml_path)")
    p.add_argument("--setup-data-sample-path", default=None, help="generator init HDF5 override (default: config data_path)")
    p.add_argument("--limit-events", type=int, default=None, help="extract only first N events per split (smoke test)")
    p.add_argument("--fixed-noise", action="store_true", help="force deterministic per-event noise (equivalence checks)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _config_sections(config_path: str) -> tuple[dict, dict]:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_args = dict(cfg["data"]["init_args"])
    kw_overrides = dict((cfg.get("model", {}).get("init_args", {}) or {}).get("kw_overrides", {}) or {})
    for key in ("u0up_cut", "u0low_cut", "rew", "dep_cut", "eps", "val_frac", "width_noise"):
        if key in data_args and data_args[key] is not None:
            data_args[key] = float(data_args[key])
    return data_args, kw_overrides


def _file_identity(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime}"


def _manifest_key(generator_ckpt: str, data_args: dict, num_dim: int, c_dim: int) -> str:
    with open(data_args["xml_path"], "rb") as f:
        xml_hash = hashlib.md5(f.read()).hexdigest()
    material = "|".join(
        [
            _file_identity(generator_ckpt),
            _file_identity(data_args["data_path"]),
            _file_identity(data_args["val_data_path"]),
            str(data_args.get("u0up_cut")),
            str(data_args.get("dep_cut")),
            str(data_args.get("eps")),
            str(data_args.get("width_noise")),
            str(data_args.get("val_frac")),
            str(data_args.get("shuffle")),
            str(num_dim),
            str(c_dim),
            xml_hash,
        ]
    )
    return hashlib.md5(material.encode()).hexdigest()[:16]


def _split_fields(prefix: str, n: int, num_dim: int, c_dim: int) -> dict:
    return {
        "z": {"file": f"{prefix}_z.dat", "shape": [n, num_dim], "dtype": "float32"},
        "c": {"file": f"{prefix}_c.dat", "shape": [n, c_dim], "dtype": "float32"},
    }


def _build_test_dataset(dm: LEMURSDataModule, width_noise: float) -> _CaloINNDataset:
    valid = dm._compute_filter_indices(dm.val_data_path)
    source = LEMURSHDF5Source(dm.val_data_path)
    return _CaloINNDataset(
        source=source,
        indices=valid,
        layer_boundaries=dm.layer_boundaries,
        xml_filename=dm.xml_path,
        particle_type=dm.xml_ptype,
        eps=dm.eps,
        u0up_cut=dm.u0up_cut,
        u0low_cut=dm.u0low_cut,
        rew=dm.rew,
        dep_cut=dm.dep_cut,
        width_noise=width_noise,
        fixed_noise=False,
    )


def extract_split(
    model: CaloINNLightningModule,
    dataset,
    out_dir: str,
    prefix: str,
    num_dim: int,
    c_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    limit: int | None,
) -> dict:
    n = len(dataset) if limit is None else min(len(dataset), int(limit))
    fields = _split_fields(prefix, n, num_dim, c_dim)
    z_mm = np.memmap(os.path.join(out_dir, fields["z"]["file"]), mode="w+", dtype="float32", shape=(n, num_dim))
    c_mm = np.memmap(os.path.join(out_dir, fields["c"]["file"]), mode="w+", dtype="float32", shape=(n, c_dim))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pbar = tqdm.tqdm(total=n, desc=f"latents {prefix}", unit="samples")
    offset = 0
    with torch.inference_mode():
        for xb, cb in loader:
            if offset >= n:
                break
            take = min(int(xb.shape[0]), n - offset)
            xb = xb[:take].to(device, non_blocking=True)
            cb_dev = cb[:take].to(device, non_blocking=True)
            z = model.model(xb, cb_dev, rev=False)[0].float().cpu().numpy()
            z_mm[offset : offset + take] = z
            c_mm[offset : offset + take] = cb[:take].float().numpy()
            offset += take
            pbar.update(take)
    pbar.close()
    z_mm.flush()
    c_mm.flush()
    if offset != n:
        raise RuntimeError(f"extracted {offset} rows but expected {n} for split '{prefix}'")
    return fields


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data_args, kw_overrides = _config_sections(args.config)
    if "data_path" not in data_args:
        raise SystemExit(
            f"{args.config}: data.init_args has no 'data_path'; pass a LEMURSDataModule-backed "
            "classifier config (e.g. clfbase_lemurs_fcceeallegro_M1.yaml)."
        )
    data_args["num_workers"] = args.num_workers
    data_args["cache_mode"] = "none"
    if args.fixed_noise:
        data_args["fixed_noise"] = True
    xml_path = args.xml_path or kw_overrides.get("xml_path") or data_args["xml_path"]
    setup_path = (
        args.setup_data_sample_path
        or kw_overrides.get("setup_data_sample_path")
        or data_args["data_path"]
    )

    existing = None
    if os.path.exists(manifest_path(args.out_dir)) and not args.overwrite:
        existing = read_manifest(args.out_dir)

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    present = set(existing.get("splits", {})) if existing else set()
    missing = requested if args.overwrite else [s for s in requested if s not in present]
    if not missing:
        print(f"[extract] all requested splits present: {requested} ({manifest_path(args.out_dir)})")
        return
    print(f"[extract] will extract: {missing} (present: {sorted(present)})")

    print(f"[extract] loading generator: {args.generator_ckpt}")
    t0 = time.time()
    model = CaloINNLightningModule.load_from_checkpoint(
        args.generator_ckpt,
        map_location="cpu",
        xml_path=xml_path,
        setup_data_sample_path=setup_path,
    )
    model = model.to(args.device).eval()
    print(f"[extract] generator loaded in {time.time() - t0:.1f}s "
          f"(params={sum(p.numel() for p in model.parameters()):,}, num_dim={model.num_dim})")

    dm = LEMURSDataModule(**data_args)
    if any(s in missing for s in ("train", "val")):
        dm.setup("fit")

    if existing and existing.get("cond_dim"):
        c_dim = int(existing["cond_dim"])
    else:
        probe_ds = dm._train_dataset if ("train" in missing and dm._train_dataset is not None) else None
        if probe_ds is None:
            probe_ds = _build_test_dataset(dm, float(data_args.get("width_noise", 0.0)))
        _, c0 = next(iter(DataLoader(probe_ds, batch_size=min(64, len(probe_ds)), shuffle=False, num_workers=0)))
        c_dim = int(c0.shape[1])

    fields = dict(existing.get("splits", {})) if existing else {}
    t0 = time.time()
    for split in missing:
        if split == "train":
            dataset = dm._train_dataset
        elif split == "val":
            dataset = dm._val_dataset
        elif split == "test":
            dataset = _build_test_dataset(dm, float(data_args.get("width_noise", 0.0)))
        else:
            raise ValueError(f"unknown split '{split}' (expected train/val/test)")
        print(f"[extract] {split}: {len(dataset):,} events")
        fields[split] = extract_split(
            model, dataset, args.out_dir, split, model.num_dim, c_dim,
            args.batch_size, args.num_workers, args.device, args.limit_events,
        )
    elapsed = time.time() - t0

    key = existing["key"] if existing else _manifest_key(args.generator_ckpt, data_args, model.num_dim, c_dim)
    manifest = existing or {
        "key": key,
        "generator": {
            "ckpt": args.generator_ckpt,
            "epoch": -1,
            "num_dim": int(model.num_dim),
        },
        "source": {
            "config": args.config,
            "data_path": data_args.get("data_path"),
            "u0up_cut": data_args.get("u0up_cut"),
            "dep_cut": data_args.get("dep_cut"),
            "eps": data_args.get("eps"),
            "val_frac": data_args.get("val_frac"),
            "shuffle": data_args.get("shuffle"),
        },
        "noise": {
            "width_noise": data_args.get("width_noise"),
            "fixed_noise": data_args.get("fixed_noise"),
        },
        "cond_dim": c_dim,
    }
    manifest["source"]["val_data_path"] = data_args.get("val_data_path")
    manifest["source"]["xml_path"] = xml_path
    manifest["key"] = key
    manifest["splits"] = fields
    write_manifest(args.out_dir, manifest)

    total = sum(int(fields[s]["z"]["shape"][0]) for s in missing)
    print(f"[extract] wrote {[(s, fields[s]['z']['shape'][0]) for s in missing]} ({total:,} events) in {elapsed:.1f}s")
    print(f"[extract] manifest: {manifest_path(args.out_dir)}")


if __name__ == "__main__":
    main()
