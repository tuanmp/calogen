#!/usr/bin/env python
"""Generate CaloINN showers from its native standard-normal latent prior.

Unlike ``sample_whitened_nce.py`` this does *no* NCE rejection sampling: for
each requested incident energy it draws a pure-Gaussian latent
``z ~ N(0, I)`` (the CaloINN prior used by ``CINN.sample`` / ``conditional_generate``)
and decodes it with the frozen generator.

The decoded showers are mapped back to the LEMURS truth grid and units
(``showers`` ``(N, 9, 16, 45)`` MeV) and written together with ``incident_energy``
(``(N,)`` MeV) under the same dataset names as the truth files.  ``z`` (generator
latents) and ``c`` (linear-GeV conditions) are written as extra datasets.

Conditions come from a reference LEMURS HDF5 (``--generate-from``, whose
``incident_energy`` is in MeV); if omitted, the generator config's validation
file is used.

Usage (run from repo root)::

    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/sample/sample_caloinn_gaussian.py \
        --gen-config configs/caloinn/train_lemurs_fcceeallegro.yaml \
        --gen-ckpt /path/to/generator.ckpt \
        --generate-from /path/to/reference.h5 \
        --n-generate 10000 --out run/caloinn/gaussian/gen.h5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from helper import initialize_gen, inv_transform_caloinn, resize_h5_and_add
from tqdm import tqdm


def now() -> str:
    return time.strftime("%H:%M:%S")


def load_reference_energies(
    path: str, n_generate: int | None = None
) -> torch.Tensor:
    """Read incident energies (MeV) from a LEMURS HDF5, return linear GeV ``(N, 1)``."""
    with h5py.File(path, "r") as f:
        energies_mev = f["incident_energy"][:]
    energies = torch.from_numpy(energies_mev.astype(np.float32)).reshape(-1, 1) / 1.0e3
    if n_generate is not None and n_generate > 0:
        energies = energies[:n_generate]
    return energies


@torch.inference_mode()
def sample_prior_and_decode(
    gen,
    gen_dm,
    c: torch.Tensor,
    save_path: str | Path,
    batch_size: int = 1024,
):
    """Decode ``z ~ N(0, I)`` at conditions ``c`` and write LEMURS-format output."""
    device = gen.device
    conditions = c.reshape(-1, 1).to(device)
    num_dim = gen.num_dim

    handler = h5py.File(save_path, "w")
    gen.eval()

    with tqdm(total=conditions.shape[0], desc="Decoding Gaussian latents", unit="ev") as pbar:
        for start in range(0, conditions.shape[0], batch_size):
            cb = conditions[start:start + batch_size]
            zb = torch.randn(cb.shape[0], num_dim, device=device)

            decoded = gen.decode_latent(zb, cb)
            post = gen.post_process_decoded(decoded, cb)
            nlay = len(gen.layer_boundaries) - 1
            flat = torch.cat(
                [post[f"layer_{i}"] for i in range(nlay)], dim=1
            ).cpu().numpy()

            showers, incident_energies = inv_transform_caloinn(
                flat, cb.cpu().numpy(), gen_dm.truth_format
            )

            resize_h5_and_add(handler, "z", zb.cpu().numpy())
            resize_h5_and_add(handler, "c", cb.cpu().numpy())
            resize_h5_and_add(handler, "showers", showers)
            resize_h5_and_add(handler, "incident_energy", incident_energies)
            pbar.update(cb.shape[0])

    return handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-config", required=True,
                   help="Generation config (YAML) with model class_path and init_args.")
    p.add_argument("--gen-ckpt", required=True,
                   help="Path to the generative model's checkpoint.")
    p.add_argument("--generate-from", required=False, default=None,
                   help="Reference LEMURS HDF5 whose incident_energy (MeV) values "
                        "seed one sample each. Default: the config's validation file.")
    p.add_argument("--n-generate", type=int, default=None,
                   help="Number of reference conditions to sample at (default: all).")
    p.add_argument("--multiplier", type=int, default=1,
                   help="Number of independent samples per condition (default 1).")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Decode batch size (default 1024).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p.add_argument("--out", required=True, help="Output HDF5 path.")
    p.add_argument("--compile", action="store_true",
                   help="Compile the model for faster inference."   )
    args = p.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔧 RNG seed set to {args.seed}")

    report = {
        "device": args.device,
        "gen_config": args.gen_config,
        "gen_ckpt": args.gen_ckpt,
        "generate_from": args.generate_from,
        "out": args.out,
        "generation": {},
    }

    h5_handler = None
    try:
        print(f"[{now()}] 🔧 Loading generator from {args.gen_config},\n {args.gen_ckpt}")
        gen, data_module = initialize_gen(args.gen_config, args.gen_ckpt, args.device)
        if args.compile:
            gen = torch.compile(gen)

        reference = (
            args.generate_from or data_module.val_data_path or data_module.data_path
        )
        if reference is None:
            raise ValueError("No reference HDF5: pass --generate-from.")
        print(f"[{now()}] 🔧 Conditions from {reference}")
        conditions = load_reference_energies(reference, args.n_generate)
        if args.multiplier > 1:
            conditions = conditions.repeat(args.multiplier, 1)
        report["generation"] = {
            "reference": reference,
            "n_conditions": int(conditions.shape[0]),
        }

        h5_handler = sample_prior_and_decode(
            gen,
            data_module,
            conditions,
            save_path=args.out,
            batch_size=args.batch_size,
        )

        torch.cuda.empty_cache()
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"[{now()}] ❌ failed:\n{traceback.format_exc()}")
    finally:
        if h5_handler is not None:
            h5_handler.close()

    json_path = str(output_path.with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[{now()}] 📄 Report written to {json_path}")


if __name__ == "__main__":
    main()
