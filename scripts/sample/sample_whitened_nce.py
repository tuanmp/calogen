#!/usr/bin/env python
"""Rejection-sample the whitened CaloINN latent space at given conditions
using a trained latent NCE classifier.

Proposals are drawn as ``w ~ q0 = N(0, I)`` in the whitened latent space and
scored by a ``PrecomputedLatentWhitenedNCE`` classifier whose input is
``[w, c]``.  A proposal is accepted with ``min(1, exp(logit) / M)``; ``M``
(``--M``) is a user-supplied normalisation ceiling that must upper-bound the
classifier's ``exp(logit)`` for exact sampling.  Accepted proposals are mapped
back to the generator latent space via ``unwhiten`` and written to HDF5
together with their condition ``c``.

The condition is the same one the generator consumed: linear incident energy in
GeV (i.e. the latent store's ``c``, raw ``incident_energy`` in MeV divided by
1e3).  Unlike the generator, the whitened NCE does not take ``log10`` of it.

The classifier and datamodule are loaded from a training config (``--config``):
the datamodule (a ``PrecomputedLatentDataModule``) provides the condition store,
and the model class is restored from ``--clf-ckpt`` with its ``init_args``.
With ``--decode``, accepted latents are decoded to showers via a CaloINN
generator (``--gen-config`` / ``--gen-ckpt``).

Usage (run from repo root)::

    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/sample/sample_whitened_nce.py \\
        --config configs/caloinn/clfbase_lemurs_fcceeallegro_whitened_M10.yaml \\
        --clf-ckpt /path/to/nce.ckpt \\
        --out results/whitened_nce/generated.hdf5 --M 1e6

Output: ``--out`` HDF5 with ``z`` (unwhitened generator latents) and ``c``
(linear-GeV conditions), plus ``scores``; when ``--decode`` is set it also
contains ``showers`` (truth grid ``(N, 9, 16, 45)``, MeV) and
``incident_energy`` (``(N,)``, MeV), matching the truth-file dataset names.
A JSON report is written alongside at the same path with a ``.json`` extension.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from calogen.sampler import WhitenedNCESampler
from helper import initialize_gen, inv_transform_caloinn, resize_h5_and_add
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%H:%M:%S")


def load_reference_energies(path: str, n_generate: int | None = None) -> torch.Tensor:
    """Read incident energies (MeV) from a LEMURS HDF5, return linear GeV ``(N, 1)``."""
    with h5py.File(path, "r") as f:
        energies_mev = f["incident_energy"][:]
    energies = torch.from_numpy(energies_mev.astype(np.float32)).reshape(-1, 1) / 1.0e3
    if n_generate is not None and n_generate > 0:
        energies = energies[:n_generate]
    return energies


@torch.inference_mode()
def sample_latent(
    sampler: WhitenedNCESampler,
    c: torch.Tensor,
    M: float,
    save_path: str | Path,
    max_steps=10000,
    eval_mode=False,
):
    device = sampler.device
    conditions = c.reshape(-1, 1).to(device)

    handler = h5py.File(save_path, "w")

    z_rejected = None
    with tqdm(range(max_steps), desc="Sampling latents", unit="step") as pbar:
        for _ in pbar:

            sample_dict = sampler.sample(conditions, Z=M, eval_mode=eval_mode)
            z_proposal = sample_dict["z"]
            survive_prob = sample_dict["survive_prob"].reshape(-1)
            accepted = sample_dict["mask"].reshape(-1)

            resize_h5_and_add(handler, "z", z_proposal[accepted].cpu().numpy())
            resize_h5_and_add(handler, "c", conditions[accepted].cpu().numpy())

            z_rejected = z_proposal[~accepted]
            conditions = conditions[~accepted]

            pbar.set_postfix({
                "99th survive quantile": torch.quantile(survive_prob, 0.99).item(),
                "remaining": conditions.shape[0],
                "rate": accepted.float().mean().item(),
            })
            if conditions.shape[0] == 0:
                break

        # Best-effort: emit the last proposal for conditions still unaccepted
        # when the step budget is exhausted (empty when all were accepted).
        if conditions.shape[0] > 0 and z_rejected is not None:
            resize_h5_and_add(handler, "z", z_rejected.cpu().numpy())
            resize_h5_and_add(handler, "c", conditions.cpu().numpy())

    return handler


@torch.inference_mode()
def sample_and_decode_latent(
    sampler: WhitenedNCESampler,
    gen,
    gen_dm,
    c: torch.Tensor,
    M: float,
    save_path: str | Path,
    max_steps=10000,
    eval_mode=False,
):
    device = sampler.device
    conditions = c.reshape(-1, 1).to(device)

    handler = h5py.File(save_path, "w")

    z_rejected = None
    logits_rejected = None
    with tqdm(range(max_steps), desc="Sampling and decoding latents", unit="step") as pbar:
        for _ in pbar:

            sample_dict = sampler.sample_and_decode(gen, conditions, Z=M, eval_mode=eval_mode)
            z_proposal = sample_dict["z"]
            survive_prob = sample_dict["survive_prob"].reshape(-1)
            accepted = sample_dict["mask"].reshape(-1)
            # decoded flat shower in GeV, already restricted to accepted rows
            raw_shower = sample_dict["shower"]
            scores = torch.sigmoid(sample_dict["logits"])

            resize_h5_and_add(handler, "z", z_proposal[accepted].cpu().numpy())
            resize_h5_and_add(handler, "c", conditions[accepted].cpu().numpy())
            resize_h5_and_add(handler, "scores", scores[accepted].cpu().numpy())
            if raw_shower is not None:
                # raw_shower already holds the accepted rows (see sample_and_decode)
                shower, incident_energies = inv_transform_caloinn(raw_shower.cpu().numpy(), conditions[accepted].cpu().numpy(), gen_dm.truth_format)
                resize_h5_and_add(handler, "showers", shower)
                resize_h5_and_add(handler, "incident_energy", incident_energies)

            z_rejected = z_proposal[~accepted]
            logits_rejected = sample_dict["logits"][~accepted]
            conditions = conditions[~accepted]

            pbar.set_postfix({
                "99th survive quantile": torch.quantile(survive_prob, 0.99).item(),
                "remaining": conditions.shape[0],
                "rate": accepted.float().mean().item(),
            })
            if conditions.shape[0] == 0:
                break

        # Best-effort leftovers: decode the last proposal for still-unaccepted
        # conditions so z / c / scores / showers / incident_energy stay row-aligned.
        if conditions.shape[0] > 0 and z_rejected is not None:
            resize_h5_and_add(handler, "z", z_rejected.cpu().numpy())
            resize_h5_and_add(handler, "c", conditions.cpu().numpy())
            resize_h5_and_add(
                handler, "scores", torch.sigmoid(logits_rejected).cpu().numpy()
            )
            shower_rejected, incident_energies_rejected = inv_transform_caloinn(sampler.decode(gen, z_rejected, conditions).cpu().numpy(), conditions.cpu().numpy(), gen_dm.truth_format)
            resize_h5_and_add(handler, "showers", shower_rejected)
            resize_h5_and_add(handler, "incident_energy", incident_energies_rejected)

    return handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clf-ckpt", required=True,
                   help="Path to a PrecomputedLatentWhitenedNCE .ckpt")
    p.add_argument("--config", "-c", required=True,
                   help="Training config (YAML) with the classifier and "
                        "PrecomputedLatentDataModule class_path/init_args.")
    p.add_argument("--gen-config", required=False, default=None,
                   help="Generation config (YAML) with model class_path and "
                        "init_args; required with --decode.")
    p.add_argument("--gen-ckpt", required=False, default=None,
                   help="Path to the generative model's checkpoint; required "
                        "with --decode.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--generate-from",
        required=False,
        default=None,
        help="Reference LEMURS HDF5 whose incident_energy (MeV) values seed one "
             "proposal each. Default: conditions from the config's test split.",
    )
    p.add_argument("--n-generate", type=int, default=None,
                   help="Number of reference conditions to propose at (default: all).")
    p.add_argument("--M", type=float, required=True,
                   help="User-supplied rejection ceiling; must upper-bound "
                        "sup_{w} exp(logit(w, c)) for exact sampling.")
    p.add_argument("--max-steps", type=int, default=100000,
                   help="Proposal-round budget (default 10000).")
    p.add_argument("--out", required=True,
                   help="Output HDF5 path.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p.add_argument("--decode", action="store_true", default=False,
                   help="Decode accepted latents to showers (default False).")
    p.add_argument("--multiplier", type=int, default=1,
                   help="Multiplier for the number of proposals per condition "
                        "(default 1).")
    p.add_argument("--eval-mode", action="store_true", default=False,
                   help="Run the NCE classifier in eval() mode (dropout off); "
                        "default keeps train() mode so dropout stays active.")
    args = p.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔧 RNG seed set to {args.seed}")

    if args.M <= 0:
        raise ValueError(f"--M must be positive, got {args.M}")

    if args.eval_mode:
        print("🔧 Running with the NCE classifier in eval() mode.")

    if args.decode and not (args.gen_config and args.gen_ckpt):
        raise ValueError("When using --decode, you must also provide --gen-config and --gen-ckpt.")

    report = {
        "device": args.device,
        "clf_ckpt": args.clf_ckpt,
        "config": args.config,
        "gen_config": args.gen_config,
        "gen_ckpt": args.gen_ckpt,
        "out": args.out,
        "M": args.M,
        "generation": {},
    }

    h5_handler = None
    try:
        print(f"[{now()}] 🔧 Loading sampler from {args.config},\n {args.clf_ckpt}")
        sampler = WhitenedNCESampler(args.config, args.clf_ckpt, args.device)

        # conditions are linear incident energies in GeV
        if args.generate_from is not None:
            conditions = load_reference_energies(args.generate_from, args.n_generate)
        else:
            conditions = sampler.collect_conditions(loader="test")
            if args.n_generate is not None and args.n_generate > 0:
                conditions = conditions[:args.n_generate]
        if args.multiplier > 1:
            conditions = conditions.repeat(args.multiplier, 1)
        report["generation"]["n_conditions"] = int(conditions.shape[0])

        if not args.decode:
            h5_handler = sample_latent(
                sampler,
                conditions,
                M=args.M,
                save_path=args.out,
                max_steps=args.max_steps,
                eval_mode=args.eval_mode,
            )
        else:
            gen, data_module = initialize_gen(args.gen_config, args.gen_ckpt, args.device)
            h5_handler = sample_and_decode_latent(
                sampler,
                gen,
                data_module,
                conditions,
                M=args.M,
                save_path=args.out,
                max_steps=args.max_steps,
                eval_mode=args.eval_mode,
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
