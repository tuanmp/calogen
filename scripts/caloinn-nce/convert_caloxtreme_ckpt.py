#!/usr/bin/env python
"""Convert a ``caloxtreme_clf`` MLP checkpoint to a native calogen checkpoint.

The old checkpoint is a ``module.lightning.MLPClassifier``: its head weights are
already the same architecture and key layout as
:class:`calogen.classifiers.caloinn.shower_nce.LemursPhysicalShowerNCE`
(``net.net.*`` / ``net.output_layer.*``), but its hyperparameters only describe
the MLP.  This script rewrites the metadata so the new class can be constructed
and loaded by ``load_from_checkpoint``, while copying the weights unchanged.

It does **not** embed the CaloINN generator (7.7 GB); the new class loads it
from ``generator_ckpt_path`` at construction time and sets
``strict_loading = False`` so a head-only checkpoint is a valid state dict.

Usage (from the repository root)::

    HDF5_USE_FILE_LOCKING=FALSE uv run python \\
        scripts/caloinn-nce/convert_caloxtreme_ckpt.py \\
        --input  .../fcceeallegro-clf-val_aucroc=0.858136-epoch-epoch=318.ckpt \\
        --output .../fcceeallegro-clf-physical-onthefly.ckpt \\
        --generator-ckpt .../fcceeallegro-caloinn-epoch=146-val_loss=7574.05.ckpt \\
        --xml-path configs/binning_lemurs_fcceeallegro.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

NEW_CLASS_PATH = "calogen.classifiers.caloinn.shower_nce.LemursPhysicalShowerNCE"
_HEAD_KEYS = ("hidden_dim", "num_layers", "batch_norm", "layer_norm",
              "output_dim", "dropout", "activation")


def _expected_head_keys(
    hidden_dim: int,
    num_layers: int,
    batch_norm: bool,
    layer_norm: bool,
    output_dim: int,
    dropout: float,
    activation: str,
    input_dim: int,
) -> set[str]:
    """Parameter names of the new classifier head, prefixed by ``net.``."""
    from calogen.classifiers.base import MLP

    mlp = MLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        batch_norm=batch_norm,
        layer_norm=layer_norm,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
    )
    mlp(torch.zeros(1, input_dim))  # materialise LazyLinear
    return {f"net.{key}" for key in mlp.state_dict()}


def convert_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    generator_ckpt: str,
    xml_path: str,
    voxel_energy_cutoff: float | None = 1.0,
    log_transform: bool = True,
    particle: str | None = None,
    keep_optimizer: bool = False,
    verify: bool = True,
) -> dict:
    """Rewrite one ``caloxtreme_clf`` checkpoint for ``LemursPhysicalShowerNCE``.

    Returns the converted checkpoint dictionary (also written to
    ``output_path``).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    old = torch.load(input_path, map_location="cpu", weights_only=False)

    old_hparams = dict(old.get("hyper_parameters") or {})
    state_dict = old["state_dict"]

    missing_arch = [key for key in _HEAD_KEYS if key not in old_hparams]
    if missing_arch:
        raise KeyError(
            f"{input_path} is missing classifier hyperparameters {missing_arch}; "
            "is this a caloxtreme MLPClassifier checkpoint?"
        )

    head_keys = [key for key in state_dict if key.startswith("net.")]
    other_keys = [key for key in state_dict if not key.startswith("net.")]
    if other_keys:
        raise KeyError(
            f"Unexpected non-head keys in {input_path}: {other_keys[:5]}; "
            "this converter only handles head-only MLP checkpoints."
        )
    if not head_keys:
        raise KeyError(f"No 'net.*' classifier weights in {input_path}")

    input_dim = int(state_dict["net.net.0.weight"].shape[1])
    head_state = {key: state_dict[key] for key in head_keys}

    expected = _expected_head_keys(
        hidden_dim=int(old_hparams["hidden_dim"]),
        num_layers=int(old_hparams["num_layers"]),
        batch_norm=bool(old_hparams["batch_norm"]),
        layer_norm=bool(old_hparams["layer_norm"]),
        output_dim=int(old_hparams["output_dim"]),
        dropout=float(old_hparams["dropout"]),
        activation=str(old_hparams["activation"]),
        input_dim=input_dim,
    )
    missing = sorted(expected - set(head_state))
    unexpected = sorted(set(head_state) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Head key mismatch for {input_path}: missing={missing}, "
            f"unexpected={unexpected}"
        )

    kw_overrides = {"xml_path": str(xml_path)}
    if particle is not None:
        kw_overrides["xml_ptype"] = str(particle)

    new_hparams = {key: old_hparams[key] for key in _HEAD_KEYS}
    new_hparams.update(
        {
            "_class_path": NEW_CLASS_PATH,
            "generator_ckpt_path": str(generator_ckpt),
            "kw_overrides": kw_overrides,
            "voxel_energy_cutoff": voxel_energy_cutoff,
            "log_transform": bool(log_transform),
        }
    )
    # Carry optimizer hyperparameters over when the old checkpoint defined them.
    for key in ("lr", "step_size", "gamma", "amsgrad"):
        if key in old_hparams:
            new_hparams[key] = old_hparams[key]

    new = {
        "state_dict": head_state,
        "hyper_parameters": new_hparams,
        "epoch": int(old.get("epoch", 0)),
        "global_step": int(old.get("global_step", 0)),
        "pytorch-lightning_version": old.get("pytorch-lightning_version", ""),
        "hparams_name": "kwargs",
    }
    if keep_optimizer:
        for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
            if key in old:
                new[key] = old[key]

    if verify:
        _verify_head(head_state, old_hparams, input_dim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, output_path)
    return new


def _verify_head(state_dict: dict, hparams: dict, input_dim: int) -> None:
    """Check that the rewritten head reproduces the original logits."""
    from calogen.classifiers.base import MLP

    reference = MLP(
        hidden_dim=int(hparams["hidden_dim"]),
        num_layers=int(hparams["num_layers"]),
        batch_norm=bool(hparams["batch_norm"]),
        layer_norm=bool(hparams["layer_norm"]),
        output_dim=int(hparams["output_dim"]),
        dropout=float(hparams["dropout"]),
        activation=str(hparams["activation"]),
    )
    reference.load_state_dict(
        {key[len("net."):]: value for key, value in state_dict.items()}
    )
    reference.eval()

    converted = MLP(
        hidden_dim=int(hparams["hidden_dim"]),
        num_layers=int(hparams["num_layers"]),
        batch_norm=bool(hparams["batch_norm"]),
        layer_norm=bool(hparams["layer_norm"]),
        output_dim=int(hparams["output_dim"]),
        dropout=float(hparams["dropout"]),
        activation=str(hparams["activation"]),
    )
    converted.load_state_dict(
        {key[len("net."):]: value for key, value in state_dict.items()}
    )
    converted.eval()

    x = torch.randn(4, input_dim, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(converted(x), reference(x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Old caloxtreme .ckpt.")
    parser.add_argument("--output", required=True, help="New calogen .ckpt.")
    parser.add_argument("--generator-ckpt", required=True,
                        help="CaloINN generator checkpoint used at sampling time.")
    parser.add_argument("--xml-path", required=True,
                        help="Binning XML for the XML-geometry high-level features.")
    parser.add_argument("--particle", default=None,
                        help="Optional particle override for the XML (default: generator ckpt).")
    parser.add_argument("--voxel-energy-cutoff", type=float, default=1.0,
                        help="MeV cell cutoff used by the old classifier (default: 1.0).")
    parser.add_argument("--no-log-transform", action="store_true",
                        help="Do not apply log1p to the energy-normalised cells.")
    parser.add_argument("--keep-optimizer", action="store_true",
                        help="Also copy optimizer/scheduler/loop state.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-conversion logit comparison.")
    args = parser.parse_args()

    converted = convert_checkpoint(
        args.input,
        args.output,
        generator_ckpt=args.generator_ckpt,
        xml_path=args.xml_path,
        voxel_energy_cutoff=args.voxel_energy_cutoff,
        log_transform=not args.no_log_transform,
        particle=args.particle,
        keep_optimizer=args.keep_optimizer,
        verify=not args.no_verify,
    )
    print(f"Wrote {args.output}")
    print(f"  class:      {converted['hyper_parameters']['_class_path']}")
    print(f"  input dim:  {converted['state_dict']['net.net.0.weight'].shape[1]}")
    print(f"  epoch/step: {converted['epoch']}/{converted['global_step']}")


if __name__ == "__main__":
    main()
