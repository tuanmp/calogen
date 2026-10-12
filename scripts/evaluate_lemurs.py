#!/usr/bin/env python
"""LEMURS truth-vs-generated evaluation CLI.

Usage:
    HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/evaluate_lemurs.py \
        --config configs/eval/lemurs_base.yaml [--gen LABEL=PATH ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from calogen.eval.config import EvalConfig
from calogen.eval.report import evaluate_lemurs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--gen", action="append", default=None, metavar="LABEL=PATH")
    p.add_argument("--out", default=None)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    if args.gen:
        cfg.generators = []
        for item in args.gen:
            label, path = item.split("=", 1)
            cfg.generators.append({"label": label, "path": path})
    if args.out:
        cfg.out_dir = args.out
    if args.max_events is not None:
        cfg.max_events = args.max_events
    if args.no_pdf:
        cfg.make_pdf = False

    evaluate_lemurs(cfg)


if __name__ == "__main__":
    main()
