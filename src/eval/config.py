"""Evaluation control config, loaded from YAML."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import List, Optional

import yaml


@dataclass
class EvalConfig:
    truth: str
    generators: List[dict]
    binning_xml: str
    particle: str = "photon"
    out_dir: str = "eval_out"
    max_events: Optional[int] = None
    threshold_mev: Optional[float] = None
    energy_range: Optional[List[float]] = None
    observables: Optional[List[str]] = None
    by_energy: bool = False
    energy_bins: int = 8
    make_pdf: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.generators:
            raise ValueError("`generators` must contain at least one entry")
        for g in self.generators:
            if "label" not in g or "path" not in g:
                raise ValueError(f"generator entry missing label/path: {g}")

    def to_dict(self) -> dict:
        return asdict(self)
