"""Canonical shower loading for LEMURS evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
from calogen.data.data_util import TruthFormat
from calogen.data.sources import HDF5Source


@dataclass
class ShowerSet:
    label: str
    showers: np.ndarray      # (N, 6480) float32, layer-major (Z,R,Phi)
    energies: np.ndarray     # (N,) float64, MeV
    grid_shape: Optional[Tuple[int, ...]]
    source_path: str


def load_showers(
    path: str,
    label: str,
    max_events: Optional[int] = None,
    threshold_mev: Optional[float] = None,
    energy_range: Optional[Tuple[float, float]] = None,
    seed: int = 42,
) -> ShowerSet:
    fmt = TruthFormat.detect_from_file(path)
    with h5py.File(path, "r") as f:
        n_total = int(f["showers"].shape[0])

    rng = np.random.RandomState(seed)
    if max_events is not None and max_events < n_total:
        idx = np.sort(rng.choice(n_total, size=int(max_events), replace=False))
    else:
        idx = np.arange(n_total)

    if fmt.showers_grid_shape is not None:
        source = HDF5Source(path)
        try:
            showers, energies = source.read_rows(idx)
            showers = source.truth_format.flatten_showers(showers)
            energies = source.truth_format.format_energy(energies)
        finally:
            source.close()

    else:
        with h5py.File(path, "r") as f:
            showers = np.asarray(f["showers"][idx], dtype=np.float32)
            energies = np.asarray(f["incident_energies"][idx], dtype=np.float64).reshape(-1)

    if energy_range is not None:
        lo, hi = energy_range
        keep = (energies >= lo) & (energies <= hi)
        showers, energies = showers[keep], energies[keep]

    if threshold_mev is not None:
        showers = np.where(showers < threshold_mev, 0.0, showers).astype(np.float32)

    return ShowerSet(
        label=label,
        showers=showers,
        energies=energies,
        grid_shape=fmt.showers_grid_shape,
        source_path=path,
    )
