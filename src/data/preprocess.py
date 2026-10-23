"""Torch-side preprocessing helpers shared by the classifier converters.

These mirror the NumPy pipeline in :mod:`calogen.data.data_util` and are used
by the *physical* (CaloChallenge) classifier-input path, which reconstructs
MeV showers before scaling.  The internal-representation path for CaloINN does
not need them: per-layer fractional cells are already the model's output and
all high-level features are energy-weighted means.
"""
from __future__ import annotations

import torch


def voxel_cutoff_torch(X_mev: torch.Tensor, cutoff_mev: float | None) -> torch.Tensor:
    """Zero out shower cells at or below ``cutoff_mev``.

    Parameters
    ----------
    X_mev : torch.Tensor  shape ``(N, cells)``
        Shower cell energies in MeV.
    cutoff_mev : float, optional
        Threshold in MeV.  Cells ``<= cutoff_mev`` are set to zero.  ``None``
        or a non-positive value leaves the input untouched.
    """
    if cutoff_mev is None or cutoff_mev <= 0:
        return X_mev
    return torch.where(X_mev > cutoff_mev, X_mev, torch.zeros_like(X_mev))


def scale_shower_torch(
    X_mev: torch.Tensor, Einc_mev: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise cells by the incident energy and convert it to GeV.

    Parameters
    ----------
    X_mev : torch.Tensor  shape ``(N, cells)``
        Raw shower cells in MeV.
    Einc_mev : torch.Tensor  shape ``(N, 1)``
        Incident energies in MeV.

    Returns
    -------
    X_proc : torch.Tensor  shape ``(N, cells)``
        Energy-normalised cells (``X_mev / Einc_mev``), dimensionless.
    cond_proc : torch.Tensor  shape ``(N, 1)``
        Incident energy in GeV.
    """
    return X_mev / Einc_mev, Einc_mev / 1000.0
