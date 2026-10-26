"""Decoding of the CaloINN internal shower representation.

The generator works on a single flat vector per event::

    x_internal = [ cells (N_CELLS) | extra_dims (n_layers) ]

* ``cells`` are per-layer-normalised energy fractions (each layer's block
  sums to ~1), in the same r-major order as the truth showers.
* ``extra_dims`` encode the layer deposits in *telescoped* form::

      extra_dims[0]     = E_tot / E_inc
      extra_dims[i + 1] = E_i / sum_{j >= i} E_j        (i = 0 .. n_layers - 2)

  with limits 0 <= extra_dims[i+1] <= 1 (the model does not enforce this, so
  it is clipped, exactly as ``calogen.core.postprocess.unnormalize_layers``
  does before reconstructing energies).

:func:`layer_fractions` inverts that telescoping into the per-layer deposit
``lambda_i = E_i / E_inc``.  Because ``cells`` are normalised *per layer*, the
absolute deposit is not recoverable from them -- ``lambda`` is the only place
the layer energy scale survives, hence it is a feature block of its own.

This module is CaloINN-specific and must not be imported by
:mod:`calogen.data`; the generic feature maths lives in
:mod:`calogen.data.hlf`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch

from calogen.data.hlf import DEFAULT_BLOCKS, compute_hlf

__all__ = [
    "split_internal",
    "layer_fractions",
    "internal_to_classifier_input",
]


def split_internal(x: torch.Tensor, n_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``(N, cells + n_layers)`` into ``(cells, extra_dims)``."""
    if x.shape[1] <= n_layers:
        raise ValueError(
            f"internal vector has {x.shape[1]} columns, too few for "
            f"{n_layers} extra dims"
        )
    return x[..., :-n_layers], x[..., -n_layers:]


def layer_fractions(extra_dims: torch.Tensor) -> torch.Tensor:
    """Per-layer fractional deposit ``lambda_i = E_i / E_inc``.  ``(N, n_layers)``.

    Closed-form inverse of the telescoping encoding used by
    :func:`calogen.core.postprocess.unnormalize_layers`::

        f_i    = clip(extra_dims[i + 1], 0, 1)
        S_0    = extra_dims[0]
        S_{i+1}= S_i * (1 - f_i)
        lam_i  = S_i * f_i   (i < n_layers - 1)
        lam_n  = S_{n_layers - 1}
    """
    n_layers = extra_dims.shape[1]
    energy_total = extra_dims[:, 0:1]
    f = extra_dims[:, 1:].clamp(0.0, 1.0)  # (N, n_layers - 1)
    surviving = energy_total * torch.cumprod(
        torch.cat([torch.ones_like(energy_total), 1.0 - f], dim=1), dim=1
    )  # (N, n_layers): surviving[i] = S_i
    return torch.cat([surviving[:, : n_layers - 1] * f, surviving[:, n_layers - 1:]], dim=1)


def internal_to_classifier_input(
    x_internal: torch.Tensor,
    c: torch.Tensor,
    layer_boundaries: Sequence[int],
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    log_transform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the classifier input straight from CaloINN's internal samples.

    The cells are passed through untouched -- they are already per-layer
    fractional deposits, which is the representation we want the classifier to
    see.  No postprocessing, quantile masking or energy rescaling happens
    here, so generator defects survive into the classifier input.

    Parameters
    ----------
    x_internal : torch.Tensor  shape ``(N, cells + n_layers)``
        Raw generator output (or a dataloader batch in the same space).
    c : torch.Tensor  shape ``(N, 1)``
        Incident energies in GeV -- the only dimensionful input.
    layer_boundaries : sequence of int
        Cumulative cell indices defining the layers.
    blocks : iterable of str
        High-level feature blocks, see :data:`calogen.data.hlf.BLOCKS`.
    log_transform : bool
        Apply ``log1p`` to the cell block (clamped at ``-1 + eps`` because
        generator cells can be negative and ``log1p`` would return NaN).

    Returns
    -------
    torch.Tensor
        ``[cells | c | hlf]`` of shape ``(N, cells + 1 + hlf_cols)``.
    """
    n_layers = len(layer_boundaries) - 1
    cells, extra_dims = split_internal(x_internal, n_layers)

    hlf = compute_hlf(
        cells,
        layer_boundaries,
        blocks=blocks,
        layer_frac=layer_fractions(extra_dims),
    )

    if log_transform:
        cells = torch.log1p(cells.clamp(min=-1.0 + 1e-6))

    if c.ndim == 1:
        c = c.reshape(-1, 1)

    return cells, c, hlf
