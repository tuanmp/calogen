"""GPU-native post-processing for CaloINN generated showers.

This module provides PyTorch equivalents of the NumPy functions in
``data_util.py`` (``postprocess`` and ``unnormalize_layers``).  It is
intended to replace the CPU/NumPy post-processing path in the Lightning
module once numerical convergence is verified.

All operations are performed in float32, matching the model's default dtype.
"""

import torch


def unnormalize_layers(x, c, layer_boundaries, eps=1.0e-10):
    """Reverse the per-layer normalization done in ``data_util.preprocess``.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(N, num_cells + num_layers)``.  The first ``num_cells``
        entries are per-cell energy fractions (each layer sums to 1), and the
        last ``num_layers`` entries are the extra dimensions encoding layer
        energies.
    c : torch.Tensor
        Shape ``(N, 1)`` incident energies in GeV.
    layer_boundaries : array-like
        1-D sequence of cell indices defining layer start/end positions.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Shape ``(N, num_cells)`` unnormalized cell energies in GeV.
    """
    number_of_layers = len(layer_boundaries) - 1

    # Extra dimensions live in the last n_layers columns.
    extra_dims = x[..., -number_of_layers:].clone()
    # Energy-fraction extra dims must lie in [0, 1].
    extra_dims[:, (-number_of_layers + 1):] = torch.clamp(
        extra_dims[:, (-number_of_layers + 1):], min=0.0, max=1.0
    )

    # Normalized cell fractions live in the remaining leading columns.
    x_frac = x[..., :-number_of_layers]
    output = torch.zeros_like(x_frac)

    # Reconstruct total deposited energy and per-layer energies from extra dims.
    incident_energy = c[..., 0]
    en_tot = incident_energy * extra_dims[:, 0]
    cum_sum = torch.zeros_like(en_tot)

    layer_energies = []
    for i in range(extra_dims.shape[-1] - 1):
        ens = (en_tot - cum_sum) * extra_dims[:, i + 1]
        layer_energies.append(ens)
        cum_sum = cum_sum + ens
    layer_energies.append(en_tot - cum_sum)
    layer_energies = torch.stack(layer_energies, dim=1)  # (N, n_layers)

    # Scale normalized cell fractions by reconstructed layer energies.
    for layer_index, (layer_start, layer_end) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:])
    ):
        layer_frac = x_frac[..., layer_start:layer_end]
        output[..., layer_start:layer_end] = (
            layer_frac
            * layer_energies[:, [layer_index]]
            / (torch.sum(layer_frac, dim=1, keepdim=True) + eps)
        )

    return output


def postprocess(x, c, layer_boundaries, quantiles, threshold=1e-4, eps=1.0e-10):
    """PyTorch equivalent of ``data_util.postprocess``.

    Parameters
    ----------
    x : torch.Tensor
        Generated samples, shape ``(N, num_cells + num_layers)``.
    c : torch.Tensor
        Incident energies in GeV, shape ``(N, 1)``.
    layer_boundaries : array-like
        Layer boundary indices.
    quantiles : torch.Tensor
        Per-cell 1% quantiles used for noise suppression.
    threshold : float
        Kept for API compatibility with the NumPy version; the actual cut is
        performed against ``quantiles``.
    eps : float
        Numerical stability constant.

    Returns
    -------
    dict
        Dictionary with keys ``"energy"`` and ``"layer_0"``, ``"layer_1"``,
        ...  Values are tensors on the same device as ``x``.
    """
    x = x.clone()
    c = c.clone()

    # Zero out sub-quantile cells (noise suppression).
    x[x < quantiles] = 0.0

    x_unnorm = unnormalize_layers(x, c, layer_boundaries, eps=eps)

    data = {"energy": c[..., [0]]}
    for layer_index, (layer_start, layer_end) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:])
    ):
        data[f"layer_{layer_index}"] = x_unnorm[..., layer_start:layer_end]

    return data
