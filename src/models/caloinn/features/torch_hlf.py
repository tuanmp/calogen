"""GPU-native high-level feature computation for CaloINN showers.

Provides PyTorch equivalents of the numpy HLF pipeline in
``convert.py`` and ``_hlf.py``.  The geometry constants (cell eta/phi
positions, layer boundaries) are loaded from the XML binning file once
and cached, enabling the full classifier-input pipeline to run on GPU
with zero CPU transfers per MH step.
"""

from __future__ import annotations

import torch

from calogen.core.xml_handler import XMLHandler

# ---------------------------------------------------------------------------
#  Geometry constants (loaded once per XML + particle + device)
# ---------------------------------------------------------------------------

_hlf_constants_cache: dict[tuple, dict] = {}


def _build_constants(xml_path: str, particle: str, device: str | torch.device):
    """Parse the XML binning file and extract tensors for HLF computation."""
    xml = XMLHandler(particle, filename=xml_path)
    bin_edges = xml.GetBinEdges()
    eta_all, phi_all = xml.GetEtaPhiAllLayers()
    relevant_layers = xml.GetRelevantLayers()
    alpha_layers = xml.GetLayersWithBinningInAlpha()

    # Convert per-layer geometry arrays to tensors on the target device.
    eta_t = [torch.tensor(e, dtype=torch.float32, device=device) for e in eta_all]
    phi_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in phi_all]

    return {
        "bin_edges": bin_edges,
        "eta": eta_t,
        "phi": phi_t,
        "relevant_layers": relevant_layers,
        "alpha_layers": alpha_layers,
    }


def get_hlf_constants(
    xml_path: str,
    particle: str,
    device: str | torch.device = "cpu",
) -> dict:
    """Return (cached) detector geometry constants for HLF computation.

    Parameters
    ----------
    xml_path : str
        Path to the XML binning file.
    particle : str
        Particle name (e.g. ``"pion"``).
    device : str or torch.device
        Target device for the geometry tensors.

    Returns
    -------
    dict
        Keys: ``bin_edges``, ``eta``, ``phi``, ``relevant_layers``,
        ``alpha_layers``.
    """
    key = (xml_path, particle, str(device))
    if key not in _hlf_constants_cache:
        _hlf_constants_cache[key] = _build_constants(xml_path, particle, device)
    return _hlf_constants_cache[key]


# ---------------------------------------------------------------------------
#  Torch-native computation functions
# ---------------------------------------------------------------------------


def voxel_cutoff_torch(
    X_mev: torch.Tensor, cutoff_mev: float | None
) -> torch.Tensor:
    """Zero out shower cells below an energy threshold.

    Parameters
    ----------
    X_mev : torch.Tensor  shape ``(N, 720)``
        Shower cell energies in MeV.
    cutoff_mev : float, optional
        Threshold in MeV.  Cells ≤ this value are set to zero.

    Returns
    -------
    torch.Tensor  shape ``(N, 720)``
    """
    if cutoff_mev is None or cutoff_mev <= 0:
        return X_mev
    return torch.where(X_mev > cutoff_mev, X_mev, torch.zeros_like(X_mev))


def scale_shower_torch(
    X_mev: torch.Tensor, Einc_mev: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy-normalise shower cells.

    Parameters
    ----------
    X_mev : torch.Tensor  shape ``(N, 720)``
        Raw shower cells in MeV.
    Einc_mev : torch.Tensor  shape ``(N, 1)``
        Incident energies in MeV.

    Returns
    -------
    X_proc : torch.Tensor  shape ``(N, 720)``
        Energy-normalised cells (X_mev / Einc_mev).
    cond_proc : torch.Tensor  shape ``(N, 1)``
        Incident energy in GeV (Einc_mev / 1000).
    """
    X_proc = X_mev / Einc_mev
    cond_gev = Einc_mev / 1000.0
    return X_proc, cond_gev


def compute_hlf_torch(
    shower_mev: torch.Tensor,
    hlf_constants: dict,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Compute 51 high-level features from a batch of showers.

    Replicates the feature extraction in
    ``HighLevelFeatures.CalculateFeatures`` and the ordering in
    ``_compute_hlf`` from ``convert.py``.

    Parameters
    ----------
    shower_mev : torch.Tensor  shape ``(N, 720)``
        Shower cell energies in MeV on the model device.
    hlf_constants : dict
        Output of ``get_hlf_constants``, with tensors on the same device.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor  shape ``(N, 51)``
        Stacked features: [E_tot (1), E_layers (n_rel), EC_etas (n_alpha),
        EC_phis (n_alpha), WidthEtas (n_alpha), WidthPhis (n_alpha)].
    """
    bin_edges = hlf_constants["bin_edges"]
    eta = hlf_constants["eta"]
    phi = hlf_constants["phi"]
    relevant = hlf_constants["relevant_layers"]
    alpha = hlf_constants["alpha_layers"]
    dev = shower_mev.device

    # -- E_tot -----------------------------------------------------------
    e_tot = shower_mev.sum(dim=1)  # (N,)

    # -- E_layers (per relevant layer) -----------------------------------
    e_layers_list = []
    for l in relevant:
        el = shower_mev[:, bin_edges[l] : bin_edges[l + 1]].sum(dim=1)
        e_layers_list.append(el)
    e_layers = torch.stack(e_layers_list, dim=1)  # (N, n_relevant)

    # -- EC and Widths (per alpha-binned layer) --------------------------
    ec_etas_list = []
    ec_phis_list = []
    w_etas_list = []
    w_phis_list = []

    for l in alpha:
        cells = shower_mev[:, bin_edges[l] : bin_edges[l + 1]]  # (N, M)
        e_sum = cells.sum(dim=1, keepdim=True) + eps            # (N, 1)

        eta_l = eta[l].to(dev)   # (M,)
        phi_l = phi[l].to(dev)   # (M,)

        # Weighted centroid
        ec_eta = (cells * eta_l).sum(dim=1) / e_sum.squeeze(1)  # (N,)
        ec_phi = (cells * phi_l).sum(dim=1) / e_sum.squeeze(1)  # (N,)

        # Weighted second moment
        e2_eta = (cells * eta_l * eta_l).sum(dim=1) / e_sum.squeeze(1)
        e2_phi = (cells * phi_l * phi_l).sum(dim=1) / e_sum.squeeze(1)

        # Std = sqrt(max(E[x^2] - E[x]^2, 0))
        w_eta = torch.sqrt(torch.clamp(e2_eta - ec_eta * ec_eta, min=0.0))
        w_phi = torch.sqrt(torch.clamp(e2_phi - ec_phi * ec_phi, min=0.0))

        ec_etas_list.append(ec_eta)
        ec_phis_list.append(ec_phi)
        w_etas_list.append(w_eta)
        w_phis_list.append(w_phi)

    ec_etas = torch.stack(ec_etas_list, dim=1)   # (N, n_alpha)
    ec_phis = torch.stack(ec_phis_list, dim=1)   # (N, n_alpha)
    w_etas = torch.stack(w_etas_list, dim=1)      # (N, n_alpha)
    w_phis = torch.stack(w_phis_list, dim=1)      # (N, n_alpha)

    # -- Stack into final feature vector ---------------------------------
    features = torch.cat(
        [
            e_tot.unsqueeze(1),  # (N, 1)
            e_layers,            # (N, n_relevant)
            ec_etas,             # (N, n_alpha)
            ec_phis,             # (N, n_alpha)
            w_etas,              # (N, n_alpha)
            w_phis,              # (N, n_alpha)
        ],
        dim=1,
    )

    return features  # (N, 1 + n_relevant + 4 * n_alpha)
