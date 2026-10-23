"""High-level (physics summary) features from calorimeter shower cells.

Model-agnostic.  Every feature block takes a batch of cell energies
``(N, cells)`` plus the per-layer cell boundaries, and nothing here knows
about any particular generator's internal representation.

Cell values may be in any consistent linear unit (MeV, ``e_i / E_inc``, or
per-layer normalised fractions):

* :func:`center_of_energy` and :func:`width` are energy-weighted means, so
  they are invariant under any common per-layer rescaling -- they give the
  same answer for absolute and fractional cells.
* :func:`layer_energy_frac` expresses the per-layer deposit as a fraction of
  the incident energy, so it needs either absolute energies
  (``energies=``) or a caller-supplied per-layer deposit (``layer_frac=``).

The LEMURS lattice is described once here (r-major within a layer,
``index = r * N_PHI + phi``) and reused by :mod:`calogen.eval.observables`.
Note that the binning XML enumerates cells phi-major, so the XML eta/phi
arrays are *not* a substitute for :func:`cell_coordinates` -- see
:func:`compute_hlf_xml_geometry` for that (legacy CaloChallenge) path.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from calogen.core.xml_handler import XMLHandler

# ═══════════════════════════════════════════════════════════════════════
#  LEMURS lattice geometry
# ═══════════════════════════════════════════════════════════════════════

N_LAYERS, N_R, N_PHI = 45, 9, 16
CELLS_PER_LAYER = N_R * N_PHI
R_EDGES = np.arange(0, 46, 5)
R_MID = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])
PHI_EDGES = np.linspace(-np.pi, np.pi, N_PHI + 1)
PHI_MID = 0.5 * (PHI_EDGES[:-1] + PHI_EDGES[1:])
# r-major within-layer coordinates (index = r * N_PHI + phi)
X_V = (R_MID[:, None] * np.cos(PHI_MID[None, :])).reshape(-1)
Y_V = (R_MID[:, None] * np.sin(PHI_MID[None, :])).reshape(-1)

_COORDS_CACHE: dict[tuple, dict] = {}


def cell_coordinates(
    cells_per_layer: int = CELLS_PER_LAYER,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Return (cached) per-cell coordinates for the LEMURS lattice.

    Returns a dict with ``x``, ``y``, ``x2``, ``y2`` (all shape
    ``(cells_per_layer,)``) in mm, on ``device``.
    """
    if cells_per_layer != CELLS_PER_LAYER:
        raise ValueError(
            f"cell_coordinates only describes the LEMURS lattice "
            f"({CELLS_PER_LAYER} cells/layer); got {cells_per_layer}. "
            f"Use compute_hlf_xml_geometry for XML-described geometries."
        )
    key = (cells_per_layer, str(device), str(dtype))
    if key not in _COORDS_CACHE:
        x = torch.as_tensor(X_V, dtype=dtype, device=device)
        y = torch.as_tensor(Y_V, dtype=dtype, device=device)
        _COORDS_CACHE[key] = {"x": x, "y": y, "x2": x * x, "y2": y * y}
    return _COORDS_CACHE[key]


# ═══════════════════════════════════════════════════════════════════════
#  Feature blocks
#  Signature: f(cells, layer_boundaries, *, energies=None, layer_frac=None)
#  Shapes:    cells (N, cells) -> (N, n_layers) or (N, 2 * n_layers)
# ═══════════════════════════════════════════════════════════════════════

def _as_layers(cells: torch.Tensor, layer_boundaries: Sequence[int]):
    """View flat cells as ``(N, n_layers, cells_per_layer)``.

    Requires uniform layer widths (true for LEMURS and CaloChallenge ds1).
    """
    boundaries = np.asarray(layer_boundaries)
    widths = np.diff(boundaries)
    if widths.size == 0 or not np.all(widths == widths[0]):
        raise ValueError(
            f"layer_boundaries must describe equally sized layers; got widths "
            f"{sorted(set(widths.tolist()))[:4]}..."
        )
    cells_per_layer = int(widths[0])
    if cells.shape[1] != boundaries[-1]:
        raise ValueError(
            f"cells has {cells.shape[1]} columns but layer_boundaries span "
            f"{int(boundaries[-1])}"
        )
    return cells.reshape(cells.shape[0], widths.size, cells_per_layer), cells_per_layer


def _moments(cells: torch.Tensor, layer_boundaries: Sequence[int], eps: float = 1e-16):
    """Per-layer energy sum and first/second energy-weighted position moments."""
    layers, _ = _as_layers(cells, layer_boundaries)
    coords = cell_coordinates(layers.shape[-1], device=cells.device, dtype=cells.dtype)
    e_sum = layers.sum(dim=2)                      # (N, L)
    inv = 1.0 / (e_sum + eps)
    m1x = (layers @ coords["x"]) * inv
    m1y = (layers @ coords["y"]) * inv
    m2x = (layers @ coords["x2"]) * inv
    m2y = (layers @ coords["y2"]) * inv
    return e_sum, m1x, m1y, m2x, m2y


def layer_energy(cells: torch.Tensor, layer_boundaries: Sequence[int], **_) -> torch.Tensor:
    """Per-layer energy sum, in whatever unit ``cells`` carries.  ``(N, L)``."""
    return _as_layers(cells, layer_boundaries)[0].sum(dim=2)


def layer_energy_frac(
    cells: torch.Tensor,
    layer_boundaries: Sequence[int],
    *,
    energies: torch.Tensor | None = None,
    layer_frac: torch.Tensor | None = None,
    **_
) -> torch.Tensor:
    """Per-layer deposit as a fraction of the incident energy.  ``(N, L)``.

    Either pass ``energies`` (incident energy per event, in the same unit as
    ``cells``) or hand in an already known deposit via ``layer_frac`` -- the
    latter is what the CaloINN converter does, whose internal representation
    stores the layer deposits explicitly.
    """
    if layer_frac is not None:
        return layer_frac
    if energies is None:
        raise ValueError(
            "layer_energy_frac needs either `energies` (incident energy, same "
            "unit as `cells`) or a precomputed `layer_frac`"
        )
    return layer_energy(cells, layer_boundaries) / energies.reshape(-1, 1)


def active_fraction(cells: torch.Tensor, layer_boundaries: Sequence[int], **_) -> torch.Tensor:
    """Fraction of non-zero cells per layer.  ``(N, L)``.

    Threshold-sensitive by construction -- with noisy showers this mostly
    measures the noise floor, so it is off by default.
    """
    layers, _ = _as_layers(cells, layer_boundaries)
    return (layers > 0).to(cells.dtype).mean(dim=2)


def center_of_energy(cells: torch.Tensor, layer_boundaries: Sequence[int], **_) -> torch.Tensor:
    """Energy-weighted mean position per layer, ``[x | y]`` in mm.  ``(N, 2L)``."""
    _, m1x, m1y, _, _ = _moments(cells, layer_boundaries)
    return torch.cat([m1x, m1y], dim=1)


def width(cells: torch.Tensor, layer_boundaries: Sequence[int], **_) -> torch.Tensor:
    """Energy-weighted position width per layer, ``[sigma_x | sigma_y]`` in mm.

    ``sigma = sqrt(max(E[x^2]/E - (E[x]/E)^2, 0))``.  Layers with no energy
    give zeros rather than NaNs.
    """
    _, m1x, m1y, m2x, m2y = _moments(cells, layer_boundaries)
    wx = torch.sqrt((m2x - m1x * m1x).clamp(min=0.0))
    wy = torch.sqrt((m2y - m1y * m1y).clamp(min=0.0))
    return torch.cat([wx, wy], dim=1)


BLOCKS: dict[str, Callable[..., torch.Tensor]] = {
    "layer_energy": layer_energy,
    "layer_frac": layer_energy_frac,
    "active_fraction": active_fraction,
    "cog": center_of_energy,
    "width": width,
}
DEFAULT_BLOCKS: tuple[str, ...] = ("layer_frac", "cog", "width")


def compute_hlf(
    cells: torch.Tensor,
    layer_boundaries: Sequence[int],
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    *,
    energies: torch.Tensor | None = None,
    layer_frac: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate the requested feature blocks.  ``(N, sum(block widths))``.

    Parameters
    ----------
    cells:
        ``(N, cells)`` cell energies in any consistent linear unit.
    blocks:
        Names from :data:`BLOCKS`, in output-column order.
    energies:
        Incident energy per event, only needed by ``"layer_frac"``.
    layer_frac:
        Precomputed per-layer deposit, used instead of summing ``cells``.
    """
    parts = [
        BLOCKS[name](cells, layer_boundaries, energies=energies, layer_frac=layer_frac)
        for name in blocks
    ]
    if not parts:
        raise ValueError("blocks must not be empty")
    return torch.cat(parts, dim=1)


def hlf_feature_names(
    layer_boundaries: Sequence[int], blocks: Iterable[str] = DEFAULT_BLOCKS
) -> list[str]:
    """Column names matching :func:`compute_hlf`, in the same order."""
    n_layers = len(np.asarray(layer_boundaries)) - 1
    names: list[str] = []
    for name in blocks:
        if name in ("cog", "width"):
            for axis in ("x", "y"):
                names += [f"{name}_{axis}_{layer:02d}" for layer in range(n_layers)]
        else:
            names += [f"{name}_{layer:02d}" for layer in range(n_layers)]
    return names


# ═══════════════════════════════════════════════════════════════════════
#  Legacy XML-geometry features (CaloChallenge), kept for the physical path
# ═══════════════════════════════════════════════════════════════════════

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
    """Return (cached) detector geometry constants read from the binning XML.

    Keys: ``bin_edges``, ``eta``, ``phi``, ``relevant_layers``,
    ``alpha_layers``.
    """
    key = (xml_path, particle, str(device))
    if key not in _hlf_constants_cache:
        _hlf_constants_cache[key] = _build_constants(xml_path, particle, device)
    return _hlf_constants_cache[key]


def compute_hlf_xml_geometry(
    shower: torch.Tensor,
    hlf_constants: dict,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Legacy high-level features from the XML eta/phi geometry.

    Reproduces ``HighLevelFeatures.CalculateFeatures`` for CaloChallenge-style
    detectors.  The XML arrays are enumerated phi-major, so these centroids
    are only meaningful for data ordered the same way -- do not use this for
    r-major LEMURS cells (use :func:`compute_hlf` instead).

    Returns ``(N, 1 + n_relevant + 4 * n_alpha)``:
    ``[E_tot, E_layers, EC_etas, EC_phis, WidthEtas, WidthPhis]``.
    """
    bin_edges = hlf_constants["bin_edges"]
    eta = hlf_constants["eta"]
    phi = hlf_constants["phi"]
    relevant = hlf_constants["relevant_layers"]
    alpha = hlf_constants["alpha_layers"]
    dev = shower.device

    e_tot = shower.sum(dim=1)  # (N,)

    e_layers_list = []
    for l in relevant:
        e_layers_list.append(shower[:, bin_edges[l]:bin_edges[l + 1]].sum(dim=1))
    e_layers = torch.stack(e_layers_list, dim=1)  # (N, n_relevant)

    ec_etas_list, ec_phis_list, w_etas_list, w_phis_list = [], [], [], []
    for l in alpha:
        cells = shower[:, bin_edges[l]:bin_edges[l + 1]]  # (N, M)
        e_sum = cells.sum(dim=1, keepdim=True) + eps  # (N, 1)

        eta_l = eta[l].to(dev)  # (M,)
        phi_l = phi[l].to(dev)  # (M,)

        ec_eta = (cells * eta_l).sum(dim=1) / e_sum.squeeze(1)
        ec_phi = (cells * phi_l).sum(dim=1) / e_sum.squeeze(1)
        e2_eta = (cells * eta_l * eta_l).sum(dim=1) / e_sum.squeeze(1)
        e2_phi = (cells * phi_l * phi_l).sum(dim=1) / e_sum.squeeze(1)

        ec_etas_list.append(ec_eta)
        ec_phis_list.append(ec_phi)
        w_etas_list.append(torch.sqrt(torch.clamp(e2_eta - ec_eta * ec_eta, min=0.0)))
        w_phis_list.append(torch.sqrt(torch.clamp(e2_phi - ec_phi * ec_phi, min=0.0)))

    features = torch.cat(
        [
            e_tot.unsqueeze(1),
            e_layers,
            torch.stack(ec_etas_list, dim=1),
            torch.stack(ec_phis_list, dim=1),
            torch.stack(w_etas_list, dim=1),
            torch.stack(w_phis_list, dim=1),
        ],
        dim=1,
    )
    return features  # (N, 1 + n_relevant + 4 * n_alpha)
