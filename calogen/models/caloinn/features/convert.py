"""CINN internal representation → classifier input conversion.

Two representations are supported, selected with ``representation=``:

``"internal"`` (native to CaloINN)
    ``[cells | extra_dims]`` -> ``[cells | E_inc | HLF]``.  The cells are
    per-layer fractional deposits and are passed through untouched; the
    high-level features are energy-weighted means computed directly on them,
    so no energy rescaling, quantile masking or voxel cut happens.  Generator
    defects survive into the classifier input, which is the point.

``"physical"`` (legacy, CaloChallenge)
    Replicates the ``caloxtreme_clf`` pipeline: ``postprocess`` -> MeV ->
    ``scale_shower`` -> 51 XML-geometry features -> 772D concat.
"""

from __future__ import annotations

import numpy as np
import torch

from calogen.core import data_util
from calogen.core import postprocess as torch_postprocess
from calogen.core.caloch_eval.HighLevelFeatures import HighLevelFeatures
from calogen.data.hlf import compute_hlf_xml_geometry, get_hlf_constants
from calogen.data.preprocess import scale_shower_torch, voxel_cutoff_torch
from calogen.models.caloinn.features.representation import internal_to_classifier_input


def _compute_hlf(
    X_mev: np.ndarray,
    Einc_mev: np.ndarray,
    xml_path: str,
    particle: str,
) -> np.ndarray:
    """Compute high-level features from raw showers in MeV.

    Replicates ``data.utils.get_high_level_features`` from caloxtreme_clf.
    The 51 features are (in order):
      - E_tot (1)
      - Per-layer energies (10)
      - Eta centroids per layer (10)
      - Phi centroids per layer (10)
      - Eta widths per layer (10)
      - Phi widths per layer (10)

    Sparsity is intentionally excluded — it is commented out in the
    classifier training pipeline and would change the input dimension.

    Parameters
    ----------
    X_mev : np.ndarray  shape (N, 720)
        Raw shower cell energies in MeV.
    Einc_mev : np.ndarray  shape (N, 1)
        Incident energies in MeV.
    xml_path : str
        Path to the XML binning file.
    particle : str
        Particle name (e.g. "pion").

    Returns
    -------
    np.ndarray  shape (N, 51)
        Stacked high-level features.
    """
    hlf = HighLevelFeatures(particle, filename=xml_path)
    # setattr(hlf, "Einc", Einc_mev)  — not used by CalculateFeatures, set for reference
    hlf.CalculateFeatures(X_mev)

    # Order must match caloxtreme_clf data/utils.py exactly
    features = [
        hlf.GetEtot(),
        *hlf.GetElayers().values(),
        *hlf.GetECEtas().values(),
        *hlf.GetECPhis().values(),
        *hlf.GetWidthEtas().values(),
        *hlf.GetWidthPhis().values(),
    ]
    return np.stack(features, axis=1, dtype=np.float32)


def _scale_shower(X_mev: np.ndarray, Einc_mev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Energy-normalise shower cells and convert energy to GeV.

    Replicates ``data.utils.scale_shower`` from caloxtreme_clf.

    Parameters
    ----------
    X_mev : np.ndarray  shape (N, 720)
        Raw shower cells in MeV.
    Einc_mev : np.ndarray  shape (N, 1)
        Incident energies in MeV.

    Returns
    -------
    X_proc : np.ndarray  shape (N, 720)
        Energy-normalised cells (X_mev / Einc_mev).
    cond_proc : np.ndarray  shape (N, 1)
        Incident energy in GeV (Einc_mev / 1000).
    """
    cond = Einc_mev.reshape(-1, 1).astype(np.float64)
    X = X_mev.astype(np.float64) / cond
    cond_gev = cond / 1000.0
    return X.astype(np.float32), cond_gev.astype(np.float32)


def _apply_voxel_cutoff(X_mev: np.ndarray, cutoff_mev: float | None) -> np.ndarray:
    """Zero out shower cells below an energy threshold (in MeV).

    Replicates the ``voxel_energy_cutoff`` step from the classifier's
    ``_preprocess_batch``: ``X[X <= cutoff] = 0.0``.
    """
    if cutoff_mev is None or cutoff_mev <= 0:
        return X_mev
    return np.where(X_mev > cutoff_mev, X_mev, 0.0)


def cinn_sample_to_classifier_input(
    x_internal: np.ndarray | torch.Tensor,
    c: np.ndarray | torch.Tensor,
    layer_boundaries: list[int],
    q: torch.Tensor,
    width_noise: float,
    xml_path: str,
    particle: str,
    log_transform: bool = False,
    voxel_energy_cutoff: float | None = None,
    representation: str = "physical",
    blocks: tuple[str, ...] = ("layer_frac", "cog", "width"),
) -> np.ndarray:
    """Convert CINN internal representation to classifier input.

    Replicates the full ``_preprocess_batch`` pipeline from the
    classifier's ``LargeHDF5MLPDataModule``, including optional
    ``log1p`` transform and voxel energy cutoff.

    Parameters
    ----------
    x_internal : np.ndarray or torch.Tensor  shape (N, 730)
        CINN internal representation (720 shower cells + 10 extra dims).
        This is the raw output of ``model.sample()``.
    c : np.ndarray or torch.Tensor  shape (N, 1)
        Incident energies in GeV.
    layer_boundaries : list[int]
        Cumulative cell indices defining layer boundaries.
    q : torch.Tensor
        Quantile thresholds for postprocessing (``"physical"`` only).
    width_noise : float
        Uniform noise width subtracted before postprocessing
        (``"physical"`` only).
    xml_path : str
        Path to the XML binning file (``"physical"`` only).
    particle : str
        Particle name matching the XML, e.g. "pion" (``"physical"`` only).
    log_transform : bool
        Apply ``log1p`` to the shower block.  Must match the classifier's
        training config.
    voxel_energy_cutoff : float, optional
        Zero out cells below this MeV threshold before HLF computation
        (``"physical"`` only).
    representation : {"physical", "internal"}
        Which classifier-input representation to build.  ``"internal"``
        ignores ``q``, ``width_noise``, ``xml_path``, ``particle`` and
        ``voxel_energy_cutoff``.
    blocks : tuple of str
        High-level feature blocks for the ``"internal"`` representation, see
        :data:`calogen.data.hlf.BLOCKS`.

    Returns
    -------
    np.ndarray
        ``"physical"``: shape (N, 772), ``[X_proc(720), cond_proc(1), X_hlf(51)]``.
        ``"internal"``: shape (N, cells + 1 + hlf_cols).
    """
    if representation not in ("physical", "internal"):
        raise ValueError(
            f"representation must be 'physical' or 'internal', got {representation!r}"
        )
    # Convert torch → numpy if needed
    if isinstance(x_internal, torch.Tensor):
        x_np = x_internal.cpu().numpy().copy()
    else:
        x_np = np.array(x_internal, copy=True)
    if isinstance(c, torch.Tensor):
        c_np = c.cpu().numpy().copy()
    else:
        c_np = np.array(c, copy=True)

    # Ensure 2D
    if x_np.ndim == 1:
        x_np = x_np.reshape(1, -1)
    if c_np.ndim == 1:
        c_np = c_np.reshape(-1, 1)

    if representation == "internal":
        # CaloINN-native path: the cells are already the representation we want.
        return internal_to_classifier_input(
            torch.as_tensor(x_np, dtype=torch.float32),
            torch.as_tensor(c_np, dtype=torch.float32),
            layer_boundaries,
            blocks=blocks,
            log_transform=log_transform,
        ).numpy()

    # Step 1: Subtract width noise (matches predict_step / generate_single_energy)
    x_np = x_np - width_noise

    # Step 2: Postprocess CINN internal → physical layers in GeV
    data = data_util.postprocess(
        x_np,
        c_np,
        layer_boundaries=layer_boundaries,
        quantiles=q.detach().cpu().numpy(),
        threshold=width_noise,
    )
    # data = {"energy": (N,1) in GeV, "layer_0": (N, cells_0), ..., "layer_9": (N, cells_9)}

    # Step 3: Reconstruct flat shower array in MeV
    n_layers = len(layer_boundaries) - 1
    layers_gev = [data[f"layer_{i}"] for i in range(n_layers)]
    X_gev = np.concatenate(layers_gev, axis=1)  # (N, 720) in GeV
    X_mev = X_gev * 1e3                          # (N, 720) in MeV
    Einc_gev = data["energy"]                     # (N, 1) in GeV
    Einc_mev = Einc_gev * 1e3                     # (N, 1) in MeV

    # Step 4: Apply voxel energy cutoff (before HLF, matches _preprocess_batch)
    X_mev = _apply_voxel_cutoff(X_mev, voxel_energy_cutoff)

    # Step 5: Compute high-level features (requires MeV, after cutoff)
    X_hlf = _compute_hlf(X_mev, Einc_mev, xml_path, particle)  # (N, 51)

    # Step 6: Scale shower (energy-normalise, MeV→GeV for condition)
    X_proc, cond_proc = _scale_shower(X_mev, Einc_mev)          # (N, 720), (N, 1)

    # Step 7: Optional log transform (matches log_transform config)
    if log_transform:
        X_proc = np.log1p(X_proc)

    # Step 8: Stack into classifier input
    classifier_input = np.concatenate(
        [X_proc, cond_proc, X_hlf],
        axis=1,
    ).astype(np.float32)

    return classifier_input  # (N, 772)


def cinn_sample_to_classifier_input_torch(
    x_internal: torch.Tensor,
    c: torch.Tensor,
    layer_boundaries: list[int],
    q: torch.Tensor,
    width_noise: float,
    xml_path: str,
    particle: str,
    log_transform: bool = False,
    voxel_energy_cutoff: float | None = None,
    representation: str = "physical",
    blocks: tuple[str, ...] = ("layer_frac", "cog", "width"),
) -> torch.Tensor:
    """Fully GPU-native variant of ``cinn_sample_to_classifier_input``.

    Accepts GPU tensors and performs the entire conversion pipeline on
    GPU: postprocessing, voxel cutoff, high-level features, shower
    scaling, and stacking.  Detector geometry constants are loaded from
    the XML once and cached.  Designed as a drop-in replacement inside
    ``IMHSampler._density_ratio`` to eliminate all CPU-GPU transfers.

    Parameters
    ----------
    x_internal : torch.Tensor  shape (N, 730)
        CINN internal samples on the model device.
    c : torch.Tensor  shape (N, 1)
        Incident energies in GeV on the model device.
    layer_boundaries : list[int]
    q : torch.Tensor
        Per-cell quantile thresholds (on the same device as x_internal).
        Unused when ``representation="internal"``.
    width_noise : float
        Unused when ``representation="internal"``.
    xml_path : str
        Unused when ``representation="internal"``.
    particle : str
        Unused when ``representation="internal"``.
    log_transform : bool
    voxel_energy_cutoff : float, optional
        Unused when ``representation="internal"``.
    representation : {"physical", "internal"}
        ``"internal"`` builds ``[cells | E_inc | HLF]`` straight from the
        generator's per-layer fractional cells.
    blocks : tuple of str
        Feature blocks for the ``"internal"`` representation.

    Returns
    -------
    torch.Tensor
        ``"physical"``: shape (N, 772).  ``"internal"``: shape
        (N, cells + 1 + hlf_cols), on the same device as ``x_internal``.
    """
    if representation not in ("physical", "internal"):
        raise ValueError(
            f"representation must be 'physical' or 'internal', got {representation!r}"
        )

    if representation == "internal":
        return internal_to_classifier_input(
            x_internal,
            c,
            layer_boundaries,
            blocks=blocks,
            log_transform=log_transform,
        )

    device = x_internal.device

    # Step 1: Subtract width noise
    x = x_internal - width_noise

    # Step 2: Postprocess on GPU
    data = torch_postprocess.postprocess(
        x, c,
        layer_boundaries=layer_boundaries,
        quantiles=q,
    )

    # Step 3: Reconstruct flat shower array in MeV (on GPU)
    n_layers = len(layer_boundaries) - 1
    layers_gev = [data[f"layer_{i}"] for i in range(n_layers)]
    X_gev = torch.cat(layers_gev, dim=1)  # (N, 720) in GeV
    X_mev = X_gev * 1e3                    # (N, 720) in MeV
    Einc_gev = data["energy"]              # (N, 1) in GeV
    Einc_mev = Einc_gev * 1e3              # (N, 1) in MeV

    # Step 4: All remaining steps on GPU (zero CPU round-trips)
    hlf_consts = get_hlf_constants(xml_path, particle, device=device)

    # Voxel cutoff (before HLF)
    X_mev = voxel_cutoff_torch(X_mev, voxel_energy_cutoff)

    # High-level features (on GPU, geometry constants cached)
    X_hlf = compute_hlf_xml_geometry(X_mev, hlf_consts)  # (N, 51)

    # Scale shower (energy-normalise, MeV → GeV)
    X_proc, cond_proc = scale_shower_torch(X_mev, Einc_mev)  # (N, 720), (N, 1)

    # Optional log transform
    if log_transform:
        X_proc = torch.log1p(X_proc)

    # Stack into classifier input
    classifier_input = torch.cat(
        [X_proc, cond_proc, X_hlf], dim=1
    )

    return classifier_input  # (N, 772)
