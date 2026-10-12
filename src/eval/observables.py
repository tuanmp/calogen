"""Physics observables for LEMURS truth-vs-generated evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from calogen.core.caloch_eval.HighLevelFeatures import HighLevelFeatures

N_LAYERS, N_R, N_PHI = 45, 9, 16
CELLS_PER_LAYER = N_R * N_PHI
R_EDGES = np.arange(0, 46, 5)
R_MID = 0.5 * (R_EDGES[:-1] + R_EDGES[1:])
PHI_EDGES = np.linspace(-np.pi, np.pi, N_PHI + 1)
PHI_MID = 0.5 * (PHI_EDGES[:-1] + PHI_EDGES[1:])
# r-major within-layer coordinates (index = r * N_PHI + phi)
X_V = (R_MID[:, None] * np.cos(PHI_MID[None, :])).reshape(-1)
Y_V = (R_MID[:, None] * np.sin(PHI_MID[None, :])).reshape(-1)


@dataclass
class EvalContext:
    hlf: HighLevelFeatures
    layer_boundaries: np.ndarray
    energies: np.ndarray
    showers: np.ndarray


def make_context(binning_xml: str, particle: str, showers, energies) -> EvalContext:
    hlf = HighLevelFeatures(particle, filename=binning_xml)
    hlf.CalculateFeatures(showers)
    hlf.Einc = np.asarray(energies).reshape(-1)
    boundaries = np.unique(hlf.bin_edges)
    return EvalContext(hlf=hlf, layer_boundaries=boundaries,
                       energies=hlf.Einc, showers=showers)


def e_tot_frac(ctx: EvalContext) -> np.ndarray:
    return ctx.hlf.GetEtot() / np.maximum(ctx.energies, 1.0)


def layer_energy_frac(ctx: EvalContext) -> np.ndarray:
    layers = [ctx.hlf.GetElayers()[l] for l in sorted(ctx.hlf.GetElayers())]
    return np.stack(layers, axis=1) / np.maximum(ctx.energies[:, None], 1.0)


def active_fraction(showers: np.ndarray, n_layers: int = N_LAYERS,
                    cells_per_layer: int = CELLS_PER_LAYER) -> np.ndarray:
    data = showers.reshape(showers.shape[0], n_layers, cells_per_layer)
    return (data > 0).mean(axis=2)


def native_center_of_energy(showers):
    data = showers.reshape(showers.shape[0], N_LAYERS, CELLS_PER_LAYER)
    esum = data.sum(axis=2) + 1e-16
    x = (data * X_V).sum(axis=2) / esum
    y = (data * Y_V).sum(axis=2) / esum
    wx = np.sqrt(((X_V ** 2 * data).sum(axis=2) / esum - x ** 2).clip(min=0.0))
    wy = np.sqrt(((Y_V ** 2 * data).sum(axis=2) / esum - y ** 2).clip(min=0.0))
    return x, y, wx, wy


def layer_energy_corr(showers: np.ndarray, n_layers: int = N_LAYERS,
                      cells_per_layer: int = CELLS_PER_LAYER) -> np.ndarray:
    data = showers.reshape(showers.shape[0], n_layers, cells_per_layer)
    layer_frac = data.sum(axis=2)
    return np.corrcoef(layer_frac, rowvar=False)


@dataclass
class Observable:
    name: str
    per_layer: bool
    xlabel: str
    xscale: str
    unit: str
    compute: Callable[[EvalContext], np.ndarray]


def _cog_x(ctx):
    return native_center_of_energy(ctx.showers)[0]


def _cog_y(ctx):
    return native_center_of_energy(ctx.showers)[1]


def _w_x(ctx):
    return native_center_of_energy(ctx.showers)[2]


def _w_y(ctx):
    return native_center_of_energy(ctx.showers)[3]


def _cell_energy(ctx):
    return ctx.showers[ctx.showers > 0].reshape(-1)


OBSERVABLES = {
    "e_tot_frac": Observable("e_tot_frac", False, r"$E_{tot}/E_{inc}$", "linear", "", e_tot_frac),
    "layer_energy_frac": Observable("layer_energy_frac", True, r"$E_l/E_{inc}$", "linear", "", layer_energy_frac),
    "active_fraction": Observable("active_fraction", True, "active cell fraction", "linear", "",
                                  lambda ctx: active_fraction(ctx.showers)),
    "cell_energy": Observable("cell_energy", False, "cell energy [MeV]", "log", "MeV", _cell_energy),
    "center_of_energy_x": Observable("center_of_energy_x", True, r"$\langle x \rangle$ [mm]", "linear", "mm", _cog_x),
    "center_of_energy_y": Observable("center_of_energy_y", True, r"$\langle y \rangle$ [mm]", "linear", "mm", _cog_y),
    "width_x": Observable("width_x", True, r"$\sigma_x$ [mm]", "linear", "mm", _w_x),
    "width_y": Observable("width_y", True, r"$\sigma_y$ [mm]", "linear", "mm", _w_y),
    "layer_energy_corr": Observable("layer_energy_corr", False, "layer", "linear", "",
                                    lambda ctx: layer_energy_corr(ctx.showers)),
}
