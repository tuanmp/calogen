import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pytest

from calogen.eval import observables as obs

N_LAYERS, N_R, N_PHI = 45, 9, 16
CPL = N_R * N_PHI


def _make_showers(n=4):
    rng = np.random.RandomState(0)
    return rng.random((n, N_LAYERS * CPL)).astype(np.float32)


def test_registry_names_present():
    for name in ["e_tot_frac", "layer_energy_frac", "active_fraction",
                 "cell_energy", "center_of_energy_x", "center_of_energy_y",
                 "width_x", "width_y", "layer_energy_corr"]:
        assert name in obs.OBSERVABLES


def test_native_center_of_energy_known():
    # single hot cell in layer 0 at (r_mid[0], phi_mid[8])
    s = np.zeros((1, N_LAYERS * CPL), dtype=np.float32)
    r_idx, phi_idx = 0, 8
    s[0, r_idx * N_PHI + phi_idx] = 1.0
    x, y, wx, wy = obs.native_center_of_energy(s)
    assert x.shape == (1, N_LAYERS)
    assert abs(x[0, 0] - obs.R_MID[0] * np.cos(obs.PHI_MID[8])) < 1e-5
    assert abs(y[0, 0] - obs.R_MID[0] * np.sin(obs.PHI_MID[8])) < 1e-5
    assert abs(wx[0, 0]) < 1e-5
    assert abs(wy[0, 0]) < 1e-5


def test_active_fraction_bounds():
    s = _make_showers()
    s[:, :100] = 0.0
    frac = obs.active_fraction(s, N_LAYERS, CPL)
    assert frac.shape == (4, N_LAYERS)
    assert (frac >= 0).all() and (frac <= 1).all()


def test_layer_energy_corr_shape():
    s = _make_showers()
    corr = obs.layer_energy_corr(s, N_LAYERS, CPL)
    assert corr.shape == (N_LAYERS, N_LAYERS)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)


_REF = ("/pscratch/sd/p/pmtuan/lemurs/FCCeeALLEGRO/validation/"
        "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part10.h5")


@pytest.mark.skipif(not os.path.exists(_REF), reason="LEMURS reference not found")
def test_native_cog_matches_old_geometry():
    from calogen.eval.io import load_showers
    import h5py
    s = load_showers(_REF, "truth", max_events=200, seed=42)
    x, y, wx, wy = obs.native_center_of_energy(s.showers)
    with h5py.File(_REF, "r") as f:
        n_total = f["showers"].shape[0]
    idx = np.sort(np.random.RandomState(42).choice(n_total, 200, replace=False))
    with h5py.File(_REF, "r") as f:
        raw = f["showers"][idx].transpose(0, 3, 2, 1).reshape(200, 45, -1)
    X_old = (obs.R_MID[None, :] * np.cos(obs.PHI_MID[:, None])).reshape(-1)
    esum = raw.sum(2) + 1e-16
    x_old = (raw * X_old).sum(2) / esum
    assert np.allclose(x, x_old, atol=1e-3)
