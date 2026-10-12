import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pytest

_REF = ("/pscratch/sd/p/pmtuan/lemurs/FCCeeALLEGRO/validation/"
        "LEMURS_FCCeeALLEGRO_gamma_100kEvents_1GeV100GeV_GPSflat_part10.h5")


@pytest.mark.skipif(not os.path.exists(_REF), reason="LEMURS reference not found")
def test_load_matches_source_flattening():
    import h5py
    from calogen.core.lemurs import LEMURSHDF5Source
    from calogen.eval.io import load_showers

    s = load_showers(_REF, "truth", max_events=50, seed=42)
    with h5py.File(_REF, "r") as f:
        n_total = f["showers"].shape[0]
    idx = np.sort(np.random.RandomState(42).choice(n_total, 50, replace=False))

    src = LEMURSHDF5Source(_REF)
    try:
        flat, _ = src.read_rows(idx)
    finally:
        src.close()
    assert np.allclose(s.showers, flat, atol=1e-5)
