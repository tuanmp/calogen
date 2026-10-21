"""Phase-2 parity: calogen classifier inputs must equal caloinn_lightning.

Cross-imports the original ``mcmc.convert`` (read-only, via sys.path) and
compares it against ``calogen.models.caloinn.features.convert`` on fixed
inputs.  Skips if the original repo is unavailable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

OLD_REPO = "/global/cfs/cdirs/m3443/usr/pmtuan/caloinn_lightning"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(OLD_REPO, "src")),
    reason="original caloinn_lightning repo not available",
)

XML = os.path.join(OLD_REPO, "binning_pion_odd_coarse_hcal.xml")
LAYER_BOUNDARIES = [0, 48, 96, 144, 192, 240, 336, 432, 528, 624, 720]


def _old_convert():
    # append (not insert) so calogen modules keep resolving
    for p in (OLD_REPO, os.path.join(OLD_REPO, "src")):
        if p not in sys.path:
            sys.path.append(p)
    import mcmc.convert as old_convert  # noqa: E402
    return old_convert


@pytest.mark.parametrize("log_transform,cutoff", [(False, None), (True, 1.0)])
def test_convert_matches_original(log_transform, cutoff):
    from calogen.models.caloinn.features.convert import cinn_sample_to_classifier_input

    old_convert = _old_convert()

    rng = np.random.RandomState(42)
    x = rng.uniform(0.01, 0.5, size=(4, 730)).astype(np.float32)
    c = np.array([[60.0], [70.0], [80.0], [90.0]], dtype=np.float32)
    q = torch.tensor(5e-6)

    kwargs = dict(
        q=q, width_noise=5e-6, xml_path=XML, particle="pion",
        log_transform=log_transform, voxel_energy_cutoff=cutoff,
    )
    new = cinn_sample_to_classifier_input(x, c, LAYER_BOUNDARIES, **kwargs)
    ref = old_convert.cinn_sample_to_classifier_input(x, c, LAYER_BOUNDARIES, **kwargs)

    np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-6)
