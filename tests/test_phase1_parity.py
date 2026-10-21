"""Phase-1 parity: calogen must reproduce caloinn_lightning numerically.

The reference tensors are produced read-only from the original repo by
``reference/capture_phase1.py`` and stored in ``reference/phase1_ref.pt``.
This test rebuilds the same modules with the new ``calogen`` code, loads the
original state dicts, and asserts identical state-dict structure and outputs.
"""

from __future__ import annotations

import os

import pytest
import torch

from calogen.models.caloinn.cinn import CINN
from calogen.models.icaloflow.model import (
    build_flow,
    build_student_flow,
    chi2_loss,
    logabsdet_of_base,
)

REF_PATH = os.path.join(os.path.dirname(__file__), "..", "reference", "phase1_ref.pt")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REF_PATH), reason="run reference/capture_phase1.py first"
)


@pytest.fixture(scope="module")
def ref():
    return torch.load(REF_PATH, map_location="cpu", weights_only=False)


def _load_state_dict(module, sd):
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"


def test_caloinn_state_dict_structure(ref):
    old_sd = ref["caloinn"]["state_dict"]
    new = CINN(ref["cinn_params"], torch.rand(8, 20) + 0.1, torch.rand(8, 1) + 1.0)
    assert set(new.state_dict()) == set(old_sd)


def test_caloinn_log_prob_and_sample_match(ref):
    c_ref = ref["caloinn"]
    new = CINN(
        ref["cinn_params"],
        torch.rand(8, 20, generator=torch.Generator().manual_seed(2026)) * 10 + 0.1,
        torch.rand(8, 1, generator=torch.Generator().manual_seed(2026)) * 100 + 1,
    )
    _load_state_dict(new, c_ref["state_dict"])
    new.eval()

    log_prob = new.log_prob(c_ref["x"], c_ref["c"]).detach()
    torch.testing.assert_close(log_prob, c_ref["log_prob"], rtol=1e-5, atol=1e-6)

    torch.manual_seed(999)
    sample = new.sample(3, c_ref["c"][:4]).detach()
    torch.testing.assert_close(sample, c_ref["sample"], rtol=1e-5, atol=1e-6)


def test_icaloflow_state_dict_and_density_match(ref):
    f_ref = ref["flow"]
    torch.manual_seed(21)
    flow = build_flow(
        features=20, context_features=2, num_layers=2,
        hidden_size=32, n_blocks=3, n_bins=8, tail_bound=14.0,
    ).double()
    assert set(flow.state_dict()) == set(f_ref["state_dict"])
    _load_state_dict(flow, f_ref["state_dict"])
    flow.eval()

    log_prob = flow.log_prob(f_ref["x"], f_ref["c"]).detach()
    inverse = flow.inverse(f_ref["x"], f_ref["c"]).detach()
    torch.testing.assert_close(log_prob, f_ref["log_prob"], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(inverse, f_ref["inverse"], rtol=1e-9, atol=1e-10)


def test_icaloflow_student_state_dict_structure(ref):
    f_ref = ref["flow"]
    torch.manual_seed(21)
    teacher = build_flow(
        features=20, context_features=2, num_layers=2,
        hidden_size=32, n_blocks=3, n_bins=8, tail_bound=14.0,
    ).double()
    _load_state_dict(teacher, f_ref["state_dict"])

    torch.manual_seed(41)
    student = build_student_flow(
        teacher, features=20, context_features=2, num_layers=2,
        hidden_size=32, n_blocks=3, n_bins=8, tail_bound=14.0,
    ).double()
    assert set(student.state_dict()) == set(f_ref["student_state_dict"])
    _load_state_dict(student, f_ref["student_state_dict"])


def test_pure_helpers_match(ref):
    f_ref = ref["flow"]
    torch.testing.assert_close(
        chi2_loss(f_ref["chi2_a"], f_ref["chi2_b"]).detach(),
        f_ref["chi2"], rtol=1e-9, atol=1e-10,
    )
    torch.testing.assert_close(
        logabsdet_of_base(f_ref["chi2_a"], width=1.0).detach(),
        f_ref["logabsdet"], rtol=1e-9, atol=1e-10,
    )
