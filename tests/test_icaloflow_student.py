"""Unit tests for the iCaloFlow student IAF builder and PDD helpers."""

import time

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.models.icaloflow.lightning import ICaloFlowLightningModule
from calogen.models.icaloflow.model import (
    build_flow,
    build_student_flow,
    chi2_loss,
    guided_cascade,
    logabsdet_of_base,
)
from normflows.flows.mixing import Permute

_N_BLOCKS = 2
_HIDDEN_SIZE = 32
_TAIL_BOUND = 3.0


def _build_teacher(features, context_features, num_layers):
    torch.manual_seed(0)
    return build_flow(
        features,
        context_features,
        num_layers,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=_N_BLOCKS,
        tail_bound=_TAIL_BOUND,
    )


def _build_student(teacher, features, context_features, num_layers):
    return build_student_flow(
        teacher,
        features,
        context_features,
        num_layers,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=_N_BLOCKS,
        tail_bound=_TAIL_BOUND,
    )


# ---------------------------------------------------------------------------
# IAFRQS speed asymmetry
# ---------------------------------------------------------------------------


def test_student_sample_faster_than_log_prob():
    teacher = _build_teacher(45, 1, 1)
    student = _build_student(teacher, 45, 1, 1).double()
    context = torch.randn(8, 1, dtype=torch.float64)

    # Warm up + time the sequential log_prob (density) direction.
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(2):
        student.log_prob(torch.randn(8, 45, dtype=torch.float64), context)
    t_density = (time.perf_counter() - t0) / 2

    # Time the parallel sample (generation) direction.
    t0 = time.perf_counter()
    for _ in range(2):
        student.sample(8, context)
    t_sample = (time.perf_counter() - t0) / 2

    assert t_sample < t_density, (
        f"IAF student sampling ({t_sample:.4f}s) should be faster than "
        f"log_prob ({t_density:.4f}s) (parallel vs sequential)"
    )


# ---------------------------------------------------------------------------
# build_student_flow shapes + permutation parity
# ---------------------------------------------------------------------------


def test_build_student_flow_sample_and_log_prob_shapes():
    teacher = _build_teacher(144, 2, 2)
    student = _build_student(teacher, 144, 2, 2)
    context = torch.randn(4, 2)
    x, _ = student.sample(4, context)
    assert x.shape == (4, 144)
    lp = student.log_prob(torch.randn(4, 144), context)
    assert lp.shape == (4,)
    assert torch.isfinite(lp).all()


def test_build_student_flow_single_block_no_perms():
    torch.manual_seed(0)
    teacher = build_flow(
        45, 1, 1, hidden_size=_HIDDEN_SIZE, n_blocks=1, tail_bound=_TAIL_BOUND
    )
    student = build_student_flow(
        teacher, 45, 1, 1, hidden_size=_HIDDEN_SIZE, n_blocks=1, tail_bound=_TAIL_BOUND
    )
    assert all(not isinstance(f, Permute) for f in student.flows)
    context = torch.randn(4, 1)
    x, _ = student.sample(4, context)
    assert x.shape == (4, 45)
    assert torch.isfinite(student.log_prob(torch.randn(4, 45), context)).all()


def test_build_student_flow_rejects_block_count_mismatch():
    teacher = _build_teacher(45, 1, 1)  # n_blocks=2 -> 1 perm
    with pytest.raises(ValueError, match="n_blocks"):
        build_student_flow(
            teacher, 45, 1, 1, hidden_size=_HIDDEN_SIZE, n_blocks=4, tail_bound=_TAIL_BOUND
        )


def test_student_permutations_match_teacher():
    teacher = _build_teacher(144, 2, 2)
    student = _build_student(teacher, 144, 2, 2)

    teacher_perms = [p for p in teacher.flows if isinstance(p, Permute)]
    student_perms = [p for p in student.flows if isinstance(p, Permute)]
    assert len(teacher_perms) == len(student_perms) == _N_BLOCKS - 1
    for tp, sp in zip(teacher_perms, student_perms):
        assert torch.equal(tp.perm, sp.perm)
        assert torch.equal(tp.inv_perm, sp.inv_perm)


@pytest.mark.parametrize("n_blocks", [4, 8])
def test_student_permutations_match_teacher_many_blocks(n_blocks):
    torch.manual_seed(0)
    teacher = build_flow(
        144, 2, 2, hidden_size=_HIDDEN_SIZE, n_blocks=n_blocks, tail_bound=_TAIL_BOUND
    )
    student = build_student_flow(
        teacher,
        144,
        2,
        2,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=n_blocks,
        tail_bound=_TAIL_BOUND,
    )

    teacher_perms = [p for p in teacher.flows if isinstance(p, Permute)]
    student_perms = [p for p in student.flows if isinstance(p, Permute)]
    assert len(teacher_perms) == len(student_perms) == n_blocks - 1
    for tp, sp in zip(teacher_perms, student_perms):
        assert torch.equal(tp.perm, sp.perm)
        assert torch.equal(tp.inv_perm, sp.inv_perm)


# ---------------------------------------------------------------------------
# guided_cascade
# ---------------------------------------------------------------------------


def test_guided_cascade_inverse_matches_flow_inverse():
    teacher = _build_teacher(45, 1, 1)
    x = torch.randn(4, 45)
    context = torch.randn(4, 1)
    z = teacher.inverse(x, context)
    out, logdet, steps, p = guided_cascade(
        teacher.flows, x, context, direction="inverse",
        return_steps=True, return_p=True,
    )
    assert torch.allclose(out, z, atol=1e-5)
    assert len(steps) == len(teacher.flows) + 1
    assert len(p) == _N_BLOCKS  # one MADE output per RQS block
    assert torch.isfinite(logdet).all()
    for pi in p:
        assert torch.isfinite(pi).all()


def test_guided_cascade_forward_matches_flow_forward():
    teacher = _build_teacher(45, 1, 1)
    student = _build_student(teacher, 45, 1, 1)
    z = torch.randn(4, 45)
    context = torch.randn(4, 1)
    x = student.forward(z, context)
    out, logdet, steps, p = guided_cascade(
        student.flows, z, context, direction="forward",
        return_steps=True, return_p=True,
    )
    assert torch.allclose(out, x, atol=1e-5)
    assert len(steps) == len(student.flows) + 1
    assert len(p) == _N_BLOCKS
    assert torch.isfinite(logdet).all()


def test_guided_cascade_no_flags_returns_pair():
    teacher = _build_teacher(45, 1, 1)
    x = torch.randn(4, 45)
    context = torch.randn(4, 1)
    out, logdet = guided_cascade(teacher.flows, x, context, direction="inverse")
    assert torch.isfinite(logdet).all()
    assert out.shape == (4, 45)


# ---------------------------------------------------------------------------
# PDD loss helpers
# ---------------------------------------------------------------------------


def test_chi2_loss_matches_hand_computed():
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[1.0, 2.0, 4.0], [4.0, 5.0, 5.0]])
    # Row 0: only index 2 differs (3-4)^2 = 1.  Row 1: only index 2 (6-5)^2 = 1.
    expected = 1.0
    assert torch.allclose(chi2_loss(a, b), torch.tensor(expected))


def test_logabsdet_of_base_matches_closed_form():
    noise = torch.zeros(2, 4)
    out = logabsdet_of_base(noise, width=1.0)
    expected = -0.5 * 4 * np.log(2 * np.pi)
    assert torch.allclose(out, torch.full((2,), expected), atol=1e-5)

    # Nonzero noise exercises the quadratic (noise/width)^2 term at width=1.
    noise = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    out = logabsdet_of_base(noise, width=1.0)
    expected = np.array(
        [-0.5 * (1 + 4) - 0.5 * 2 * np.log(2 * np.pi),
         -0.5 * (0 + 1) - 0.5 * 2 * np.log(2 * np.pi)]
    )
    assert torch.allclose(out, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# Lightning module student mode (PDD distillation)
# ---------------------------------------------------------------------------

_NUM_LAYERS = 45
_CELLS_PER_LAYER = 144
_TOTAL_CELLS = _NUM_LAYERS * _CELLS_PER_LAYER  # 6480


def _make_student_module(which_flow, **kwargs):
    """Small student-mode module for fast CPU tests (dummy data constants)."""
    return ICaloFlowLightningModule(
        which_flow=which_flow,
        student_mode=True,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=kwargs.pop("n_blocks", _N_BLOCKS),
        tail_bound=_TAIL_BOUND,
        normalization_constant=1500.0,
        layer_boundaries=[0, 144, 288, 432],
        **kwargs,
    )


def test_student_mode_builds_teacher_and_student():
    torch.manual_seed(0)
    mod2 = _make_student_module(which_flow=2)
    assert mod2.flow2 is not None
    assert mod2.flow2_student is not None
    mod4 = _make_student_module(which_flow=4)
    assert mod4.flow3 is not None
    assert mod4.flow3_student is not None
    mod7 = _make_student_module(which_flow=7, n_blocks=1)
    assert mod7.flow1 is not None and mod7.flow2 is not None and mod7.flow3 is not None
    assert mod7.flow2_student is not None and mod7.flow3_student is not None
    # Default (teacher-only) mode builds no students.
    teacher_mod = ICaloFlowLightningModule(
        which_flow=2, hidden_size=_HIDDEN_SIZE, n_blocks=_N_BLOCKS, tail_bound=_TAIL_BOUND
    )
    assert teacher_mod.flow2_student is None and teacher_mod.flow3_student is None


def test_configure_optimizers_targets_student():
    torch.manual_seed(0)
    mod = _make_student_module(which_flow=2)
    cfg = mod.configure_optimizers()
    assert set(cfg.keys()) == {"optimizer", "lr_scheduler"}
    optimizer = cfg["optimizer"]
    student_ids = set(map(id, mod.flow2_student.parameters()))
    teacher_ids = set(map(id, mod.flow2.parameters()))
    param_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert param_ids
    assert all(pid in student_ids for pid in param_ids)
    assert all(pid not in teacher_ids for pid in param_ids)
    assert cfg["lr_scheduler"]["interval"] == "step"


def test_freeze_teacher_requires_grad_flags():
    torch.manual_seed(0)
    mod = _make_student_module(which_flow=2)
    mod._freeze_teacher()
    teacher_params = list(mod.flow2.parameters())
    student_params = list(mod.flow2_student.parameters())
    assert teacher_params and student_params
    assert all(not p.requires_grad for p in teacher_params)
    assert all(p.requires_grad for p in student_params)


def test_student_training_step_flow2_finite():
    torch.manual_seed(0)
    mod = _make_student_module(which_flow=2).double()
    mod._freeze_teacher()
    batch = {
        "layer": torch.randn(8, _CELLS_PER_LAYER),
        "energy": torch.randn(8, 1),
        "energy_dep": torch.randn(8, 1),
    }
    out = mod.training_step(batch, 0)
    assert set(out.keys()) == {"loss", "batch_size"}
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["batch_size"] == 8


def test_student_training_step_flow3_finite():
    torch.manual_seed(0)
    mod = _make_student_module(which_flow=4).double()
    mod._freeze_teacher()
    batch = {
        "layer": torch.randn(8, _CELLS_PER_LAYER),
        "layer_p": torch.randn(8, _CELLS_PER_LAYER),
        "energy": torch.randn(8, 1),
        "energy_dep": torch.randn(8, 1),
        "energy_dep_p": torch.randn(8, 1),
        "layer_number": torch.randint(1, _NUM_LAYERS, (8,)),
    }
    out = mod.training_step(batch, 0)
    assert set(out.keys()) == {"loss", "batch_size"}
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])
    assert out["batch_size"] == 8


def test_pdd_training_updates_student_weights():
    # Both teacher and student start identity-initialized, so the PDD loss is
    # ~0 and static; perturb the teacher so the student has real signal to
    # learn.  A few Adam steps must move the student weights and keep the
    # loss finite (OneCycleLR's first steps run at base_lr, so a strict
    # monotone-loss-decrease assertion would be flaky here; real convergence
    # is exercised by the Task-5 smoke test on data).
    torch.manual_seed(0)
    mod = _make_student_module(
        which_flow=2, n_blocks=1, scheduler_params={"lr": 1e-3}
    )
    with torch.no_grad():
        for p in mod.flow2.parameters():
            if p.dtype.is_floating_point:
                p.add_(0.05 * torch.randn_like(p))
    mod._freeze_teacher()
    optimizer = mod.configure_optimizers()["optimizer"]

    student_before = {
        k: v.detach().clone() for k, v in mod.flow2_student.state_dict().items()
    }
    batch = {
        "layer": torch.randn(8, _CELLS_PER_LAYER),
        "energy": torch.randn(8, 1),
        "energy_dep": torch.randn(8, 1),
    }
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        loss = mod.training_step(batch, 0)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[0] > 1e-3, "perturbed teacher should create PDD signal"

    moved = any(
        not torch.equal(mod.flow2_student.state_dict()[k], v)
        for k, v in student_before.items()
    )
    assert moved, "PDD training did not update any student weights"


def test_predict_step_with_students(tmp_path):
    torch.manual_seed(0)
    # Save one ckpt from a full generation module: it carries all five
    # submodule prefixes (flow1./flow2./flow3./flow2_student./flow3_student.).
    src = _make_student_module(which_flow=7, n_blocks=1)
    ckpt_path = str(tmp_path / "gen.ckpt")
    torch.save({"state_dict": src.state_dict()}, ckpt_path)

    gen_mod = _make_student_module(which_flow=7, n_blocks=1)
    gen_mod.load_flows_for_generation(
        flow1_path=ckpt_path,
        flow2_path=ckpt_path,
        flow3_path=ckpt_path,
        flow2_student_path=ckpt_path,
        flow3_student_path=ckpt_path,
    )
    batch = {"energy_inc": torch.tensor([[-1.5], [0.5]])}
    energies, showers, t = gen_mod.predict_step(batch, 0)
    assert energies.shape == (2, 1)
    assert showers.shape == (2, _TOTAL_CELLS)
    assert isinstance(t, float)
    assert np.isfinite(energies).all() and (energies > 0).all()
    assert np.isfinite(showers).all() and (showers >= 0).all()


def test_load_flows_for_generation_loads_student_prefixes(tmp_path):
    torch.manual_seed(0)
    src = _make_student_module(which_flow=7, n_blocks=1)
    ckpt_path = str(tmp_path / "gen.ckpt")
    torch.save({"state_dict": src.state_dict()}, ckpt_path)

    gen_mod = _make_student_module(which_flow=7, n_blocks=1)
    gen_mod.load_flows_for_generation(
        flow1_path=ckpt_path,
        flow2_path=ckpt_path,
        flow3_path=ckpt_path,
        flow2_student_path=ckpt_path,
        flow3_student_path=ckpt_path,
    )
    expected = {
        k[len("flow2_student."):]: v
        for k, v in src.state_dict().items()
        if k.startswith("flow2_student.")
    }
    loaded = gen_mod.flow2_student.state_dict()
    assert set(loaded.keys()) == set(expected.keys())
    for key in loaded:
        assert torch.equal(loaded[key], expected[key])


def test_setup_fit_student_requires_teacher_checkpoint():
    mod = _make_student_module(which_flow=2)
    with pytest.raises(ValueError, match="teacher checkpoint"):
        mod.setup("fit")


def _full_lb(num_layers=45):
    """Full LEMURS layer boundaries (num_layers+1 offsets, 144 cells/layer)."""
    return list(range(0, num_layers * _CELLS_PER_LAYER + 1, _CELLS_PER_LAYER))


def test_encode_joint_use_student():
    # A which_flow=7 student module (random flows) encodes a shower with the
    # teacher by default and with the students when use_student=True. Both
    # paths must produce finite latents of the full joint dimension; the
    # student path exercises the sequential (slow) density direction and is
    # used to quantify the student's own invertibility.  A small layer count
    # keeps the sequential-density encode fast in the unit test.
    torch.manual_seed(0)
    num_layers = 5
    n_cells = num_layers * _CELLS_PER_LAYER
    mod = ICaloFlowLightningModule(
        which_flow=7,
        student_mode=True,
        hidden_size=_HIDDEN_SIZE,
        n_blocks=1,
        tail_bound=_TAIL_BOUND,
        normalization_constant=1500.0,
        num_layers=num_layers,
        layer_boundaries=_full_lb(num_layers),
    )
    mod.flow1.double(); mod.flow2.double(); mod.flow3.double()
    mod.flow2_student.double(); mod.flow3_student.double()
    mod.normalization_constant = 1500.0

    b = 2
    shower_mev = torch.rand(b, n_cells, dtype=torch.float64) * 10.0
    energy_mev = torch.tensor([[1000.0], [100000.0]], dtype=torch.float64)

    z_t = mod.encode_joint(shower_mev, energy_mev)  # teacher
    z_s = mod.encode_joint(shower_mev, energy_mev, use_student=True)  # student

    assert z_t.shape == (b, mod.num_dim)
    assert z_s.shape == (b, mod.num_dim)
    assert torch.isfinite(z_t).all()
    assert torch.isfinite(z_s).all()
    # With identity-initialized flows teacher and student agree; the student
    # path is exercised (not an error) and is what enables quantifying the
    # student's own invertibility after PDD training.
