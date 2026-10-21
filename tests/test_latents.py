"""Unit tests for the precomputed latent store and identity-latent NCE.

No GPU or checkpoints required: a tiny synthetic store is written to tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.classifiers import (
    PrecomputedLatentNCE,
    PrecomputedLatentWhitenedNCE,
)
from calogen.core.latents import (
    PrecomputedLatentDataModule,
    PrecomputedLatentDataset,
    manifest_path,
    read_manifest,
    save_latent_q0,
    write_manifest,
)


def _write_store(latents_dir, n_train=10, n_val=4, n_test=None, z_dim=5, c_dim=1, seed=0):
    rng = np.random.RandomState(seed)
    splits = {}
    sizes = [("train", n_train), ("val", n_val)]
    if n_test is not None:
        sizes.append(("test", n_test))
    for prefix, n in sizes:
        z = rng.randn(n, z_dim).astype("float32")
        c = rng.randn(n, c_dim).astype("float32")
        np.memmap(latents_dir / f"{prefix}_z.dat", mode="w+", dtype="float32",
                  shape=(n, z_dim))[:] = z
        np.memmap(latents_dir / f"{prefix}_c.dat", mode="w+", dtype="float32",
                  shape=(n, c_dim))[:] = c
        splits[prefix] = {
            "z": {"file": f"{prefix}_z.dat", "shape": [n, z_dim], "dtype": "float32"},
            "c": {"file": f"{prefix}_c.dat", "shape": [n, c_dim], "dtype": "float32"},
        }
    manifest = {"key": "test", "cond_dim": c_dim, "splits": splits}
    write_manifest(str(latents_dir), manifest)
    return manifest


def test_dataset_reads_shapes_dtypes(tmp_path):
    _write_store(tmp_path, n_train=10, n_val=4, z_dim=5)
    ds = PrecomputedLatentDataset(str(tmp_path), "train")
    assert len(ds) == 10
    item = ds[3]
    assert set(item) == {"z", "c"}
    assert item["z"].shape == (5,) and item["c"].shape == (1,)
    assert item["z"].dtype == torch.float32 and item["c"].dtype == torch.float32
    pairs = ds.__getitems__([1, 2, 3])
    assert len(pairs) == 3
    assert pairs[0]["z"].shape == (5,)
    assert torch.equal(pairs[1]["z"], ds[2]["z"])


def test_manifest_written_and_read(tmp_path):
    _write_store(tmp_path)
    path = manifest_path(str(tmp_path))
    manifest = read_manifest(str(tmp_path))
    assert manifest["splits"]["train"]["z"]["shape"] == [10, 5]
    with open(path) as f:
        assert json.load(f)["key"] == "test"


def test_missing_split_raises(tmp_path):
    _write_store(tmp_path)
    with pytest.raises(KeyError):
        PrecomputedLatentDataset(str(tmp_path), "nope")


def test_datamodule_batches(tmp_path):
    _write_store(tmp_path, n_train=10, n_val=4, z_dim=5)
    dm = PrecomputedLatentDataModule(str(tmp_path), batch_size=4, num_workers=0, shuffle=False)
    dm.setup("fit")
    assert dm.num_train_samples == 10
    batches = list(dm.train_dataloader())
    assert len(batches) == 3  # 4 + 4 + 2
    first = batches[0]
    assert set(first) == {"z", "c"}
    assert first["z"].shape == (4, 5) and first["c"].shape == (4, 1)
    assert len(list(dm.val_dataloader())) == 1


def test_datamodule_uses_test_split(tmp_path):
    _write_store(tmp_path, n_train=10, n_val=4, n_test=3, z_dim=5)
    dm = PrecomputedLatentDataModule(str(tmp_path), batch_size=4, num_workers=0)
    dm.setup("test")
    test_ds = dm._test_dataset
    assert test_ds.split == "test"
    assert len(test_ds) == 3
    assert list(dm.predict_dataloader())


def test_datamodule_test_falls_back_to_val(tmp_path):
    _write_store(tmp_path, n_train=10, n_val=4, z_dim=5)
    dm = PrecomputedLatentDataModule(str(tmp_path), batch_size=4, num_workers=0)
    with pytest.warns(UserWarning, match="using 'val' for test/predict"):
        dm.setup("test")
    assert dm._test_dataset.split == "val"
    assert len(dm._test_dataset) == 4


def test_precomputed_latent_nce_identity_and_batch(tmp_path):
    _write_store(tmp_path, n_train=6, n_val=2, z_dim=5)
    ds = PrecomputedLatentDataset(str(tmp_path), "train")
    mod = PrecomputedLatentNCE(hidden_dim=8, num_layers=1, dropout=0.0, noise_amplifier=2)

    zs = torch.stack([ds[i]["z"] for i in range(len(ds))])
    cs = torch.stack([ds[i]["c"] for i in range(len(ds))])
    batch = {"z": zs, "c": cs}

    assert mod._get_latent(zs, cs) is zs

    X_input, y, z_truth = mod.get_input_from_batch(batch)
    assert torch.equal(z_truth, zs)
    assert X_input.shape == (len(ds) * 3, 5 + 1)
    assert y.shape == (len(ds) * 3, 1)
    assert torch.equal(y[: len(ds)], torch.ones(len(ds), 1))
    assert torch.equal(y[len(ds):], torch.zeros(len(ds) * 2, 1))

    out = mod.training_step(batch, 0)
    assert torch.isfinite(out["loss"])


def _write_q0(tmp_path, d=5, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randn(d, d)
    cov = a @ a.T + d * np.eye(d)
    mean = rng.randn(d)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    out = tmp_path / "q0"
    save_latent_q0(
        str(out),
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(vecs, dtype=torch.float32),
        torch.tensor(vals, dtype=torch.float32),
        {"D": d},
    )
    return torch.tensor(mean, dtype=torch.float32), cov


def test_whiten_unwhiten_roundtrip(tmp_path):
    _write_q0(tmp_path, d=5)
    mod = PrecomputedLatentWhitenedNCE(q0_dir=str(tmp_path / "q0"), hidden_dim=8, num_layers=1)
    z = torch.randn(7, 5)
    assert torch.allclose(mod.unwhiten(mod.whiten(z)), z, atol=1e-4)


def test_whitening_standardizes_base_samples(tmp_path):
    mean, cov = _write_q0(tmp_path, d=5)
    mod = PrecomputedLatentWhitenedNCE(q0_dir=str(tmp_path / "q0"), hidden_dim=8, num_layers=1)
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=torch.tensor(cov, dtype=torch.float32))
    w = mod.whiten(dist.sample((20000,)))
    assert torch.allclose(w.mean(dim=0), torch.zeros(5), atol=0.05)
    assert torch.allclose(w.std(dim=0), torch.ones(5), atol=0.05)
    offdiag = (w.T @ w / w.shape[0]) - torch.eye(5)
    assert offdiag.abs().max() < 0.05


def test_whitened_nce_forward_and_log_density(tmp_path):
    _write_q0(tmp_path, d=5)
    mod = PrecomputedLatentWhitenedNCE(
        q0_dir=str(tmp_path / "q0"), hidden_dim=8, num_layers=1, dropout=0.0, noise_amplifier=2
    )
    z = torch.randn(6, 5)
    c = torch.randn(6, 1)
    batch = {"z": z, "c": c}
    X_input, y, w = mod.get_input_from_batch(batch)
    assert X_input.shape == (6 * 3, 5 + 1)
    assert torch.allclose(w, mod.whiten(z), atol=1e-5)
    out = mod.training_step(batch, 0)
    assert torch.isfinite(out["loss"])
    logp = mod.estimate_log_density(z, c)
    assert logp.shape == (6,)
    assert torch.isfinite(logp).all()
