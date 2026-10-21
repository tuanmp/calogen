"""Precomputed latent-space store for NCE classifiers.

An offline extraction pass runs a frozen generator once over the full dataset
and writes the resulting latents ``z`` together with their condition ``c`` to
memmap files.  :class:`PrecomputedLatentDataset` / :class:`PrecomputedLatentDataModule`
memory-map that store and yield ``{"z": ..., "c": ...}`` batches, so the
generator forward (which dominates the on-the-fly NCE cost) runs once instead of
every epoch.

The train/val split is baked into the store at extraction time; this module
only reads it.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

MANIFEST_NAME = "manifest.json"


def manifest_path(latents_dir: str) -> str:
    """Path of the manifest for a latent store directory."""
    return os.path.join(latents_dir, MANIFEST_NAME)


def write_manifest(latents_dir: str, manifest: dict) -> str:
    """Write ``manifest.json`` (creating ``latents_dir`` if needed)."""
    os.makedirs(latents_dir, exist_ok=True)
    path = manifest_path(latents_dir)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(latents_dir: str) -> dict:
    """Read ``manifest.json`` for a latent store directory."""
    with open(manifest_path(latents_dir), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data-derived base (q0) statistics: eigen-whitening N(mu, Sigma) -> N(0, I)
# ---------------------------------------------------------------------------

Q0_MEAN = "mean.pt"
Q0_EIGVECS = "eigvecs.pt"
Q0_EIGVALS = "eigvals.pt"
Q0_META = "meta.json"


def save_latent_q0(out_dir: str, mean, eigvecs, eigvals, meta: dict) -> str:
    """Save mean + eigendecomposition of the regularized covariance."""
    os.makedirs(out_dir, exist_ok=True)
    import torch

    torch.save(mean, os.path.join(out_dir, Q0_MEAN))
    torch.save(eigvecs, os.path.join(out_dir, Q0_EIGVECS))
    torch.save(eigvals, os.path.join(out_dir, Q0_EIGVALS))
    path = os.path.join(out_dir, Q0_META)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def load_latent_q0(out_dir: str):
    """Load ``(mean, eigvecs, eigvals, meta)`` written by :func:`save_latent_q0`."""
    import torch

    mean = torch.load(os.path.join(out_dir, Q0_MEAN), map_location="cpu")
    eigvecs = torch.load(os.path.join(out_dir, Q0_EIGVECS), map_location="cpu")
    eigvals = torch.load(os.path.join(out_dir, Q0_EIGVALS), map_location="cpu")
    meta = {}
    meta_path = os.path.join(out_dir, Q0_META)
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return mean, eigvecs, eigvals, meta


def _mmap_field(latents_dir: str, field: dict) -> np.memmap:
    return np.memmap(
        os.path.join(latents_dir, field["file"]),
        mode="r",
        dtype=field["dtype"],
        shape=tuple(field["shape"]),
    )


class PrecomputedLatentDataset(Dataset):
    """Map-style dataset over one latent-store split.

    Each item is a dict ``{"z": latent, "c": condition}``; the default
    collate then yields ``{"z": (B, d), "c": (B, c_dim)}`` batches.
    """

    def __init__(self, latents_dir: str, split: str):
        self.latents_dir = latents_dir
        self.split = split
        manifest = read_manifest(latents_dir)
        splits = manifest.get("splits", {})
        if split not in splits:
            raise KeyError(
                f"split '{split}' not in {manifest_path(latents_dir)}; "
                f"available: {sorted(splits)}"
            )
        fields = splits[split]
        self.z = _mmap_field(latents_dir, fields["z"])
        self.c = _mmap_field(latents_dir, fields["c"])
        if self.z.shape[0] != self.c.shape[0]:
            raise ValueError(
                f"z/c length mismatch for split '{split}': "
                f"{self.z.shape[0]} vs {self.c.shape[0]}"
            )
        self._length = int(self.z.shape[0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return {
            "z": torch.from_numpy(np.asarray(self.z[index])),
            "c": torch.from_numpy(np.asarray(self.c[index])),
        }

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        z = torch.from_numpy(np.asarray(self.z[idx]))
        c = torch.from_numpy(np.asarray(self.c[idx]))
        return [{"z": z[i], "c": c[i]} for i in range(len(idx))]


class PrecomputedLatentDataModule(LightningDataModule):
    """Lightning data module yielding precomputed ``(z, c)`` latent batches.

    ``latents_dir`` must contain a ``manifest.json`` written by the extraction
    pass.  Lightning roles are mapped to manifest split keys via
    ``train_split`` / ``val_split`` / ``test_split``.  Test/predict use the
    ``test`` split (the independent data file) when present, falling back to
    ``val`` for stores that only carry train/val.
    """

    def __init__(
        self,
        latents_dir: str,
        batch_size: int = 1024,
        num_workers: int = 8,
        shuffle: bool = True,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        **kwargs,
    ):
        super().__init__()
        self.latents_dir = latents_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self.num_train_samples = 0

    def _resolve_test_split(self) -> str:
        """Return the manifest split to use for test/predict.

        Prefers ``test_split`` (default ``"test"``), then ``"test"``, then
        ``"val"`` (with a warning) so train/val-only stores still work.
        """
        splits = read_manifest(self.latents_dir).get("splits", {})
        for name in (self.test_split, "test", "val"):
            if name and name in splits:
                if name != self.test_split:
                    warnings.warn(
                        f"split '{self.test_split}' not in "
                        f"{manifest_path(self.latents_dir)}; using '{name}' for test/predict.",
                        stacklevel=2,
                    )
                return name
        raise KeyError(
            f"no usable test/val split in {manifest_path(self.latents_dir)}; "
            f"have {sorted(splits)}"
        )

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self._train_dataset = PrecomputedLatentDataset(
                self.latents_dir, self.train_split
            )
            self._val_dataset = PrecomputedLatentDataset(
                self.latents_dir, self.val_split
            )
            self.num_train_samples = len(self._train_dataset)
        if stage in (None, "test", "predict"):
            self._test_dataset = PrecomputedLatentDataset(
                self.latents_dir, self._resolve_test_split()
            )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        assert self._train_dataset is not None, "call setup('fit') before train_dataloader"
        return self._loader(self._train_dataset, self.shuffle)

    def val_dataloader(self):
        assert self._val_dataset is not None, "call setup('fit') before val_dataloader"
        return self._loader(self._val_dataset, False)

    def test_dataloader(self):
        assert self._test_dataset is not None, "call setup('test'|'predict') before test_dataloader"
        return self._loader(self._test_dataset, False)

    def predict_dataloader(self):
        assert self._test_dataset is not None, "call setup('test'|'predict') before predict_dataloader"
        return self._loader(self._test_dataset, False)
