import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import data_util
from ..sources import HDF5Source, MemmapSource

dtype = torch.get_default_dtype()


class CaloINNDataset(Dataset):
    """Processes raw HDF5 data on-the-fly per batch.

    Mirror legacy _CaloINNDataset 

    Stores ONLY split indices (no data).  Preprocessing happens in
    __getitems__, which is called once per batch by the DataLoader.

    Preprocessing pipeline (matches legacy data_util.preprocess):
      raw showers → _build_data_dict → preprocess → add noise → tensor
    """

    def __init__(
        self,
        source: HDF5Source | MemmapSource,
        indices: np.ndarray,
        layer_boundaries: np.ndarray,
        xml_filename: str,
        particle_type: str,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
    ):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.layer_boundaries = layer_boundaries
        self.xml_filename = xml_filename
        self.particle_type = particle_type
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    # ------------------------------------------------------------------
    #  Per-element (DataLoader fallback — not the fast path)
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        x, c = self._get_numpy_batch(np.array([index]))
        return (
            torch.from_numpy(x[0]).to(dtype),
            torch.from_numpy(c[0]).to(dtype),
        )

    def __getitems__(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        x, c = self._get_numpy_batch(idx)
        return [
            (
                torch.from_numpy(x[i]).to(dtype), 
                torch.from_numpy(c[i]).to(dtype)
            )
            for i in range(len(idx))
        ]

    # ------------------------------------------------------------------
    #  Core preprocessing — the same as legacy data_util pipeline
    # ------------------------------------------------------------------

    def _get_numpy_batch(
        self, local_indices: np.ndarray
    ) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Load raw data, preprocess, add noise — all vectorised."""
        source_indices = local_indices
        if isinstance(self.source, HDF5Source):
            source_indices = self.indices[local_indices]
        showers, energies = self.source.read_rows(source_indices)

        # 3-D LEMURS (N, R, Phi, Z) → flat (N, cells); no-op for flat input.
        # Must precede _build_data_dict, which slices the last axis.
        showers = self.source.truth_format.flatten_showers(showers)

        # Build data dict (matches legacy load_data pattern)
        data = data_util._build_data_dict(showers, energies, self.layer_boundaries)

        # Legacy preprocessing
        x, c = data_util.preprocess(
            data,
            self.layer_boundaries,
            self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
        )

        # Convert to float32 tensors for noise + output
        x = x.astype(np.float32, copy=False)
        c = c.astype(np.float32, copy=False)

        # Add noise (matches legacy MyDataLoader)
        if self.width_noise > 0:
            if self.fixed_noise:
                # Per-sample deterministic noise: each HDF5 index gets its
                # own seeded RNG, making noise independent of batch composition.
                noise = np.empty_like(x, dtype=np.float32)
                for i, h5idx in enumerate(source_indices.astype(np.int64)):
                    rng = np.random.RandomState(int(h5idx))
                    noise[i] = rng.uniform(0, 1, x.shape[1:]).astype(np.float32)
                noise *= self.width_noise
                x += noise
            else:
                x = data_util.add_noise(x, self.width_noise)  # vectorised random noise

        return x, c

