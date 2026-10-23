"""LEMURS dataset adapter for CaloINN.

Provides an HDF5 source wrapper that adapts LEMURS-format files
(3D showers: (S, R, Phi, Z) = (S, 9, 16, 45), scalar incident_energy)
to the CaloINN flat format, then reuses the existing sharded data pipeline.

Usage (LightningCLI):
    data:
      class_path: calogen.core.lemurs.LEMURSDataModule
      init_args:
        data_path: /path/to/LEMURS_Par04SiW_...h5
        val_data_path: /path/to/LEMURS_Par04SiW_...h5
        batch_size: 2048
        xml_path: binning_lemurs_par04siw.xml
        xml_ptype: photon
        ...

Or via CaloINNDataModule (auto-detected):
    data:
      class_path: calogen.models.caloinn.datamodule.CaloINNDataModule
      init_args:
        data_path: /path/to/LEMURS_...h5
        use_lemurs: true
        ...
"""

from __future__ import annotations

import threading
from typing import Tuple

import h5py
import numpy as np
import tqdm
from lightning.pytorch import LightningDataModule

from calogen.core import data_util
from calogen.core.data_util import TruthFormat
from calogen.core.sharded import (
    ShardedCaloINNDataModule,
    _CaloINNDataset,
    _MemmapDataset,
    _build_data_dict,
    _H5_RDCC,
)

# LEMURS grid: 9 radial × 16 phi bins = 144 cells per layer
_LEMURS_R_BINS = 9
_LEMURS_PHI_BINS = 16
_LEMURS_N_LAYERS = 45
_LEMURS_CELLS_PER_LAYER = _LEMURS_R_BINS * _LEMURS_PHI_BINS  # 144
_LEMURS_TOTAL_CELLS = _LEMURS_N_LAYERS * _LEMURS_CELLS_PER_LAYER  # 6480


class LEMURSHDF5Source:
    """Lazy HDF5 reader for LEMURS-format files.

    Adapts the LEMURS 3D shower representation (S, 9, 16, 45) to the
    CaloINN flat format (S, 6480).  Equivalent to _RawHDF5Source but
    reads different keys and transposes the shower grid.

    Opens the file on first access in each worker.  __getstate__ drops
    the file handle so DataLoader multiprocessing works.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._lock = threading.Lock()
        with h5py.File(self.file_path, "r") as handle:
            self._length = int(handle["showers"].shape[0])
            showers_ndim = handle["showers"].ndim
            grid_shape = (
                tuple(int(s) for s in handle["showers"].shape[1:])
                if showers_ndim >= 3 else None
            )
        self.truth_format = TruthFormat(
            energy_key="incident_energy",
            energy_is_1d=True,
            showers_grid_shape=grid_shape,
        )

    def __len__(self) -> int:
        return self._length

    def _get_file(self):
        if self._file is None:
            with self._lock:
                if self._file is None:
                    self._file = h5py.File(self.file_path, "r", **_H5_RDCC)
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def read_rows(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Read and adapt LEMURS rows to CaloINN format.

        Returns:
            showers:  (len(indices), 6480) float32 in MeV
            energies: (len(indices), 1)     float32 in MeV
        """
        handle = self._get_file()

        showers_3d = _read_rows(handle["showers"], indices)
        energies_1d = _read_rows(handle["incident_energy"], indices)

        showers_flat = _transpose_and_flatten(showers_3d)
        energies_2d = np.asarray(energies_1d, dtype=np.float32).reshape(-1, 1)

        return showers_flat, energies_2d

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


def _transpose_and_flatten(showers: np.ndarray) -> np.ndarray:
    """Transpose LEMURS 3D showers (..., 9, 16, 45) to flat (..., 6480).

    The LEMURS grid is stored as (..., R, Phi, Z).  We transpose to
    (..., Z, R, Phi) then flatten so each layer's 144 cells are
    contiguous in the output.  This matches the layer-boundary slicing
    expected by data_util.preprocess.
    """
    return np.transpose(showers, (0, 3, 1, 2)).reshape(showers.shape[0], -1)


def _read_rows(dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows with internal sort/unsort for I/O speed."""
    if len(indices) == 0:
        return dataset[:0]

    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.size == 1:
        return np.asarray(dataset[ordered], dtype=np.float32)

    if np.all(ordered[1:] >= ordered[:-1]):
        return np.asarray(dataset[ordered], dtype=np.float32)

    sort_order = np.argsort(ordered)
    sorted_indices = ordered[sort_order]
    data = np.asarray(dataset[sorted_indices], dtype=np.float32)
    inverse = np.argsort(sort_order)
    return data[inverse]


class LEMURSDataModule(ShardedCaloINNDataModule):
    """Lightning DataModule for LEMURS-format HDF5 files.

    Subclasses ShardedCaloINNDataModule, overriding only the HDF5
    source creation to use LEMURSHDF5Source.  All preprocessing,
    filtering, caching, and DataLoader logic is inherited unchanged.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        batch_size: int,
        xml_path: str,
        xml_ptype: str = "photon",
        val_frac: float = 0.01,
        eps: float = 1e-10,
        u0up_cut: float = 7.0,
        u0low_cut: float = 0.0,
        rew: float = 1.0,
        dep_cut: float = 1e10,
        width_noise: float = 0.0,
        fixed_noise: bool = False,
        shuffle: bool = True,
        num_workers: int = 0,
        predict_batch_size: int = 1000,
        eval_dataset: str = "1-photons",
        cache_mode: str = "none",
        cache_dir: str = "",
        cache_chunk: int = 1024,
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            val_data_path=val_data_path,
            batch_size=batch_size,
            xml_path=xml_path,
            xml_ptype=xml_ptype,
            val_frac=val_frac,
            eps=eps,
            u0up_cut=u0up_cut,
            u0low_cut=u0low_cut,
            rew=rew,
            dep_cut=dep_cut,
            width_noise=width_noise,
            fixed_noise=fixed_noise,
            shuffle=shuffle,
            num_workers=num_workers,
            predict_batch_size=predict_batch_size,
            eval_dataset=eval_dataset,
            cache_mode=cache_mode,
            cache_dir=cache_dir,
            cache_chunk=cache_chunk,
            **kwargs,
        )

    def _compute_filter_indices(self, data_path: str) -> np.ndarray:
        """Compute valid indices using LEMURSHDF5Source."""
        import hashlib
        import os

        param_str = (
            f"{self.u0up_cut}_{self.u0low_cut}_{self.dep_cut}_{self.eps}_"
            f"{self.xml_ptype}_{self.xml_path}"
        )
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        cache_path = f"{data_path}.filter_{param_hash}.npy"

        if os.path.exists(cache_path):
            print(f"📦 Loading cached filter indices from {cache_path}...")
            return np.load(cache_path)

        source = LEMURSHDF5Source(data_path)
        try:
            n_samples = len(source)
            chunk_size = 100000
            all_valid = []

            pbar = tqdm.tqdm(
                total=n_samples, unit="samples",
                desc="Computing filter indices (will cache)",
            )

            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                idx = np.arange(start, end)
                showers, energies = source.read_rows(idx)

                data = _build_data_dict(showers, energies, self.layer_boundaries)

                energy, layers = data_util.get_energy_and_sorted_layers(data)
                x_raw = np.concatenate(layers, axis=1)
                c_raw = energy
                c_raw, extra_dims = data_util.get_energy_dims(
                    x_raw, c_raw, self.layer_boundaries, self.eps
                )

                mask = np.sum(x_raw, axis=1) >= 0
                mask &= extra_dims[:, 0] < self.u0up_cut
                mask &= extra_dims[:, 0] >= self.u0low_cut
                mask &= ((x_raw < self.dep_cut).prod(-1) != 0)

                all_valid.append(idx[mask])
                pbar.update(len(idx))

            pbar.close()
            valid = np.concatenate(all_valid)
            print(f" Number of valid samples: {len(valid)} / {n_samples}")
            np.save(cache_path, valid)
            return valid
        finally:
            source.close()

    def setup(self, stage=None):

        # Detect truth format once from the primary data file
        if self.truth_format is None:
            self.truth_format = TruthFormat.detect_from_file(self.data_path)

        if stage in (None, "fit"):
            train_valid = self._compute_filter_indices(self.data_path)

            if self.shuffle:
                rng = np.random.RandomState(42)
                train_valid = train_valid[rng.permutation(len(train_valid))]

            n_total = len(train_valid)
            n_val = int(n_total * self.val_frac)
            n_train = n_total - n_val

            self.num_train_samples = n_train
            if self.cache_mode == "memmap" and self._cache_exists("train"):
                self._train_dataset = _MemmapDataset(
                    self._cache_path("train"),
                    width_noise=self.width_noise, fixed_noise=self.fixed_noise,
                )
                self._val_dataset = _MemmapDataset(
                    self._cache_path("val"),
                    width_noise=self.width_noise, fixed_noise=False,
                )
            else:
                source = LEMURSHDF5Source(self.data_path)
                ds_kwargs = dict(
                    source=source,
                    layer_boundaries=self.layer_boundaries,
                    xml_filename=self.xml_path,
                    particle_type=self.xml_ptype,
                    eps=self.eps, u0up_cut=self.u0up_cut,
                    u0low_cut=self.u0low_cut, rew=self.rew, dep_cut=self.dep_cut,
                    width_noise=self.width_noise, fixed_noise=self.fixed_noise,
                )
                self._train_dataset = _CaloINNDataset(
                    indices=train_valid[:n_train], **ds_kwargs,
                )
                self._val_dataset = _CaloINNDataset(
                    indices=train_valid[n_train:],
                    width_noise=self.width_noise, fixed_noise=False,
                    **{k: v for k, v in ds_kwargs.items()
                       if k not in ("width_noise", "fixed_noise")},
                )
                if self.cache_mode == "memmap":
                    self._write_memmap_cache("train", self._train_dataset)
                    self._write_memmap_cache("val", self._val_dataset)
                    source.close()

        if stage in (None, "test", "predict"):
        
            print(f"📦 Building test/predict dataset from {self.val_data_path}...")
            val_valid = self._compute_filter_indices(self.val_data_path)
            val_source = LEMURSHDF5Source(self.val_data_path)
            self._test_dataset = _CaloINNDataset(
                source=val_source,
                indices=val_valid,
                layer_boundaries=self.layer_boundaries,
                xml_filename=self.xml_path,
                particle_type=self.xml_ptype,
                eps=self.eps,
                u0up_cut=self.u0up_cut,
                u0low_cut=self.u0low_cut,
                rew=self.rew,
                dep_cut=self.dep_cut,
                width_noise=0.0,
                fixed_noise=False,
            )
