import json
import os
import threading
from typing import Tuple

import h5py
import numpy as np

from .data_util import TruthFormat
from .io import _read_rows

# HDF5 read-cache settings — small enough for per-worker handles
_H5_RDCC = {"rdcc_nbytes": 64 * 1024 * 1024, "rdcc_nslots": 4093}


class HDF5Source:
    """Lazy HDF5 reader — one per worker, pickle-safe.

    Opens the file on first access in each worker.  __getstate__ drops
    the file handle so DataLoader multiprocessing works.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._lock = threading.Lock()
        with h5py.File(self.file_path, "r") as handle:
            self._length = int(handle["showers"].shape[0])
        self.truth_format = TruthFormat.detect_from_file(file_path)

    def __len__(self) -> int:
        return self._length

    def _get_file(self):
        if self._file is None:
            with self._lock:
                if self._file is None:  # double-check under lock
                    self._file = h5py.File(self.file_path, "r", **_H5_RDCC)
        return self._file

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def read_rows(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        handle = self._get_file()
        showers = _read_rows(handle["showers"], indices)
        energies = _read_rows(handle[self.truth_format.energy_key], indices)
        return showers, energies

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # don't pickle the file handle
        state["_lock"] = None  # locks aren't picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

class MemmapSource:
    def __init__(self, meta_path: str, truth_format: TruthFormat) -> None:
        self.truth_format = truth_format
        self.meta_path = meta_path
        self._lock = threading.Lock()
        self._arrays = None
        self._length = None

    def __len__(self) -> int:
        if self._arrays is None:
            self._get_arrays()

        return int(self._arrays["x"].shape[0])

    def _get_arrays(self):
        if self._arrays is None:
            with self._lock:
                if self._arrays is None:          # double-check under lock
                    with open(self.meta_path, "r") as f:
                        meta = json.load(f)
                    arrays = {}
                    for name, info in meta["fields"].items():
                        fp = os.path.join(os.path.dirname(self.meta_path), info["file"])
                        arrays[name] = np.memmap(
                            fp, mode="r",
                            dtype=info["dtype"], shape=tuple(info["shape"]),
                        )
                    self._arrays = arrays
                    self._length = int(arrays["x"].shape[0])
        return self._arrays


    def read_rows(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arrays = self._get_arrays()
        return arrays["x"][indices], arrays["c"][indices]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None      # memmaps aren't picklable
        state["_lock"] = None        # locks aren't picklable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def close(self):
        # Memmap files don't need to be closed, but we can delete the arrays to free memory
        pass
    