

import hashlib
import os

import numpy as np
from calogen.data.sources import HDF5Source
from tqdm import tqdm


class MemmapMixin:

    def _cache_key(self) -> str:
        """Content-based key for the raw memmap cache.
    
        The cache stores *raw* rows, so it depends only on the data file's
        identity (realpath + size + mtime) and the train/val split
        (``val_frac`` + ``shuffle``); it is independent of preprocessing
        parameters (cuts, log_transform, noise), which are applied on the fly.
        """
    
        try:
            st = os.stat(self.data_path)
            file_id = f"{os.path.realpath(self.data_path)}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            file_id = os.path.realpath(self.data_path)
        material = f"{file_id}|val_frac={self.val_frac}|shuffle={self.shuffle}"
        return hashlib.md5(material.encode()).hexdigest()[:16]

    def _cache_path(self, split: str) -> str:
        """Return the memmap metadata path for a split."""
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{split}_{self._cache_key()}.json")
    
    def _cache_exists(self, split: str) -> bool:
        return os.path.exists(self._cache_path(split))
    
    def _write_memmap_cache(self, split: str, source: HDF5Source, indices: np.ndarray):
        """Write *raw* split rows to memmap files once.

        Reads raw ``(showers, energies)`` rows from ``source`` and writes them
        to ``.dat`` files with a companion ``.json`` manifest.  Later runs
        memory-map them; preprocessing happens on the fly downstream, so the
        cache is independent of preprocessing parameters.
        """
        meta_path = self._cache_path(split)
        if os.path.exists(meta_path):
            return
    
        n = len(indices)
        if n == 0:
            return
    
        # Probe first batch for feature dimensions
        x0, c0 = source.read_rows(np.arange(min(256, n)))
    
        dat_dir = os.path.dirname(meta_path)
        meta = {
            "fields": {
                "x": {"file": f"{split}_x.dat", "shape": [n, *x0.shape[1:]], "dtype": "float32"},
                "c": {"file": f"{split}_c.dat", "shape": [n, *c0.shape[1:]], "dtype": "float32"},
            },
    
        }
        # y is unused dummy — kept for _MemmapDataset compatibility
        x_arr = np.memmap(os.path.join(dat_dir, f"{split}_x.dat"),
                          mode="w+", dtype="float32", shape=(n, *x0.shape[1:]))
        c_arr = np.memmap(os.path.join(dat_dir, f"{split}_c.dat"),
                          mode="w+", dtype="float32", shape=(n, *c0.shape[1:]))
    
        pbar = tqdm(total=n, desc=f"memmap cache '{split}'", unit="samples")
        offset = 0
        indices = np.sort(indices)  # ensure sequential access for speed
        for start in range(0, n, self.cache_chunk):
            end = min(start + self.cache_chunk, n)
            idx = indices[start:end]
            xb, cb = source.read_rows(idx)
            m = len(xb)
            x_arr[offset:offset+m] = np.array(xb)
            c_arr[offset:offset+m] = np.array(cb)
            offset += m
            pbar.update(m)
        pbar.close()
    
        x_arr.flush(); c_arr.flush()
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
