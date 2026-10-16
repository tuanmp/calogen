
import os

import numpy as np
from calogen.core.xml_handler import XMLHandler
from calogen.data.data_util import TruthFormat
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.caloinn import CaloINNDataset
from ..sources import HDF5Source, MemmapSource


class CaloINNDataModule(LightningDataModule):
    """Lightning DataModule backed by map-style HDF5 datasets.

    Replaces IterableDataset-based LegacyStreamingDataModule with
    proper multi-worker DataLoader support.  Preprocessing and noise
    happen on-the-fly in workers.

    Can be used directly with LightningCLI or via CaloINNDataModule
    with use_sharded=True.
    """

    def __init__(
        self,
        data_path: str,
        val_data_path: str,
        batch_size: int,
        xml_path: str,
        xml_ptype: str = "pion",
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
        eval_dataset: str = "1-pions",
        cache_mode: str = "none",
        cache_dir: str = "",
        cache_chunk: int = 1024,
        predict_multiplier: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.xml_path = xml_path
        self.xml_ptype = xml_ptype
        self.val_frac = val_frac
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.fixed_noise = fixed_noise
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.predict_batch_size = predict_batch_size
        self.eval_dataset = eval_dataset
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir or "/tmp/caloinn_cache"
        self.cache_chunk = cache_chunk


        # Load layer boundaries once (shared across splits)
        xml_handler = XMLHandler(particle_name=xml_ptype, filename=xml_path)
        self.layer_boundaries = np.unique(xml_handler.GetBinEdges())

        self.num_train_samples = 0
        self.truth_format = TruthFormat.detect_from_file(self.data_path)  # set during setup() from source
        self.predict_multiplier = predict_multiplier

    # ------------------------------------------------------------------
    #  Memmap cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Content-based key for the raw memmap cache.

        The cache stores *raw* rows, so it depends only on the data file's
        identity (realpath + size + mtime) and the train/val split
        (``val_frac`` + ``shuffle``); it is independent of preprocessing
        parameters (cuts, log_transform, noise), which are applied on the fly.
        """
        import hashlib

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
        """Write preprocessed split data to memmap files once.

        Iterates the full dataset, extracts (x, c) pairs, and writes
        them to .dat files with a companion .json metadata file.
        Subsequent runs load via _MemmapDataset — 20-40x faster.
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

    # ------------------------------------------------------------------
    #  Lightning DataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage=None):

        # no compute filter indices. usee all
        source = HDF5Source(self.data_path)

        train_valid = np.arange(len(source), dtype=np.int64)

        # One-time shuffle before split (matching legacy)
        if self.shuffle:
            rng = np.random.RandomState(42)
            train_valid = train_valid[rng.permutation(len(train_valid))]

        n_total = len(train_valid)
        n_val = int(n_total * self.val_frac)
        n_train = n_total - n_val

        self.num_train_samples = n_train

        train_indices = np.sort(train_valid[:n_train])
        val_indices = np.sort(train_valid[n_train:])

        # build train val sets
        if self.cache_mode == "memmap":
            if not self._cache_exists("train"):
                self._write_memmap_cache("train", source, train_indices)
            if not self._cache_exists("val"):
                self._write_memmap_cache("val", source, val_indices)
            train_source = MemmapSource(self._cache_path("train"), source.truth_format)
            val_source = MemmapSource(self._cache_path("val"), source.truth_format)
        else:
            train_source = source
            val_source = source

        
        ds_kwargs = dict(
            layer_boundaries=self.layer_boundaries,
            xml_filename=self.xml_path,
            particle_type=self.xml_ptype,
            eps=self.eps, 
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut, 
            rew=self.rew, 
            dep_cut=self.dep_cut,
            width_noise=self.width_noise, 
            fixed_noise=self.fixed_noise,
        )
        self._train_dataset = CaloINNDataset(
            source=train_source,
            indices=train_indices, 
            **ds_kwargs,
        )
        self._val_dataset = CaloINNDataset(
            source=val_source,
            indices=val_indices,
            **ds_kwargs,
        )

        # build test set
        print(f"📦 Building test/predict dataset from {self.val_data_path}...")
        # val_valid = self._compute_filter_indices(self.val_data_path)
        val_source = HDF5Source(self.val_data_path)
        val_valid = np.arange(len(val_source), dtype=np.int64)
        self._test_dataset = CaloINNDataset(
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

    def _dl_kwargs(self, shuffle: bool = False, batch_size: int | None = None) -> dict:
        batch_size = batch_size or self.batch_size
        kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self):
        return DataLoader(self._train_dataset, **self._dl_kwargs(shuffle=self.shuffle))

    def val_dataloader(self):
        return DataLoader(self._val_dataset, **self._dl_kwargs(shuffle=False))

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            **self._dl_kwargs(shuffle=False, batch_size=self.predict_batch_size)
        )

    def predict_dataloader(self):

        assert self._test_dataset is not None, (
            "Call setup('predict') before requesting predict_dataloader"
        )
        return [DataLoader(
            self._test_dataset,
            **self._dl_kwargs(shuffle=False, batch_size=self.predict_batch_size)
        ) for _ in range(self.predict_multiplier)]

    def teardown(self, stage=None):
        """Clean up HDF5 handles opened by this DataModule."""
        for attr in ("_train_dataset", "_val_dataset", "_test_dataset"):
            ds = getattr(self, attr, None)
            if ds is not None and hasattr(ds, 'source') and ds.source is not None:
                ds.source.close()