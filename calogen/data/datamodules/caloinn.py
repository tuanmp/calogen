import numpy as np
from calogen.core.xml_handler import XMLHandler
from calogen.data.data_util import TruthFormat

from ..datasets.caloinn import CaloINNDataset
from ..sources import HDF5Source, MemmapSource
from .base_datamodule import BaseDataModule
from .memmap_mixin import MemmapMixin


class CaloINNDataModule(BaseDataModule, MemmapMixin):
    """Lightning DataModule backed by map-style HDF5 datasets.

    Multi-worker DataLoader over raw HDF5 rows; preprocessing and noise
    happen on the fly in workers.  With ``cache_mode="memmap"`` the raw
    rows are written once to disk and memory-mapped afterwards.
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
        shuffle: bool = True,
        num_workers: int = 0,
        predict_batch_size: int = 1000,
        eval_dataset: str = "1-pions",
        cache_mode: str = "none",
        cache_dir: str = "",
        cache_chunk: int = 1024,
        predict_multiplier: int = 1,
        num_noise_doses: int = 1,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            predict_batch_size=predict_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            predict_multiplier=predict_multiplier,
        )
        self.save_hyperparameters(ignore=["kwargs"])
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.xml_path = xml_path
        self.xml_ptype = xml_ptype
        self.val_frac = val_frac
        self.eps = eps
        self.u0up_cut = u0up_cut
        self.u0low_cut = u0low_cut
        self.rew = rew
        self.dep_cut = dep_cut
        self.width_noise = width_noise
        self.eval_dataset = eval_dataset
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir or "/tmp/caloinn_cache"
        self.cache_chunk = cache_chunk
        self.num_noise_doses = num_noise_doses

        # Load layer boundaries once (shared across splits)
        xml_handler = XMLHandler(particle_name=xml_ptype, filename=xml_path)
        self.layer_boundaries = np.unique(xml_handler.GetBinEdges())

        self.num_train_samples = 0
        self.truth_format = TruthFormat.detect_from_file(self.data_path)  # set during setup() from source
    
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
            source.close()
            train_source = MemmapSource(self._cache_path("train"), source.truth_format)
            val_source = MemmapSource(self._cache_path("val"), source.truth_format)
        else:
            train_source = source
            val_source = source

        
        ds_kwargs = dict(
            layer_boundaries=self.layer_boundaries,
            eps=self.eps, 
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut, 
            rew=self.rew, 
            dep_cut=self.dep_cut,
            width_noise=self.width_noise, 
            num_noise_doses=self.num_noise_doses
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
            eps=self.eps,
            u0up_cut=self.u0up_cut,
            u0low_cut=self.u0low_cut,
            rew=self.rew,
            dep_cut=self.dep_cut,
            width_noise=0.0,
            num_noise_doses=self.num_noise_doses,
        )

    def teardown(self, stage=None):
        """Clean up HDF5 handles opened by this DataModule."""
        for attr in ("_train_dataset", "_val_dataset", "_test_dataset"):
            ds = getattr(self, attr, None)
            if ds is not None and hasattr(ds, 'source') and ds.source is not None:
                ds.source.close()