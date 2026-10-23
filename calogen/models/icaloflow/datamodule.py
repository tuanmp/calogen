"""iCaloFlow data module for LEMURS FCCeeALLEGRO (arXiv:2305.11934).

Three flow-specific views over flat showers (N, 6480) MeV + incident
energy (N, 1) MeV, preprocessed in the Dataset __getitem__ (noise lives
in the dataloader, matching the reference):

  Flow-1  features: energy_dep (B, 45)  logit((E_dep + noise) / norm)
          cond:     energy_inc (B, 1)   log10(E_inc / 1e4.5)
  Flow-2  features: layer (B, 144)      logit(sum-norm (layer_0 + noise))
          cond:     [log10(E_inc), logit((E_0 + noise) / norm) / 4]
  Flow-3  features: layer (B, 144)      logit(sum-norm (layer_i + noise))
          cond:     [log10(E_inc), logit(E_i / norm), logit(E_{i-1} / norm),
                     logit(sum-norm layer_{i-1}), one_hot(i - 1, 44)]

The Flow-2 ``energy_dep`` field stores the raw logit((E_0+noise)/norm); the
/4 condition scaling is applied in ``preprocess_cond_flow2``.

Setup loads a capped slice of the flat HDF5 showers (default cap 200k rows,
~5 GB float32, configurable via ``default_max_events``).  With
``cache_mode="memmap"`` the raw showers/energies are written once to a
memmap cache on disk (bounded peak memory) and memory-mapped on later runs,
so RAM stays near zero and startup is fast even for the full 899k-event file.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping, Optional

import numpy as np
import torch
import tqdm
from calogen.core.xml_handler import XMLHandler
from calogen.data.data_util import TruthFormat
from calogen.data.sources import HDF5Source
from calogen.models.icaloflow.model import (
    add_noise,
    compute_normalization_constant,
    logit_trafo,
)
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset, default_collate

# which_flow bitmask values (one flow active per run)
FLOW_1 = 1
FLOW_2 = 2
FLOW_3 = 4

from enum import IntEnum


class Flow(IntEnum):
    FLOW_1 = 1
    FLOW_2 = 2
    FLOW_3 = 3

# ---------------------------------------------------------------------------
# Preprocessing helpers (torch tensors)
# ---------------------------------------------------------------------------

def collate_noise(batch):
    """Collate a batch of dicts with noise samples into a single dict.

    Expects a list of dicts, each containing the same keys.  The values
    are tensors, and the noise samples are lists of tensors.  This function
    collates the tensors and concatenates the noise samples along the batch
    dimension.
    """
    collated = default_collate(batch)
    if isinstance(collated, dict): 
        for key, item in collated.items():
            if "noise" in key:
                collated[key] = torch.reshape(item, (-1, item.shape[-1]))
    return collated

def log10_energy(energy_mev: torch.Tensor, energy_ref_mev: float = 10**4.5) -> torch.Tensor:
    """log10(E / energy_ref_mev), the flow condition encoding of incident energy.

    The reference energy ``1e4.5 MeV`` (≈ 31.6 GeV) centers the 1 GeV–1 TeV
    LEMURS/CaloChallenge incident-energy range to roughly [-1.5, +1.5].
    """
    return torch.log10(energy_mev / energy_ref_mev)


def preprocess_energy_dep(energy_dep, normalization_constant, noise_level):
    """Per-layer energy -> logit((E + U[0, noise_level]) / norm)."""
    return logit_trafo(add_noise(energy_dep, noise_level) / normalization_constant)


def preprocess_voxel_layer(layer, noise_level):
    """Layer voxels -> logit(sum-norm (layer + U[0, noise_level]))."""
    noisy = add_noise(layer, noise_level)
    return logit_trafo(noisy / noisy.sum(dim=-1, keepdim=True))


def one_hot(layer_number: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot of 1-based layer numbers (index = layer_number - 1)."""
    return torch.nn.functional.one_hot(layer_number.long() - 1, num_classes).float()


def preprocess_cond_flow2(batch):
    """Flow-2 condition: [log10(E_inc), logit((E_0 + noise)/norm) / 4].

    Expects a *batched* dict (values carry a leading batch dim).  The /4
    scaling is applied here (not in the dataset) so the raw logit value is
    what the reference keeps in ``energy_dep``.
    """
    return torch.cat([batch["energy"], batch["energy_dep"] / 4.0], dim=1)


def preprocess_cond_flow3(batch, num_layers: Optional[int] = None):
    """Flow-3 condition (191 dims): energy + E_i + E_{i-1} + layer_p + onehot.

    Expects a *batched* dict (values carry a leading batch dim).  The one-hot
    has ``num_layers - 1`` classes (layers 2..45, i.e. 1-based 2..45 index
    into 1..44); if ``num_layers`` is omitted it is inferred from
    ``layer_number`` values.
    """
    if num_layers is None:
        num_layers = int(batch["layer_number"].max()) + 1
    onehot_layer = one_hot(batch["layer_number"], num_classes=num_layers - 1)
    return torch.cat(
        [
            batch["energy"],
            batch["energy_dep"],
            batch["energy_dep_p"],
            batch["layer_p"],
            onehot_layer,
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ICaloFlowDataset(Dataset):
    """Map-style dataset over flat showers; noise applied per item.

    ``showers`` / ``energies`` may be in-memory ndarrays or memmap-backed
    arrays; ``torch.as_tensor`` shares storage (zero-copy) for memmaps, so
    slicing in ``__getitem__`` reads only the rows it touches.  Per-layer
    deposited energies are computed on demand per item rather than
    precomputed, so no full-file pass is needed at construction (important
    when the arrays are memmap-backed).
    """

    def __init__(
        self,
        showers,
        energies,
        layer_boundaries,
        which_flow,
        normalization_constant,
        noise_level,
        energy_ref_mev: float = 10**4.5,
        return_raw: bool = False,
    ):
        self.showers = torch.as_tensor(showers, dtype=torch.float32)
        self.energies = torch.as_tensor(energies, dtype=torch.float32)
        self.layer_boundaries = np.asarray(layer_boundaries, dtype=int)
        self.num_layers = len(self.layer_boundaries) - 1
        self.which_flow = which_flow
        self.normalization_constant = float(normalization_constant)
        self.noise_level = float(noise_level)
        self.energy_ref_mev = float(energy_ref_mev)
        self.return_raw = return_raw

    def _layer_energy(self, idx, layer):
        """Sum of the deposited energy in ``layer`` of event ``idx`` (0-d tensor)."""
        lo = self.layer_boundaries[layer]
        hi = self.layer_boundaries[layer + 1]
        return self.showers[idx, lo:hi].sum()

    def __len__(self) -> int:
        return len(self.showers)

    def __getitem__(self, idx):
        energy = self.energies[idx]  # (1,)
        if self.return_raw:
            # Raw physical showers (MeV) + incident energy (MeV), no preprocessing.
            # Used by the density-ratio classifier truth data.
            return self.showers[idx], energy
        if self.which_flow == FLOW_1:
            # Per-layer deposited energies for all 45 layers, computed on demand.
            e_dep = torch.stack(
                [self._layer_energy(idx, l) for l in range(self.num_layers)]
            )
            return {
                "energy_dep": preprocess_energy_dep(
                    e_dep, self.normalization_constant, self.noise_level
                ),
                "energy_inc": log10_energy(energy, self.energy_ref_mev),
                "energy_mev": energy,
                "energy_mev_scaled": energy / self.energy_ref_mev
            }
        if self.which_flow == FLOW_2:
            lo, hi = self.layer_boundaries[0], self.layer_boundaries[1]
            e_0 = self._layer_energy(idx, 0).reshape(1)
            return {
                "layer": preprocess_voxel_layer(self.showers[idx, lo:hi], self.noise_level),
                "energy": log10_energy(energy, self.energy_ref_mev),
                "energy_dep": preprocess_energy_dep(
                    e_0, self.normalization_constant, self.noise_level
                ),
                "energy_mev": energy,
                "energy_mev_scaled": energy / self.energy_ref_mev
            }
        if self.which_flow == FLOW_3:
            # Uniformly sample the current layer 1..num_layers-1 (previous = current - 1).
            layer_number = int(torch.randint(1, self.num_layers, ()).item())
            lo, hi = self.layer_boundaries[layer_number], self.layer_boundaries[layer_number + 1]
            lo_p, hi_p = self.layer_boundaries[layer_number - 1], self.layer_boundaries[layer_number]
            e_i = self._layer_energy(idx, layer_number).reshape(1)
            e_im1 = self._layer_energy(idx, layer_number - 1).reshape(1)
            return {
                "layer": preprocess_voxel_layer(self.showers[idx, lo:hi], self.noise_level),
                "layer_p": preprocess_voxel_layer(self.showers[idx, lo_p:hi_p], self.noise_level),
                "energy": log10_energy(energy, self.energy_ref_mev),
                "energy_dep": preprocess_energy_dep(
                    e_i, self.normalization_constant, self.noise_level
                ),
                "energy_dep_p": preprocess_energy_dep(
                    e_im1, self.normalization_constant, self.noise_level
                ),
                "layer_number": torch.tensor(layer_number, dtype=torch.long),
                "energy_mev_scaled": energy / self.energy_ref_mev,
                "energy_mev": energy,
                "energy_mev_p": self._layer_energy(idx, layer_number - 1).reshape(1),
                "layer_mev": self.showers[idx, lo:hi],
                "layer_mev_p": self.showers[idx, lo_p:hi_p],
                "energy_mev_i": e_i,
            }
        raise ValueError(f"which_flow must be one of 1, 2, 4; got {self.which_flow}")

class ICaloFlowNCEDataset(ICaloFlowDataset):
    def __init__(
        self,
        showers,
        energies,
        layer_boundaries,
        which_flow,
        normalization_constant,
        noise_level,
        energy_ref_mev: float = 10**4.5,
        return_raw: bool = False,
        noise_amplifier: int = 1,
        q0_mean: Mapping[int, str] | None = None,
        q0_cov: Mapping[int, str] | None = None,
    ):
        super().__init__(
            showers,
            energies,
            layer_boundaries,
            which_flow,
            normalization_constant,
            noise_level,
            energy_ref_mev,
            return_raw
        )
        self.noise_amplifier = noise_amplifier
        self.noise = self._setup_noise_distribution(q0_mean, q0_cov)

    def _setup_noise_distribution(self, q0_mean: Mapping[int, str] | None, q0_cov: Mapping[int, str] | None):
        """Set up the noise distribution for NCE.

        The noise distribution is a multivariate Gaussian with mean and covariance
        specified by q0_mean and q0_cov. If q0_mean or q0_cov is None, the noise
        distribution is set to a standard normal distribution.
        """
        # get a sample to obtain the latent dimension
        print("Obtaining a sample to determine latent dimensions...")
        sample = super().__getitem__(0)
        latent_dims = {}
        if self.return_raw:
            raise NotImplementedError("NCE dataset does not support return_raw=True.")
        else:
            match self.which_flow:
                case Flow.FLOW_1:
                    latent_dims[self.which_flow] = sample["energy_dep"].shape[0]
                    print(f"Latent dimensions for Flow-1: {latent_dims[self.which_flow]}")
                case Flow.FLOW_2:
                    raise NotImplementedError("NCE for Flow-2 is not implemented yet.")
                case Flow.FLOW_3:
                    raise NotImplementedError("NCE for Flow-3 is not implemented yet.")
                case _:
                    raise ValueError(f"which_flow must be one of 1, 2, 4; got {self.which_flow}")

        self.q0 = {}
        noise = {}
        for flow, dim in latent_dims.items():
            mean = torch.zeros(dim, dtype=torch.float32)
            cov = torch.eye(dim, dtype=torch.float32)
            if q0_mean is not None and q0_mean.get(flow) is not None and os.path.exists(q0_mean[flow]):
                mean = torch.load(q0_mean[flow], map_location=torch.device('cpu')).to(torch.float32)
            if q0_cov is not None and q0_cov.get(flow) is not None and os.path.exists(q0_cov[flow]):
                cov = torch.load(q0_cov[flow], map_location=torch.device('cpu')).to(torch.float32)
            self.q0[flow] = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            print(f"Sampling noise distribution for Flow-{flow}: mean shape = {mean.shape}, cov shape = {cov.shape}, N = {self.__len__()} x {self.noise_amplifier}")
            noise[flow] = self.q0[flow].sample((self.__len__() * self.noise_amplifier ,)) 
        return noise
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # for nce, also return the latent variables for the current flow
        for flow, noise_sample in self.noise.items():
            noise_idx = [idx + i * self.__len__() for i in range(self.noise_amplifier)]
            noise = noise_sample[noise_idx]
            match flow:
                case Flow.FLOW_1:
                    data[f"energy_dep_noise"] = noise
                case Flow.FLOW_2:
                    raise NotImplementedError("NCE for Flow-2 is not implemented yet.")
                case Flow.FLOW_3:
                    raise NotImplementedError("NCE for Flow-3 is not implemented yet.")

        return data


# ---------------------------------------------------------------------------
# Lightning data module
# ---------------------------------------------------------------------------


class ICaloFlowDataModule(LightningDataModule):
    """Data module for a single iCaloFlow training view (Flow-1/2/3).

    Loads a capped slice of the flat HDF5 showers and derives the flow
    views per __getitem__, matching the reference (noise lives in the
    dataloader).  With ``cache_mode="memmap"`` the raw showers/energies are
    written to a disk cache once and memory-mapped afterwards, keeping RAM
    near zero (important on low-memory clusters) and making setup fast.

    Geometry (``num_layers`` x ``cells_per_layer``) is read from the XML
    layer boundaries, not hardcoded.  ``seed`` fixes the train/val split;
    ``default_max_events`` / ``load_chunk`` bound setup memory; and
    ``energy_ref_mev`` (1e4.5 MeV, ≈ 31.6 GeV) centers the incident-energy
    conditioning.
    """

    def __init__(
        self,
        which_flow: int,
        data_path: str,
        val_data_path: str,
        xml_path: str,
        xml_ptype: str = "photon",
        batch_size: int = 1000,
        val_frac: float = 0.01,
        noise_level: float = 5e-3,
        safety_factor: float = 1.5,
        num_workers: int = 0,
        max_events: Optional[int] = None,
        max_events_val: Optional[int] = None,
        norm_sample_size: int = 20000,
        seed: int = 42,
        default_max_events: int = 200000,
        load_chunk: int = 20000,
        energy_ref_mev: float = 10**4.5,
        cache_mode: str = "none",
        cache_dir: str = "",
        cache_chunk: Optional[int] = None,
        return_raw: bool = False,
    ):
        super().__init__()
        if which_flow not in (FLOW_1, FLOW_2, FLOW_3):
            raise ValueError("which_flow must be 1 (flow 1), 2 (flow 2) or 4 (flow 3)")
        if cache_mode not in ("none", "memmap"):
            raise ValueError(f"cache_mode must be 'none' or 'memmap'; got {cache_mode}")
        if cache_mode == "memmap" and not cache_dir:
            raise ValueError("cache_dir is required when cache_mode='memmap'")
        self.which_flow = which_flow
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.xml_path = xml_path
        self.xml_ptype = xml_ptype
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.noise_level = noise_level
        self.safety_factor = safety_factor
        self.num_workers = num_workers
        self.max_events = max_events
        self.max_events_val = max_events_val
        self.norm_sample_size = norm_sample_size
        self.seed = seed
        self.default_max_events = default_max_events
        self.load_chunk = load_chunk
        self.energy_ref_mev = energy_ref_mev
        self.cache_mode = cache_mode
        self.cache_dir = cache_dir
        self.cache_chunk = cache_chunk or load_chunk
        self.return_raw = return_raw

        self.layer_boundaries = None
        self.normalization_constant = None
        self.num_train_samples = 0
        self.truth_format: TruthFormat = None
        self._train_dataset = None
        self._val_dataset = None
        self._predict_dataset = None

    # -- setup --------------------------------------------------------------

    def setup(self, stage=None):
        self._load_layer_boundaries()
        if stage in (None, "fit"):
            showers, energies = self._load_or_build(
                self.data_path, self.max_events, split=True
            )
            sample = min(self.norm_sample_size, len(showers))
            self.normalization_constant = compute_normalization_constant(
                showers[:sample], self.layer_boundaries, self.safety_factor
            )
            self.num_train_samples = int(len(showers))
            self._train_dataset = self._build_dataset(showers, energies)
            self._val_dataset = self._build_dataset(
                self._val_showers, self._val_energies
            )
            rank_zero_info(
                f"📦 iCaloFlow Flow-{self.which_flow} fit ready: "
                f"{self.num_train_samples} train / {len(self._val_dataset)} val events, "
                f"norm_constant={self.normalization_constant:.2f} MeV, "
                f"cache_mode={self.cache_mode}"
            )
        if stage in (None, "test", "predict"):
            self._ensure_normalization_constant()
            showers, energies = self._load_or_build(
                self.val_data_path, self.max_events_val, split=False
            )
            self._predict_dataset = self._build_dataset(showers, energies)
            rank_zero_info(
                f"📦 iCaloFlow predict ready: {len(self._predict_dataset)} events "
                f"from {os.path.basename(self.val_data_path)}"
            )

    def _build_dataset(self, showers, energies):
        return ICaloFlowDataset(
            showers,
            energies,
            self.layer_boundaries,
            self.which_flow,
            self.normalization_constant,
            self.noise_level,
            self.energy_ref_mev,
            return_raw=self.return_raw,
        )

    def _load_layer_boundaries(self):
        if self.layer_boundaries is None:
            xml = XMLHandler(particle_name=self.xml_ptype, filename=self.xml_path)
            self.layer_boundaries = np.unique(xml.GetBinEdges()).astype(np.int64)

    def _ensure_normalization_constant(self):
        """Compute from a capped sample of the train file if not already set."""
        if self.normalization_constant is not None:
            return
        source = HDF5Source(self.data_path)
        try:
            n = min(self.norm_sample_size, len(source))
            showers, _ = source.read_rows(np.arange(n))
            showers = source.truth_format.flatten_showers(showers)
            self.truth_format = source.truth_format
        finally:
            source.close()
        self.normalization_constant = compute_normalization_constant(
            showers, self.layer_boundaries, self.safety_factor
        )
        rank_zero_info(
            f"Computed normalization_constant = {self.normalization_constant:.2f} MeV "
            f"from {n} samples of {os.path.basename(self.data_path)}"
        )

    # ------------------------------------------------------------------
    #  Data loading: in-memory (cache_mode="none") or memmap cache
    # ------------------------------------------------------------------

    def _load_or_build(self, path, max_events, split):
        """Return (showers, energies) for a data file.

        With ``cache_mode="memmap"``: if a cache for this file+config exists,
        memory-map it (near-zero RSS, fast startup); otherwise stream the HDF5
        once into memmap files (bounded peak memory).  When ``split`` is True,
        the train/val split is baked into the cache at write time so loading
        needs no in-RAM fancy-index copies.

        The cache holds only *raw* showers/energies in MeV — no preprocessing
        (logit/normalization/noise) is cached; that all happens per __getitem__.
        """
        if self.cache_mode != "memmap":
            return self._load_array(path, max_events, split)

        meta_path = self._cache_path(path, max_events, split)
        if os.path.exists(meta_path):
            rank_zero_info(
                f"📦 Memmap cache hit: {os.path.basename(path)} "
                f"(key={os.path.basename(meta_path)[:-5]})"
            )
            return self._open_cache(meta_path, split)
        rank_zero_info(
            f"📦 Memmap cache MISS for {os.path.basename(path)} "
            f"(key={os.path.basename(meta_path)[:-5]}) — writing cache to "
            f"{self.cache_dir} (one-time cost)"
        )
        self._write_cache(path, max_events, split)
        return self._open_cache(meta_path, split)

    def _cache_path(self, path, max_events, split):
        key = self._cache_key(path, max_events, split)
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{key}.json")

    def _cache_key(self, path, max_events, split):
        """Stable content-based key for the memmap cache.

        Data-file identity uses realpath + size + mtime: content-hashing a
        multi-GB HDF5 is too slow, and size+mtime invalidate the cache when
        the file is replaced or regenerated.  The XML is hashed by *content*
        (a tiny file) rather than its path, so a relative vs absolute
        reference never spuriously rebuilds the cache.  The split is keyed by
        seed + val_frac because it is baked into the cache layout.
        """
        st = os.stat(path)
        with open(self.xml_path, "rb") as f:
            xml_hash = hashlib.md5(f.read()).hexdigest()
        material = (
            f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime}|{max_events}"
        )
        if split:
            material += f"|{self.seed}|{self.val_frac}"
        material += f"|{xml_hash}"
        return hashlib.md5(material.encode()).hexdigest()[:16]

    def _write_cache(self, path, max_events, split):
        """Stream HDF5 -> raw memmap .dat files (train/val split if requested)."""
        meta_path = self._cache_path(path, max_events, split)
        if os.path.exists(meta_path):
            return

        source = HDF5Source(path)
        try:
            n_total = len(source)
            cap = self.default_max_events if max_events is None else max_events
            n = int(min(n_total, cap))
            total_cells = int(self.layer_boundaries[-1])

            # Deterministic split baked into the cache (mirrors setup seeding).
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(n) if split else np.arange(n)
            n_val = int(n * self.val_frac) if split else 0
            n_train = n - n_val
            self.num_train_samples = n_train

            dat_dir = os.path.dirname(meta_path)
            key = os.path.basename(meta_path)[:-5]
            trn_s = np.memmap(os.path.join(dat_dir, f"{key}_train_showers.dat"),
                              mode="w+", dtype="float32", shape=(n_train, total_cells))
            trn_e = np.memmap(os.path.join(dat_dir, f"{key}_train_energies.dat"),
                              mode="w+", dtype="float32", shape=(n_train, 1))
            val_s = np.memmap(os.path.join(dat_dir, f"{key}_val_showers.dat"),
                              mode="w+", dtype="float32", shape=(n_val, total_cells))
            val_e = np.memmap(os.path.join(dat_dir, f"{key}_val_energies.dat"),
                              mode="w+", dtype="float32", shape=(n_val, 1))

            # Assign rows in chunks so only one load_chunk is resident at a time.
            # Rows are scattered to their permuted output positions so the cache
            # is bit-identical to the in-memory path (same split ordering, which
            # also fixes the normalization-constant sample rows).
            inv_perm = np.empty(n, dtype=np.int64)
            inv_perm[perm] = np.arange(n)
            pbar = tqdm.tqdm(
                total=n, desc=f"memmap cache {os.path.basename(path)}", unit="samples"
            )
            for start in range(0, n, self.load_chunk):
                end = min(start + self.load_chunk, n)
                showers, energies = source.read_rows(np.arange(start, end))
                showers = source.truth_format.flatten_showers(showers)
                energies = source.truth_format.format_energy(energies)
                out_pos = inv_perm[start:end]
                is_trn = out_pos < n_train
                trn_s[out_pos[is_trn]] = showers[is_trn]
                trn_e[out_pos[is_trn]] = energies[is_trn]
                nv = int((~is_trn).sum())
                val_s[out_pos[~is_trn] - n_train] = showers[~is_trn]
                val_e[out_pos[~is_trn] - n_train] = energies[~is_trn]
                pbar.update(end - start)
            pbar.close()

            trn_s.flush(); trn_e.flush(); val_s.flush(); val_e.flush()
            self.truth_format = source.truth_format
        finally:
            source.close()

        rank_zero_info(
            f"✅ Cache written: {n_train} train / {n_val} val events "
            f"(key={os.path.basename(meta_path)[:-5]})"
        )

        meta = {
            "fields": {
                "train_showers": {"file": f"{key}_train_showers.dat",
                                  "shape": [n_train, total_cells], "dtype": "float32"},
                "train_energies": {"file": f"{key}_train_energies.dat",
                                   "shape": [n_train, 1], "dtype": "float32"},
                "val_showers": {"file": f"{key}_val_showers.dat",
                                "shape": [n_val, total_cells], "dtype": "float32"},
                "val_energies": {"file": f"{key}_val_energies.dat",
                                 "shape": [n_val, 1], "dtype": "float32"},
            },
            "truth_format": self.truth_format.__dict__ if self.truth_format else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _open_cache(self, meta_path, split):
        """Memory-map the cached .dat files and return (showers, energies)."""
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if self.truth_format is None and meta.get("truth_format"):
            self.truth_format = TruthFormat(**meta["truth_format"])
        dat_dir = os.path.dirname(meta_path)
        fields = meta["fields"]

        def _mmap(name):
            info = fields[name]
            # Read-only: concurrent training jobs safely share the cache file
            # (the OS/Lustre page cache is coherent for read-only mmap).  No
            # per-node /tmp copy: nothing writes to these arrays (noise is
            # applied to copies in the Dataset), and a read-only open makes
            # any accidental in-place mutation fail instead of silently
            # corrupting the shared cache.
            return np.memmap(
                os.path.join(dat_dir, info["file"]),
                mode="r",
                dtype=info["dtype"],
                shape=tuple(info["shape"]),
            )

        if split:
            trn_s = _mmap("train_showers")
            trn_e = _mmap("train_energies")
            self._val_showers = _mmap("val_showers")
            self._val_energies = _mmap("val_energies")
            self.num_train_samples = int(trn_s.shape[0])
            rank_zero_info(
                f"Memory-mapped train cache: {self.num_train_samples} train / "
                f"{int(self._val_showers.shape[0])} val events"
            )
            return trn_s, trn_e

        # Non-split (predict) cache stores a single block under "train_*".
        showers = _mmap("train_showers")
        energies = _mmap("train_energies")
        rank_zero_info(f"Memory-mapped predict cache: {int(showers.shape[0])} events")
        return showers, energies

    def _load_array(self, path, max_events, split):
        """In-memory fallback (cache_mode="none"): read the capped rows + split.

        Rows are read in ``self.load_chunk`` blocks to bound peak memory.
        """
        source = HDF5Source(path)
        try:
            n_total = len(source)
            cap = self.default_max_events if max_events is None else max_events
            n = int(min(n_total, cap))
            parts = []
            pbar = tqdm.tqdm(
                total=n, desc=f"loading {os.path.basename(path)}", unit="samples"
            )
            for start in range(0, n, self.load_chunk):
                end = min(start + self.load_chunk, n)
                showers, energies = source.read_rows(np.arange(start, end))
                showers = source.truth_format.flatten_showers(showers)
                energies = source.truth_format.format_energy(energies)
                parts.append((showers, energies))
                pbar.update(end - start)
            pbar.close()
            if not parts:
                total_cells = int(self.layer_boundaries[-1])
                empty_s = np.empty((0, total_cells), dtype=np.float32)
                empty_e = np.empty((0, 1), dtype=np.float32)
                return empty_s, empty_e
            showers = np.concatenate([p[0] for p in parts], axis=0)
            energies = np.concatenate([p[1] for p in parts], axis=0)
            self.truth_format = source.truth_format
            rank_zero_info(
                f"Loaded {n} events from {os.path.basename(path)} into memory"
            )
        finally:
            source.close()

        if not split:
            return showers, energies

        # Deterministic train/val split (mirrors LEMURSDataModule's seed).
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(n)
        n_val = int(n * self.val_frac)
        n_train = n - n_val
        self.num_train_samples = n_train
        trn_s = showers[perm[:n_train]]
        trn_e = energies[perm[:n_train]]
        self._val_showers = showers[perm[n_train:]]
        self._val_energies = energies[perm[n_train:]]
        return trn_s, trn_e

    # -- dataloaders --------------------------------------------------------

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class ICaloFlowNCEDataModule(ICaloFlowDataModule):
    """Data module for a single iCaloFlow training view (Flow-1/2/3) with NCE.

    Loads a capped slice of the flat HDF5 showers and derives the flow
    views per __getitem__, matching the reference (noise lives in the
    dataloader).  With ``cache_mode="memmap"`` the raw showers/energies are
    written to a disk cache once and memory-mapped afterwards, keeping RAM
    near zero (important on low-memory clusters) and making setup fast.

    Geometry (``num_layers`` x ``cells_per_layer``) is read from the XML
    layer boundaries, not hardcoded.  ``seed`` fixes the train/val split;
    ``default_max_events`` / ``load_chunk`` bound setup memory; and
    ``energy_ref_mev`` (1e4.5 MeV, ≈ 31.6 GeV) centers the incident-energy
    conditioning.
    """

    def __init__(
        self,
        which_flow: int,
        data_path: str,
        val_data_path: str,
        xml_path: str,
        xml_ptype: str = "photon",
        batch_size: int = 1000,
        val_frac: float = 0.01,
        noise_level: float = 5e-3,
        safety_factor: float = 1.5,
        num_workers: int = 0,
        max_events: Optional[int] = None,
        max_events_val: Optional[int] = None,
        norm_sample_size: int = 20000,
        seed: int = 42,
        default_max_events: int = 200000,
        load_chunk: int = 20000,
        energy_ref_mev: float = 10**4.5,
        cache_mode: str = "none",
        cache_dir: str = "",
        cache_chunk: Optional[int] = None,
        return_raw: bool = False,
        noise_amplifier: int = 1,
        q0_mean: Mapping[int, str] | None = None,
        q0_cov: Mapping[int, str]  | None = None,
    ):
        super().__init__(
            which_flow,
            data_path,
            val_data_path,
            xml_path,
            xml_ptype,
            batch_size,
            val_frac,
            noise_level,
            safety_factor,
            num_workers,
            max_events,
            max_events_val,
            norm_sample_size,
            seed,
            default_max_events,
            load_chunk,
            energy_ref_mev,
            cache_mode,
            cache_dir,
            cache_chunk,
            return_raw
        )
        self.noise_amplifier = noise_amplifier
        self.q0_mean = q0_mean or {}
        self.q0_cov = q0_cov or {}

    def _build_dataset(self, showers, energies):
            return ICaloFlowNCEDataset(
                showers,
                energies,
                self.layer_boundaries,
                self.which_flow,
                self.normalization_constant,
                self.noise_level,
                self.energy_ref_mev,
                return_raw=self.return_raw,
                noise_amplifier=self.noise_amplifier,
                q0_mean=self.q0_mean,
                q0_cov=self.q0_cov
            )

    # -- dataloaders --------------------------------------------------------
    
    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_noise
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_noise
        )

    def predict_dataloader(self):
        return DataLoader(
            self._predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_noise
        )