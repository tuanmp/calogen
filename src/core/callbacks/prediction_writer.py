import os
import shutil

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from calogen.core.data_util import TruthFormat


class BaseWriter(L.pytorch.callbacks.BasePredictionWriter):

    def __init__(self, predict_output_file="samples.hdf5"):

        super().__init__()

        self.save_dir = "./predictions"
        self.predict_output_file = predict_output_file
        self.batch_dir = os.path.join(self.save_dir, "batches")
        self.all_dls = set()

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):

        super().setup(trainer, pl_module, stage)

        if stage != "predict":
            return

        # if self.save_dir is None:
        #     self.save_dir = os.path.join(trainer.default_root_dir, "predictions")
        #     self.batch_dir = os.path.join(self.save_dir, "batches")

        if trainer.is_global_zero:
            self.save_dir = os.path.join(trainer.default_root_dir, "predictions")
            self.batch_dir = os.path.join(self.save_dir, "batches")
            os.makedirs(self.save_dir, exist_ok=True)
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)

            # make a subdirectory to store batches, then combine them at the end of prediction
            os.makedirs(self.batch_dir, exist_ok=True)

        rank_zero_info(f"PredictionWriter will save predictions to {self.save_dir}")

    def save_batch(self, incident_energies, showers, batch_idx: str, dataloader_idx: str):

        save_path = os.path.join(self.batch_dir, f"dl_{dataloader_idx}_batch_{batch_idx}.npz")

        np.savez(save_path, incident_energies=incident_energies, showers=showers)

        self.all_dls.add(dataloader_idx)


class PredictionWriter(BaseWriter):

    def __init__(self, predict_output_file="samples.hdf5"):
        super().__init__(predict_output_file)

        self.inference_time = []

    def write_on_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, prediction, batch_indices, batch, batch_idx, dataloader_idx):

        incident_energies, showers, t = prediction

        self.save_batch(
            incident_energies=incident_energies,
            showers=showers,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        self.inference_time.append(t)

    def on_predict_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):

        # combine batches into one hdf5 file and save to self.save_dir
        rank_zero_info("Combining batch predictions into one file...")
        if trainer.is_global_zero:
            # Get truth format from datamodule (if available)
            dm = getattr(trainer, "datamodule", None)
            truth_format = getattr(dm, "truth_format", None) if dm is not None else None

            # combine batches into one file
            batch_dir = self.batch_dir
            batch_files = sorted(os.listdir(batch_dir))

            for dataloader_idx in self.all_dls:

                num_predictions = 0
                all_energies = []
                all_showers = []

                h5_file = os.path.join(self.save_dir, self.predict_output_file)
                h5_file = h5_file.replace(".hdf5", f"_dl{dataloader_idx}.hdf5")
                with h5py.File(h5_file, "w") as f:

                    for batch_file in batch_files:
                        if not batch_file.startswith(f"dl_{dataloader_idx}_"):
                            continue
                        batch_path = os.path.join(batch_dir, batch_file)
                        data = np.load(batch_path, allow_pickle=True)

                        incident_energies = data["incident_energies"]
                        showers = data["showers"]

                        # Apply truth format: reshape showers + energy
                        if truth_format is not None:
                            if truth_format.showers_grid_shape is not None:
                                showers = truth_format.unflatten_showers(showers)
                            incident_energies = truth_format.format_energy(incident_energies)

                        energy_key = truth_format.energy_key if truth_format else "incident_energies"

                        if num_predictions == 0:
                            maxshape_energy = (None, incident_energies.shape[1]) if incident_energies.ndim > 1 else (None,)
                            maxshape_showers = (None, *showers.shape[1:])
                            f.create_dataset(energy_key, data=incident_energies, maxshape=maxshape_energy, chunks=True)
                            f.create_dataset("showers", data=showers, maxshape=maxshape_showers, chunks=True)
                        else:
                            f[energy_key].resize(num_predictions + incident_energies.shape[0], axis=0)
                            f[energy_key][-incident_energies.shape[0]:] = incident_energies
                            f["showers"].resize(num_predictions + showers.shape[0], axis=0)
                            f["showers"][-showers.shape[0]:] = showers

                        num_predictions += incident_energies.shape[0]

                rank_zero_info(f"Saved {num_predictions} showers to {h5_file}")
        rank_zero_info(f"Average inference time per sample: {np.mean(self.inference_time):.6f} seconds")