"""Tests for the GPU-native post-processing implementation.

These tests validate that ``torch_postprocess`` produces numerically
identical results to the legacy NumPy implementation in ``data_util.py``.
"""

import os
import unittest

import numpy as np
import torch

import calogen.core.data_util as data_util
import calogen.core.postprocess as torch_postprocess
from calogen.models.caloinn.lightning import CaloINNLightningModule

# Must match the CINN model's expected dtype.
torch.set_default_dtype(torch.float32)

CHECKPOINT_PATH = "/pscratch/sd/p/pmtuan/caloinn/caloinn_lightning/53921130/checkpoints/epoch=491-val_loss=895.95.ckpt"
DATA_PATH = "/pscratch/sd/p/pmtuan/ddsim/single_pion_discrete_coarse_hcal/trainset_flat_good_order.hdf5"
XML_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "binning_pion_odd_coarse_hcal.xml")

# Tolerances for float32 comparisons.  The operations are deterministic and
# algebraically identical, so differences should be near machine epsilon.
RTOL = 1e-5
ATOL = 1e-6


def _numpy_dict_to_float32(data):
    """Recursively cast NumPy arrays in a dict to float32."""
    return {k: v.astype(np.float32) if isinstance(v, np.ndarray) else v for k, v in data.items()}


def _torch_dict_to_float32(data):
    """Move torch tensors to CPU and convert to float32 numpy arrays."""
    return {
        k: v.detach().cpu().to(torch.float32).numpy()
        if isinstance(v, torch.Tensor)
        else v
        for k, v in data.items()
    }


class TestTorchUnnormalizeLayersSynthetic(unittest.TestCase):
    """Unit tests for ``unnormalize_layers`` using synthetic data."""

    def _make_synthetic(self, batch_size=8, num_cells=720, num_layers=36, seed=42):
        """Build a synthetic normalized sample + conditions.

        The cells within each layer are normalized to sum to 1, and the extra
        dimensions encode layer energies consistent with those cells.
        """
        rng = np.random.default_rng(seed)

        # Layer boundaries: 36 layers of equal size for simplicity.
        layer_boundaries = np.linspace(0, num_cells, num_layers + 1, dtype=int)

        # Random incident energies in GeV.
        incident_energies = rng.uniform(10.0, 1000.0, size=(batch_size, 1)).astype(np.float32)

        # Random cell energies, then normalize per layer.
        cell_energies = rng.exponential(scale=1.0, size=(batch_size, num_cells)).astype(np.float32)
        for ls, le in zip(layer_boundaries[:-1], layer_boundaries[1:]):
            layer_sum = cell_energies[:, ls:le].sum(axis=1, keepdims=True) + 1e-10
            cell_energies[:, ls:le] /= layer_sum

        layer_energies = np.array(
            [cell_energies[:, ls:le].sum(axis=1) for ls, le in zip(layer_boundaries[:-1], layer_boundaries[1:])]
        ).T.astype(np.float32)

        # Build extra dimensions matching ``data_util.get_energy_dims``.
        extra_dims = [layer_energies.sum(axis=1, keepdims=True) / incident_energies]
        for i in range(num_layers - 1):
            remaining = layer_energies[:, i:].sum(axis=1, keepdims=True) + 1e-10
            extra_dims.append(layer_energies[:, [i]] / remaining)
        extra_dims = np.concatenate(extra_dims, axis=1).astype(np.float32)

        x = np.concatenate([cell_energies, extra_dims], axis=1)
        return x, incident_energies, layer_boundaries

    def test_unnormalize_layers_matches_numpy(self):
        x_np, c_np, layer_boundaries = self._make_synthetic()

        out_np = data_util.unnormalize_layers(x_np, c_np, layer_boundaries)
        out_torch = torch_postprocess.unnormalize_layers(
            torch.from_numpy(x_np),
            torch.from_numpy(c_np),
            layer_boundaries,
        )

        # The NumPy implementation returns a padded array whose trailing
        # columns (the former extra-dim slots) are all zeros.  The PyTorch
        # implementation drops those columns.  Only the leading num_cells
        # columns are meaningful.
        num_cells = x_np.shape[1] - (len(layer_boundaries) - 1)
        np.testing.assert_allclose(
            out_np[:, :num_cells].astype(np.float32),
            out_torch.detach().cpu().numpy().astype(np.float32),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_postprocess_matches_numpy(self):
        x_np, c_np, layer_boundaries = self._make_synthetic()

        # Use a simple per-cell quantile for the synthetic test.
        quantiles_np = np.full(x_np.shape[1], 1e-4, dtype=np.float32)
        quantiles_torch = torch.from_numpy(quantiles_np)

        data_np = data_util.postprocess(
            x_np,
            c_np,
            layer_boundaries=layer_boundaries,
            threshold=1e-4,
            quantiles=quantiles_np,
        )
        data_torch = torch_postprocess.postprocess(
            torch.from_numpy(x_np),
            torch.from_numpy(c_np),
            layer_boundaries=layer_boundaries,
            quantiles=quantiles_torch,
        )

        data_np = _numpy_dict_to_float32(data_np)
        data_torch = _torch_dict_to_float32(data_torch)

        self.assertEqual(set(data_np.keys()), set(data_torch.keys()))
        for key in data_np:
            np.testing.assert_allclose(
                data_np[key],
                data_torch[key],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Mismatch in key '{key}'",
            )


class TestTorchPostprocessIntegration(unittest.TestCase):
    """End-to-end test comparing post-processing on real CINN samples."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CHECKPOINT_PATH):
            raise unittest.SkipTest(f"Checkpoint not found: {CHECKPOINT_PATH}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; this test runs the model on GPU")

        torch.set_default_dtype(torch.float32)
        cls.module = CaloINNLightningModule.load_from_checkpoint(CHECKPOINT_PATH)
        cls.module.eval()
        cls.module = cls.module.to("cuda")

    def test_generation_postprocess_matches_numpy(self):
        energy_log2 = 16  # 2^16 MeV ≈ 65.5 GeV
        energy_gev = 2.0 ** energy_log2 / 1e3
        num_samples = 32

        energies = torch.full(
            (num_samples, 1),
            energy_gev,
            dtype=torch.float32,
            device=self.module.device,
        )

        # Raw CINN samples (same shape as inside predict_step before post-processing).
        with torch.inference_mode():
            samples = self.module.model.sample(1, energies)
            samples = samples - self.module.width_noise
            samples = samples[:, 0, ...]

        # NumPy post-processing path (current production code).
        data_np = data_util.postprocess(
            samples.detach().cpu().numpy(),
            energies.detach().cpu().numpy(),
            layer_boundaries=self.module.layer_boundaries,
            threshold=self.module.width_noise,
            quantiles=self.module.q.detach().cpu().numpy(),
        )

        # PyTorch post-processing path (candidate replacement).
        data_torch = torch_postprocess.postprocess(
            samples,
            energies,
            layer_boundaries=self.module.layer_boundaries,
            quantiles=self.module.q,
        )

        data_np = _numpy_dict_to_float32(data_np)
        data_torch = _torch_dict_to_float32(data_torch)

        self.assertEqual(set(data_np.keys()), set(data_torch.keys()))
        for key in data_np:
            np.testing.assert_allclose(
                data_np[key],
                data_torch[key],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Mismatch in key '{key}'",
            )


if __name__ == "__main__":
    unittest.main()
