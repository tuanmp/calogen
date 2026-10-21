"""Tests for the LEMURS data module adapter.

Uses the LEMURS testing dataset (downloaded to /pscratch/sd/p/pmtuan/lemurs/).
Skips if data is not available.
"""

import os
import unittest

import numpy as np
import torch

torch.set_default_dtype(torch.float32)

from calogen.core.lemurs import (
    LEMURSHDF5Source,
    LEMURSDataModule,
    _transpose_and_flatten,
    _LEMURS_TOTAL_CELLS,
    _LEMURS_N_LAYERS,
    _LEMURS_CELLS_PER_LAYER,
)
from calogen.core.sharded import _build_data_dict
from calogen.core.xml_handler import XMLHandler

# Test file paths (testing dataset, 50 GeV, theta=pi/2, phi=0, Par04SiW)
_TEST_DATA_DIR = "/pscratch/sd/p/pmtuan/lemurs/extracted"
_SIW_50 = os.path.join(_TEST_DATA_DIR, "Par04SiW/testing/LEMURS_Par04SiW_gamma_1000events_50GeV_phi0.0_theta1.57.h5")
_ALLEGRO_50 = os.path.join(_TEST_DATA_DIR, "FCCeeALLEGRO/testing/LEMURS_FCCeeALLEGRO_gamma_1000events_50GeV_phi0.0_theta1.57.h5")

_HAS_DATA = os.path.exists(_SIW_50)


@unittest.skipUnless(_HAS_DATA, "LEMURS test data not found")
class TestLEMURSHDF5Source(unittest.TestCase):
    """Test the raw HDF5 source adapter."""

    def setUp(self):
        self.source = LEMURSHDF5Source(_SIW_50)

    def tearDown(self):
        self.source.close()

    def test_length(self):
        self.assertEqual(len(self.source), 1000)

    def test_read_rows_shape(self):
        showers, energies = self.source.read_rows(np.array([0, 1, 2]))
        self.assertEqual(showers.shape, (3, _LEMURS_TOTAL_CELLS))
        self.assertEqual(energies.shape, (3, 1))
        self.assertEqual(showers.dtype, np.float32)
        self.assertEqual(energies.dtype, np.float32)

    def test_read_rows_values(self):
        import h5py
        showers, energies = self.source.read_rows(np.array([0]))
        with h5py.File(_SIW_50, "r") as f:
            orig = f["showers"][0]
            orig_flat = np.transpose(orig, (2, 0, 1)).reshape(-1)
            self.assertTrue(np.allclose(showers[0], orig_flat),
                            "Flattened showers must match original transposed data")

            orig_energy = f["incident_energy"][0]
            self.assertAlmostEqual(energies[0, 0], orig_energy, places=3)

    def test_transpose_and_flatten(self):
        import h5py
        with h5py.File(_SIW_50, "r") as f:
            showers_3d = f["showers"][:5]
        flat = _transpose_and_flatten(showers_3d)
        self.assertEqual(flat.shape, (5, _LEMURS_TOTAL_CELLS))

        for i in range(5):
            for layer in range(_LEMURS_N_LAYERS):
                start = layer * _LEMURS_CELLS_PER_LAYER
                end = (layer + 1) * _LEMURS_CELLS_PER_LAYER
                layer_flat = flat[i, start:end].reshape(9, 16)
                layer_3d = showers_3d[i, :, :, layer]
                self.assertTrue(np.allclose(layer_flat, layer_3d),
                                f"Layer {layer} mismatch for shower {i}")

    def test_pickle_safe(self):
        import pickle
        state = pickle.dumps(self.source)
        restored = pickle.loads(state)
        self.assertIsNone(restored._file)
        showers, _ = restored.read_rows(np.array([0]))
        self.assertEqual(showers.shape, (1, _LEMURS_TOTAL_CELLS))
        restored.close()

    def test_out_of_order_read(self):
        indices = np.array([5, 0, 3])
        showers, energies = self.source.read_rows(indices)
        import h5py
        with h5py.File(_SIW_50, "r") as f:
            s0 = np.transpose(f["showers"][0], (2, 0, 1)).reshape(-1)
            s5 = np.transpose(f["showers"][5], (2, 0, 1)).reshape(-1)
            s3 = np.transpose(f["showers"][3], (2, 0, 1)).reshape(-1)
            self.assertTrue(np.allclose(showers[0], s5), "First should be index 5")
            self.assertTrue(np.allclose(showers[1], s0), "Second should be index 0")
            self.assertTrue(np.allclose(showers[2], s3), "Third should be index 3")


@unittest.skipUnless(_HAS_DATA, "LEMURS test data not found")
class TestLEMURSDataModule(unittest.TestCase):
    """Test the full data module pipeline."""

    def setUp(self):
        self.dm = LEMURSDataModule(
            data_path=_SIW_50,
            val_data_path=_SIW_50,
            batch_size=64,
            xml_path="binning_lemurs_par04siw.xml",
            xml_ptype="photon",
            val_frac=0.2,
            eps=1.0e-10,
            u0up_cut=0.05,
            u0low_cut=0.0,
            dep_cut=600,
            width_noise=5.0e-6,
            shuffle=True,
            num_workers=0,
        )

    def tearDown(self):
        self.dm.teardown()

    def test_setup_fit(self):
        self.dm.setup("fit")
        self.assertGreater(len(self.dm._train_dataset), 0)
        self.assertGreater(len(self.dm._val_dataset), 0)
        self.assertEqual(
            len(self.dm._train_dataset) + len(self.dm._val_dataset),
            1000,
        )

    def test_train_batch_shape(self):
        self.dm.setup("fit")
        dl = self.dm.train_dataloader()
        x, c = next(iter(dl))
        expected_dim = _LEMURS_TOTAL_CELLS + _LEMURS_N_LAYERS  # 6480 + 45 = 6525
        self.assertEqual(x.shape[1], expected_dim)
        self.assertEqual(c.shape[1], 1)
        self.assertEqual(x.dtype, torch.float32)

    def test_val_batch_shape(self):
        self.dm.setup("fit")
        dl = self.dm.val_dataloader()
        x, c = next(iter(dl))
        expected_dim = _LEMURS_TOTAL_CELLS + _LEMURS_N_LAYERS
        self.assertEqual(x.shape[1], expected_dim)
        self.assertEqual(c.shape[1], 1)

    def test_no_nan_inf(self):
        self.dm.setup("fit")
        dl = self.dm.train_dataloader()
        for x, c in dl:
            self.assertFalse(torch.isnan(x).any(), "NaN in x")
            self.assertFalse(torch.isinf(x).any(), "Inf in x")
            self.assertFalse(torch.isnan(c).any(), "NaN in c")
            self.assertFalse(torch.isinf(c).any(), "Inf in c")
            break

    def test_extra_dims_range(self):
        """Energy ratio (extra_dims[:,0]) should match sampling fraction."""
        self.dm.setup("fit")
        dl = self.dm.train_dataloader()
        x, _ = next(iter(dl))
        extra = x[:, -_LEMURS_N_LAYERS:]
        ratio = extra[:, 0]  # E_dep / E_inc
        self.assertGreater(ratio.mean().item(), 0.01)
        self.assertLess(ratio.mean().item(), 0.05)
        self.assertLess(ratio.max().item(), self.dm.u0up_cut)

    def test_predict_dataloader(self):
        self.dm.setup("predict")
        dl = self.dm.predict_dataloader()
        x, c = next(iter(dl))
        self.assertEqual(x.shape[1], _LEMURS_TOTAL_CELLS + _LEMURS_N_LAYERS)

    def test_filter_cache(self):
        self.dm.setup("fit")
        cache_path = f"{_SIW_50}.filter_*.npy"
        import glob
        cached = glob.glob(cache_path)
        self.assertGreater(len(cached), 0, "Filter cache should be created")

    def test_deterministic_split(self):
        dm1 = LEMURSDataModule(
            data_path=_SIW_50, val_data_path=_SIW_50, batch_size=64,
            xml_path="binning_lemurs_par04siw.xml", xml_ptype="photon",
            val_frac=0.2, u0up_cut=0.05, num_workers=0,
        )
        dm2 = LEMURSDataModule(
            data_path=_SIW_50, val_data_path=_SIW_50, batch_size=64,
            xml_path="binning_lemurs_par04siw.xml", xml_ptype="photon",
            val_frac=0.2, u0up_cut=0.05, num_workers=0,
        )
        dm1.setup("fit")
        dm2.setup("fit")
        self.assertEqual(len(dm1._train_dataset), len(dm2._train_dataset))
        self.assertEqual(len(dm1._val_dataset), len(dm2._val_dataset))
        dm1.teardown()
        dm2.teardown()


@unittest.skipUnless(_HAS_DATA, "LEMURS test data not found")
class TestLEMURSMultiDetector(unittest.TestCase):
    """Test that different LEMURS detectors work correctly."""

    @unittest.skipUnless(os.path.exists(_ALLEGRO_50), "FCCeeALLEGRO not extracted")
    def test_allegro_sampling_fraction(self):
        dm = LEMURSDataModule(
            data_path=_ALLEGRO_50, val_data_path=_ALLEGRO_50, batch_size=64,
            xml_path="binning_lemurs_fcceeallegro.xml", xml_ptype="photon",
            val_frac=0.2, u0up_cut=0.20, num_workers=0,
        )
        dm.setup("fit")
        dl = dm.train_dataloader()
        x, _ = next(iter(dl))
        ratio = x[:, -_LEMURS_N_LAYERS:][:, 0]
        self.assertAlmostEqual(ratio.mean().item(), 0.144, places=2)
        dm.teardown()


@unittest.skipUnless(_HAS_DATA, "LEMURS test data not found")
class TestLEMURSWithBuildDataDict(unittest.TestCase):
    """Verify _build_data_dict works with LEMURS-flattened data."""

    def setUp(self):
        self.source = LEMURSHDF5Source(_SIW_50)
        xh = XMLHandler(particle_name="photon", filename="binning_lemurs_par04siw.xml")
        self.layer_boundaries = np.unique(xh.GetBinEdges())

    def tearDown(self):
        self.source.close()

    def test_build_data_dict(self):
        showers, energies = self.source.read_rows(np.arange(10))
        data = _build_data_dict(showers, energies, self.layer_boundaries)

        self.assertIn("energy", data)
        self.assertAlmostEqual(data["energy"][0, 0], 50.0, places=1)
        self.assertEqual(len(data) - 1, _LEMURS_N_LAYERS)

        for i in range(_LEMURS_N_LAYERS):
            self.assertIn(f"layer_{i}", data)
            self.assertEqual(data[f"layer_{i}"].shape[1], 144)

    def test_preprocess_pipeline(self):
        import calogen.core.data_util as data_util
        showers, energies = self.source.read_rows(np.arange(100))
        data = _build_data_dict(showers, energies, self.layer_boundaries)
        x, c = data_util.preprocess(
            data, self.layer_boundaries, 1.0e-10,
            u0up_cut=0.05, u0low_cut=0.0, dep_cut=600,
        )
        expected_dim = _LEMURS_TOTAL_CELLS + _LEMURS_N_LAYERS
        self.assertEqual(x.shape[1], expected_dim)
        self.assertEqual(c.shape[1], 1)
        self.assertGreater(x.shape[0], 0, "Preprocessing should not discard all events")


if __name__ == "__main__":
    unittest.main()
