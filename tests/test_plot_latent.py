"""Tests for the PlotLatentCallback (latent-Gaussianity histogram)."""

import os

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float32)

from calogen.core.callbacks.plot_latent import PlotLatentCallback


class _FakeTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.is_global_zero = True
        self.log_dir = None
        self.default_root_dir = "/tmp"


class _FakeModule:
    pass


def _make_outputs(n_batch, n_latent):
    return {"z": torch.randn(n_batch, n_latent)}


def test_default_latent_variables_is_first_three():
    cb = PlotLatentCallback()
    assert cb.latent_variables == [0, 1, 2]


def test_latent_variables_respected_when_provided():
    cb = PlotLatentCallback(latent_variables=[4, 7])
    assert cb.latent_variables == [4, 7]


def test_callback_plots_every_epoch(tmp_path):
    trainer = _FakeTrainer()
    trainer.default_root_dir = str(tmp_path)
    cb = PlotLatentCallback(latent_variables=[0, 1])

    for epoch in range(2):
        trainer.current_epoch = epoch
        cb.on_validation_epoch_start(trainer, _FakeModule())
        cb.on_validation_batch_end(
            trainer, _FakeModule(), _make_outputs(8, 5), None, 0
        )
        cb.on_validation_batch_end(
            trainer, _FakeModule(), _make_outputs(4, 5), None, 1
        )
        cb.on_validation_epoch_end(trainer, _FakeModule())
        path = os.path.join(tmp_path, f"val_latent_hist_epoch{epoch}.png")
        assert os.path.exists(path), f"expected plot at {path}"


def test_callback_skips_when_no_latents(tmp_path):
    trainer = _FakeTrainer()
    trainer.default_root_dir = str(tmp_path)
    cb = PlotLatentCallback(latent_variables=[0, 1])
    cb.on_validation_epoch_start(trainer, _FakeModule())
    # No on_validation_batch_end calls: _z stays empty.
    cb.on_validation_epoch_end(trainer, _FakeModule())
    assert not os.path.exists(
        os.path.join(tmp_path, f"val_latent_hist_epoch{0}.png")
    )
