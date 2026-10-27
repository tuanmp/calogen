from importlib import import_module
from pathlib import Path

import h5py
import torch
import yaml
from tqdm import tqdm


def _class_from_str(class_path: str):
    """Get a class from a string representation of its path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def create_h5_handler(h5_path: str | Path, mode: str = "a"):
    """Create an HDF5 file handler."""
    h5_path = Path(h5_path)
    if not h5_path.parent.exists():
        h5_path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(h5_path, mode)

class BaseSampler:
    def __init__(self, config: str | Path, model_path: str | Path, device: str = "cuda"):

        # config to train the NCE model
        self.config = config if isinstance(config, Path) else Path(config)
        self.model_path = model_path
        self.device = device

        # open config to get the model parameters
        self.cfg = self.load_config()

        self.model = self.load_model()

        self.datamodule = self.load_data_module()

    def load_config(self):
        if not self.config.exists():
            raise FileNotFoundError(f"Config file {self.config} does not exist.")
        with open(self.config, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg

    def load_model(self):
        model_class_path = self.cfg["model"]["class_path"]
        model_class = _class_from_str(model_class_path)
        init_args = dict(self.cfg["model"].get("init_args") or {})
        # Weights always come from ``model_path``; never let the config silently
        # point the sampler at a different classifier.
        init_args.pop("classifier_ckpt_path", None)

        print(f"Loading model from {self.model_path}...")
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters") or {}
        saved_class = (
            hparams.get("_class_path")
            if isinstance(hparams, dict)
            else getattr(hparams, "_class_path", None)
        )

        if saved_class == model_class_path:
            model = model_class.load_from_checkpoint(self.model_path, **init_args)
        else:
            # Checkpoint from another repo (e.g. caloxtreme_clf): its
            # ``_class_path`` / ``_instantiator`` cannot build a calogen class,
            # so instantiate from the config and load the weights directly.
            print(f"  foreign checkpoint ({saved_class}); loading weights only")
            model = model_class(**init_args)
            if hasattr(model, "load_classifier_weights"):
                model.load_classifier_weights(self.model_path)
            else:
                from calogen.classifiers.base import load_state_dict_from_checkpoint

                load_state_dict_from_checkpoint(model, self.model_path)

        model.to(self.device)
        model.setup(stage="fit")
        return model

    def load_data_module(self):
        data_cfg = self.cfg["data"]["init_args"]
        data_module_class_path = self.cfg["data"]["class_path"]
        data_module_class = _class_from_str(data_module_class_path)
        print(f"Loading data module {data_module_class_path}...")
        data_module = data_module_class(**data_cfg)
        data_module.setup()
        assert data_module.train_dataloader() is not None, "No train dataloader found. Please check your data module."
        assert data_module.val_dataloader() is not None, "No validation dataloader found. Please check your data module."
        assert data_module.test_dataloader() is not None, "No test dataloader found. Please check your data module."
        return data_module

    def collect_conditions(self, loader: str = "test", max_samples: int | None = None) -> torch.Tensor:
        """Collect conditions from the dataloader.

        Datamodules may expose ``condition_scale`` when their batches carry
        conditions in a different unit than the generator expects (e.g. raw
        MeV truth energies); the returned tensor is always in generator units
        (GeV).
        """
        match loader:
            case "train":
                dataloader = self.datamodule.train_dataloader()
            case "val":
                dataloader = self.datamodule.val_dataloader()
            case "test":
                dataloader = self.datamodule.test_dataloader()
            case _:
                raise ValueError(f"Unknown loader type: {loader}. Must be one of 'train', 'val', or 'test'.")
        scale = float(getattr(self.datamodule, "condition_scale", 1.0))
        conditions = []
        for batch in tqdm(dataloader, desc=f"Collecting conditions from {loader} dataloader"):
            if isinstance(batch, (list, tuple)):
                c = batch[1]
            elif isinstance(batch, dict):
                c = batch["c"]
            conditions.append(c * scale)
            if max_samples is not None and sum(len(c) for c in conditions) >= max_samples:
                break
        return torch.cat(conditions, dim=0)

