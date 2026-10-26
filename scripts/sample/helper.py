


import torch
import yaml

from calogen.sampler.base import _class_from_str


def initialize_gen(config, checkpoint, device):
    """Initialize the generator model from the config and checkpoint."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    model_class_path = cfg["model"]["class_path"]
    model_class = _class_from_str(model_class_path)
    print(f"Loading model from {checkpoint}...")
    # model = model_class.load_from_checkpoint(checkpoint, weights_only=True, strict=False, **cfg["model"]["init_args"])
    model = model_class(**cfg["model"]["init_args"])
    sd = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd["state_dict"], strict=False)
    model.to(device)
    model.setup(stage="val")

    print(f"Loading data module {cfg['data']['class_path']}...")
    data_module_class_path = cfg["data"]["class_path"]
    data_module_class = _class_from_str(data_module_class_path)
    data_module = data_module_class(**cfg["data"]["init_args"])
    return model, data_module

def resize_h5_and_add(h5_handler, dataset_name, new_data):
    """Resize an HDF5 dataset and add new data to it."""
    if new_data.shape[0] == 0:
        return
    if dataset_name not in h5_handler:
        # Create the dataset if it doesn't exist
        maxshape = (None,) + new_data.shape[1:]  # Allow unlimited growth in the first dimension
        h5_handler.create_dataset(dataset_name, data=new_data, maxshape=maxshape)
    else:
        # Resize the existing dataset and append new data
        dataset = h5_handler[dataset_name]
        current_size = dataset.shape[0]
        new_size = current_size + new_data.shape[0]
        dataset.resize((new_size,) + dataset.shape[1:])
        dataset[current_size:new_size] = new_data

def inv_transform_caloinn(x, c, truth_format):
    """Inverse transform for CaloINN decoded output.

    ``x`` is the flattened decoded shower and ``c`` the incident energy, both
    in GeV.  Returns the shower on the truth grid and the energy, both in MeV,
    using the truth-file dataset conventions.
    """
    # Convert to MeV
    incident_energy_mev = c * 1.e3
    showers_mev = x* 1.e3

    # Apply truth format

    showers_mev = truth_format.unflatten_showers(showers_mev)

    energy_out = truth_format.format_energy(incident_energy_mev)

    return showers_mev, energy_out