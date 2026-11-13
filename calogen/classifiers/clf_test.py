from functools import partial

import torch

from calogen.models.caloinn.features.convert import showers_to_classifier_input_torch

from .base import Classifier


class HybirdClf(Classifier):

    def __init__(self, 
                 hidden_dim: int = 256, 
                 num_layers: int = 4, 
                 batch_norm: bool = False, 
                 layer_norm: bool = True, 
                 output_dim: int = 1, 
                 dropout: float = 0, 
                 activation: str = "relu", 
                 lr: float = 0.001, step_size: int = 10, gamma: float = 0.95, 
                 amsgrad: bool = True,
                 xml_path: str | None = None,
                 xml_ptype  : str | None = None,
                 log_transform: bool = False,
                 voxel_energy_cutoff: float = 0.0,
                 ):
        super().__init__(hidden_dim, num_layers, batch_norm, layer_norm, output_dim, dropout, activation, lr, step_size, gamma, amsgrad)

        self.convert_func = partial(
            showers_to_classifier_input_torch,
            xml_path=xml_path,
            particle=xml_ptype,
            log_transform=log_transform,
            voxel_energy_cutoff=voxel_energy_cutoff
        )

    def get_input_from_batch(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, c, y = batch

        X_input = self.convert_func(x, c)

        return X_input, y, None


