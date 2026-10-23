"""CaloINN shower -> classifier-input feature extraction (numpy + torch)."""

from calogen.models.caloinn.features.convert import (
    cinn_sample_to_classifier_input,
    cinn_sample_to_classifier_input_torch,
)

__all__ = ["cinn_sample_to_classifier_input", "cinn_sample_to_classifier_input_torch"]
