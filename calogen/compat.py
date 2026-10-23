"""Legacy module-name aliases for loading ``caloinn_lightning`` checkpoints.

LightningCLI checkpoints embed ``_instantiator`` / ``_class_path`` entries such
as ``lightning_module.CaloINNLightningModule``.  Registering those legacy
top-level names in :data:`sys.modules` lets ``importlib`` / jsonargparse resolve
them to the equivalent ``calogen`` classes, so existing generator checkpoints
load unchanged.  Import this module for its side effect before loading such
checkpoints.
"""

from __future__ import annotations

import sys

from calogen.core import data_util as _data_util
from calogen.core import lemurs as _lemurs
from calogen.core import postprocess as _torch_postprocess
from calogen.core import sharded as _sharded
from calogen.core import xml_handler as _xml_handler
from calogen.data.datamodules import caloinn as _caloinn_datamodule
from calogen.models.caloinn import classifier as _caloinn_classifier
from calogen.models.caloinn import lightning as _caloinn_lightning
from calogen.models.icaloflow import classifier as _icaloflow_classifier
from calogen.models.icaloflow import datamodule as _icaloflow_datamodule
from calogen.models.icaloflow import lightning as _icaloflow_lightning
from calogen.models.icaloflow import model as _icaloflow_model

_ALIASES = {
    # generator model / datamodule module paths
    "lightning_module": _caloinn_lightning,
    "lightning_data": _caloinn_datamodule,
    "icaloflow_lightning": _icaloflow_lightning,
    "icaloflow_data": _icaloflow_datamodule,
    "icaloflow_model": _icaloflow_model,
    # classifier module paths
    "classifier": _caloinn_classifier,
    "icaloflow_classifier": _icaloflow_classifier,
    # shared low-level modules referenced by old checkpoints
    "lemurs_data": _lemurs,
    "sharded_data": _sharded,
    "XMLHandler": _xml_handler,
    "data_util": _data_util,
    "torch_postprocess": _torch_postprocess,
}

for _name, _mod in _ALIASES.items():
    sys.modules.setdefault(_name, _mod)

del _name, _mod
