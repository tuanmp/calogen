import torch

# Force float32 before anything instantiates the CINN model (legacy requirement).
torch.set_default_dtype(torch.float32)

try:
    from lightning.pytorch.cli import LightningCLI
except ImportError:
    from pytorch_lightning.cli import LightningCLI

from calogen.core.trainer import Trainer

# Register legacy checkpoint module aliases (side-effect import).
import calogen.compat  # noqa: F401,E402  (legacy checkpoint module aliases)


def main():
    LightningCLI(
        trainer_class=Trainer,
        auto_configure_optimizers=False,
        trainer_defaults={"num_sanity_val_steps": 0},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
