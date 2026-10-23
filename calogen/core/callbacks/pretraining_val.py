# Callback to run a pretraining validation pass

import os

import torch
from calogen.core.plotting import plot_latent_histo
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info
from tqdm import tqdm


class PretrainingValidationCallback(Callback):

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == 0:
            rank_zero_info("📈 Running pretraining validation pass...")
            latent = []
            losses = []
            dataloader = trainer.val_dataloaders[0] if \
                isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            pl_module.eval()
            with torch.inference_mode():
                for i, batch in tqdm(enumerate(dataloader)):
                    batch = [x.to(pl_module.device) for x in batch]
                    dim = batch[0].shape[1]
                    outputs = pl_module.validation_step(batch, i)
                    loss = outputs["loss"]
                    z = outputs["z"]
                    latent.append(z)
                    losses.append(loss)
                loss = torch.stack(losses).mean().item()
                latent = torch.cat(latent, dim=0).cpu().numpy()
                if getattr(trainer, "log_dir") is not None:
                    save_path = os.path.join(trainer.log_dir, f"pretraining_latent_histogram_pretraining_perdim_loss_{loss / dim:.4f}.png")
                else:
                    save_path = f"./pretraining_latent_histogram_pretraining_perdim_loss_{loss / dim:.4f}.png"
                plot_latent_histo(
                    latent_array=latent,
                    save_path=save_path,
                    latent_variables=pl_module.inspect_dimensions,
                    bins=100,
                    xrange=(-4, 4),
                )
            pl_module.train()
            rank_zero_info(f"📊 Saved latent histogram to {save_path}")


        return super().on_train_start(trainer, pl_module)
