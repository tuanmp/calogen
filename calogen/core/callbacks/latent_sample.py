import lightning as L
import torch

from calogen.core import plotting


class LatentSampler(L.pytorch.callbacks.Callback):

    def __init__(self, sampling_interval: int):
        super().__init__()
        self.sampling_interval = sampling_interval


    def on_train_epoch_start(self, trainer, pl_module):
        
        if trainer.current_epoch % self.sampling_interval != 0:
            return
        
        pl_module.eval()
        self.latent_samples(trainer, pl_module)
        pl_module.train()

    @torch.no_grad()
    def latent_samples(self, trainer, pl_module):
        """
            Plot latent space distribution. 

            Parameters:
            epoch (int): current epoch
        """

        samples = []

        for x, c in trainer.train_dataloader:
            x = x.to(pl_module.device)
            c = c.to(pl_module.device)
            sample = pl_module.model(x, c)[0].cpu()
            samples.append(sample)

        samples = torch.concat(samples).numpy()
        plotting.plot_latent(samples, trainer.default_root_dir, trainer.current_epoch)