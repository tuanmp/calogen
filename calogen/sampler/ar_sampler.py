from pathlib import Path

import torch

from calogen.models.caloinn.lightning import CaloINNLightningModule
from calogen.sampler.base import BaseSampler


class WhitenedNCESampler(BaseSampler):
    def __init__(self, config: str | Path, model_path: str | Path, device: str = "cuda"):
        super().__init__(config, model_path, device)

    def sample_from_base(self, num_samples: int):
        """Sample from the base distribution (whitened latent space)."""
        return self.model.q0.sample((num_samples, )).to(self.device)

    @torch.inference_mode()
    def sample(self, c: torch.Tensor, Z: float, eval_mode: bool = False):
        """Sample from the model given conditions c.

        Proposals are drawn in the whitened base space ``w ~ q0 = N(0, I)`` and
        the classifier scores ``[w, c]``.  The returned ``"z"`` is mapped back
        to the generator latent space (``unwhiten(w)``) so it matches the truth
        latents stored by the offline extraction pass.
        """
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()
        num_samples = c.shape[0]
        w = self.sample_from_base(num_samples).to(c.device)
        x_in = torch.cat([w, c], dim=1).to(self.device)
        logits = self.model(x_in)
        survive_prob = torch.clamp(torch.exp(logits), max=Z) / Z
        mask = torch.rand(num_samples, device=c.device) < survive_prob.squeeze()
        z = self.model.unwhiten(w)
        return {
            "z": z,
            "w": w,
            "c": c,
            "survive_prob": survive_prob,
            "mask": mask,
            "logits": logits,
        }

    @torch.inference_mode()
    def decode(self, gen: CaloINNLightningModule, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode generator latents ``z`` at conditions ``c`` to flat showers."""
        gen.eval()
        x_decoded = gen.decode_latent(z, c)
        data = gen.post_process_decoded(x_decoded, c)
        num_layers = len(gen.layer_boundaries) - 1
        return torch.cat([data[f"layer_{i}"] for i in range(num_layers)], dim=1)

    @torch.inference_mode()
    def sample_and_decode(self, gen: CaloINNLightningModule, c: torch.Tensor, Z: float, eval_mode: bool = False):
        """Sample from the model given conditions c and decode to original space."""
        sample_dict = self.sample(c, Z, eval_mode)
        mask = sample_dict["mask"]
        z = sample_dict["z"][mask]
        c = sample_dict["c"][mask]
        if mask.sum() == 0:
            shower = None
        else:
            shower = self.decode(gen, z, c)

        sample_dict["shower"] = shower
        return sample_dict


class InternalShowerSampler(BaseSampler):
    def __init__(self, config: str | Path, model_path: str | Path, device: str = "cuda"):
        super().__init__(config, model_path, device)

    @torch.inference_mode()
    def sample(self, c: torch.Tensor, Z: float, eval_mode: bool = False):
        """Sample from the generator given conditions c."""
        if eval_mode:
            self.model.eval()
        else:
            self.model.train()
        num_samples = c.shape[0]
        # z = self.model.sample_latent(num_samples).to(c.device)
        x_gen = self.model.generator.conditional_generate(c, measure_gen_time=False).squeeze(1)
        density_ratio = self.model.estimate_density_ratio(x_gen, c)
        logits = torch.log(density_ratio)
        survival_prob = torch.clamp(density_ratio, max=Z) / Z
        mask = torch.rand(num_samples, device=c.device) < survival_prob.squeeze()

        return {
            "x_gen": x_gen,
            "c": c,
            "logits": logits,
            "survival_prob": survival_prob,
            "mask": mask,
        }

    @torch.inference_mode()
    def decode(self, x_gen: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode generator latents ``x_gen`` at conditions ``c`` to flat showers."""
        data = self.model.generator.post_process_decoded(x_gen, c)
        num_layers = len(self.model.generator.layer_boundaries) - 1
        return torch.cat([data[f"layer_{i}"] for i in range(num_layers)], dim=1)

    @torch.inference_mode()
    def sample_and_decode(self, c: torch.Tensor, Z: float, eval_mode: bool = False):
        """Sample from the generator given conditions c and decode to original space."""
        sample_dict = self.sample(c, Z, eval_mode)
        mask = sample_dict["mask"]
        x_gen = sample_dict["x_gen"][mask]
        c = sample_dict["c"][mask]
        if mask.sum() == 0:
            shower = None
        else:
            shower = self.decode(x_gen, c)

        sample_dict["shower"] = shower
        return sample_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whitened NCE Sampler")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the sampler on.")
    args = parser.parse_args()

    sampler = WhitenedNCESampler(args.config, args.model_path, args.device)
    print("Sampler initialized successfully.")