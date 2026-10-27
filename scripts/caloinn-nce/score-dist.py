import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from calogen.eval.plots import plot_calibration, plot_score_histo
from calogen.sampler.base import _class_from_str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clf(config, checkpoint, device, batch_size=1024):
    """Initialize the classifier model from the config and checkpoint."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    model_class_path = cfg["model"]["class_path"]
    model_class = _class_from_str(model_class_path)
    print(f"Loading model from {checkpoint}...")
    model = model_class(**cfg["model"]["init_args"])
    if hasattr(model, "load_classifier_weights"):
        # Head-only load: works for checkpoints that do not embed the generator
        # (converted caloxtreme checkpoints) and for native ones.
        model.load_classifier_weights(checkpoint)
    else:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["state_dict"])
    model.to(device)
    model.eval()

    print(f"Loading data module {cfg['data']['class_path']}...")
    data_module_class_path = cfg["data"]["class_path"]
    data_module_class = _class_from_str(data_module_class_path)
    cfg["data"]["init_args"]["predict_batch_size"] = batch_size
    data_module = data_module_class(**cfg["data"]["init_args"])
    data_module.setup(stage="test")
    return model, data_module

@torch.inference_mode()
def get_score(clf, dm, eval_mode=False, multiplier=1):

    clf.train()
    if eval_mode:
        clf.eval()

    y_hat = []
    y = []
    for i in range(multiplier):
        for batch in tqdm(dm.test_dataloader(), desc=f"Scoring pass {i+1}", unit="batch"):
            batch = [b.to(device) for b in batch]
            X_batch, y_batch, _ = clf.get_input_from_batch(batch)

            y_hat.append(torch.sigmoid(clf(X_batch).cpu()).reshape(-1))
            y.append(y_batch.cpu().reshape(-1))

    return torch.cat(y_hat).numpy(), torch.cat(y).numpy()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clf-config", required=True,
                   help="Generation config (YAML) with model class_path and init_args.")
    p.add_argument("--clf-ckpt", required=True,
                   help="Path to the generative model's checkpoint.")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Decode batch size (default 1024).")
    p.add_argument("--out-dir", required=True, help="Output numpy array path.")
    p.add_argument("--eval-mode", action="store_true", help="Run in evaluation mode.")
    p.add_argument("--multiplier", type=int, default=1, help="Number of times to repeat the test dataset for prediction.")
    p.add_argument("--plot-only", action="store_true", help="Only plot the score histogram and calibration curve from existing scores.npz.")
    args = p.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "clf_config": args.clf_config,
        "clf_ckpt": args.clf_ckpt,
        "out_dir": args.out_dir,
        "eval_mode": args.eval_mode,
        "multiplier": args.multiplier,
    }

    if args.plot_only:
        assert (output_dir / "scores.npz").exists(), "scores.npz not found in the output directory."
        data = np.load(output_dir / "scores.npz")
        y_hat = data["y_hat"]
        y = data["y"]
    else:
        clf, dm = load_clf(args.clf_config, args.clf_ckpt, device, args.batch_size)

        y_hat, y = get_score(clf, dm, eval_mode=args.eval_mode, multiplier=args.multiplier)

        np.savez_compressed(output_dir / "scores.npz", y_hat=y_hat, y=y)

    plot_score_histo(y_hat, y, log_scale=True, save_path=output_dir / "score_histo.png")
    plot_calibration(y_hat, y, save_path=output_dir / "calibration.png")

    report["score_histo"] = str(output_dir / "score_histo.png")
    report["calibration"] = str(output_dir / "calibration.png")
    report["scores"] = str(output_dir / "scores.npz")

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()