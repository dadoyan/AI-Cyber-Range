from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from patchguard.attack import generate_adversarial_img_at
from patchguard.model import BasicCNN


SEED = 20260927
PATCH_SIZE = 12
HIDDEN_START = 500
HIDDEN_COUNT = 100
HIDDEN_LOCATIONS = [(0, 0), (0, 16), (16, 0), (16, 16), (8, 8)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-source", default="/opt/private/patchguard/hidden_seed.pt")
    parser.add_argument("--reference", default="/opt/artifacts/patchguard/reference_model.pt")
    parser.add_argument("--baseline", default="/opt/artifacts/patchguard/baseline_model.pt")
    parser.add_argument("--output-dir", default="/opt/artifacts/patchguard/eval_assets")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed_everything(SEED + 2)
    hidden_source = torch.load(args.hidden_source, map_location="cpu", weights_only=True)
    images = hidden_source["images"].to(torch.float32).unsqueeze(1).div_(255.0)
    labels = hidden_source["labels"].to(torch.long)
    if images.shape[0] != HIDDEN_COUNT or labels.shape != (HIDDEN_COUNT,):
        raise RuntimeError("private hidden MNIST subset has an unexpected shape")
    baseline = BasicCNN(patch_size=28, in_channels=1)
    baseline.load_state_dict(torch.load(args.baseline, map_location="cpu", weights_only=True), strict=True)
    baseline.eval()
    adversarial_images = []
    adversarial_labels = []
    for top, left in HIDDEN_LOCATIONS:
        for start in range(0, HIDDEN_COUNT, 10):
            stop = min(start + 10, HIDDEN_COUNT)
            adv = generate_adversarial_img_at(
                baseline,
                images[start:stop],
                labels[start:stop],
                PATCH_SIZE,
                top,
                left,
                steps=20,
                step_size=8 / 255,
            )
            adversarial_images.append(adv.detach())
            adversarial_labels.append(labels[start:stop])
    payload = {
        "clean_images": images.contiguous(),
        "clean_labels": labels.contiguous(),
        "adversarial_images": torch.cat(adversarial_images).contiguous(),
        "adversarial_labels": torch.cat(adversarial_labels).contiguous(),
    }
    torch.save(payload, out / "hidden_eval.pt")
    torch.save(
        torch.load(args.reference, map_location="cpu", weights_only=True),
        out / "reference_model.pt",
    )
    metadata = {
        "seed": SEED,
        "mnist_test_indices": list(range(HIDDEN_START, HIDDEN_START + HIDDEN_COUNT)),
        "location_top_left_y_x": HIDDEN_LOCATIONS,
        "patch_size": PATCH_SIZE,
        "window_size": 16,
        "stride": 2,
        "attack": "untargeted CE maximization against fixed clean baseline",
        "steps": 20,
        "step_size": 8 / 255,
        "reference_filename": "reference_model.pt",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "clean_examples": len(images),
        "adversarial_examples": len(payload["adversarial_images"]),
        "hidden_location_count": len(HIDDEN_LOCATIONS),
        "hidden_assets": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
