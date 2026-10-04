from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms

from patchguard.attack import generate_adversarial_img
from patchguard.model import BasicCNN
from patchguard.predict import patchguard_batch_predict
from patchguard.windows import extract_windows, extract_windows_batch
from range_device import select_device


SEED = 20260927
TRAIN_COUNT = 3000
TEST_COUNT = 500
BATCH_SIZE = 10
WINDOW_SIZE = 16
STRIDE = 2
PATCH_SIZE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(root: str):
    transform = transforms.ToTensor()
    train = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    train_images = torch.stack([train[i][0] for i in range(TRAIN_COUNT)])
    train_labels = torch.tensor([train[i][1] for i in range(TRAIN_COUNT)])
    test_images = torch.stack([test[i][0] for i in range(TEST_COUNT)])
    test_labels = torch.tensor([test[i][1] for i in range(TEST_COUNT)])
    return train_images, train_labels, test_images, test_labels


def train_baseline(model, images, labels, epochs: int, device: torch.device) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    synchronize(images)
    started = time.perf_counter()
    n = images.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=images.device)
        for start in range(0, n, BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            inputs = images[indices].to(device)
            targets = labels[indices].to(device)
            loss = F.cross_entropy(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    synchronize(images)
    return time.perf_counter() - started


def train_window_model(model, images, labels, epochs: int, device: torch.device) -> float:
    """One Adam update per 10 source images, over their 490 cropped windows."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    synchronize(images)
    started = time.perf_counter()
    n = images.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, device=images.device)
        for start in range(0, n, BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            source_batch = images[indices]
            targets = labels[indices]
            windows = extract_windows_batch(source_batch, WINDOW_SIZE, STRIDE)
            batch, count, channels, height, width = windows.shape
            inputs = windows.reshape(batch * count, channels, height, width).to(device)
            window_labels = targets[:, None].expand(batch, count).reshape(-1).to(device)
            loss = F.cross_entropy(model(inputs), window_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    synchronize(images)
    return time.perf_counter() - started


def classifier_accuracy(model, images, labels, batch_size: int = 128) -> float:
    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            logits = model(images[start : start + batch_size])
            correct += int((logits.argmax(dim=1) == labels[start : start + batch_size]).sum())
    return correct / images.shape[0]


def source_adversarial_images(model, images, labels) -> torch.Tensor:
    batches = []
    # Independent per-image patch gradients make batching an exact memory bound.
    for start in range(0, images.shape[0], 25):
        batches.append(
            generate_adversarial_img(
                model,
                images[start : start + 25],
                labels[start : start + 25],
                PATCH_SIZE,
                patch_x=12,
                patch_y=12,
            )
        )
    return torch.cat(batches)


def synchronize(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def extraction_benchmark(images: torch.Tensor) -> dict[str, float]:
    sample = images[:500]
    synchronize(sample)
    started = time.perf_counter()
    for image in sample:
        extract_windows(image, WINDOW_SIZE, STRIDE)
    synchronize(sample)
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    extract_windows_batch(sample, WINDOW_SIZE, STRIDE)
    synchronize(sample)
    vectorized_seconds = time.perf_counter() - started
    return {
        "reference_python_500_images_seconds": reference_seconds,
        "vectorized_500_images_seconds": vectorized_seconds,
        "speedup": reference_seconds / max(vectorized_seconds, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", default="/opt/mnist")
    parser.add_argument("--output-dir", default="/opt/artifacts/patchguard")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = select_device()
    train_images, train_labels, test_images, test_labels = load_data(args.mnist_root)
    train_images, train_labels, test_images, test_labels = (
        tensor.to(device) for tensor in (train_images, train_labels, test_images, test_labels)
    )

    seed_everything(SEED)
    baseline = BasicCNN(patch_size=28, in_channels=1).to(device)
    baseline_train_seconds = train_baseline(baseline, train_images, train_labels, 30, device)
    baseline.eval()
    torch.save(baseline.state_dict(), out / "baseline_model.pt")
    baseline_clean = classifier_accuracy(baseline, test_images, test_labels)
    adversarial_test = source_adversarial_images(baseline, test_images, test_labels)
    baseline_adv = classifier_accuracy(baseline, adversarial_test, test_labels)

    seed_everything(SEED + 1)
    reference = BasicCNN(patch_size=WINDOW_SIZE, in_channels=1).to(device)
    reference_train_seconds = train_window_model(
        reference, train_images, train_labels, 100, device
    )
    reference.eval()
    torch.save(reference.state_dict(), out / "reference_model.pt")
    reference_clean = float(
        (patchguard_batch_predict(reference, test_images, WINDOW_SIZE, STRIDE) == test_labels)
        .float()
        .mean()
    )
    reference_source_adv = float(
        (patchguard_batch_predict(reference, adversarial_test, WINDOW_SIZE, STRIDE) == test_labels)
        .float()
        .mean()
    )
    seed_everything(SEED + 2)
    random_model = BasicCNN(patch_size=WINDOW_SIZE, in_channels=1).to(device)
    torch.save(random_model.state_dict(), out / "random_model.pt")
    seed_everything(SEED + 3)
    weak_model = BasicCNN(patch_size=WINDOW_SIZE, in_channels=1).to(device)
    weak_training_seconds = train_window_model(
        weak_model, train_images, train_labels, 1, device
    )
    torch.save(weak_model.state_dict(), out / "weak_model.pt")
    weak_clean = float(
        (patchguard_batch_predict(weak_model, test_images, WINDOW_SIZE, STRIDE) == test_labels)
        .float()
        .mean()
    )
    metrics = {
        "seed": SEED,
        "device": str(device),
        "training_examples": TRAIN_COUNT,
        "source_test_examples": TEST_COUNT,
        "baseline_epochs": 30,
        "window_model_epochs": 100,
        "patch_size": PATCH_SIZE,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "default_source_patch_top_left": [2, 2],
        "baseline_training_seconds": baseline_train_seconds,
        "window_model_training_seconds": reference_train_seconds,
        "baseline_clean_accuracy": baseline_clean,
        "baseline_adversarial_accuracy": baseline_adv,
        "reference_clean_accuracy": reference_clean,
        "reference_source_adversarial_accuracy": reference_source_adv,
        "weak_one_epoch_training_seconds": weak_training_seconds,
        "weak_clean_accuracy": weak_clean,
        "window_extraction_benchmark": extraction_benchmark(train_images),
    }
    (out / "source_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
