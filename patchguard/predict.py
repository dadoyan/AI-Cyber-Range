from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .windows import extract_windows, extract_windows_batch

NUM_CLASSES = 10


def get_window_predictions(model: nn.Module, windows: Tensor) -> Tensor:
    with torch.no_grad():
        return model(windows).argmax(dim=1)


def patchguard_predict(
    model: nn.Module, img: Tensor, window_size: int = 16, stride: int = 2
) -> int:
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = img.device
    windows = extract_windows(img, window_size, stride).to(device)
    predictions = get_window_predictions(model, windows)
    # torch.argmax returns the first maximum, so ties select the smallest class.
    return int(torch.bincount(predictions, minlength=NUM_CLASSES).argmax().item())


def patchguard_accuracy(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    window_size: int = 16,
    stride: int = 2,
) -> float:
    if images.shape[0] == 0:
        raise ValueError("accuracy is undefined for an empty dataset")
    model.eval()
    correct = 0
    for i, image in enumerate(images):
        correct += int(patchguard_predict(model, image, window_size, stride) == int(labels[i]))
    return correct / images.shape[0]


def patchguard_batch_predict(
    model: nn.Module,
    images: Tensor,
    window_size: int = 16,
    stride: int = 2,
    inference_batch_size: int = 512,
) -> Tensor:
    """Vectorized image-batch evaluator used by the protected service."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if images.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=images.device)
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = images.device
    window_batches = extract_windows_batch(images, window_size, stride)
    batch, num_windows, channels, height, width = window_batches.shape
    flat = window_batches.reshape(batch * num_windows, channels, height, width)
    window_predictions = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], inference_batch_size):
            logits = model(flat[start : start + inference_batch_size].to(model_device))
            window_predictions.append(logits.argmax(dim=1).to(images.device))
    votes = F.one_hot(
        torch.cat(window_predictions).reshape(batch, num_windows),
        num_classes=NUM_CLASSES,
    ).sum(dim=1)
    return votes.argmax(dim=1)
