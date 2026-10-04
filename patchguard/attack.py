from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def generate_adversarial_img_at(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    patch_size: int,
    top: int,
    left: int,
    steps: int = 20,
    step_size: float = 8 / 255,
) -> Tensor:
    """Source-derived per-image untargeted CE-ascent patch attack."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    _, _, height, width = images.shape
    if patch_size <= 0 or top < 0 or left < 0 or top + patch_size > height or left + patch_size > width:
        raise ValueError("patch location must fit inside each image")
    model.eval()
    base = images.detach()
    patch = torch.zeros(
        images.shape[0], images.shape[1], patch_size, patch_size,
        device=images.device,
    )
    row_slice = slice(top, top + patch_size)
    col_slice = slice(left, left + patch_size)
    batch_slice = (slice(None), slice(None), row_slice, col_slice)
    for _ in range(steps):
        patch.requires_grad_(True)
        patched = base.clone()
        patched[batch_slice] = (base[batch_slice] + patch).clamp(0, 1)
        loss = F.cross_entropy(model(patched), labels)
        gradient, = torch.autograd.grad(loss, patch)
        with torch.no_grad():
            patch = (patch + step_size * gradient.sign()).clamp(-1.0, 1.0)
    adversarial = base.clone()
    adversarial[batch_slice] = (base[batch_slice] + patch).clamp(0, 1)
    return adversarial


def generate_adversarial_img(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    patch_size: int,
    patch_x: int,
    patch_y: int,
    steps: int = 20,
    step_size: float = 8 / 255,
) -> Tensor:
    """Keep the notebook's offset-from-center coordinate convention."""
    height, width = images.shape[-2:]
    top = height // 2 - patch_y
    left = width // 2 - patch_x
    return generate_adversarial_img_at(
        model, images, labels, patch_size, top, left, steps, step_size
    )
