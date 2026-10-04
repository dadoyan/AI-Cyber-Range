from __future__ import annotations

import torch
from torch import nn


class BasicCNN(nn.Module):
    """The fixed LeNet-style classifier from the supplied MNIST notebook."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        channels: tuple[int, int] = (32, 64),
        fc_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if patch_size < 4:
            raise ValueError("patch_size must be at least 4")
        c1, c2 = channels
        feat_dim = c2 * (patch_size // 4) * (patch_size // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(feat_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
