from __future__ import annotations

import torch
from torch import Tensor


def _validate_image_and_window(img: Tensor, window_size: int, stride: int) -> None:
    if img.ndim != 3:
        raise ValueError("img must have shape [channels, height, width]")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if window_size > img.shape[-2] or window_size > img.shape[-1]:
        raise ValueError("window_size must fit inside the image")


def extract_windows_reference(img: Tensor, window_size: int, stride: int) -> Tensor:
    """Small, direct reference implementation in row-major crop order."""
    _validate_image_and_window(img, window_size, stride)
    windows = []
    for top in range(0, img.shape[-2] - window_size + 1, stride):
        for left in range(0, img.shape[-1] - window_size + 1, stride):
            windows.append(img[:, top : top + window_size, left : left + window_size])
    return torch.stack(windows)


def extract_windows(img: Tensor, window_size: int, stride: int) -> Tensor:
    """Vectorized equivalent of the notebook's sliding crop operation.

    Output order is row-major: all horizontal positions for the top row, then
    the next vertical position. The returned values are cropped windows; pixels
    outside a crop are not represented.
    """
    _validate_image_and_window(img, window_size, stride)
    view = img.unfold(1, window_size, stride).unfold(2, window_size, stride)
    rows, cols = view.shape[1:3]
    return view.permute(1, 2, 0, 3, 4).reshape(
        rows * cols, img.shape[0], window_size, window_size
    )


def extract_windows_batch(images: Tensor, window_size: int, stride: int) -> Tensor:
    """Return [batch, windows, channels, window, window] sliding crops."""
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if images.shape[0] == 0:
        return images.new_empty((0, 0, images.shape[1], window_size, window_size))
    _validate_image_and_window(images[0], window_size, stride)
    view = images.unfold(2, window_size, stride).unfold(3, window_size, stride)
    batch, channels, rows, cols, _, _ = view.shape
    return view.permute(0, 2, 3, 1, 4, 5).reshape(
        batch, rows * cols, channels, window_size, window_size
    )
