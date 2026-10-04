"""Shared device selection for evaluators and participant notebooks."""

import os

import torch


def select_device(requested: str | None = None) -> torch.device:
    choice = (requested or os.getenv("TORCH_DEVICE", "auto")).strip().lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(choice)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("TORCH_DEVICE must be auto, cpu, cuda, or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Check the CUDA PyTorch build and Docker GPU access.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device does not exist")
        # Preserve the calibrated float32 scoring behavior across CPU and GPU.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    return device


def describe_device(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
