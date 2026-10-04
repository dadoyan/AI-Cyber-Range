"""Small, deterministic image transforms used by the X-ray Blue team."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


PREPROCESSORS = {"none", "median3", "gaussian", "quantize"}


def preprocess_batch(images: np.ndarray, name: str, bits: int = 5) -> np.ndarray:
    """Apply an allow-listed preprocessing transform to NCHW RGB images."""
    if name not in PREPROCESSORS:
        raise ValueError(f"preprocess must be one of: {', '.join(sorted(PREPROCESSORS))}")
    x = np.asarray(images, dtype=np.float32)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    x = np.clip(x, 0.0, 1.0)
    if name == "none":
        return x.copy()
    if name == "quantize":
        if isinstance(bits, bool) or not 2 <= int(bits) <= 8:
            raise ValueError("quantize bits must be between 2 and 8")
        levels = (1 << int(bits)) - 1
        return (np.round(x * levels) / levels).astype(np.float32)

    result = np.empty_like(x)
    for index, image in enumerate(x):
        pixels = np.rint(image.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(pixels, mode="RGB")
        if name == "median3":
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
        else:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.6))
        result[index] = np.asarray(pil_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return result


def prediction_consistency(
    classifier,
    images: np.ndarray,
    *,
    samples: int = 4,
    sigma: float = 0.03,
    threshold: float = 0.75,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return majority labels, vote confidence, and low-consensus flags."""
    if isinstance(samples, bool) or not 2 <= int(samples) <= 8:
        raise ValueError("samples must be between 2 and 8")
    if not np.isfinite(sigma) or not 0.0 <= float(sigma) <= 0.1:
        raise ValueError("sigma must be between 0 and 0.1")
    if not np.isfinite(threshold) or not 0.5 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0.5 and 1.0")

    x = np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0)
    generator = np.random.default_rng(2026)
    predictions = []
    for _ in range(int(samples)):
        noise = generator.normal(0.0, float(sigma), size=x.shape).astype(np.float32)
        probabilities = classifier.predict(np.clip(x + noise, 0.0, 1.0), batch_size=batch_size)
        predictions.append(np.argmax(probabilities, axis=1))

    votes = np.stack(predictions, axis=0)
    class_count = int(np.max(votes)) + 1
    counts = np.zeros((len(x), class_count), dtype=np.int32)
    for sample_index in range(votes.shape[0]):
        counts[np.arange(len(x)), votes[sample_index]] += 1
    labels = counts.argmax(axis=1)
    confidence = counts.max(axis=1).astype(np.float32) / float(samples)
    flagged = confidence < float(threshold)
    return labels, confidence, flagged


def randomized_smoothing(
    classifier,
    images: np.ndarray,
    *,
    samples: int,
    sigma: float,
    seed: int,
    batch_size: int = 8,
    max_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return majority labels and vote confidence for bounded Gaussian views.

    This uses the same noisy-view voting method as the static exercise's
    prediction-consistency detector, with a caller-supplied reproducible seed
    and a higher cap for the interactive Arena experiment.
    """
    if isinstance(samples, bool) or not 1 <= int(samples) <= int(max_samples):
        raise ValueError(f"samples must be between 1 and {max_samples}")
    if not np.isfinite(sigma) or not 0.0 <= float(sigma) <= 0.1:
        raise ValueError("sigma must be between 0 and 0.1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(batch_size, bool) or not 1 <= int(batch_size) <= 64:
        raise ValueError("batch_size must be between 1 and 64")

    x = np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    generator = np.random.default_rng(int(seed))
    votes: list[np.ndarray] = []
    # Keep memory bounded and submit small batches to the CPU classifier.
    remaining = int(samples)
    while remaining:
        count = min(int(batch_size), remaining)
        noisy = np.repeat(x, count, axis=0)
        noise = generator.normal(0.0, float(sigma), size=noisy.shape).astype(np.float32)
        probabilities = classifier.predict(
            np.clip(noisy + noise, 0.0, 1.0), batch_size=int(batch_size)
        )
        predictions = np.argmax(probabilities, axis=1).reshape(len(x), count)
        votes.extend(predictions[:, index] for index in range(count))
        remaining -= count

    vote_array = np.stack(votes, axis=0)
    class_count = int(np.max(vote_array)) + 1
    counts = np.zeros((len(x), class_count), dtype=np.int32)
    for sample_index in range(vote_array.shape[0]):
        counts[np.arange(len(x)), vote_array[sample_index]] += 1
    return counts.argmax(axis=1), counts.max(axis=1).astype(np.float32) / float(samples)
