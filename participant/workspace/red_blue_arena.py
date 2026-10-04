"""Notebook helpers for the paired X-Ray Red/Blue Arena exercise."""

from __future__ import annotations

import hashlib
from html import escape
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import requests
from PIL import Image
from range_device import select_device

ARENA_URL = os.getenv("ARENA_URL", "http://arena-service:5000").rstrip("/")
ARENA_TOKEN = os.getenv("ARENA_TOKEN") or os.getenv("JUPYTER_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {ARENA_TOKEN}"} if ARENA_TOKEN else {}
WORKSPACE = Path(__file__).resolve().parent
SOURCE_DIR = WORKSPACE / "arena_sources"
INBOX_DIR = WORKSPACE / "inbox"
MODEL_PATH = WORKSPACE / "arena_models" / "efficientnet_b0.pth"
_MODEL = None


def _request(method: str, path: str, **kwargs) -> requests.Response:
    if not ARENA_TOKEN:
        raise RuntimeError("No Arena workspace token was injected. Launch this notebook from your paired CTFd workspace.")
    response = requests.request(method, f"{ARENA_URL}{path}", headers={**HEADERS, **kwargs.pop("headers", {})}, timeout=kwargs.pop("timeout", 180), **kwargs)
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Arena request failed ({response.status_code}): {detail}")
    return response


def identity() -> dict[str, Any]:
    """Show the server-assigned CTFd account, match, and role."""
    return _request("GET", "/api/identity").json()


def get_sources() -> dict[str, Any]:
    """List the evaluator-approved, correctly classified clean sources."""
    return _request("GET", "/api/sources").json()


def download_source(source_id: str | int, destination: str | Path | None = None) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(destination) if destination else SOURCE_DIR / f"source_{source_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    response = _request("GET", f"/api/sources/{source_id}")
    path.write_bytes(response.content)
    return path


def download_standard_model(destination: str | Path = MODEL_PATH) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(_request("GET", "/api/model").content)
    return path


def load_standard_model():
    """Load the protected standard EfficientNet-B0 checkpoint for local experiments."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0

    path = download_standard_model()
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    model.load_state_dict(checkpoint)

    class ImageNetNormalizedModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, images):
            return self.backbone((images - self.mean) / self.std)

    _MODEL = ImageNetNormalizedModel(model).to(select_device()).eval()
    return _MODEL


def _source_tensor(source_id: str | int):
    import torch

    path = download_source(source_id)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (256, 256):
            raise ValueError("The approved source image must be 256 by 256 pixels.")
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def predict_image(image: str | Path | Image.Image) -> dict[str, Any]:
    import torch

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            array = np.asarray(opened.convert("RGB"), dtype=np.float32).copy() / 255.0
    else:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    model = load_standard_model()
    tensor = tensor.to(next(model.parameters()).device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    labels = get_sources()["class_names"]
    return {"class_index": index, "class_label": labels[index], "probability": float(probabilities[index])}


def make_adversarial_candidate(
    source_id: str | int,
    *,
    method: str = "PGD",
    epsilon: float = 0.02,
    step_size: float = 0.0025,
    iterations: int = 8,
    sparse_fraction: float = 0.01,
    destination: str | Path | None = None,
) -> Path:
    """Create untargeted FGSM, sparse-FGSM, or PGD within the L-infinity budget."""
    import torch
    import torch.nn.functional as F

    if method.upper() not in {"FGSM", "SPARSE_FGSM", "PGD"}:
        raise ValueError("method must be FGSM, SPARSE_FGSM, or PGD")
    if not math.isfinite(epsilon) or not 0 < epsilon <= 0.02:
        raise ValueError("epsilon must be between 0 and 0.02")
    if not math.isfinite(step_size) or not 0 < step_size <= epsilon:
        raise ValueError("step_size must be positive and no larger than epsilon")
    if isinstance(iterations, bool) or not 1 <= int(iterations) <= 1000:
        raise ValueError("iterations must be between 1 and 1000")
    if not math.isfinite(sparse_fraction) or not 0 < sparse_fraction <= 1:
        raise ValueError("sparse_fraction must be between 0 and 1")

    info = next((item for item in get_sources()["sources"] if str(item["source_id"]) == str(source_id)), None)
    if info is None:
        raise ValueError("source_id is not in the approved source list")
    clean = _source_tensor(source_id)
    label = torch.tensor([int(info["class_index"])], dtype=torch.long)
    model = load_standard_model()
    device = next(model.parameters()).device
    clean, label = clean.to(device), label.to(device)
    # PNG stores integer RGB values; use the largest exactly representable
    # normalized budget that does not exceed the evaluator's 0.02 limit.
    pixel_budget = math.floor(epsilon * 255.0 + 1e-9) / 255.0
    if method.upper() in {"FGSM", "SPARSE_FGSM"}:
        adversarial = clean.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), label)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        direction = gradient.sign()
        if method.upper() == "SPARSE_FGSM":
            spatial_scores = gradient.abs().sum(dim=1).flatten()
            changed_pixels = max(1, round(float(sparse_fraction) * spatial_scores.numel()))
            chosen = torch.topk(spatial_scores, changed_pixels).indices
            spatial_mask = torch.zeros_like(spatial_scores)
            spatial_mask[chosen] = 1.0
            direction = direction * spatial_mask.reshape(1, 1, 256, 256)
        candidate = (clean + pixel_budget * direction).clamp(0, 1)
    else:
        candidate = clean + torch.empty_like(clean).uniform_(-pixel_budget, pixel_budget)
        candidate = candidate.clamp(0, 1)
        for _ in range(int(iterations)):
            candidate = candidate.detach().requires_grad_(True)
            loss = F.cross_entropy(model(candidate), label)
            gradient = torch.autograd.grad(loss, candidate)[0]
            candidate = candidate.detach() + step_size * gradient.sign()
            candidate = torch.maximum(torch.minimum(candidate, clean + pixel_budget), clean - pixel_budget).clamp(0, 1)

    pixels = candidate.detach()[0].permute(1, 2, 0).cpu().numpy()
    output = np.rint(np.clip(pixels, 0, 1) * 255.0).astype(np.uint8)
    path = Path(destination) if destination else WORKSPACE / f"arena_candidate_{source_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(path, format="PNG")
    return path


def submit_red_attack(
    image: str | Path,
    source_id: str | int,
    *,
    attack_name: str = "PGD",
    epsilon: float = 0.02,
    step_size: float = 0.0025,
    iterations: int = 8,
) -> dict[str, Any]:
    path = Path(image)
    raw = path.read_bytes()
    encoded = __import__("base64").b64encode(raw).decode("ascii")
    started = time.perf_counter()
    result = _request("POST", "/api/red/attacks", json={
        "source_id": str(source_id), "image_b64": encoded,
        "attack_name": attack_name, "epsilon": epsilon,
        "step_size": step_size, "iterations": iterations,
    }).json()
    result["request_elapsed_seconds"] = time.perf_counter() - started
    return result


def get_red_flag() -> str:
    """Recover the CTFd Red flag after this account has a valid Arena attack."""
    return _request("GET", "/api/red/flag").json()["flag"]


def get_pending_attack() -> dict[str, Any]:
    """Refresh the Blue inbox. A pending entry is the current unresolved attack."""
    return _request("GET", "/api/blue/pending").json()


def download_pending_attack() -> tuple[Path, dict[str, Any]]:
    info = get_pending_attack()
    if not info.get("pending"):
        raise RuntimeError("There is no pending Red attack in the Blue inbox.")
    response = _request("GET", info["artifact_url"])
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != info["sha256"] or response.headers.get("X-Artifact-SHA256") != digest:
        raise RuntimeError("Downloaded artifact SHA256 does not match the Arena record.")
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    path = INBOX_DIR / f"red_attack_{int(info['sequence_number']):03d}.png"
    path.write_bytes(response.content)
    return path, info


def local_randomized_smoothing(
    image: str | Path,
    *,
    sigma: float = 0.05,
    num_samples: int = 100,
    batch_size: int = 8,
    seed: int = 2026,
) -> dict[str, Any]:
    """Experiment locally; Arena independently repeats protected evaluation."""
    import torch

    if not math.isfinite(sigma) or not 0 <= sigma <= 0.1:
        raise ValueError("sigma must be between 0 and 0.1")
    if isinstance(num_samples, bool) or not 10 <= int(num_samples) <= 500:
        raise ValueError("num_samples must be between 10 and 500")
    with Image.open(image) as opened:
        pixels = np.asarray(opened.convert("RGB"), dtype=np.float32).copy() / 255.0
    clean = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    counts = torch.zeros(2, dtype=torch.int64)
    model = load_standard_model()
    remaining = int(num_samples)
    while remaining:
        count = min(int(batch_size), remaining)
        batch = clean.repeat(count, 1, 1, 1)
        noise = torch.randn(batch.shape, generator=generator) * float(sigma)
        with torch.inference_mode():
            inputs = (batch + noise).clamp(0, 1).to(next(model.parameters()).device)
            predictions = model(inputs).argmax(dim=1).cpu()
        counts += torch.bincount(predictions, minlength=2)
        remaining -= count
    index = int(counts.argmax())
    labels = get_sources()["class_names"]
    return {"class_index": index, "class_label": labels[index],
            "confidence": float(counts[index] / int(num_samples)),
            "votes": counts.tolist(), "sigma": float(sigma), "num_samples": int(num_samples)}


def submit_blue_response(
    attack_id: str,
    *,
    sigma: float = 0.05,
    num_samples: int = 100,
    abstain_threshold: float = 0.75,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = _request("POST", "/api/blue/responses", json={
        "attack_id": attack_id, "sigma": sigma,
        "num_samples": num_samples, "abstain_threshold": abstain_threshold,
    }).json()
    result["request_elapsed_seconds"] = time.perf_counter() - started
    return result


def get_blue_flag() -> str:
    """Recover the CTFd Blue flag after this account has a successful defense."""
    return _request("GET", "/api/blue/flag").json()["flag"]


def get_blue_response() -> dict[str, Any]:
    return _request("GET", "/api/red/latest-response").json()


def get_match_history() -> dict[str, Any]:
    return _request("GET", "/api/match/history").json()


def _info_card(
    title: str,
    fields: list[tuple[str, Any]],
    *,
    accent: str = "#64748b",
    extra_html: str = "",
    details: list[tuple[str, Any]] | None = None,
) -> str:
    field_html = "".join(
        "<div style='min-width:130px;flex:1;padding:9px 12px;background:#fff;"
        "border:1px solid #e2e8f0;border-radius:7px'>"
        f"<div style='font-size:12px;color:#64748b;margin-bottom:4px'>{escape(label)}</div>"
        f"<div style='font-weight:600;color:#172033'>{escape(str(value))}</div></div>"
        for label, value in fields
    )
    details_html = ""
    if details:
        detail_rows = "".join(
            "<div style='margin:4px 0;overflow-wrap:anywhere'>"
            f"<span style='color:#64748b'>{escape(label)}:</span> "
            f"<code>{escape(str(value))}</code></div>"
            for label, value in details if value is not None
        )
        details_html = (
            "<details style='margin-top:9px;color:#475569'><summary style='cursor:pointer'>"
            "Technical details</summary><div style='padding:7px 2px 0'>" + detail_rows + "</div></details>"
        )
    return (
        "<div style='border:1px solid #dbe2ea;border-left:5px solid " + accent +
        ";border-radius:8px;padding:12px 14px;margin:8px 0;background:#f8fafc'>"
        f"<div style='font-size:16px;font-weight:700;color:#172033;margin-bottom:9px'>{escape(title)}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px'>{field_html}</div>"
        f"{extra_html}{details_html}</div>"
    )


def _linf_meter_html(measured: Any, maximum: Any) -> str:
    if measured is None or maximum is None:
        return ""
    measured_value, maximum_value = float(measured), float(maximum)
    if maximum_value <= 0:
        return ""
    ratio = max(0.0, min(1.0, measured_value / maximum_value))
    fill = "#15803d" if measured_value <= maximum_value else "#dc2626"
    return (
        "<div style='margin-top:10px;padding:9px 11px;background:#fff;"
        "border:1px solid #e2e8f0;border-radius:7px'>"
        f"<div style='font-size:12px;color:#475569;margin-bottom:6px'>"
        f"Measured L∞: {measured_value:.6f} / exercise maximum {maximum_value:.6f}</div>"
        "<div style='height:9px;background:#e2e8f0;border-radius:99px;overflow:hidden'>"
        f"<div style='width:{ratio * 100:.2f}%;height:100%;background:{fill};border-radius:99px'></div>"
        "</div></div>"
    )


def _confidence_panel(label: str, prediction: dict[str, Any], color: str) -> str:
    probability = prediction.get("probability")
    if probability is None:
        return ""
    probability = max(0.0, min(1.0, float(probability)))
    return (
        "<div style='flex:1;min-width:190px;padding:10px;background:#fff;"
        "border:1px solid #e2e8f0;border-radius:7px'>"
        f"<div style='font-size:12px;color:#64748b'>{escape(label)}</div>"
        f"<div style='font-weight:700;color:#172033;margin:3px 0'>{escape(str(prediction.get('class_label', '—')))} · {probability:.1%}</div>"
        "<div style='height:9px;background:#e2e8f0;border-radius:99px;overflow:hidden'>"
        f"<div style='width:{probability * 100:.2f}%;height:100%;background:{color};border-radius:99px'></div>"
        "</div></div>"
    )


def attack_result_card_html(result: dict[str, Any]) -> str:
    """Render a concise result card for a successful Red submission."""
    prediction = f"{result.get('clean_prediction', '—')} → {result.get('adversarial_prediction', '—')}"
    fields = [
        ("Sequence", result.get("sequence_number", "—")),
        ("Attack", result.get("attack_name", "—")),
        ("Prediction", prediction),
        ("Declared ε", f"{float(result['declared_epsilon']):.6f}" if result.get("declared_epsilon") is not None else "—"),
        ("Status", result.get("status", "Submitted")),
    ]
    if result.get("flag"):
        fields.append(("Challenge flag", result["flag"]))
    details = [
        ("Exercise maximum L∞", result.get("max_linf")),
        ("Request round-trip", f"{float(result['request_elapsed_seconds']):.2f} s" if result.get("request_elapsed_seconds") is not None else None),
        ("Attack ID", result.get("attack_id")),
        ("SHA-256", result.get("sha256")),
    ]
    return _info_card(
        "Red attack submitted", fields, accent="#dc2626",
        extra_html=_linf_meter_html(result.get("measured_linf"), result.get("max_linf")),
        details=details,
    )


def pending_attack_card_html(pending: dict[str, Any]) -> str:
    """Render the current Blue inbox entry without dumping the API dictionary."""
    if not pending.get("pending"):
        return _info_card("Blue inbox", [("Status", "No Red attack is waiting")])
    evaluation = pending.get("evaluation") or {}
    prediction = f"{evaluation.get('true_class_label', '—')} → {evaluation.get('adversarial_class_label', '—')}"
    fields = [
        ("Sequence", pending.get("sequence_number", "—")),
        ("Attack", pending.get("attack_name", "—")),
        ("Source", pending.get("source_id", "—")),
        ("Prediction", prediction),
        ("Declared ε", f"{float(pending['declared_epsilon']):.6f}" if pending.get("declared_epsilon") is not None else "—"),
        ("Status", "Ready for Blue"),
    ]
    details = [
        ("Exercise maximum L∞", pending.get("max_linf")),
        ("Filename", pending.get("filename")),
        ("SHA-256", pending.get("sha256")),
    ]
    return _info_card(
        "Incoming Red attack", fields, accent="#2563eb",
        extra_html=_linf_meter_html(pending.get("measured_linf"), pending.get("max_linf")),
        details=details,
    )


def blue_result_card_html(result: dict[str, Any]) -> str:
    """Render the protected Blue response as labeled result fields."""
    if not result.get("available", True):
        return _info_card("Blue response", [("Status", "No response is available yet")])
    protected = result.get("protected_result") or {}
    if not protected:
        return _info_card("Blue response", [("Status", result.get("status", "No protected result returned"))])
    settings = result.get("defense_parameters") or protected.get("parameters") or {}
    defense_success = protected.get("defense_success")
    defense_label = "Succeeded" if defense_success is True else "Did not succeed" if defense_success is False else "Unknown"
    confidence = protected.get("defended_vote_confidence")
    fields = [
        ("Sequence", result.get("sequence_number", "—")),
        ("Defense", defense_label),
        ("Clean utility", "Passed" if protected.get("clean_utility_pass") else "Failed"),
        ("Defended prediction", protected.get("defended_prediction_label", "—")),
        ("Vote confidence", f"{float(confidence):.3f}" if confidence is not None else "—"),
        ("Attack survived", "Yes" if protected.get("attack_survived") else "No"),
        ("Blue settings", f"σ={settings.get('sigma', '—')}, n={settings.get('num_samples', '—')}, threshold={settings.get('abstain_threshold', '—')}"),
    ]
    if result.get("flag"):
        fields.append(("Challenge flag", result["flag"]))
    details = [
        ("Request round-trip", f"{float(result['request_elapsed_seconds']):.2f} s" if result.get("request_elapsed_seconds") is not None else None),
        ("Response ID", result.get("response_id")),
        ("Attack ID", result.get("attack_id")),
    ]
    return _info_card(
        "Protected Blue evaluation", fields, accent="#2563eb",
        extra_html=(
            "<div style='margin-top:7px;font-weight:700;color:" +
            ("#15803d" if defense_success else "#dc2626") + "'>" +
            ("✓ Defense succeeded" if defense_success else "✗ Defense did not succeed") +
            "</div>"
        ),
        details=details,
    )


def prediction_card_html(clean: dict[str, Any], changed: dict[str, Any], *, role: str = "red") -> str:
    """Render local clean and changed-image predictions in one card."""
    accent = "#dc2626" if role == "red" else "#2563eb"
    return _info_card(
        "Local model predictions", [], accent=accent,
        extra_html=(
            "<div style='display:flex;flex-wrap:wrap;gap:8px'>" +
            _confidence_panel("Clean source", clean, accent) +
            _confidence_panel("Submitted image", changed, accent) +
            "</div>"
        ),
    )


def source_info_card_html(source: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Render the selected approved source and its clean prediction."""
    probability = prediction.get("probability")
    predicted = (
        f"{prediction.get('class_label', '—')} ({float(probability):.3f})"
        if probability is not None else prediction.get("class_label", "—")
    )
    fields = [
        ("Source ID", source.get("source_id", "—")),
        ("Expected class", source.get("class_label", "—")),
        ("Clean prediction", predicted),
    ]
    return _info_card(
        "Selected approved source", fields, accent="#dc2626",
        extra_html=_confidence_panel("Clean prediction confidence", prediction, "#dc2626"),
    )


def source_gallery(sources: list[dict[str, Any]], *, columns: int = 5) -> Image.Image:
    """Build a thumbnail contact sheet of approved sources and clean predictions."""
    from PIL import ImageDraw

    if not sources:
        raise ValueError("No approved source images are available.")
    columns = max(1, int(columns))
    cell_width, cell_height, thumb_size = 146, 145, 116
    rows = (len(sources) + columns - 1) // columns
    gallery = Image.new("RGB", (columns * cell_width, rows * cell_height), "#f1f5f9")
    draw = ImageDraw.Draw(gallery)
    for index, source in enumerate(sources):
        image_path = download_source(source["source_id"])
        with Image.open(image_path) as opened:
            thumbnail = opened.convert("RGB")
            thumbnail.thumbnail((thumb_size, thumb_size))
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image_x = x + (cell_width - thumbnail.width) // 2
        gallery.paste(thumbnail, (image_x, y + 4))
        probability = source.get("clean_probability")
        probability_label = f"{float(probability):.2f}" if probability is not None else "—"
        draw.text((x + 7, y + 123), f"ID {source.get('source_id', '?')} · {source.get('class_label', '?')}", fill="#172033")
        draw.text((x + 7, y + 137), f"Clean confidence: {probability_label}", fill="#475569")
        draw.rectangle((x + 2, y + 2, x + cell_width - 3, y + cell_height - 3), outline="#cbd5e1")
    return gallery


def compare_images(clean: str | Path | Image.Image, changed: str | Path | Image.Image) -> Image.Image:
    """Create a clean / submitted / amplified-difference panel for notebook display."""
    from PIL import ImageDraw

    def load_rgb(value: str | Path | Image.Image) -> Image.Image:
        if isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                return opened.convert("RGB").copy()
        return value.convert("RGB").copy()

    clean_image = load_rgb(clean)
    changed_image = load_rgb(changed)
    if clean_image.size != changed_image.size:
        raise ValueError("The clean and submitted images must have the same dimensions.")
    clean_pixels = np.asarray(clean_image, dtype=np.uint8)
    changed_pixels = np.asarray(changed_image, dtype=np.uint8)
    difference = np.clip(np.abs(clean_pixels.astype(np.int16) - changed_pixels.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    difference_image = Image.fromarray(difference, mode="RGB")

    width, height = clean_image.size
    header = 26
    panel = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(panel)
    for index, label in enumerate(("Clean source", "Red submission", "Difference ×8")):
        draw.text((index * width + 8, 7), label, fill="#172033")
    panel.paste(clean_image, (0, header))
    panel.paste(changed_image, (width, header))
    panel.paste(difference_image, (width * 2, header))
    return panel


def smoothing_result_card_html(result: dict[str, Any]) -> str:
    """Render a local randomized-smoothing run as labeled values."""
    confidence = result.get("confidence")
    fields = [
        ("Prediction", result.get("class_label", "—")),
        ("Vote confidence", f"{float(confidence):.3f}" if confidence is not None else "—"),
        ("σ", result.get("sigma", "—")),
        ("Samples", result.get("num_samples", "—")),
    ]
    votes = result.get("votes")
    details = [("Class vote counts", votes)] if votes is not None else None
    return _info_card("Local smoothing estimate", fields, accent="#2563eb", details=details)


def round_summary_markdown() -> str:
    """Build a compact participant-facing table for every Red/Blue sequence."""
    history = get_match_history().get("history", [])
    if not history:
        return "**No attacks have been submitted in this match yet.**"

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def utc_timestamp(value: Any) -> str:
        if not value:
            return "—"
        try:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(value)

    rows = [
        "| Sequence | Red attack result | Declared ε | Measured L∞ | Blue settings | Defense succeeded | Timing (UTC) |",
        "|---:|---|---:|---:|---|:---:|---|",
    ]
    for item in history:
        evaluation = item.get("attack_evaluation") or {}
        clean_label = evaluation.get("true_class_label", "?")
        adversarial_label = evaluation.get("adversarial_class_label", "?")
        attack_result = f"{item.get('attack_name', 'Attack')}: {clean_label} → {adversarial_label}"
        declared = item.get("declared_epsilon")
        measured = item.get("measured_linf", evaluation.get("measured_linf"))
        declared_text = f"{float(declared):.6f}" if declared is not None else "—"
        measured_text = f"{float(measured):.6f}" if measured is not None else "—"

        settings = item.get("defense_parameters")
        if settings:
            settings_text = (
                f"σ={settings.get('sigma', '—')}, "
                f"samples={settings.get('num_samples', '—')}, "
                f"threshold={settings.get('abstain_threshold', '—')}"
            )
        else:
            settings_text = "Awaiting Blue" if item.get("status") == "pending_blue" else "—"

        result = item.get("protected_result")
        if result is None:
            defense_text = "🟡 Pending" if item.get("status") == "pending_blue" else "—"
        else:
            defense_text = "🟢 Succeeded" if result.get("defense_success") else "🔴 Failed"

        red_time = utc_timestamp(item.get("attack_created_at"))
        blue_time = utc_timestamp(item.get("response_created_at"))
        timing_text = f"Red: {red_time}; Blue: {blue_time if blue_time != '—' else 'pending'}"

        rows.append(
            f"| {cell(item.get('sequence_number', '—'))} | {cell(attack_result)} "
            f"| {cell(declared_text)} | {cell(measured_text)} | {cell(settings_text)} "
            f"| {cell(defense_text)} | {cell(timing_text)} |"
        )
    return "\n".join(rows)


def get_match_status() -> dict[str, Any]:
    return _request("GET", "/api/match/status").json()


def turn_indicator_markdown() -> str:
    """Explain whose turn it is and the next sequence using Arena match state."""
    status = get_match_status()
    role = status.get("role")
    pending_sequence = status.get("pending_attack_sequence")
    latest_sequence = int(status.get("latest_sequence", 0))

    if pending_sequence is not None:
        sequence = int(pending_sequence)
        if role == "solo":
            return (
                f"### Your turn — Blue, sequence {sequence}\n\n"
                "Open the Blue Team notebook in this workspace, inspect the verified image, and submit a defense."
            )
        if role == "blue":
            return (
                f"### Your turn — Blue, sequence {sequence}\n\n"
                "Refresh the inbox, inspect Red’s verified image, then submit your defense settings."
            )
        if role == "red":
            return (
                f"### Waiting for Blue — sequence {sequence}\n\n"
                "Blue is evaluating the submitted attack. Refresh the response cell to see the result."
            )
        return f"### Sequence {sequence} is waiting for Blue."

    next_sequence = latest_sequence + 1
    if role == "solo":
        return (
            f"### Your turn — Red, sequence {next_sequence}\n\n"
            "Use the Red Team notebook to prepare and submit an adversarial image."
        )
    if role == "red":
        return (
            f"### Your turn — Red, sequence {next_sequence}\n\n"
            "Choose an approved source, prepare an adversarial image, and submit it to start the next sequence."
        )
    if role == "blue":
        return (
            f"### Waiting for Red — next sequence {next_sequence}\n\n"
            "Refresh the inbox after Red submits an attack."
        )
    return f"### The match is ready for Red to start sequence {next_sequence}."


def turn_indicator_html() -> str:
    """Render a color-coded action/waiting banner for the assigned participant."""
    status = get_match_status()
    role = status.get("role")
    pending_sequence = status.get("pending_attack_sequence")
    latest_sequence = int(status.get("latest_sequence", 0))

    if pending_sequence is not None and role == "solo":
        title = f"Your turn — Blue · Sequence {int(pending_sequence)}"
        message = "Open the Blue Team notebook in this workspace, inspect the verified image, and submit a defense."
        accent, background = "#2563eb", "#eff6ff"
    elif pending_sequence is not None and role == "blue":
        title = f"Your turn — Blue · Sequence {int(pending_sequence)}"
        message = "Refresh the inbox, inspect Red’s verified image, then submit your defense settings."
        accent, background = "#2563eb", "#eff6ff"
    elif pending_sequence is not None and role == "red":
        title = f"Waiting for Blue · Sequence {int(pending_sequence)}"
        message = "Blue is evaluating the submitted attack. Refresh the response cell to see the result."
        accent, background = "#64748b", "#f1f5f9"
    elif role == "solo":
        title = f"Your turn — Red · Sequence {latest_sequence + 1}"
        message = "Use the Red Team notebook to prepare and submit an adversarial image."
        accent, background = "#dc2626", "#fef2f2"
    elif role == "red":
        title = f"Your turn — Red · Sequence {latest_sequence + 1}"
        message = "Choose an approved source, prepare an adversarial image, and submit it to start the next sequence."
        accent, background = "#dc2626", "#fef2f2"
    elif role == "blue":
        title = f"Waiting for Red · Next sequence {latest_sequence + 1}"
        message = "Refresh the inbox after Red submits an attack."
        accent, background = "#64748b", "#f1f5f9"
    else:
        title = f"Ready for Red · Sequence {latest_sequence + 1}"
        message = "The match is ready for Red to submit an attack."
        accent, background = "#64748b", "#f1f5f9"

    return (
        f"<div style='border-left:6px solid {accent};background:{background};"
        "border-radius:8px;padding:13px 16px;margin:10px 0'>"
        f"<div style='font-size:17px;font-weight:700;color:#172033'>{escape(title)}</div>"
        f"<div style='margin-top:4px;color:#475569'>{escape(message)}</div></div>"
    )


def progress_header_html(role: str, current_step: str) -> str:
    """Render clickable, role-specific notebook progress steps."""
    if role == "red":
        steps = [
            ("setup", "1 · Source", "red-setup"),
            ("prepare", "2 · Candidate", "red-prepare"),
            ("submit", "3 · Submit", "red-submit"),
            ("review", "4 · Review", "red-review"),
        ]
        accent = "#dc2626"
    else:
        steps = [
            ("inbox", "1 · Inbox", "blue-inbox"),
            ("experiment", "2 · Experiment", "blue-experiment"),
            ("submit", "3 · Submit", "blue-submit"),
            ("review", "4 · Review", "blue-review"),
        ]
        accent = "#2563eb"
    active_index = next((i for i, step in enumerate(steps) if step[0] == current_step), 0)
    chips = []
    for index, (_key, label, anchor) in enumerate(steps):
        if index < active_index:
            background, foreground, border = "#dcfce7", "#166534", "#86efac"
        elif index == active_index:
            background, foreground, border = accent, "#ffffff", accent
        else:
            background, foreground, border = "#f1f5f9", "#64748b", "#cbd5e1"
        chips.append(
            f"<a href='#{anchor}' style='text-decoration:none;display:inline-block;"
            f"padding:7px 11px;border:1px solid {border};border-radius:999px;"
            f"background:{background};color:{foreground};font-weight:600;font-size:13px'>"
            f"{escape(label)}</a>"
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:7px;margin:10px 0 14px'>" + "".join(chips) + "</div>"
