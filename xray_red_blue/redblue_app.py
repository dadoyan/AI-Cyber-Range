"""Opt-in Red/Blue round service layered on the existing AICR exercise image."""

from __future__ import annotations

import hmac
import hashlib
import io
import json
import logging
import os
import threading
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image

import attack_core
from range_device import select_device
from mlflow_audit import log_blue_attempt, log_red_attempt
from xray_defenses import PREPROCESSORS, prediction_consistency, preprocess_batch, randomized_smoothing


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("xray-red-blue")
# The base image's model factory uses this shared device when loading models.
attack_core.device = select_device()

try:
    torch.set_num_threads(max(1, int(os.getenv("XRAY_TORCH_THREADS", "2"))))
except (TypeError, ValueError):
    torch.set_num_threads(2)

APP_ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("XRAY_STATE_DIR", "/app/redblue-state"))
STATE_FILE = STATE_DIR / "round.json"
MODEL_NAME = "efficientnet_b0.pth"
BLUE_MODELS = {"standard": MODEL_NAME, "robust": "efficientnet_b0_robust.pth"}
BATCH_SIZE = 8
SAMPLES_PER_CLASS = max(4, min(32, int(os.getenv("XRAY_SAMPLES_PER_CLASS", "16"))))
RED_MAX_EPSILON = 0.02
RED_LINF_TOLERANCE = 1e-6
ARENA_LINF_TOLERANCE = 1e-8
RED_MIN_ASR = float(os.getenv("XRAY_RED_MIN_ASR", "0.40"))
BLUE_MIN_CLEAN_ACCURACY = float(os.getenv("XRAY_BLUE_MIN_CLEAN_ACCURACY", "0.65"))
BLUE_MIN_ADV_RESOLVED = float(os.getenv("XRAY_BLUE_MIN_ADV_RESOLVED", "0.55"))
BLUE_MIN_ADV_GAIN = float(os.getenv("XRAY_BLUE_MIN_ADV_GAIN", "0.05"))
BLUE_MIN_DETECTION_RATE = float(os.getenv("XRAY_BLUE_MIN_DETECTION_RATE", "0.05"))
BLUE_MAX_FALSE_POSITIVE_RATE = float(os.getenv("XRAY_BLUE_MAX_FALSE_POSITIVE_RATE", "0.25"))
RED_FLAG = os.getenv("XRAY_RED_FLAG", "")
BLUE_FLAG = os.getenv("XRAY_BLUE_FLAG", "")
ADMIN_TOKEN = os.getenv("XRAY_ADMIN_TOKEN", "")
ARENA_INTERNAL_KEY = os.getenv("XRAY_ARENA_INTERNAL_KEY", "")


def _audit_decision(success: bool, criteria: list[dict], *, challenge: str,
                    flag_returned: bool) -> dict:
    failures = [item["name"] for item in criteria if not item["passed"]]
    return {
        "schema_version": 1,
        "challenge": challenge,
        "success": bool(success),
        "flag_returned": bool(flag_returned),
        "summary": "All acceptance criteria passed." if success else "Evaluation rejected: " + ", ".join(failures),
        "criteria": criteria,
        "failure_reasons": failures,
    }

app = Flask(__name__)
# Static requests are small JSON bodies; Arena also posts one bounded PNG.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xray-round")
_batch_cache: tuple[np.ndarray, np.ndarray, list[str], int] | None = None
_arena_classifier = None
_arena_source_cache: dict[str, dict[str, Any]] | None = None
_arena_source_metadata: list[dict[str, Any]] | None = None
_arena_class_names: list[str] | None = None
_arena_lock = threading.RLock()


def _default_state() -> dict[str, Any]:
    return {
        "round_id": uuid.uuid4().hex,
        "phase": "red",
        "red_run_id": None,
        "red_result": None,
        "blue_result": None,
        "updated_at": None,
    }


def _load_state() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        state = _default_state()
        _save_state(state)
        return state
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Round state is unreadable; starting a fresh round")
        state = _default_state()
        _save_state(state)
        return state


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.replace(STATE_FILE)


_state = _load_state()


def _challenge_batch() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    global _batch_cache
    with _lock:
        if _batch_cache is not None:
            return _batch_cache

    loader, class_names, num_classes = attack_core.build_test_loader()
    targets = np.asarray(loader.dataset.targets, dtype=np.int64)
    selected: list[int] = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(targets == class_id)
        selected.extend(class_indices[:SAMPLES_PER_CLASS].tolist())
    subset = torch.utils.data.Subset(loader.dataset, selected)
    subset_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    images, labels = attack_core.numpy_from_loader(subset_loader)
    value = (images, labels, list(class_names), int(num_classes))
    with _lock:
        _batch_cache = value
    return value


def _number(body: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    value = body.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be numeric") from None
    if not np.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return parsed


def _integer(body: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    value = body.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    if str(parsed) != str(value).strip() and not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return parsed


def _prediction_disagreement(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Flag examples whose standard and robust checkpoint predictions differ."""
    primary = np.asarray(primary)
    secondary = np.asarray(secondary)
    if primary.shape != secondary.shape or primary.ndim != 1:
        raise ValueError("prediction arrays must be matching one-dimensional vectors")
    return primary != secondary


def _confidence_gated_disagreement(
    primary_probabilities: np.ndarray,
    secondary_probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Flag confident primary predictions that disagree with the other model."""
    primary = np.asarray(primary_probabilities, dtype=np.float32)
    secondary = np.asarray(secondary_probabilities, dtype=np.float32)
    if primary.ndim != 2 or primary.shape != secondary.shape:
        raise ValueError("model probability arrays must have matching (N, classes) shapes")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    disagreements = _prediction_disagreement(primary.argmax(axis=1), secondary.argmax(axis=1))
    return disagreements & (primary.max(axis=1) >= float(threshold))


def _job_view(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {key: value for key, value in job.items() if key not in {"future", "identity"}}


def _resolve_workspace_identity(authorization: str | None, remote_addr: str | None) -> dict[str, Any] | None:
    """Resolve identity via the Launcher; never accept CTFd identity claims from a client."""
    resolve_key = os.getenv("XRAY_IDENTITY_RESOLVE_KEY", "")
    launcher_url = os.getenv("WORKSPACE_LAUNCHER_URL", "http://launcher:7000").rstrip("/")
    if not resolve_key:
        return None
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    payload = json.dumps({"token": token, "remote_addr": remote_addr or ""}).encode("utf-8")
    lookup = Request(
        f"{launcher_url}/internal/identity/resolve",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Workspace-Identity-Key": resolve_key,
        },
        method="POST",
    )
    try:
        with urlopen(lookup, timeout=2.5) as response:
            identity = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code not in {401, 403, 404}:
            logger.warning("Launcher identity lookup returned HTTP %s", exc.code)
        return None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        logger.warning("Launcher identity lookup failed; X-Ray evaluation will continue anonymously", exc_info=True)
        return None
    except Exception:
        logger.warning("Unexpected Launcher identity lookup failure; evaluation will continue anonymously", exc_info=True)
        return None

    try:
        user_id = int(identity["ctfd_user_id"])
        username = str(identity["ctfd_username"]).strip()
        workspace_id = str(identity["workspace_id"]).strip()
        source = str(identity.get("identity_source", "launcher"))
        if user_id <= 0 or not username or len(username) > 64 or not workspace_id:
            return None
    except (KeyError, TypeError, ValueError):
        logger.warning("Launcher returned an invalid identity mapping")
        return None
    return {
        "ctfd_user_id": user_id,
        "ctfd_username": username,
        "workspace_id": workspace_id,
        "identity_source": source,
    }


def _public_result(result: Any) -> Any:
    """Remove challenge flags from round-wide status and summary responses."""
    if not isinstance(result, dict):
        return result
    return {key: value for key, value in result.items() if key != "flag"}


def _set_job(job_id: str, **values: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(values)


def _run_red(job_id: str, parameters: dict[str, Any], identity: dict[str, Any] | None = None) -> None:
    _set_job(job_id, status="running")
    try:
        images, labels, _class_names, num_classes = _challenge_batch()
        classifier = attack_core.build_classifier(MODEL_NAME, num_classes)
        clean_probabilities = classifier.predict(images, batch_size=BATCH_SIZE)
        clean_predictions = clean_probabilities.argmax(axis=1)

        if parameters["attack"] == "FGSM":
            attack = FastGradientMethod(
                estimator=classifier,
                eps=parameters["epsilon"],
                batch_size=BATCH_SIZE,
            )
        else:
            attack = ProjectedGradientDescent(
                estimator=classifier,
                norm=np.inf,
                eps=parameters["epsilon"],
                eps_step=parameters["step"],
                max_iter=parameters["iterations"],
                targeted=False,
                num_random_init=0,
                batch_size=BATCH_SIZE,
            )
        adversarial = attack.generate(images).astype(np.float32)
        adversarial = np.clip(adversarial, 0.0, 1.0)
        adv_probabilities = classifier.predict(adversarial, batch_size=BATCH_SIZE)
        adv_predictions = adv_probabilities.argmax(axis=1)

        clean_correct = clean_predictions == labels
        denominator = int(clean_correct.sum())
        flipped = clean_correct & (adv_predictions != clean_predictions)
        asr = float(flipped.sum() / denominator) if denominator else 0.0
        clean_accuracy = float(np.mean(clean_predictions == labels))
        adv_accuracy = float(np.mean(adv_predictions == labels))
        measured_linf = float(np.max(np.abs(adversarial - images)))
        computed_success = denominator > 0 and asr >= RED_MIN_ASR and measured_linf <= RED_MAX_EPSILON + RED_LINF_TOLERANCE

        # Keep the public artifact/run identifier distinct from the private job-poll ID.
        run_id = uuid.uuid4().hex
        round_id = parameters["round_id"]
        result = {
            "run_id": run_id,
            "attack": parameters["attack"],
            "epsilon": parameters["epsilon"],
            "step": parameters["step"],
            "iterations": parameters["iterations"],
            "total_samples": int(len(labels)),
            "originally_correct": denominator,
            "flipped": int(flipped.sum()),
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "attack_success_rate": asr,
            "measured_linf": measured_linf,
            "success": computed_success,
            "artifact_url": None,
        }
        published = False
        qualified = False
        with _lock:
            if computed_success and _state["round_id"] == round_id and _state["phase"] == "red":
                artifact_path = STATE_DIR / f"red-{run_id}.npz"
                np.savez_compressed(
                    artifact_path,
                    x_clean=images,
                    x_adv=adversarial,
                    labels=labels,
                    clean_predictions=clean_predictions,
                    adv_predictions=adv_predictions,
                )
                result["artifact_url"] = f"/api/round/latest/artifact?run_id={run_id}"
                _state["updated_at"] = datetime.now(timezone.utc).isoformat()
                _state["phase"] = "blue"
                _state["red_run_id"] = run_id
                # Keep the participant-only flag in the job result, not the shared round summary.
                _state["red_result"] = dict(result)
                _state["blue_result"] = None
                _save_state(_state)
                published = True
                qualified = True
            elif computed_success and _state["round_id"] == round_id and _state["phase"] in {"blue", "complete"}:
                active_run = _state.get("red_run_id")
                active_result = _state.get("red_result")
                qualified = bool(
                    active_run and isinstance(active_result, dict)
                    and active_result.get("success") is True
                    and (STATE_DIR / f"red-{active_run}.npz").is_file()
                )
            elif computed_success:
                result["success"] = False
                result["error"] = "The round changed before this Red attempt finished"
        if computed_success and not qualified:
            result["success"] = False
            result.setdefault("error", "No active Red artifact is available for this round")
        result["shared_artifact_published"] = published
        red_criteria = [
            {"name": "clean_correct_samples_available", "passed": denominator > 0,
             "observed": denominator, "minimum": 1,
             "rule": "at least one clean sample must be correctly classified"},
            {"name": "minimum_attack_success_rate", "passed": asr >= RED_MIN_ASR,
             "observed": asr, "minimum": RED_MIN_ASR,
             "rule": "attack success rate must be >= configured minimum"},
            {"name": "maximum_measured_linf", "passed": measured_linf <= RED_MAX_EPSILON + RED_LINF_TOLERANCE,
             "observed": measured_linf, "maximum": RED_MAX_EPSILON,
             "tolerance": RED_LINF_TOLERANCE,
             "effective_maximum": RED_MAX_EPSILON + RED_LINF_TOLERANCE,
             "rule": "measured L-infinity distance must be <= exercise budget plus numeric tolerance"},
        ]
        if computed_success:
            red_criteria.append({
                "name": "published_to_active_round" if published else "qualified_with_active_round",
                "passed": qualified,
                "observed": qualified,
                "rule": "the first successful attack publishes the shared artifact; later successful attacks qualify without replacing it",
            })
        if qualified and RED_FLAG:
            result["flag"] = RED_FLAG
        red_audit = _audit_decision(
            bool(result["success"]), red_criteria, challenge="xray-redblue-red",
            flag_returned="flag" in result,
        )
        red_audit.update({
            "attack_success_rate": asr,
            "clean_accuracy": clean_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "originally_correct": denominator,
            "flipped": int(flipped.sum()),
            "measured_linf": measured_linf,
            "max_linf": RED_MAX_EPSILON,
            "linf_tolerance": RED_LINF_TOLERANCE,
            "declared_epsilon": parameters["epsilon"],
            "shared_artifact_published": published,
        })
        log_red_attempt(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "xray-redblue"),
            round_id=round_id,
            public_run_id=run_id,
            parameters={key: value for key, value in parameters.items() if key != "round_id"},
            metrics={
                "attack_success_rate": result["attack_success_rate"],
                "clean_accuracy": result["clean_accuracy"],
                "adversarial_accuracy": result["adversarial_accuracy"],
                "measured_linf": result["measured_linf"],
                "originally_correct": result["originally_correct"],
                "flipped": result["flipped"],
                "success": result["success"],
                "shared_artifact_published": published,
            },
            evaluation={
                **{key: value for key, value in result.items() if key not in {"flag", "error"}},
                "decision_audit": red_audit,
            },
            identity=identity,
            arrays={
                "x_clean": images,
                "x_adv": adversarial,
                "labels": labels,
                "clean_predictions": clean_predictions,
                "adv_predictions": adv_predictions,
            },
        )
        _set_job(job_id, status="completed", result=result)
    except Exception as exc:
        logger.exception("Red attack job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(exc))


def _run_blue(
    job_id: str,
    parameters: dict[str, Any],
    run_id: str,
    identity: dict[str, Any] | None = None,
) -> None:
    _set_job(job_id, status="running")
    try:
        artifact_path = STATE_DIR / f"red-{run_id}.npz"
        if not artifact_path.is_file():
            raise ValueError("The active Red artifact is unavailable")
        with np.load(artifact_path, allow_pickle=False) as data:
            x_clean = data["x_clean"].astype(np.float32)
            x_adv = data["x_adv"].astype(np.float32)
            labels = data["labels"].astype(np.int64)

        _images, _labels, _class_names, num_classes = _challenge_batch()
        classifier = attack_core.build_classifier(BLUE_MODELS[parameters["model"]], num_classes)
        x_clean = preprocess_batch(x_clean, parameters["preprocess"], parameters["bits"])
        x_adv = preprocess_batch(x_adv, parameters["preprocess"], parameters["bits"])

        clean_flagged = np.zeros(len(labels), dtype=bool)
        adv_flagged = np.zeros(len(labels), dtype=bool)
        if parameters["detector"] == "consistency":
            clean_predictions, clean_confidence, clean_flagged = prediction_consistency(
                classifier,
                x_clean,
                samples=parameters["samples"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                batch_size=BATCH_SIZE,
            )
            adv_predictions, adv_confidence, adv_flagged = prediction_consistency(
                classifier,
                x_adv,
                samples=parameters["samples"],
                sigma=parameters["sigma"],
                threshold=parameters["threshold"],
                batch_size=BATCH_SIZE,
            )
        elif parameters["detector"] == "disagreement":
            alternate_name = "robust" if parameters["model"] == "standard" else "standard"
            alternate_classifier = attack_core.build_classifier(BLUE_MODELS[alternate_name], num_classes)
            clean_probabilities = classifier.predict(x_clean, batch_size=BATCH_SIZE)
            adv_probabilities = classifier.predict(x_adv, batch_size=BATCH_SIZE)
            clean_other = alternate_classifier.predict(x_clean, batch_size=BATCH_SIZE)
            adv_other = alternate_classifier.predict(x_adv, batch_size=BATCH_SIZE)
            clean_predictions = clean_probabilities.argmax(axis=1)
            adv_predictions = adv_probabilities.argmax(axis=1)
            clean_flagged = _confidence_gated_disagreement(
                clean_probabilities, clean_other, parameters["confidence"]
            )
            adv_flagged = _confidence_gated_disagreement(
                adv_probabilities, adv_other, parameters["confidence"]
            )
            clean_confidence = adv_confidence = None
        else:
            clean_predictions = classifier.predict(x_clean, batch_size=BATCH_SIZE).argmax(axis=1)
            adv_predictions = classifier.predict(x_adv, batch_size=BATCH_SIZE).argmax(axis=1)
            clean_confidence = adv_confidence = None

        clean_accuracy = float(np.mean(clean_predictions == labels))
        adv_accuracy = float(np.mean(adv_predictions == labels))
        false_positive_rate = float(clean_flagged.mean())
        detection_rate = float(adv_flagged.mean())
        clean_usable_accuracy = float(np.mean((clean_predictions == labels) & ~clean_flagged))
        adv_resolved_accuracy = float(np.mean((adv_predictions == labels) | adv_flagged))
        with _lock:
            red_result = _state.get("red_result") or {}
        baseline_clean = float(red_result.get("clean_accuracy", 0.0))
        baseline_adv = float(red_result.get("adversarial_accuracy", 0.0))
        computed_success = (
            clean_usable_accuracy >= BLUE_MIN_CLEAN_ACCURACY
            and adv_resolved_accuracy >= max(BLUE_MIN_ADV_RESOLVED, baseline_adv + BLUE_MIN_ADV_GAIN)
            and detection_rate >= BLUE_MIN_DETECTION_RATE
            and false_positive_rate <= BLUE_MAX_FALSE_POSITIVE_RATE
        )
        result = {
            # A public round summary must not reveal the private job-poll ID.
            "run_id": uuid.uuid4().hex,
            "red_run_id": run_id,
            "model": parameters["model"],
            "preprocess": parameters["preprocess"],
            "detector": parameters["detector"],
            "confidence_threshold": parameters["confidence"] if parameters["detector"] == "disagreement" else None,
            "total_samples": int(len(labels)),
            "clean_accuracy": clean_accuracy,
            "clean_usable_accuracy": clean_usable_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "adv_resolved_accuracy": adv_resolved_accuracy,
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "success": computed_success,
        }
        if clean_confidence is not None:
            result["mean_clean_consistency"] = float(clean_confidence.mean())
            result["mean_adversarial_consistency"] = float(adv_confidence.mean())
        with _lock:
            current_round = _state.get("round_id") == parameters["round_id"]
            current_red_run = _state.get("red_run_id") == run_id
            qualified = current_round and current_red_run and _state["phase"] in {"blue", "complete"}
            published = qualified and _state["phase"] == "blue"
            if published:
                _state["updated_at"] = datetime.now(timezone.utc).isoformat()
                # Keep the participant-only flag in the job result, not the shared round summary.
                _state["blue_result"] = dict(result)
            elif not qualified:
                result["success"] = False
                result["error"] = "The round changed before this Blue attempt finished"
            if result["success"] and published:
                _state["phase"] = "complete"
                _save_state(_state)
        result["shared_result_published"] = published
        if result["success"] and BLUE_FLAG:
            result["flag"] = BLUE_FLAG
        required_adv_resolved = max(BLUE_MIN_ADV_RESOLVED, baseline_adv + BLUE_MIN_ADV_GAIN)
        blue_criteria = [
            {"name": "minimum_clean_usable_accuracy", "passed": clean_usable_accuracy >= BLUE_MIN_CLEAN_ACCURACY,
             "observed": clean_usable_accuracy, "minimum": BLUE_MIN_CLEAN_ACCURACY,
             "rule": "clean usable accuracy must be >= configured minimum"},
            {"name": "minimum_adversarial_resolved_accuracy", "passed": adv_resolved_accuracy >= required_adv_resolved,
             "observed": adv_resolved_accuracy, "minimum": required_adv_resolved,
             "baseline_adversarial_accuracy": baseline_adv,
             "minimum_gain": BLUE_MIN_ADV_GAIN,
             "rule": "adversarial resolved accuracy must meet the absolute floor and baseline gain"},
            {"name": "minimum_detection_rate", "passed": detection_rate >= BLUE_MIN_DETECTION_RATE,
             "observed": detection_rate, "minimum": BLUE_MIN_DETECTION_RATE,
             "rule": "detection rate must be >= configured minimum"},
            {"name": "maximum_false_positive_rate", "passed": false_positive_rate <= BLUE_MAX_FALSE_POSITIVE_RATE,
             "observed": false_positive_rate, "maximum": BLUE_MAX_FALSE_POSITIVE_RATE,
             "rule": "false-positive rate must be <= configured maximum"},
        ]
        if computed_success:
            blue_criteria.append({
                "name": "published_to_active_round" if published else "qualified_with_completed_round",
                "passed": qualified,
                "observed": qualified,
                "rule": "the first successful defense completes the shared round; later defenses qualify without replacing it",
            })
        blue_audit = _audit_decision(
            bool(result["success"]), blue_criteria, challenge="xray-redblue-blue",
            flag_returned="flag" in result,
        )
        blue_audit.update({
            "clean_accuracy": clean_accuracy,
            "clean_usable_accuracy": clean_usable_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "adversarial_resolved_accuracy": adv_resolved_accuracy,
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "red_baseline_adversarial_accuracy": baseline_adv,
            "shared_result_published": published,
            "parameters": {key: value for key, value in parameters.items() if key != "round_id"},
        })
        log_blue_attempt(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "xray-redblue"),
            round_id=parameters["round_id"],
            public_run_id=result["run_id"],
            related_red_run_id=run_id,
            parameters={key: value for key, value in parameters.items() if key != "round_id"},
            metrics={
                "clean_accuracy": result["clean_accuracy"],
                "clean_usable_accuracy": result["clean_usable_accuracy"],
                "adversarial_accuracy": result["adversarial_accuracy"],
                "adv_resolved_accuracy": result["adv_resolved_accuracy"],
                "detection_rate": result["detection_rate"],
                "false_positive_rate": result["false_positive_rate"],
                "success": result["success"],
                "shared_result_published": published,
            },
            evaluation={
                **{key: value for key, value in result.items() if key not in {"flag", "error"}},
                "decision_audit": blue_audit,
            },
            identity=identity,
        )
        _set_job(job_id, status="completed", result=result)
    except Exception as exc:
        logger.exception("Blue defense job %s failed", job_id)
        _set_job(job_id, status="failed", error=str(exc))


@app.get("/api/health")
@app.get("/health")
def health():
    models = [name for name in (MODEL_NAME, "efficientnet_b0_robust.pth") if (APP_ROOT / "models" / name).is_file()]
    data_dir = Path(attack_core.test_data)
    return jsonify({
        "status": "ready" if data_dir.is_dir() and MODEL_NAME in models else "missing_resources",
        "dataset_available": data_dir.is_dir(),
        "models_available": models,
        "device": str(attack_core.device),
        "samples_per_class": SAMPLES_PER_CLASS,
    })


def _require_arena_internal_key() -> bool:
    supplied = request.headers.get("X-Arena-Internal-Key", "")
    return bool(ARENA_INTERNAL_KEY) and hmac.compare_digest(supplied, ARENA_INTERNAL_KEY)


def _arena_model():
    global _arena_classifier
    with _arena_lock:
        if _arena_classifier is None:
            _arena_classifier = attack_core.build_classifier(MODEL_NAME, 2)
        return _arena_classifier


def _arena_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert ART classifier scores to stable probabilities for Arena metadata."""
    values = np.asarray(scores, dtype=np.float64)
    values = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(values)
    return (exp_values / np.sum(exp_values, axis=1, keepdims=True)).astype(np.float32)


def _arena_sources() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Cache the first fixed challenge examples that the standard model gets right."""
    global _arena_source_cache, _arena_source_metadata, _arena_class_names
    with _arena_lock:
        if _arena_source_cache is not None and _arena_source_metadata is not None:
            return _arena_source_cache, _arena_source_metadata

        loader, class_names, num_classes = attack_core.build_test_loader()
        _arena_class_names = list(class_names)
        targets = np.asarray(loader.dataset.targets, dtype=np.int64)
        selected: list[int] = []
        for class_id in range(num_classes):
            selected.extend(np.flatnonzero(targets == class_id)[:SAMPLES_PER_CLASS].tolist())
        batch = torch.utils.data.DataLoader(
            torch.utils.data.Subset(loader.dataset, selected),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        images, labels = attack_core.numpy_from_loader(batch)
        probabilities = _arena_probabilities(_arena_model().predict(images, batch_size=BATCH_SIZE))
        sources: dict[str, dict[str, Any]] = {}
        metadata: list[dict[str, Any]] = []
        for offset, (dataset_index, image, label) in enumerate(zip(selected, images, labels)):
            label_index = int(label)
            clean_probability = probabilities[offset]
            predicted_index = int(np.argmax(clean_probability))
            if predicted_index != label_index:
                continue
            source_id = str(dataset_index)
            source = {
                "source_id": source_id,
                "image": np.asarray(image, dtype=np.float32),
                "label_index": label_index,
                "label": str(class_names[label_index]),
            }
            image_bytes = io.BytesIO()
            pixels = np.rint(np.clip(source["image"].transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(image_bytes, format="PNG")
            source["png"] = image_bytes.getvalue()
            sources[source_id] = source
            metadata.append({
                "source_id": source_id,
                "class_index": label_index,
                "class_label": str(class_names[label_index]),
                "clean_probability": float(clean_probability[label_index]),
                "predicted_class_index": predicted_index,
                "width": int(source["image"].shape[2]),
                "height": int(source["image"].shape[1]),
            })

        if not sources:
            raise RuntimeError("No approved source image is correctly classified by the standard model")
        _arena_source_cache = sources
        _arena_source_metadata = metadata
        return sources, metadata


def _decode_arena_png(raw: bytes) -> tuple[np.ndarray, str]:
    if not raw or len(raw) > 1024 * 1024:
        raise ValueError("PNG must be non-empty and at most 1 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format != "PNG" or opened.mode != "RGB" or opened.size != (256, 256):
                raise ValueError("candidate must be a 256x256 RGB PNG")
            opened.load()
            pixels = np.asarray(opened, dtype=np.uint8).copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("candidate is not a valid PNG image") from exc
    return pixels.astype(np.float32).transpose(2, 0, 1) / 255.0, hashlib.sha256(raw).hexdigest()


@app.get("/internal/arena/sources")
def arena_sources():
    if not _require_arena_internal_key():
        return jsonify({"error": "Unauthorized"}), 403
    try:
        _sources, metadata = _arena_sources()
    except Exception:
        logger.exception("Arena approved source initialization failed")
        return jsonify({"error": "Approved source set is unavailable"}), 503
    return jsonify({"sources": metadata, "max_linf": RED_MAX_EPSILON, "class_names": _arena_class_names or []})


@app.get("/internal/arena/sources/<source_id>")
def arena_source_image(source_id: str):
    if not _require_arena_internal_key():
        return jsonify({"error": "Unauthorized"}), 403
    sources, _metadata = _arena_sources()
    source = sources.get(source_id)
    if source is None:
        return jsonify({"error": "Unknown approved source"}), 404
    return Response(source["png"], mimetype="image/png")


@app.get("/internal/arena/model")
def arena_model_checkpoint():
    if not _require_arena_internal_key():
        return jsonify({"error": "Unauthorized"}), 403
    checkpoint = APP_ROOT / "models" / MODEL_NAME
    if not checkpoint.is_file():
        return jsonify({"error": "Standard model checkpoint is unavailable"}), 503
    return send_file(checkpoint, mimetype="application/octet-stream", as_attachment=True, download_name=MODEL_NAME)


@app.post("/internal/arena/validate-attack")
def arena_validate_attack():
    if not _require_arena_internal_key():
        return jsonify({"error": "Unauthorized"}), 403
    source_id = str(request.form.get("source_id", ""))
    sources, _metadata = _arena_sources()
    source = sources.get(source_id)
    if source is None:
        return jsonify({"error": "Unknown approved source"}), 422
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "An adversarial PNG is required"}), 400
    raw = upload.read(1024 * 1024 + 1)
    try:
        candidate, digest = _decode_arena_png(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    reference = source["image"]
    measured_linf = float(np.max(np.abs(candidate - reference)))
    linf_tolerance = ARENA_LINF_TOLERANCE
    if measured_linf > RED_MAX_EPSILON + linf_tolerance:
        return jsonify({"error": "Candidate exceeds the exercise L-infinity budget", "measured_linf": measured_linf, "max_linf": RED_MAX_EPSILON, "linf_tolerance": linf_tolerance}), 422
    probabilities = _arena_probabilities(_arena_model().predict(np.stack([reference, candidate]), batch_size=2))
    clean_index = int(np.argmax(probabilities[0]))
    adv_index = int(np.argmax(probabilities[1]))
    if clean_index != int(source["label_index"]):
        return jsonify({"error": "Approved source failed its clean classification check"}), 503
    if adv_index == int(source["label_index"]):
        return jsonify({
            "error": "Candidate does not fool the standard model",
            "clean_prediction": source["label"],
            "adversarial_prediction": source["label"],
            "measured_linf": measured_linf,
        }), 422
    return jsonify({
        "valid": True,
        "source_id": source_id,
        "true_class_index": int(source["label_index"]),
        "true_class_label": source["label"],
        "clean_probability": float(probabilities[0][clean_index]),
        "adversarial_class_index": adv_index,
        "adversarial_class_label": str((_arena_class_names or [])[adv_index]),
        "adversarial_probability": float(probabilities[1][adv_index]),
        "measured_linf": measured_linf,
        "max_linf": RED_MAX_EPSILON,
        "linf_tolerance": linf_tolerance,
        "sha256": digest,
        "width": 256,
        "height": 256,
    })


@app.post("/internal/arena/evaluate-defense")
def arena_evaluate_defense():
    if not _require_arena_internal_key():
        return jsonify({"error": "Unauthorized"}), 403
    source_id = str(request.form.get("source_id", ""))
    sources, _metadata = _arena_sources()
    source = sources.get(source_id)
    if source is None:
        return jsonify({"error": "Unknown approved source"}), 422
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "Authoritative attack artifact is required"}), 400
    try:
        candidate, _digest = _decode_arena_png(upload.read(1024 * 1024 + 1))
        sigma = float(request.form.get("sigma", ""))
        samples = int(request.form.get("num_samples", ""))
        threshold = float(request.form.get("abstain_threshold", ""))
        seed = int(request.form.get("seed", ""))
        if not np.isfinite(sigma) or not 0.0 <= sigma <= 0.1:
            raise ValueError("sigma must be between 0 and 0.1")
        if not 10 <= samples <= 500:
            raise ValueError("num_samples must be between 10 and 500")
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("abstain_threshold must be between 0 and 1")
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed is outside the supported range")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422

    classifier = _arena_model()
    clean = source["image"][None, ...]
    adv = candidate[None, ...]
    clean_label, clean_conf = randomized_smoothing(
        classifier, clean, samples=samples, sigma=sigma, seed=seed, batch_size=BATCH_SIZE
    )
    adv_label, adv_conf = randomized_smoothing(
        classifier, adv, samples=samples, sigma=sigma, seed=(seed ^ 0xA5A5A5A5), batch_size=BATCH_SIZE
    )
    raw_adv = classifier.predict(adv, batch_size=1)[0]
    true_index = int(source["label_index"])
    clean_utility = int(clean_label[0]) == true_index and float(clean_conf[0]) >= threshold
    attack_survived = int(adv_label[0]) != true_index and float(adv_conf[0]) >= threshold
    result = {
        "source_id": source_id,
        "true_class_index": true_index,
        "true_class_label": source["label"],
        "clean_prediction_index": int(clean_label[0]),
        "clean_prediction_label": str((_arena_class_names or [])[int(clean_label[0])]),
        "clean_vote_confidence": float(clean_conf[0]),
        "clean_utility_pass": bool(clean_utility),
        "undefended_prediction_index": int(np.argmax(raw_adv)),
        "undefended_prediction_label": str((_arena_class_names or [])[int(np.argmax(raw_adv))]),
        "defended_prediction_index": int(adv_label[0]),
        "defended_prediction_label": str((_arena_class_names or [])[int(adv_label[0])]) if float(adv_conf[0]) >= threshold else "ABSTAIN",
        "defended_vote_confidence": float(adv_conf[0]),
        "attack_success": bool(int(np.argmax(raw_adv)) != true_index),
        "defense_success": bool(clean_utility and not attack_survived),
        "attack_survived": bool(attack_survived),
        "evaluator_version": "arena-1",
        "parameters": {"sigma": sigma, "num_samples": samples, "abstain_threshold": threshold},
    }
    return jsonify(result)


@app.get("/api/identity")
def participant_identity():
    """Let a managed workspace confirm which CTFd account MLflow attribution uses."""
    identity = _resolve_workspace_identity(
        request.headers.get("Authorization"), request.remote_addr
    )
    if identity is None:
        return jsonify({"error": "This request is not mapped to a managed CTFd workspace."}), 404
    return jsonify(identity)


@app.get("/api/config")
def config():
    return jsonify({
        "dataset": "Bundled chest X-ray test set; a balanced fixed subset is used for this round",
        "models": [MODEL_NAME],
        "blue_models": {name: checkpoint for name, checkpoint in BLUE_MODELS.items() if (APP_ROOT / "models" / checkpoint).is_file()},
        "attacks": ["FGSM", "PGD"],
        "preprocessors": sorted(PREPROCESSORS),
        "detectors": ["none", "consistency", "disagreement"],
        "defaults": {"epsilon": 0.01, "step": 0.0025, "iterations": 3, "samples": 4, "sigma": 0.03, "threshold": 0.75, "disagreement_confidence": 0.6},
        "phase_rules": {"red_min_asr": RED_MIN_ASR, "red_max_epsilon": RED_MAX_EPSILON},
        "scoring": {
            "blue_min_clean_accuracy": BLUE_MIN_CLEAN_ACCURACY,
            "blue_min_adv_resolved_accuracy": BLUE_MIN_ADV_RESOLVED,
            "blue_min_adv_gain": BLUE_MIN_ADV_GAIN,
            "blue_min_detection_rate": BLUE_MIN_DETECTION_RATE,
            "blue_max_false_positive_rate": BLUE_MAX_FALSE_POSITIVE_RATE,
        },
    })


@app.get("/api/round")
def round_status():
    with _lock:
        payload = dict(_state)
        payload["red_result"] = _public_result(payload.get("red_result"))
        payload["blue_result"] = _public_result(payload.get("blue_result"))
        return jsonify(payload)


@app.post("/api/red/attack")
def create_red_attack():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    identity = _resolve_workspace_identity(
        request.headers.get("Authorization"), request.remote_addr
    )
    with _lock:
        phase = _state["phase"]
        active_run = _state.get("red_run_id")
        if phase not in {"red", "blue", "complete"} or (
            phase != "red" and (not active_run or not (STATE_DIR / f"red-{active_run}.npz").is_file())
        ):
            return jsonify({"error": "No active Red round is available", "phase": phase}), 409
        if any(job.get("role") == "red" and job.get("status") in {"queued", "running"} for job in _jobs.values()):
            return jsonify({"error": "A Red attack is already running for this round"}), 409
        round_id = _state["round_id"]
    try:
        attack_type = str(body.get("attack", "FGSM")).upper()
        if attack_type not in {"FGSM", "PGD"}:
            raise ValueError("attack must be FGSM or PGD")
        epsilon = _number(body, "epsilon", 0.01, 0.0001, RED_MAX_EPSILON)
        iterations = _integer(body, "iterations", 3, 1, 10)
        step = _number(body, "step", min(0.0025, epsilon), 0.0001, epsilon)
        parameters = {"attack": attack_type, "epsilon": epsilon, "iterations": iterations, "step": step}
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    job_id = uuid.uuid4().hex
    with _lock:
        # Recheck while reserving the single worker slot so concurrent requests
        # cannot both pass the earlier check and queue duplicate Red jobs.
        phase = _state["phase"]
        active_run = _state.get("red_run_id")
        if _state["round_id"] != round_id or phase not in {"red", "blue", "complete"} or (
            phase != "red" and (not active_run or not (STATE_DIR / f"red-{active_run}.npz").is_file())
        ):
            return jsonify({"error": "The Red round changed before this attack was queued", "phase": phase}), 409
        if any(job.get("role") == "red" and job.get("status") in {"queued", "running"} for job in _jobs.values()):
            return jsonify({"error": "A Red attack is already running for this round"}), 409
        parameters["round_id"] = round_id
        _jobs[job_id] = {
            "job_id": job_id, "status": "queued", "role": "red",
            "parameters": parameters, "identity": identity,
        }
        future = _executor.submit(_run_red, job_id, parameters, identity)
        _jobs[job_id]["future"] = future
    return jsonify({"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}), 202


@app.get("/api/round/latest")
def latest_red_attack():
    with _lock:
        run_id = _state.get("red_run_id")
        result = _state.get("red_result")
        phase = _state["phase"]
    if not run_id or not result:
        return jsonify({"error": "No successful Red artifact is available yet", "phase": phase}), 404
    if not (STATE_DIR / f"red-{run_id}.npz").is_file():
        return jsonify({"error": "The active Red artifact is missing", "phase": phase}), 404
    return jsonify({"phase": phase, "result": _public_result(result), "artifact_url": f"/api/round/latest/artifact?run_id={run_id}"})


@app.get("/api/round/latest/artifact")
def latest_red_artifact():
    requested_id = request.args.get("run_id", "")
    with _lock:
        active_id = _state.get("red_run_id")
        phase = _state["phase"]
    if not active_id or requested_id != active_id or phase not in {"blue", "complete"}:
        return jsonify({"error": "No active Red artifact is available"}), 404
    artifact_path = STATE_DIR / f"red-{active_id}.npz"
    if not artifact_path.is_file():
        return jsonify({"error": "The active Red artifact is missing"}), 404
    return send_file(artifact_path, mimetype="application/octet-stream", as_attachment=True, download_name="red_submission.npz")


@app.get("/api/round/latest/successes")
def latest_successful_red_examples():
    """Give Blue only clean-correct samples that the active Red attack flipped."""
    requested_id = request.args.get("run_id", "")
    with _lock:
        active_id = _state.get("red_run_id")
        if not active_id or requested_id != active_id or _state["phase"] not in {"blue", "complete"}:
            return jsonify({"error": "No active Red artifact is available"}), 404
        artifact_path = STATE_DIR / f"red-{active_id}.npz"
        if not artifact_path.is_file():
            return jsonify({"error": "The active Red artifact is missing"}), 404
        with np.load(artifact_path, allow_pickle=False) as data:
            labels = data["labels"]
            clean_predictions = data["clean_predictions"]
            adv_predictions = data["adv_predictions"]
            accepted = (clean_predictions == labels) & (adv_predictions != clean_predictions)
            output = io.BytesIO()
            np.savez_compressed(
                output,
                x_clean=data["x_clean"][accepted],
                x_adv=data["x_adv"][accepted],
                labels=labels[accepted],
                clean_predictions=clean_predictions[accepted],
                adv_predictions=adv_predictions[accepted],
            )
    output.seek(0)
    return send_file(output, mimetype="application/octet-stream", as_attachment=True,
                     download_name="successful_red_examples.npz")


@app.post("/api/blue/defend")
def create_blue_defense():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    identity = _resolve_workspace_identity(
        request.headers.get("Authorization"), request.remote_addr
    )
    with _lock:
        if _state["phase"] not in {"blue", "complete"} or not _state.get("red_run_id"):
            return jsonify({"error": "Blue can start after a successful Red artifact is available", "phase": _state["phase"]}), 409
        if any(job.get("role") == "blue" and job.get("status") in {"queued", "running"} for job in _jobs.values()):
            return jsonify({"error": "A Blue defense evaluation is already running for this round"}), 409
        red_run_id = _state["red_run_id"]
        round_id = _state["round_id"]
    try:
        preprocess = str(body.get("preprocess", "none"))
        if preprocess not in PREPROCESSORS:
            raise ValueError(f"preprocess must be one of: {', '.join(sorted(PREPROCESSORS))}")
        detector = str(body.get("detector", "none"))
        if detector not in {"none", "consistency", "disagreement"}:
            raise ValueError("detector must be none, consistency, or disagreement")
        model = str(body.get("model", "robust"))
        if model not in BLUE_MODELS:
            raise ValueError("model must be standard or robust")
        if not (APP_ROOT / "models" / BLUE_MODELS[model]).is_file():
            raise ValueError(f"The {model} Blue checkpoint is unavailable")
        parameters = {
            "model": model,
            "preprocess": preprocess,
            "bits": _integer(body, "bits", 5, 2, 8),
            "detector": detector,
            "samples": _integer(body, "samples", 4, 2, 8),
            "sigma": _number(body, "sigma", 0.03, 0.0, 0.1),
            "threshold": _number(body, "threshold", 0.75, 0.5, 1.0),
            "confidence": _number(body, "confidence", 0.6, 0.0, 1.0),
            "round_id": round_id,
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    job_id = uuid.uuid4().hex
    with _lock:
        # The round might have reset while request parameters were validated.
        if (_state["phase"] not in {"blue", "complete"}
                or _state.get("red_run_id") != red_run_id
                or _state["round_id"] != round_id):
            return jsonify({"error": "The Blue phase changed before this defense was queued", "phase": _state["phase"]}), 409
        if any(job.get("role") == "blue" and job.get("status") in {"queued", "running"} for job in _jobs.values()):
            return jsonify({"error": "A Blue defense evaluation is already running for this round"}), 409
        parameters["round_id"] = round_id
        _jobs[job_id] = {
            "job_id": job_id, "status": "queued", "role": "blue",
            "parameters": parameters, "identity": identity,
        }
        future = _executor.submit(_run_blue, job_id, parameters, red_run_id, identity)
        _jobs[job_id]["future"] = future
    return jsonify({"job_id": job_id, "status": "queued", "poll_url": f"/api/jobs/{job_id}"}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _job_view(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.post("/api/admin/round/reset")
def reset_round():
    if not ADMIN_TOKEN:
        return jsonify({"error": "Round reset is disabled until XRAY_ADMIN_TOKEN is configured"}), 503
    supplied = request.headers.get("X-Exercise-Admin-Token", "")
    if not hmac.compare_digest(supplied, ADMIN_TOKEN):
        return jsonify({"error": "Invalid admin token"}), 403
    with _lock:
        previous_id = _state.get("round_id")
        previous_red_run = _state.get("red_run_id")
        _state.clear()
        _state.update(_default_state())
        _state["previous_round_id"] = previous_id
        _save_state(_state)
        if previous_red_run:
            (STATE_DIR / f"red-{previous_red_run}.npz").unlink(missing_ok=True)
        return jsonify({"status": "reset", "phase": _state["phase"], "round_id": _state["round_id"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
