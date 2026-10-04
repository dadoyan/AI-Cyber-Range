from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
import torch

from patchguard.model import BasicCNN
from patchguard.predict import patchguard_batch_predict
from range_device import select_device


logger = logging.getLogger("patchguard.evaluator")
ASSETS_DIR = Path(os.environ.get("PATCHGUARD_ASSETS_DIR", "/opt/patchguard-private"))
MODEL_LIMIT = int(os.environ.get("PATCHGUARD_MAX_MODEL_BYTES", str(16 * 1024 * 1024)))
WINDOW_SIZE = 16
STRIDE = 2
PATCH_SIZE = 12


def _evaluation_audit(clean_accuracy: float, robust_accuracy: float,
                      clean_threshold: float, robust_threshold: float,
                      clean_sample_count: int, robust_sample_count: int,
                      evaluation_seconds: float, evaluator_version: str) -> dict:
    criteria = [
        {
            "name": "clean_accuracy_minimum",
            "passed": clean_accuracy >= clean_threshold,
            "observed": float(clean_accuracy),
            "minimum": float(clean_threshold),
            "sample_count": int(clean_sample_count),
            "rule": "clean accuracy must be >= clean_threshold",
        },
        {
            "name": "adversarial_robust_accuracy_minimum",
            "passed": robust_accuracy >= robust_threshold,
            "observed": float(robust_accuracy),
            "minimum": float(robust_threshold),
            "sample_count": int(robust_sample_count),
            "rule": "accuracy on the hidden adversarial set must be >= robust_threshold",
        },
    ]
    failures = [criterion["name"] for criterion in criteria if not criterion["passed"]]
    success = not failures
    return {
        "schema_version": 1,
        "challenge": "patchguard",
        "success": success,
        "flag_returned": success,
        "summary": "All model acceptance criteria passed." if success else "Model rejected: " + ", ".join(failures),
        "criteria": criteria,
        "failure_reasons": failures,
        "evaluation_seconds": float(evaluation_seconds),
        "evaluator_version": str(evaluator_version),
    }


def _load_hidden_assets(directory: Path) -> dict[str, torch.Tensor]:
    path = directory / "hidden_eval.pt"
    if not path.is_file():
        raise RuntimeError("private hidden PatchGuard evaluation asset is missing")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("private PatchGuard evaluation asset has an invalid format")
    required = {
        "clean_images": (torch.float32, 4),
        "clean_labels": (torch.int64, 1),
        "adversarial_images": (torch.float32, 4),
        "adversarial_labels": (torch.int64, 1),
    }
    result: dict[str, torch.Tensor] = {}
    for name, (dtype, ndim) in required.items():
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.ndim != ndim:
            raise RuntimeError(f"private PatchGuard tensor {name} has an invalid format")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"private PatchGuard tensor {name} contains non-finite values")
        result[name] = tensor.contiguous()
    clean_images, clean_labels = result["clean_images"], result["clean_labels"]
    adv_images, adv_labels = result["adversarial_images"], result["adversarial_labels"]
    if (
        clean_images.shape[0] == 0
        or clean_images.shape[1:] != (1, 28, 28)
        or clean_labels.shape != (clean_images.shape[0],)
        or adv_images.shape[0] == 0
        or adv_images.shape[1:] != (1, 28, 28)
        or adv_labels.shape != (adv_images.shape[0],)
        or (clean_labels < 0).any()
        or (clean_labels > 9).any()
        or (adv_labels < 0).any()
        or (adv_labels > 9).any()
    ):
        raise RuntimeError("private PatchGuard evaluation tensors have inconsistent dimensions")
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    flag = os.environ.get("PATCHGUARD_FLAG")
    if not flag:
        raise RuntimeError("PATCHGUARD_FLAG must be configured")
    try:
        clean_threshold = float(os.environ["PATCHGUARD_CLEAN_THRESHOLD"])
        robust_threshold = float(os.environ["PATCHGUARD_ROBUST_THRESHOLD"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("PatchGuard accuracy thresholds must be configured as numbers") from exc
    if not 0.0 <= clean_threshold <= 1.0 or not 0.0 <= robust_threshold <= 1.0:
        raise RuntimeError("PatchGuard accuracy thresholds must be between 0 and 1")
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    reference = BasicCNN(patch_size=WINDOW_SIZE, in_channels=1).cpu()
    hidden = _load_hidden_assets(ASSETS_DIR)
    device = select_device()
    hidden = {name: tensor.to(device) for name, tensor in hidden.items()}
    app.state.device = device
    app.state.reference_model = reference
    app.state.hidden = hidden
    app.state.clean_threshold = clean_threshold
    app.state.robust_threshold = robust_threshold
    app.state.flag = flag
    yield


app = FastAPI(
    title="PatchGuard Protected Evaluator",
    description="Evaluates submitted BasicCNN state_dicts on private MNIST examples.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(app.state.device)}


def _decode_state_dict(data: bytes, reference_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(__import__("io").BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload must be a PyTorch state_dict.") from exc
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
        raise HTTPException(status_code=400, detail="Upload must be a PyTorch state_dict.")
    expected = reference_model.state_dict()
    if set(state) != set(expected):
        raise HTTPException(status_code=400, detail="Model keys do not match the required classifier.")
    validated: dict[str, torch.Tensor] = {}
    for key, expected_tensor in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected_tensor.shape:
            raise HTTPException(status_code=400, detail="Model tensor shapes do not match the required classifier.")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise HTTPException(status_code=400, detail="Model tensors must contain finite floating-point values.")
        validated[key] = value.detach().to(dtype=expected_tensor.dtype, device="cpu")
    return validated


def _mlflow_log(
    *,
    tracking_uri: str,
    experiment_name: str,
    state_dict_bytes: bytes,
    model_sha256: str,
    clean_accuracy: float,
    robust_accuracy: float,
    success: bool,
    evaluation_seconds: float,
    clean_threshold: float,
    robust_threshold: float,
    evaluator_version: str,
    evaluation_audit: dict,
    ctfd_user_id: str,
    ctfd_username: str,
) -> None:
    """Best-effort MLflow audit; telemetry never controls the challenge result."""
    if not tracking_uri:
        return
    run_id = None
    client = None
    temp_dir: Path | None = None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(experiment_name)
            except Exception:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
        run = client.create_run(
            experiment_id=str(experiment_id),
            tags={
                "mlflow.runName": f"patchguard-attempt-{uuid.uuid4().hex[:10]}",
                "challenge": "patchguard",
                "ctfd_user_id": str(ctfd_user_id),
                "ctfd_username": ctfd_username,
                "model_sha256": model_sha256,
                "window_size": str(WINDOW_SIZE),
                "stride": str(STRIDE),
                "patch_size": str(PATCH_SIZE),
                "evaluator_version": evaluator_version,
                "clean_threshold": str(clean_threshold),
                "robust_threshold": str(robust_threshold),
                "evaluation_success": str(bool(success)).lower(),
                "failure_reasons": ",".join(evaluation_audit["failure_reasons"]) or "none",
            },
        )
        run_id = run.info.run_id
        for key, value in {
            "clean_accuracy": clean_accuracy,
            "robust_accuracy": robust_accuracy,
            "success": float(success),
            "evaluation_seconds": evaluation_seconds,
        }.items():
            client.log_metric(run_id, key, float(value))
        temp_dir = Path(tempfile.mkdtemp(prefix="patchguard-audit-"))
        model_path = temp_dir / "model_state_dict.pt"
        evaluation_path = temp_dir / "evaluation.json"
        model_path.write_bytes(state_dict_bytes)
        evaluation_path.write_text(
            json.dumps(evaluation_audit, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        client.log_artifact(run_id, str(model_path), artifact_path="submitted")
        client.log_artifact(run_id, str(evaluation_path), artifact_path="evaluation")
        client.set_terminated(run_id, status="FINISHED")
        run_id = None
    except Exception:
        logger.warning("MLflow PatchGuard logging failed; evaluation is unaffected", exc_info=True)
    finally:
        if run_id is not None and client is not None:
            try:
                client.set_terminated(run_id, status="FINISHED")
            except Exception:
                logger.warning("Could not close incomplete PatchGuard MLflow run", exc_info=True)
        if temp_dir is not None:
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except OSError:
                logger.warning("Could not remove temporary PatchGuard audit artifacts", exc_info=True)


@app.post("/score")
async def score(
    background_tasks: BackgroundTasks,
    model: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    username: str | None = Form(default=None),
):
    data = await model.read(MODEL_LIMIT + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Model upload is empty.")
    if len(data) > MODEL_LIMIT:
        raise HTTPException(status_code=413, detail="Model upload exceeds the size limit.")
    reference_model: torch.nn.Module = app.state.reference_model
    state = _decode_state_dict(data, reference_model)
    candidate = BasicCNN(patch_size=WINDOW_SIZE, in_channels=1).cpu()
    try:
        candidate.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Model is incompatible with the required classifier.") from exc
    candidate.eval()
    hidden: dict[str, torch.Tensor] = app.state.hidden
    candidate.to(hidden["clean_images"].device)
    started = time.perf_counter()
    clean_predictions = patchguard_batch_predict(
        candidate, hidden["clean_images"], WINDOW_SIZE, STRIDE
    )
    adv_predictions = patchguard_batch_predict(
        candidate, hidden["adversarial_images"], WINDOW_SIZE, STRIDE
    )
    clean_accuracy = float(
        (clean_predictions == hidden["clean_labels"]).float().mean().item()
    )
    robust_accuracy = float(
        (adv_predictions == hidden["adversarial_labels"]).float().mean().item()
    )
    evaluation_seconds = time.perf_counter() - started
    success = (
        clean_accuracy >= app.state.clean_threshold
        and robust_accuracy >= app.state.robust_threshold
    )
    evaluator_version = os.environ.get("PATCHGUARD_EVALUATOR_VERSION", "1")
    clean_sample_count = int(hidden["clean_labels"].numel())
    robust_sample_count = int(hidden["adversarial_labels"].numel())
    evaluation_audit = _evaluation_audit(
        clean_accuracy, robust_accuracy,
        app.state.clean_threshold, app.state.robust_threshold,
        clean_sample_count, robust_sample_count, evaluation_seconds,
        evaluator_version,
    )
    result = {
        "clean_accuracy": round(clean_accuracy, 6),
        "robust_accuracy": round(robust_accuracy, 6),
        "success": success,
        "evaluation_seconds": round(evaluation_seconds, 4),
    }
    if success:
        result["flag"] = app.state.flag
    background_tasks.add_task(
        _mlflow_log,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "patchguard"),
        state_dict_bytes=data,
        model_sha256=hashlib.sha256(data).hexdigest(),
        clean_accuracy=clean_accuracy,
        robust_accuracy=robust_accuracy,
        success=success,
        evaluation_seconds=evaluation_seconds,
        clean_threshold=app.state.clean_threshold,
        robust_threshold=app.state.robust_threshold,
        evaluator_version=evaluator_version,
        evaluation_audit=evaluation_audit,
        ctfd_user_id=(user_id or "anonymous").strip() or "anonymous",
        ctfd_username=(username or "anonymous").strip() or "anonymous",
    )
    return result
