import asyncio
from io import BytesIO
import json
from pathlib import Path

import pytest
from types import SimpleNamespace
import torch
from fastapi import BackgroundTasks, HTTPException
from starlette.datastructures import UploadFile

from patchguard.evaluator import app as evaluator
from patchguard.model import BasicCNN
from patchguard.predict import patchguard_batch_predict


def state_bytes(state):
    stream = BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


class FakeMlflowClient:
    def __init__(self):
        self.artifacts = {}
        self.metrics = {}

    def get_experiment_by_name(self, _name):
        return SimpleNamespace(experiment_id="pg-test")

    def create_run(self, experiment_id, tags):
        return SimpleNamespace(info=SimpleNamespace(run_id="pg-run"))

    def log_metric(self, run_id, key, value):
        self.metrics[key] = value

    def log_artifact(self, run_id, path, artifact_path=None):
        name = Path(path).name
        key = f"{artifact_path}/{name}" if artifact_path else name
        self.artifacts[key] = Path(path).read_bytes()

    def set_terminated(self, run_id, status):
        self.status = status


def initialize_service(monkeypatch, *, reject=False):
    torch.manual_seed(11)
    candidate = BasicCNN(16, 1).cpu().eval()
    images = torch.zeros(3, 1, 28, 28)
    predictions = patchguard_batch_predict(candidate, images)
    labels = (predictions + 1) % 10 if reject else predictions
    hidden = {
        "clean_images": images,
        "clean_labels": labels,
        "adversarial_images": images.clone(),
        "adversarial_labels": labels.clone(),
    }
    evaluator.app.state.reference_model = BasicCNN(16, 1).cpu()
    evaluator.app.state.hidden = hidden
    evaluator.app.state.clean_threshold = 0.01 if reject else 0.0
    evaluator.app.state.robust_threshold = 0.0
    evaluator.app.state.flag = "flag{test-only}"
    return candidate


def upload(data):
    return UploadFile(filename="untrusted-name.pt", file=BytesIO(data))


def run_score(data, *, user_id="4", username="student4"):
    tasks = BackgroundTasks()
    result = asyncio.run(evaluator.score(tasks, upload(data), user_id, username))
    asyncio.run(tasks())
    return result


def test_valid_state_dict_passes_both_metrics_and_returns_flag(monkeypatch):
    model = initialize_service(monkeypatch)
    logged = []
    monkeypatch.setattr(evaluator, "_mlflow_log", lambda **kwargs: logged.append(kwargs))
    result = run_score(state_bytes(model.state_dict()))
    assert result["success"] is True
    assert result["flag"] == "flag{test-only}"
    assert result["clean_accuracy"] == 1.0
    assert result["robust_accuracy"] == 1.0
    assert set(result) == {"clean_accuracy", "robust_accuracy", "success", "evaluation_seconds", "flag"}
    assert logged[0]["ctfd_user_id"] == "4"
    assert logged[0]["ctfd_username"] == "student4"
    audit = logged[0]["evaluation_audit"]
    assert audit["success"] is True
    assert audit["flag_returned"] is True
    assert [criterion["name"] for criterion in audit["criteria"]] == [
        "clean_accuracy_minimum", "adversarial_robust_accuracy_minimum"
    ]
    assert all(criterion["passed"] for criterion in audit["criteria"])


def test_valid_model_that_fails_hidden_clean_threshold_gets_no_flag(monkeypatch):
    model = initialize_service(monkeypatch, reject=True)
    logged = []
    monkeypatch.setattr(evaluator, "_mlflow_log", lambda **kwargs: logged.append(kwargs))
    result = run_score(state_bytes(model.state_dict()))
    assert result["clean_accuracy"] == 0.0
    assert result["success"] is False
    assert "flag" not in result
    audit = logged[0]["evaluation_audit"]
    assert audit["success"] is False
    assert audit["flag_returned"] is False
    assert audit["failure_reasons"] == ["clean_accuracy_minimum"]


def test_patchguard_evaluation_audit_records_each_comparison():
    audit = evaluator._evaluation_audit(
        clean_accuracy=0.6,
        robust_accuracy=0.4,
        clean_threshold=0.7,
        robust_threshold=0.5,
        clean_sample_count=20,
        robust_sample_count=25,
        evaluation_seconds=1.25,
        evaluator_version="test-version",
    )
    assert audit["success"] is False
    assert audit["failure_reasons"] == [
        "clean_accuracy_minimum", "adversarial_robust_accuracy_minimum"
    ]
    assert audit["criteria"][0]["observed"] == 0.6
    assert audit["criteria"][0]["minimum"] == 0.7
    assert audit["criteria"][1]["sample_count"] == 25


def test_patchguard_mlflow_run_contains_evaluation_json(monkeypatch):
    import mlflow.tracking

    client = FakeMlflowClient()
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda **_kwargs: client)
    audit = evaluator._evaluation_audit(
        clean_accuracy=0.6, robust_accuracy=0.8,
        clean_threshold=0.7, robust_threshold=0.75,
        clean_sample_count=10, robust_sample_count=12,
        evaluation_seconds=0.4, evaluator_version="test-version",
    )
    evaluator._mlflow_log(
        tracking_uri="http://mlflow:5000", experiment_name="patchguard-test",
        state_dict_bytes=b"state-dict", model_sha256="digest",
        clean_accuracy=0.6, robust_accuracy=0.8, success=False,
        evaluation_seconds=0.4, clean_threshold=0.7, robust_threshold=0.75,
        evaluator_version="test-version", evaluation_audit=audit,
        ctfd_user_id="9", ctfd_username="student9",
    )
    stored = json.loads(client.artifacts["evaluation/evaluation.json"])
    assert stored["failure_reasons"] == ["clean_accuracy_minimum"]
    assert stored["criteria"][0]["observed"] == 0.6
    assert "submitted/model_state_dict.pt" in client.artifacts
    assert client.status == "FINISHED"


@pytest.mark.parametrize(
    "state_factory",
    [
        lambda model: {"wrong": torch.zeros(1)},
        lambda model: {key: value for key, value in list(model.state_dict().items())[1:]},
        lambda model: {**model.state_dict(), "extra": torch.zeros(1)},
        lambda model: {
            **model.state_dict(),
            next(iter(model.state_dict())): torch.zeros(1),
        },
        lambda model: {
            **model.state_dict(),
            next(iter(model.state_dict())): torch.full_like(
                next(iter(model.state_dict().values())), float("nan")
            ),
        },
        lambda model: {
            **model.state_dict(),
            next(iter(model.state_dict())): torch.full_like(
                next(iter(model.state_dict().values())), float("inf")
            ),
        },
    ],
)
def test_malformed_state_dicts_are_rejected_without_flag(monkeypatch, state_factory):
    model = initialize_service(monkeypatch)
    data = state_bytes(state_factory(model))
    with pytest.raises(HTTPException) as exc:
        evaluator._decode_state_dict(data, evaluator.app.state.reference_model)
    assert exc.value.status_code == 400


@pytest.mark.parametrize("data", [b"", b"random bytes", b"\x80\x04not-a-torch-file"])
def test_empty_and_random_uploads_are_rejected(monkeypatch, data):
    model = initialize_service(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        evaluator._decode_state_dict(data, evaluator.app.state.reference_model)
    assert exc.value.status_code == 400


def test_mlflow_failure_is_swallowed_and_success_flag_remains(monkeypatch, caplog):
    model = initialize_service(monkeypatch)
    import mlflow.tracking

    def unavailable(**_kwargs):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", unavailable)
    result = run_score(state_bytes(model.state_dict()))
    assert result["success"] is True
    assert result["flag"] == "flag{test-only}"
    assert "MLflow PatchGuard logging failed" in caplog.text
