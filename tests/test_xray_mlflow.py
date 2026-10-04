import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from xray_red_blue import mlflow_audit


class FakeMlflowClient:
    def __init__(self):
        self.run_id = "fake-run-id"
        self.tags = None
        self.metrics = []
        self.params = []
        self.artifacts = {}
        self.source_paths = []
        self.terminal_status = None

    def get_experiment_by_name(self, _name):
        return SimpleNamespace(experiment_id="17")

    def create_run(self, _experiment_id, tags=None, run_name=None):
        self.tags = dict(tags or {})
        self.tags["mlflow.runName"] = run_name
        return SimpleNamespace(info=SimpleNamespace(run_id=self.run_id))

    def log_batch(self, _run_id, metrics=(), params=(), **_kwargs):
        self.metrics.extend(metrics)
        self.params.extend(params)

    def log_artifact(self, _run_id, path, artifact_path=None):
        source = Path(path)
        self.source_paths.append(source)
        key = f"{artifact_path}/{source.name}" if artifact_path else source.name
        self.artifacts[key] = source.read_bytes()

    def set_terminated(self, _run_id, status):
        self.terminal_status = status


def test_red_logs_metrics_npz_and_safe_evaluation_artifact():
    client = FakeMlflowClient()
    images = np.zeros((2, 3, 4, 4), dtype=np.float32)
    adversarial = np.ones_like(images)
    with patch.object(mlflow_audit, "_new_client", return_value=client):
        run_id = mlflow_audit.log_red_attempt(
            tracking_uri="http://mlflow:5000",
            experiment_name="xray-redblue",
            round_id="round-1",
            public_run_id="red-public-id",
            parameters={"attack": "FGSM", "epsilon": 0.01},
            metrics={"attack_success_rate": 0.5, "success": True},
            evaluation={
                "success": True,
                "decision_audit": {"success": True, "failure_reasons": [],
                                   "criteria": [{"name": "minimum_asr", "passed": True}]},
                "flag": "must-not-be-logged",
            },
            arrays={"x_clean": images, "x_adv": adversarial},
        )

    assert run_id == "fake-run-id"
    assert client.tags["phase"] == "red"
    assert client.tags["challenge"] == "xray-redblue"
    assert client.tags["role"] == "red"
    assert client.terminal_status == "FINISHED"
    assert {metric.key for metric in client.metrics} == {"attack_success_rate", "success"}
    assert "submission/red_submission.npz" in client.artifacts
    with np.load(io.BytesIO(client.artifacts["submission/red_submission.npz"])) as artifact:
        np.testing.assert_array_equal(artifact["x_adv"], adversarial)
    evaluation = json.loads(client.artifacts["submission/red_evaluation.json"])
    assert evaluation["evaluation"]["success"] is True
    assert evaluation["evaluation"]["decision_audit"]["failure_reasons"] == []
    assert client.tags["evaluation_success"] == "true"
    assert client.tags["failure_reasons"] == "none"
    assert "must-not-be-logged" not in client.artifacts["submission/red_evaluation.json"].decode()
    assert all(not path.exists() for path in client.source_paths)


def test_blue_logs_defense_configuration_and_metrics():
    client = FakeMlflowClient()
    with patch.object(mlflow_audit, "_new_client", return_value=client):
        run_id = mlflow_audit.log_blue_attempt(
            tracking_uri="http://mlflow:5000",
            experiment_name="xray-redblue",
            round_id="round-1",
            public_run_id="blue-public-id",
            related_red_run_id="red-public-id",
            parameters={"model": "standard", "detector": "disagreement"},
            metrics={"detection_rate": 0.2, "false_positive_rate": 0.1, "success": True},
            evaluation={
                "success": True,
                "decision_audit": {"success": True, "failure_reasons": [],
                                   "criteria": [{"name": "clean_accuracy", "passed": True}]},
                "flag": "must-not-be-logged",
            },
            identity={
                "ctfd_user_id": 42,
                "ctfd_username": "blue_student",
                "workspace_id": "u42",
                "identity_source": "arena_token",
            },
        )

    assert run_id == "fake-run-id"
    assert client.tags["phase"] == "blue"
    assert client.tags["role"] == "blue"
    assert client.tags["red_run_id"] == "red-public-id"
    assert client.tags["ctfd_user_id"] == "42"
    assert client.tags["ctfd_username"] == "blue_student"
    assert client.tags["workspace_id"] == "u42"
    assert client.tags["identity_source"] == "arena_token"
    assert client.tags["identity_status"] == "mapped"
    assert client.tags["evaluation_audit_status"] == "complete"
    assert client.terminal_status == "FINISHED"
    assert {metric.key for metric in client.metrics} == {"detection_rate", "false_positive_rate", "success"}
    evaluation = json.loads(client.artifacts["submission/blue_evaluation.json"])
    assert evaluation["parameters"] == {"model": "standard", "detector": "disagreement"}
    assert evaluation["evaluation"]["success"] is True
    assert evaluation["evaluation"]["decision_audit"]["criteria"][0]["passed"] is True


def test_unresolved_workspace_identity_is_explicitly_tagged_anonymous():
    client = FakeMlflowClient()
    with patch.object(mlflow_audit, "_new_client", return_value=client):
        mlflow_audit.log_red_attempt(
            tracking_uri="http://mlflow:5000",
            experiment_name="xray-redblue",
            round_id="round-1",
            public_run_id="red-public-id",
            parameters={},
            metrics={"success": False},
            evaluation={"success": False},
            identity=None,
            arrays=None,
        )

    assert client.tags["ctfd_user_id"] == "anonymous"
    assert client.tags["ctfd_username"] == "anonymous"
    assert client.tags["identity_status"] == "unresolved"
    assert client.tags["failure_reasons"] == "evaluation_audit_missing"
    assert client.tags["evaluation_audit_status"] == "missing"


def test_mlflow_unavailable_does_not_raise_or_change_evaluation(caplog):
    with patch.object(mlflow_audit, "_new_client", side_effect=RuntimeError("offline")):
        run_id = mlflow_audit.log_blue_attempt(
            tracking_uri="http://mlflow:5000",
            experiment_name="xray-redblue",
            round_id="round-1",
            public_run_id="blue-public-id",
            related_red_run_id="red-public-id",
            parameters={},
            metrics={"success": False},
            evaluation={"success": False},
        )
    assert run_id is None
    assert "MLflow X-Ray blue logging failed" in caplog.text
