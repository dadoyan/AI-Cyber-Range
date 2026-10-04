"""Best-effort MLflow audit logging for X-Ray Red/Blue evaluations."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

import numpy as np


logger = logging.getLogger("xray-red-blue.mlflow")


def _new_client(tracking_uri: str):
    # Lazy import keeps MLflow optional for challenge evaluation.
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def _experiment_id(client, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(name)
    except Exception:
        # Another worker may have created the experiment at the same time.
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return experiment.experiment_id


def _safe_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Keep flags and internal errors out of administrator telemetry artifacts."""
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"flag", "error"}
    }


def _log_attempt(
    *,
    phase: str,
    tracking_uri: str,
    experiment_name: str,
    round_id: str,
    public_run_id: str,
    parameters: dict[str, Any],
    metrics: dict[str, float | int | bool],
    evaluation: dict[str, Any],
    identity: dict[str, Any] | None = None,
    arrays: dict[str, np.ndarray] | None = None,
    related_red_run_id: str | None = None,
) -> str | None:
    client = None
    mlflow_run_id = None
    try:
        from mlflow.entities import Metric, Param

        client = _new_client(tracking_uri)
        experiment_id = _experiment_id(client, experiment_name)
        safe_params = {
            key: str(value)
            for key, value in parameters.items()
            if value is not None and key != "round_id"
        }
        run_name = f"xray-{phase}-attempt-{uuid.uuid4().hex[:10]}"
        tags = {
            "challenge": "xray-redblue",
            "phase": phase,
            "role": phase,
            "round_id": str(round_id),
            "public_run_id": str(public_run_id),
            "success": str(bool(evaluation.get("success", False))).lower(),
        }
        identity_user_id = identity.get("ctfd_user_id") if identity else None
        identity_username = str(identity.get("ctfd_username", "")).strip() if identity else ""
        tags["ctfd_user_id"] = str(identity_user_id) if identity_user_id is not None else "anonymous"
        tags["ctfd_username"] = identity_username or "anonymous"
        tags["identity_status"] = (
            "mapped" if identity_user_id is not None and identity_username else "unresolved"
        )
        decision_audit = evaluation.get("decision_audit", {})
        if decision_audit:
            tags["evaluation_success"] = str(bool(decision_audit.get("success", evaluation.get("success", False)))).lower()
            tags["failure_reasons"] = ",".join(decision_audit.get("failure_reasons", [])) or "none"
            tags["evaluation_audit_status"] = "complete"
        else:
            evaluation_success = bool(evaluation.get("success", False))
            tags["evaluation_success"] = str(evaluation_success).lower()
            tags["failure_reasons"] = "none" if evaluation_success else "evaluation_audit_missing"
            tags["evaluation_audit_status"] = "missing"
        if identity:
            # Identity is resolved by the Launcher from an opaque workspace
            # credential or a managed Docker peer address, never from request JSON.
            for key in ("workspace_id", "identity_source"):
                value = identity.get(key)
                if value is not None and str(value).strip():
                    tags[key] = str(value)[:128]
        if related_red_run_id:
            tags["red_run_id"] = str(related_red_run_id)
        run = client.create_run(experiment_id, tags=tags, run_name=run_name)
        mlflow_run_id = run.info.run_id

        timestamp = int(time.time() * 1000)
        metric_entities = []
        for key, value in metrics.items():
            numeric = float(value)
            if math.isfinite(numeric):
                metric_entities.append(Metric(key, numeric, timestamp, 0))
        params = [Param(key, value[:500]) for key, value in safe_params.items()]
        if metric_entities or params:
            client.log_batch(
                mlflow_run_id,
                metrics=metric_entities,
                params=params,
                synchronous=True,
            )

        with tempfile.TemporaryDirectory(prefix="xray-mlflow-") as temporary:
            directory = Path(temporary)
            evaluation_path = directory / f"{phase}_evaluation.json"
            evaluation_path.write_text(
                json.dumps(
                    {
                        "challenge": "xray-redblue",
                        "phase": phase,
                        "parameters": safe_params,
                        "evaluation": _safe_evaluation(evaluation),
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            client.log_artifact(mlflow_run_id, str(evaluation_path), artifact_path="submission")
            if arrays is not None:
                submission_path = directory / "red_submission.npz"
                np.savez_compressed(submission_path, **arrays)
                client.log_artifact(mlflow_run_id, str(submission_path), artifact_path="submission")

        client.set_terminated(mlflow_run_id, status="FINISHED")
        return mlflow_run_id
    except Exception:
        logger.warning(
            "MLflow X-Ray %s logging failed; evaluation is unaffected",
            phase,
            exc_info=True,
        )
        if client is not None and mlflow_run_id is not None:
            try:
                client.set_terminated(mlflow_run_id, status="FAILED")
            except Exception:
                logger.warning("Could not close incomplete X-Ray MLflow run", exc_info=True)
        return None


def log_red_attempt(
    *,
    tracking_uri: str,
    experiment_name: str,
    round_id: str,
    public_run_id: str,
    parameters: dict[str, Any],
    metrics: dict[str, float | int | bool],
    evaluation: dict[str, Any],
    arrays: dict[str, np.ndarray],
    identity: dict[str, Any] | None = None,
) -> str | None:
    """Log Red's evaluated adversarial batch and its result, without the flag."""
    return _log_attempt(
        phase="red",
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        round_id=round_id,
        public_run_id=public_run_id,
        parameters=parameters,
        metrics=metrics,
        evaluation=evaluation,
        identity=identity,
        arrays=arrays,
    )


def log_blue_attempt(
    *,
    tracking_uri: str,
    experiment_name: str,
    round_id: str,
    public_run_id: str,
    related_red_run_id: str,
    parameters: dict[str, Any],
    metrics: dict[str, float | int | bool],
    evaluation: dict[str, Any],
    identity: dict[str, Any] | None = None,
) -> str | None:
    """Log Blue's submitted defense configuration and measured result."""
    return _log_attempt(
        phase="blue",
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        round_id=round_id,
        public_run_id=public_run_id,
        related_red_run_id=related_red_run_id,
        parameters=parameters,
        metrics=metrics,
        evaluation=evaluation,
        identity=identity,
    )
