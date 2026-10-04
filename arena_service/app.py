from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("red-blue-arena")

DATA_DIR = Path(os.getenv("ARENA_DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "arena.db"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
EVALUATOR_URL = os.getenv("XRAY_EVALUATOR_URL", "http://xray-redblue:5000").rstrip("/")
ARENA_INTERNAL_KEY = os.getenv("ARENA_INTERNAL_KEY", "")
ARENA_LAUNCHER_KEY = os.getenv("ARENA_LAUNCHER_KEY", "")
ARENA_ADMIN_KEY = os.getenv("ARENA_ADMIN_KEY", "")
XRAY_RED_FLAG = os.getenv("XRAY_RED_FLAG", "")
XRAY_BLUE_FLAG = os.getenv("XRAY_BLUE_FLAG", "")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "red_blue_arena")
MAX_IMAGE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 1400 * 1024
EVALUATOR_TIMEOUT = max(5, int(os.getenv("XRAY_EVALUATOR_TIMEOUT", "180")))
EVALUATOR_VERSION = "arena-1"

app = FastAPI(title="AI Cyber Range Red/Blue Arena", version="1.0.0")
_match_locks: dict[str, threading.RLock] = {}
_match_locks_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=20000")
    return connection


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paired' CHECK(mode IN ('paired','solo')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS participants (
                ctfd_user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                workspace_id TEXT,
                token_hash TEXT UNIQUE,
                match_id TEXT REFERENCES matches(id) ON DELETE SET NULL,
                role TEXT CHECK(role IN ('red','blue') OR role IS NULL),
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attacks (
                id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                sequence_number INTEGER NOT NULL,
                red_user_id INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                attack_name TEXT NOT NULL,
                attack_parameters_json TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                mlflow_run_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(match_id, sequence_number)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending_attack_per_match
                ON attacks(match_id) WHERE status='pending_blue';
            CREATE TABLE IF NOT EXISTS blue_responses (
                id TEXT PRIMARY KEY,
                match_id TEXT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                attack_id TEXT NOT NULL UNIQUE REFERENCES attacks(id) ON DELETE CASCADE,
                sequence_number INTEGER NOT NULL,
                blue_user_id INTEGER NOT NULL,
                defense_parameters_json TEXT NOT NULL,
                protected_result_json TEXT NOT NULL,
                mlflow_run_id TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS attacks_match_sequence
                ON attacks(match_id, sequence_number DESC);
            """
        )
        # Existing range volumes predate solo matches. Preserve their history.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(matches)")}
        if "mode" not in columns:
            conn.execute("ALTER TABLE matches ADD COLUMN mode TEXT NOT NULL DEFAULT 'paired' CHECK(mode IN ('paired','solo'))")


init_db()


def _constant_key_matches(provided: str, expected: str) -> bool:
    import hmac

    return bool(expected) and hmac.compare_digest(provided, expected)


def require_key(provided: str, expected: str, message: str) -> None:
    if not _constant_key_matches(provided, expected):
        raise HTTPException(status_code=403, detail=message)


def arena_match_lock(match_id: str) -> threading.RLock:
    with _match_locks_lock:
        return _match_locks.setdefault(match_id, threading.RLock())


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A workspace Arena token is required.")
    token = authorization[7:].strip()
    if not 24 <= len(token) <= 256:
        raise HTTPException(status_code=401, detail="The Arena token is invalid.")
    return token


def participant(authorization: str | None, role: str | None = None) -> dict[str, Any]:
    token = bearer_token(authorization)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with db_connection() as conn:
        row = conn.execute(
            """SELECT p.*, m.status AS match_status, m.mode AS match_mode
               FROM participants p LEFT JOIN matches m ON m.id=p.match_id
               WHERE p.token_hash=?""",
            (token_hash,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="The workspace is not registered with Arena.")
    identity = dict(row)
    if not identity.get("match_id") or not identity.get("role"):
        raise HTTPException(status_code=403, detail="An instructor has not assigned this user to a Red/Blue match yet.")
    if identity.get("match_status") != "active":
        raise HTTPException(status_code=403, detail="This Arena match is not active.")
    if role and identity["role"] != role and identity["match_mode"] != "solo":
        raise HTTPException(status_code=403, detail=f"This operation is available only to the {role.title()} team.")
    return identity


def evaluator_request(method: str, path: str, **kwargs):
    if not ARENA_INTERNAL_KEY:
        raise HTTPException(status_code=503, detail="Protected evaluator is not configured.")
    headers = dict(kwargs.pop("headers", {}) or {})
    headers["X-Arena-Internal-Key"] = ARENA_INTERNAL_KEY
    try:
        return requests.request(
            method,
            f"{EVALUATOR_URL}{path}",
            headers=headers,
            timeout=kwargs.pop("timeout", EVALUATOR_TIMEOUT),
            **kwargs,
        )
    except requests.RequestException as exc:
        logger.warning("Protected X-Ray evaluator request failed: %s", exc)
        raise HTTPException(status_code=503, detail="Protected evaluator is temporarily unavailable.") from exc


def _json_response_from_evaluator(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": "Protected evaluator returned an invalid response."}
    if response.status_code >= 400:
        raise HTTPException(status_code=422 if response.status_code in {400, 422} else 503, detail=payload.get("error", "Protected evaluation failed."))
    return payload


def _validate_png_envelope(raw: bytes) -> None:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="PNG must be non-empty and no larger than 1 MiB.")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=422, detail="Upload a PNG image.")


def _read_safe_json(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return value


async def read_bounded_json(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_JSON_BYTES:
                raise HTTPException(status_code=413, detail="Request body is too large.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length.") from None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_JSON_BYTES:
            raise HTTPException(status_code=413, detail="Request body is too large.")
        chunks.append(chunk)
    return _read_safe_json(b"".join(chunks))


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{name} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} must be numeric.") from None
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be between {minimum} and {maximum}.")
    return result


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{name} must be a whole number.")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} must be a whole number.") from None
    if str(result) != str(value).strip() or not minimum <= result <= maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be between {minimum} and {maximum}.")
    return result


def _store_attack_artifact(match_id: str, sequence: int, raw: bytes) -> Path:
    directory = (ARTIFACTS_DIR / match_id / "red").resolve()
    root = ARTIFACTS_DIR.resolve()
    if root not in directory.parents:
        raise RuntimeError("Resolved Arena artifact directory escaped the storage root")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"attack_{sequence:03d}.png"
    temporary = directory / f".{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return destination


def _metric_values(run_id: str, metrics: dict[str, float]) -> None:
    from mlflow.entities import Metric

    timestamp = int(time.time() * 1000)
    entities = [Metric(name, float(value), timestamp, 0) for name, value in metrics.items() if math.isfinite(float(value))]
    if entities:
        _mlflow_client().log_batch(run_id, metrics=entities, params=[], synchronous=True)


def _mlflow_client():
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=MLFLOW_URI)


def _mlflow_experiment(client) -> str:
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment:
        return experiment.experiment_id
    try:
        return client.create_experiment(MLFLOW_EXPERIMENT)
    except Exception:
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if not experiment:
            raise
        return experiment.experiment_id


def _arena_decision_audit(role: str, evaluation: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Explain the already-computed Arena outcome for the admin MLflow artifact."""
    if role == "red":
        true_index = int(evaluation["true_class_index"])
        adversarial_index = int(evaluation["adversarial_class_index"])
        measured_linf = float(evaluation["measured_linf"])
        maximum_linf = float(evaluation["max_linf"])
        linf_tolerance = float(evaluation.get("linf_tolerance", 1e-8))
        criteria = [
            {"name": "approved_clean_source", "passed": bool(evaluation.get("valid")),
             "observed": {"valid": bool(evaluation.get("valid")),
                          "true_class_index": true_index, "true_class_label": evaluation.get("true_class_label")},
             "rule": "source must be approved and correctly classified before the attack"},
            {"name": "standard_model_misclassification", "passed": adversarial_index != true_index,
             "observed": {"class_index": adversarial_index,
                          "label": evaluation.get("adversarial_class_label"),
                          "probability": evaluation.get("adversarial_probability")},
             "expected_true_class_index": true_index,
             "rule": "adversarial top-1 class must differ from the source label"},
            {"name": "maximum_measured_linf", "passed": measured_linf <= maximum_linf + linf_tolerance,
             "observed": measured_linf, "maximum": maximum_linf,
             "tolerance": linf_tolerance,
             "effective_maximum": maximum_linf + linf_tolerance,
             "declared_epsilon": params.get("epsilon"),
             "rule": "measured L-infinity distance must be <= exercise budget plus numeric tolerance"},
        ]
        success = all(item["passed"] for item in criteria)
        challenge = "xray-red-blue-arena-red"
    else:
        true_index = int(evaluation["true_class_index"])
        threshold = float(params["abstain_threshold"])
        clean_index = int(evaluation["clean_prediction_index"])
        clean_confidence = float(evaluation["clean_vote_confidence"])
        attack_survived = bool(evaluation["attack_survived"])
        criteria = [
            {"name": "clean_label_preserved", "passed": clean_index == true_index,
             "observed": {"class_index": clean_index, "label": evaluation.get("clean_prediction_label")},
             "expected_class_index": true_index,
             "rule": "smoothed clean prediction must match the known source label"},
            {"name": "clean_confidence_threshold", "passed": clean_confidence >= threshold,
             "observed": clean_confidence, "minimum": threshold,
             "rule": "clean vote confidence must be >= abstain threshold"},
            {"name": "adversarial_attack_blocked", "passed": not attack_survived,
             "observed": {"attack_survived": attack_survived,
                          "defended_prediction": evaluation.get("defended_prediction_label"),
                          "defended_vote_confidence": evaluation.get("defended_vote_confidence")},
             "rule": "attack must not retain a confident wrong defended prediction"},
        ]
        success = bool(evaluation.get("defense_success"))
        challenge = "xray-red-blue-arena-blue"
    failures = [item["name"] for item in criteria if not item["passed"]]
    return {
        "schema_version": 1,
        "challenge": challenge,
        "success": success,
        "summary": "All acceptance criteria passed." if success else "Evaluation rejected: " + ", ".join(failures),
        "criteria": criteria,
        "failure_reasons": failures,
        "parameters": params,
        "measured_result": evaluation,
    }


def _arena_rejected_red_audit(evaluation: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Build a safe audit record for candidates rejected after protected evaluation."""
    message = str(evaluation.get("error", "Protected evaluator rejected the candidate."))
    measured = evaluation.get("measured_linf")
    maximum = evaluation.get("max_linf", 0.02)
    linf_tolerance = float(evaluation.get("linf_tolerance", 1e-8))
    budget_rejected = message == "Candidate exceeds the exercise L-infinity budget"
    attack_rejected = message == "Candidate does not fool the standard model"
    criteria = [
        {"name": "approved_clean_source", "passed": True,
         "observed": "protected evaluator reached candidate validation",
         "rule": "the approved source must pass its clean classification check"},
        {"name": "maximum_measured_linf",
         "passed": False if budget_rejected else True if attack_rejected else None,
         "observed": measured, "maximum": maximum,
         "tolerance": linf_tolerance,
         "effective_maximum": float(maximum) + linf_tolerance,
         "declared_epsilon": params.get("epsilon"),
         "rule": "measured L-infinity distance must be <= exercise budget plus numeric tolerance"},
        {"name": "standard_model_misclassification",
         "passed": False if attack_rejected else None,
         "observed": {"clean_prediction": evaluation.get("clean_prediction"),
                      "adversarial_prediction": evaluation.get("adversarial_prediction")},
         "rule": "adversarial top-1 class must differ from the source label"},
    ]
    failures = [item["name"] for item in criteria if item["passed"] is False]
    return {
        "schema_version": 1,
        "challenge": "xray-red-blue-arena-red",
        "success": False,
        "summary": "Candidate rejected: " + (", ".join(failures) or message),
        "criteria": criteria,
        "failure_reasons": failures or [message],
        "parameters": params,
        "evaluator_message": message,
        "measured_result": evaluation,
    }


def _log_rejected_red_evaluation(*, actor: dict[str, Any], match_id: str, source_id: str,
                                 raw: bytes, params: dict[str, Any], evaluation: dict[str, Any]) -> None:
    """Best-effort log only model-evaluated Red rejection responses; preserve the 422 result."""
    try:
        with db_connection() as conn:
            latest = conn.execute(
                "SELECT COALESCE(MAX(sequence_number),0) AS n FROM attacks WHERE match_id=?",
                (match_id,),
            ).fetchone()
        sequence = int(latest["n"]) + 1
        audit = _arena_rejected_red_audit(evaluation, {**params, "source_id": source_id})
        metrics = {"success": 0.0, "attack_success": 0.0, "valid": 0.0}
        measured_linf = evaluation.get("measured_linf")
        if measured_linf is not None and math.isfinite(float(measured_linf)):
            metrics["measured_linf"] = float(measured_linf)
        with tempfile.TemporaryDirectory(prefix="arena-rejected-red-") as temp_dir:
            submitted_path = Path(temp_dir) / "submitted.png"
            submitted_path.write_bytes(raw)
            _log_mlflow_run(
                role="red", match_id=match_id, sequence=sequence,
                participant_info=actor,
                params={**params, "source_id": source_id, "rejected_submission": True},
                metrics=metrics, artifact=submitted_path,
                artifact_json={"result": evaluation, "decision_audit": audit},
            )
    except Exception:
        logger.warning("Could not create Arena rejection audit; original evaluation response is preserved", exc_info=True)


def _log_mlflow_run(*, role: str, match_id: str, sequence: int, participant_info: dict[str, Any],
                    params: dict[str, Any], metrics: dict[str, float], artifact: Path | None = None,
                    artifact_json: dict[str, Any] | None = None) -> str | None:
    """Best-effort telemetry. Interaction state is committed before this is called."""
    client = None
    run_id = None
    try:
        client = _mlflow_client()
        experiment_id = _mlflow_experiment(client)
        tags = {
            "challenge": "xray-red-blue-arena",
            "role": role,
            "match_mode": participant_info.get("match_mode", "paired"),
            "match_id": match_id,
            "sequence_number": str(sequence),
            "ctfd_user_id": str(participant_info["ctfd_user_id"]),
            "ctfd_username": str(participant_info["username"]),
        }
        decision_audit = (artifact_json or {}).get("decision_audit", {})
        if decision_audit:
            tags["evaluation_success"] = str(bool(decision_audit.get("success", False))).lower()
            tags["failure_reasons"] = ",".join(decision_audit.get("failure_reasons", [])) or "none"
            tags["evaluation_audit_status"] = "complete"
        elif artifact_json is not None:
            logged_result = artifact_json.get("result", {})
            outcome = logged_result.get("success", logged_result.get("defense_success"))
            tags["evaluation_success"] = "unknown" if outcome is None else str(bool(outcome)).lower()
            tags["failure_reasons"] = "evaluation_audit_missing"
            tags["evaluation_audit_status"] = "missing"
        run_name = f"arena-{role}-{match_id[:8]}-{sequence:03d}"
        if params.get("rejected_submission"):
            run_name += f"-rejected-{uuid.uuid4().hex[:6]}"
        run = client.create_run(
            experiment_id,
            tags=tags,
            run_name=run_name,
        )
        run_id = run.info.run_id
        clean_params = {str(k): str(v)[:500] for k, v in params.items() if v is not None}
        if clean_params:
            client.log_batch(run_id, metrics=[], params=[__import__("mlflow.entities", fromlist=["Param"]).Param(k, v) for k, v in clean_params.items()], synchronous=True)
        _metric_values(run_id, metrics)
        if artifact and artifact.is_file():
            client.log_artifact(run_id, str(artifact), artifact_path="submission")
        if artifact_json is not None:
            with tempfile.TemporaryDirectory(prefix="arena-mlflow-") as temp_dir:
                path = Path(temp_dir) / "evaluation.json"
                path.write_text(json.dumps(artifact_json, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
                client.log_artifact(run_id, str(path), artifact_path="evaluation")
        client.set_terminated(run_id, status="FINISHED")
        return run_id
    except Exception:
        logger.warning("MLflow Arena %s telemetry failed; interaction continues", role, exc_info=True)
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                logger.warning("Could not close incomplete Arena MLflow run", exc_info=True)
        return None


@app.get("/health")
def health():
    try:
        with db_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return {"status": "ready", "database": "sqlite", "artifact_store": "filesystem"}
    except sqlite3.Error:
        logger.exception("Arena database health check failed")
        return JSONResponse(status_code=503, content={"status": "unavailable"})


class ParticipantRegistration(BaseModel):
    ctfd_user_id: int = Field(gt=0)
    username: str = Field(min_length=1, max_length=64)
    workspace_id: str = Field(min_length=1, max_length=64)
    token: str = Field(min_length=24, max_length=256)


@app.post("/internal/participants/register", include_in_schema=False)
def register_participant(payload: ParticipantRegistration, x_arena_launcher_key: str = Header(default="", alias="X-Arena-Launcher-Key")):
    require_key(x_arena_launcher_key, ARENA_LAUNCHER_KEY, "Invalid Launcher registration request.")
    if any(ord(ch) < 32 for ch in payload.username) or not payload.username.strip():
        raise HTTPException(status_code=422, detail="Invalid CTFd username.")
    username = payload.username.strip()
    token_hash = hashlib.sha256(payload.token.encode("utf-8")).hexdigest()
    with db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT * FROM participants WHERE ctfd_user_id=?", (payload.ctfd_user_id,)).fetchone()
        if existing and existing["username"].casefold() != username.casefold():
            raise HTTPException(status_code=409, detail="Launcher CTFd username conflicts with the registered user.")
        token_owner = conn.execute("SELECT ctfd_user_id FROM participants WHERE token_hash=?", (token_hash,)).fetchone()
        if token_owner and int(token_owner["ctfd_user_id"]) != payload.ctfd_user_id:
            raise HTTPException(status_code=409, detail="Workspace token is already assigned to another user.")
        try:
            conn.execute(
                """INSERT INTO participants(ctfd_user_id,username,workspace_id,token_hash,updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(ctfd_user_id) DO UPDATE SET
                     workspace_id=excluded.workspace_id, token_hash=excluded.token_hash,
                     updated_at=excluded.updated_at""",
                (payload.ctfd_user_id, username, payload.workspace_id, token_hash, utc_now()),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Participant username or token is already registered.") from exc
    return {"registered": True, "ctfd_user_id": payload.ctfd_user_id, "workspace_id": payload.workspace_id}


class ParticipantUnregistration(BaseModel):
    ctfd_user_id: int = Field(gt=0)
    workspace_id: str = Field(min_length=1, max_length=64)


@app.post("/internal/participants/unregister", include_in_schema=False)
def unregister_participant(payload: ParticipantUnregistration, x_arena_launcher_key: str = Header(default="", alias="X-Arena-Launcher-Key")):
    require_key(x_arena_launcher_key, ARENA_LAUNCHER_KEY, "Invalid Launcher registration request.")
    with db_connection() as conn:
        conn.execute(
            """UPDATE participants SET workspace_id=NULL,token_hash=NULL,updated_at=?
               WHERE ctfd_user_id=? AND workspace_id=?""",
            (utc_now(), payload.ctfd_user_id, payload.workspace_id),
        )
    return {"unregistered": True, "ctfd_user_id": payload.ctfd_user_id}


@app.get("/api/identity")
def identity(authorization: str | None = Header(default=None)):
    actor = participant(authorization)
    return {**{key: actor[key] for key in ("ctfd_user_id", "username", "workspace_id", "match_id", "role")},
            "mode": actor["match_mode"]}


@app.get("/api/sources")
def sources(authorization: str | None = Header(default=None)):
    participant(authorization)
    response = evaluator_request("GET", "/internal/arena/sources")
    return JSONResponse(status_code=response.status_code, content=_json_response_from_evaluator(response))


@app.get("/api/sources/{source_id}")
def source_image(source_id: str, authorization: str | None = Header(default=None)):
    participant(authorization)
    if not re.fullmatch(r"\d{1,6}", source_id):
        raise HTTPException(status_code=404, detail="Unknown source.")
    response = evaluator_request("GET", f"/internal/arena/sources/{source_id}")
    if response.status_code >= 400:
        raise HTTPException(status_code=404, detail="Unknown approved source.")
    return Response(content=response.content, media_type="image/png")


@app.get("/api/model")
def standard_model(authorization: str | None = Header(default=None)):
    participant(authorization)
    response = evaluator_request("GET", "/internal/arena/model")
    if response.status_code >= 400:
        raise HTTPException(status_code=503, detail="Standard model is unavailable.")
    return Response(content=response.content, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename=efficientnet_b0.pth"})


@app.post("/api/red/attacks")
async def submit_red_attack(request: Request, authorization: str | None = Header(default=None)):
    actor = participant(authorization, "red")
    if not XRAY_RED_FLAG:
        raise HTTPException(status_code=503, detail="X-Ray Red flag is not configured.")
    body = await read_bounded_json(request)
    source_id = str(body.get("source_id", ""))
    if not re.fullmatch(r"\d{1,6}", source_id):
        raise HTTPException(status_code=422, detail="Choose a source_id from the approved source list.")
    attack_name = str(body.get("attack_name", "PGD")).strip()
    if not attack_name or len(attack_name) > 64 or any(ord(ch) < 32 for ch in attack_name):
        raise HTTPException(status_code=422, detail="attack_name must be 1 to 64 printable characters.")
    requested_epsilon = _number(body.get("epsilon", 0.02), "epsilon", 0.000001, 0.02)
    iterations = _integer(body.get("iterations", 1), "iterations", 1, 1000)
    step_size = _number(body.get("step_size", requested_epsilon), "step_size", 0.000001, requested_epsilon)
    encoded = body.get("image_b64")
    if not isinstance(encoded, str) or len(encoded) > 4 * (MAX_IMAGE_BYTES // 3 + 2):
        raise HTTPException(status_code=413, detail="Provide a base64 PNG image no larger than 1 MiB.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=422, detail="image_b64 must contain valid base64 data.") from exc
    _validate_png_envelope(raw)

    match_id = str(actor["match_id"])
    with arena_match_lock(match_id):
        with db_connection() as conn:
            pending = conn.execute(
                "SELECT sequence_number FROM attacks WHERE match_id=? AND status='pending_blue'",
                (match_id,),
            ).fetchone()
        if pending:
            raise HTTPException(status_code=409, detail=f"Attack #{pending['sequence_number']} is still waiting for Blue. Wait for its response.")

        try:
            response = evaluator_request(
                "POST",
                "/internal/arena/validate-attack",
                data={"source_id": source_id},
                files={"image": ("candidate.png", raw, "image/png")},
            )
        except HTTPException:
            raise
        params = {"epsilon": requested_epsilon, "iterations": iterations, "step_size": step_size}
        if response.status_code == 422:
            try:
                rejected_evaluation = response.json()
            except ValueError:
                rejected_evaluation = {}
            if rejected_evaluation.get("error") in {
                "Candidate exceeds the exercise L-infinity budget",
                "Candidate does not fool the standard model",
            }:
                _log_rejected_red_evaluation(
                    actor=actor, match_id=match_id, source_id=source_id, raw=raw,
                    params=params, evaluation=rejected_evaluation,
                )
        evaluation = _json_response_from_evaluator(response)
        if not evaluation.get("valid"):
            raise HTTPException(status_code=422, detail="Protected evaluator rejected this attack.")

        sha256 = hashlib.sha256(raw).hexdigest()
        if sha256 != evaluation.get("sha256"):
            raise HTTPException(status_code=503, detail="Protected evaluator image digest did not match uploaded bytes.")
        attack_id = uuid.uuid4().hex
        created_at = utc_now()
        with db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            pending = conn.execute(
                "SELECT sequence_number FROM attacks WHERE match_id=? AND status='pending_blue'",
                (match_id,),
            ).fetchone()
            if pending:
                raise HTTPException(status_code=409, detail=f"Attack #{pending['sequence_number']} is still waiting for Blue.")
            last = conn.execute("SELECT COALESCE(MAX(sequence_number),0) AS n FROM attacks WHERE match_id=?", (match_id,)).fetchone()
            sequence = int(last["n"]) + 1
            path = _store_attack_artifact(match_id, sequence, raw)
            try:
                conn.execute(
                    """INSERT INTO attacks(id,match_id,sequence_number,red_user_id,source_id,filename,sha256,
                       attack_name,attack_parameters_json,evaluation_json,artifact_path,status,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (attack_id, match_id, sequence, int(actor["ctfd_user_id"]), source_id,
                     f"attack_{sequence:03d}.png", sha256, attack_name,
                     json.dumps(params, sort_keys=True), json.dumps(evaluation, sort_keys=True),
                     str(path), "pending_blue", created_at),
                )
            except Exception:
                path.unlink(missing_ok=True)
                raise

        run_id = _log_mlflow_run(
            role="red", match_id=match_id, sequence=sequence, participant_info=actor,
            params={**params, "source_id": source_id, "attack_name": attack_name},
            metrics={"measured_linf": float(evaluation["measured_linf"]),
                     "clean_probability": float(evaluation["clean_probability"]),
                     "adversarial_probability": float(evaluation["adversarial_probability"]),
                     "attack_success": 1.0, "valid": 1.0, "flag_awarded": 1.0},
            artifact=path,
            artifact_json={
                "result": evaluation,
                "decision_audit": _arena_decision_audit("red", evaluation, params),
            },
        )
        if run_id:
            with db_connection() as conn:
                conn.execute("UPDATE attacks SET mlflow_run_id=? WHERE id=?", (run_id, attack_id))

    return {
        "submitted": True,
        "attack_id": attack_id,
        "sequence_number": sequence,
        "source_id": source_id,
        "attack_name": attack_name,
        "filename": f"attack_{sequence:03d}.png",
        "sha256": sha256,
        "declared_epsilon": requested_epsilon,
        "measured_linf": evaluation["measured_linf"],
        "max_linf": evaluation.get("max_linf", 0.02),
        "clean_prediction": evaluation["true_class_label"],
        "adversarial_prediction": evaluation["adversarial_class_label"],
        "status": "Waiting for Blue",
        "created_at": created_at,
        "mlflow_logged": bool(run_id),
        "flag": XRAY_RED_FLAG,
    }


@app.get("/api/red/flag")
def red_flag(authorization: str | None = Header(default=None)):
    actor = participant(authorization, "red")
    if not XRAY_RED_FLAG:
        raise HTTPException(status_code=503, detail="X-Ray Red flag is not configured.")
    with db_connection() as conn:
        qualified = conn.execute(
            "SELECT 1 FROM attacks WHERE match_id=? AND red_user_id=? LIMIT 1",
            (actor["match_id"], actor["ctfd_user_id"]),
        ).fetchone()
    if not qualified:
        raise HTTPException(status_code=403, detail="Submit a valid Red attack to earn this flag.")
    return {"flag": XRAY_RED_FLAG}


@app.get("/api/blue/pending")
def blue_pending(authorization: str | None = Header(default=None)):
    actor = participant(authorization, "blue")
    with db_connection() as conn:
        row = conn.execute(
            """SELECT id,sequence_number,source_id,filename,sha256,attack_name,
                      attack_parameters_json,created_at,evaluation_json
               FROM attacks WHERE match_id=? AND status='pending_blue'""",
            (actor["match_id"],),
        ).fetchone()
    if row is None:
        return {"pending": False}
    return {
        "pending": True,
        "attack_id": row["id"],
        "sequence_number": row["sequence_number"],
        "source_id": row["source_id"],
        "filename": row["filename"],
        "sha256": row["sha256"],
        "attack_name": row["attack_name"],
        "attack_parameters": json.loads(row["attack_parameters_json"]),
        "declared_epsilon": json.loads(row["attack_parameters_json"]).get("epsilon"),
        "measured_linf": json.loads(row["evaluation_json"]).get("measured_linf"),
        "max_linf": json.loads(row["evaluation_json"]).get("max_linf", 0.02),
        "submitted_at": row["created_at"],
        "evaluation": json.loads(row["evaluation_json"]),
        "artifact_url": f"/api/attacks/{row['id']}/artifact",
    }


@app.get("/api/attacks/{attack_id}/artifact")
def download_attack(attack_id: str, authorization: str | None = Header(default=None)):
    actor = participant(authorization, "blue")
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM attacks WHERE id=? AND match_id=?", (attack_id, actor["match_id"])).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Attack not found in this match.")
    path = Path(row["artifact_path"]).resolve()
    if ARTIFACTS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Arena artifact is unavailable.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["sha256"]:
        logger.error("Arena artifact digest mismatch for %s", attack_id)
        raise HTTPException(status_code=503, detail="Arena artifact integrity check failed.")
    return Response(content=raw, media_type="image/png", headers={
        "Content-Disposition": f"attachment; filename={row['filename']}",
        "X-Artifact-SHA256": digest,
    })


class BlueResponse(BaseModel):
    attack_id: str = Field(min_length=32, max_length=32)
    sigma: float
    num_samples: int
    abstain_threshold: float = Field(default=0.75)

    @field_validator("sigma", "abstain_threshold", mode="before")
    @classmethod
    def numeric_parameters(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("defense parameters must be JSON numbers")
        return float(value)

    @field_validator("num_samples", mode="before")
    @classmethod
    def integer_sample_count(cls, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("num_samples must be a JSON integer")
        return value


@app.post("/api/blue/responses")
def submit_blue_response(payload: BlueResponse, authorization: str | None = Header(default=None)):
    actor = participant(authorization, "blue")
    if not XRAY_BLUE_FLAG:
        raise HTTPException(status_code=503, detail="X-Ray Blue flag is not configured.")
    sigma = _number(payload.sigma, "sigma", 0.0, 0.1)
    samples = _integer(payload.num_samples, "num_samples", 10, 500)
    threshold = _number(payload.abstain_threshold, "abstain_threshold", 0.0, 1.0)
    match_id = str(actor["match_id"])
    with arena_match_lock(match_id):
        with db_connection() as conn:
            attack = conn.execute(
                "SELECT * FROM attacks WHERE id=? AND match_id=? AND status='pending_blue'",
                (payload.attack_id, match_id),
            ).fetchone()
        if attack is None:
            raise HTTPException(status_code=404, detail="Pending attack not found in this match.")
        artifact = Path(attack["artifact_path"]).resolve()
        if ARTIFACTS_DIR.resolve() not in artifact.parents or not artifact.is_file():
            raise HTTPException(status_code=503, detail="Authoritative Arena attack artifact is missing.")
        seed_material = f"{match_id}:{payload.attack_id}:{sigma:.8f}:{samples}:{threshold:.8f}:{EVALUATOR_VERSION}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:4], "big")
        parameters = {"sigma": sigma, "num_samples": samples, "abstain_threshold": threshold}
        response = evaluator_request(
            "POST", "/internal/arena/evaluate-defense",
            data={"source_id": attack["source_id"], "sigma": str(sigma),
                  "num_samples": str(samples), "abstain_threshold": str(threshold), "seed": str(seed)},
            files={"image": (attack["filename"], artifact.read_bytes(), "image/png")},
            timeout=EVALUATOR_TIMEOUT,
        )
        result = _json_response_from_evaluator(response)
        response_id = uuid.uuid4().hex
        created_at = utc_now()
        with db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT status FROM attacks WHERE id=? AND match_id=?", (payload.attack_id, match_id)).fetchone()
            if current is None or current["status"] != "pending_blue":
                raise HTTPException(status_code=409, detail="This attack has already received a Blue response.")
            conn.execute(
                """INSERT INTO blue_responses(id,match_id,attack_id,sequence_number,blue_user_id,
                   defense_parameters_json,protected_result_json,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (response_id, match_id, payload.attack_id, int(attack["sequence_number"]),
                 int(actor["ctfd_user_id"]), json.dumps(parameters, sort_keys=True),
                 json.dumps(result, sort_keys=True), created_at),
            )
            conn.execute("UPDATE attacks SET status='blue_responded' WHERE id=?", (payload.attack_id,))
        run_id = _log_mlflow_run(
            role="blue", match_id=match_id, sequence=int(attack["sequence_number"]),
            participant_info=actor, params={**parameters, "attack_id": payload.attack_id,
                                             "source_id": attack["source_id"]},
            metrics={"clean_utility_pass": float(bool(result["clean_utility_pass"])),
                     "attack_success": float(bool(result["attack_success"])),
                     "defense_success": float(bool(result["defense_success"])),
                     "flag_awarded": float(bool(result["defense_success"])),
                     "attack_survived": float(bool(result["attack_survived"])),
                     "clean_vote_confidence": float(result["clean_vote_confidence"]),
                     "defended_vote_confidence": float(result["defended_vote_confidence"])},
            artifact_json={
                "result": result,
                "decision_audit": _arena_decision_audit("blue", result, parameters),
            },
        )
        if run_id:
            with db_connection() as conn:
                conn.execute("UPDATE blue_responses SET mlflow_run_id=? WHERE id=?", (run_id, response_id))
    return {
        "submitted": True,
        "response_id": response_id,
        "attack_id": payload.attack_id,
        "sequence_number": int(attack["sequence_number"]),
        "defense_parameters": parameters,
        "protected_result": result,
        "created_at": created_at,
        "mlflow_logged": bool(run_id),
        **({"flag": XRAY_BLUE_FLAG} if result["defense_success"] else {}),
    }


@app.get("/api/blue/flag")
def blue_flag(authorization: str | None = Header(default=None)):
    actor = participant(authorization, "blue")
    if not XRAY_BLUE_FLAG:
        raise HTTPException(status_code=503, detail="X-Ray Blue flag is not configured.")
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT protected_result_json FROM blue_responses WHERE match_id=? AND blue_user_id=?",
            (actor["match_id"], actor["ctfd_user_id"]),
        ).fetchall()
    if not any(json.loads(row["protected_result_json"]).get("defense_success") is True for row in rows):
        raise HTTPException(status_code=403, detail="Submit a successful Blue defense to earn this flag.")
    return {"flag": XRAY_BLUE_FLAG}


@app.get("/api/red/latest-response")
def red_latest_response(authorization: str | None = Header(default=None)):
    actor = participant(authorization, "red")
    with db_connection() as conn:
        row = conn.execute(
            """SELECT r.*, a.attack_name,a.source_id,a.sha256,a.status AS attack_status
               FROM blue_responses r JOIN attacks a ON a.id=r.attack_id
               WHERE r.match_id=? ORDER BY r.sequence_number DESC LIMIT 1""",
            (actor["match_id"],),
        ).fetchone()
    if row is None:
        return {"available": False}
    return {
        "available": True,
        "response_id": row["id"],
        "attack_id": row["attack_id"],
        "sequence_number": row["sequence_number"],
        "attack_name": row["attack_name"],
        "source_id": row["source_id"],
        "attack_sha256": row["sha256"],
        "attack_status": row["attack_status"],
        "defense_parameters": json.loads(row["defense_parameters_json"]),
        "protected_result": json.loads(row["protected_result_json"]),
        "created_at": row["created_at"],
    }


@app.get("/api/match/history")
def match_history(authorization: str | None = Header(default=None)):
    actor = participant(authorization)
    with db_connection() as conn:
        rows = conn.execute(
            """SELECT a.id AS attack_id,a.sequence_number,a.source_id,a.attack_name,a.sha256,
                      a.status AS attack_status,a.created_at AS attack_created,
                      a.attack_parameters_json,a.evaluation_json,
                      r.id AS response_id,r.defense_parameters_json,
                      r.protected_result_json,r.created_at AS response_created
               FROM attacks a LEFT JOIN blue_responses r ON r.attack_id=a.id
               WHERE a.match_id=? ORDER BY a.sequence_number""",
            (actor["match_id"],),
        ).fetchall()
    return {"match_id": actor["match_id"], "history": [
        {"attack_id": row["attack_id"], "sequence_number": row["sequence_number"],
         "source_id": row["source_id"], "attack_name": row["attack_name"],
         "sha256": row["sha256"], "status": row["attack_status"],
         "attack_created_at": row["attack_created"],
         "attack_parameters": json.loads(row["attack_parameters_json"]),
         "declared_epsilon": json.loads(row["attack_parameters_json"]).get("epsilon"),
         "attack_evaluation": json.loads(row["evaluation_json"]),
         "measured_linf": json.loads(row["evaluation_json"]).get("measured_linf"),
         "max_linf": json.loads(row["evaluation_json"]).get("max_linf", 0.02),
         "defense_parameters": json.loads(row["defense_parameters_json"]) if row["defense_parameters_json"] else None,
         "protected_result": json.loads(row["protected_result_json"]) if row["protected_result_json"] else None,
         "response_created_at": row["response_created"]}
        for row in rows
    ]}


@app.get("/api/match/status")
def match_status(authorization: str | None = Header(default=None)):
    actor = participant(authorization)
    with db_connection() as conn:
        pending = conn.execute("SELECT sequence_number FROM attacks WHERE match_id=? AND status='pending_blue'", (actor["match_id"],)).fetchone()
        latest = conn.execute("SELECT COALESCE(MAX(sequence_number),0) AS n FROM attacks WHERE match_id=?", (actor["match_id"],)).fetchone()
        members = conn.execute("SELECT role,ctfd_user_id,username,workspace_id FROM participants WHERE match_id=? ORDER BY role", (actor["match_id"],)).fetchall()
        latest_attack = conn.execute(
            "SELECT sequence_number,status FROM attacks WHERE match_id=? ORDER BY sequence_number DESC LIMIT 1",
            (actor["match_id"],),
        ).fetchone()
        latest_response = conn.execute(
            "SELECT sequence_number,defense_parameters_json,created_at FROM blue_responses WHERE match_id=? ORDER BY sequence_number DESC LIMIT 1",
            (actor["match_id"],),
        ).fetchone()
    member_by_role = {row["role"]: dict(row) for row in members}
    solo = actor["match_mode"] == "solo"
    return {"match_id": actor["match_id"], "role": "solo" if solo else actor["role"],
            "mode": actor["match_mode"],
            "participants": [dict(row) for row in members],
            "red_user": member_by_role.get("red"),
            "blue_user": member_by_role.get("red") if solo else member_by_role.get("blue"),
            "latest_sequence": int(latest["n"]),
            "pending_attack_sequence": int(pending["sequence_number"]) if pending else None,
            "last_response_status": latest_attack["status"] if latest_attack else "no_attacks",
            "last_response_sequence": int(latest_response["sequence_number"]) if latest_response else None,
            "current_blue_defense": json.loads(latest_response["defense_parameters_json"]) if latest_response else None,
            "last_response_at": latest_response["created_at"] if latest_response else None,
            "state": "waiting_for_blue" if pending else "ready_for_red"}


class MatchCreate(BaseModel):
    red_user_id: int = Field(gt=0)
    red_username: str = Field(min_length=1, max_length=64)
    blue_user_id: int = Field(gt=0)
    blue_username: str = Field(min_length=1, max_length=64)


def _validate_admin_identity(user_id: int, username: str) -> str:
    cleaned = username.strip()
    if not cleaned or any(ord(ch) < 32 for ch in cleaned):
        raise HTTPException(status_code=422, detail="Usernames must be printable and non-empty.")
    return cleaned


@app.post("/admin/matches")
def create_match(payload: MatchCreate, x_arena_admin_key: str = Header(default="", alias="X-Arena-Admin-Key")):
    require_key(x_arena_admin_key, ARENA_ADMIN_KEY, "Invalid Arena administrator request.")
    if payload.red_user_id == payload.blue_user_id:
        raise HTTPException(status_code=422, detail="Red and Blue must be different CTFd users.")
    red_name = _validate_admin_identity(payload.red_user_id, payload.red_username)
    blue_name = _validate_admin_identity(payload.blue_user_id, payload.blue_username)
    if red_name.casefold() == blue_name.casefold():
        raise HTTPException(status_code=422, detail="Red and Blue usernames must be different.")
    match_id = uuid.uuid4().hex
    created_at = utc_now()
    with db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT ctfd_user_id,username,match_id FROM participants WHERE ctfd_user_id IN (?,?)",
            (payload.red_user_id, payload.blue_user_id),
        ).fetchall()
        for row in existing:
            if row["match_id"]:
                match = conn.execute("SELECT status FROM matches WHERE id=?", (row["match_id"],)).fetchone()
                if match and match["status"] == "active":
                    raise HTTPException(status_code=409, detail=f"CTFd user {row['ctfd_user_id']} is already assigned to an active match.")
        for user_id, username in ((payload.red_user_id, red_name), (payload.blue_user_id, blue_name)):
            mapped = conn.execute("SELECT username FROM participants WHERE ctfd_user_id=?", (user_id,)).fetchone()
            if mapped and mapped["username"].casefold() != username.casefold():
                raise HTTPException(status_code=409, detail=f"CTFd user ID {user_id} is already registered as {mapped['username']}.")
        conn.execute("INSERT INTO matches(id,status,mode,created_at,updated_at) VALUES(?,?,?,?,?)", (match_id,"active","paired",created_at,created_at))
        for user_id, username, role in ((payload.red_user_id, red_name, "red"), (payload.blue_user_id, blue_name, "blue")):
            conn.execute(
                """INSERT INTO participants(ctfd_user_id,username,match_id,role,updated_at)
                   VALUES(?,?,?,?,?) ON CONFLICT(ctfd_user_id) DO UPDATE SET
                   username=excluded.username,match_id=excluded.match_id,role=excluded.role,updated_at=excluded.updated_at""",
                (user_id, username, match_id, role, created_at),
            )
    return {"match_id": match_id, "status": "active", "created_at": created_at,
            "red": {"ctfd_user_id": payload.red_user_id, "username": red_name},
            "blue": {"ctfd_user_id": payload.blue_user_id, "username": blue_name}}


class SoloMatchCreate(BaseModel):
    user_id: int = Field(gt=0)
    username: str = Field(min_length=1, max_length=64)


@app.post("/admin/matches/solo")
def create_solo_match(payload: SoloMatchCreate, x_arena_admin_key: str = Header(default="", alias="X-Arena-Admin-Key")):
    require_key(x_arena_admin_key, ARENA_ADMIN_KEY, "Invalid Arena administrator request.")
    username = _validate_admin_identity(payload.user_id, payload.username)
    match_id = uuid.uuid4().hex
    created_at = utc_now()
    with db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT username,match_id FROM participants WHERE ctfd_user_id=?", (payload.user_id,)
        ).fetchone()
        if existing:
            if existing["username"].casefold() != username.casefold():
                raise HTTPException(status_code=409, detail=f"CTFd user ID {payload.user_id} is already registered as {existing['username']}.")
            if existing["match_id"]:
                current = conn.execute("SELECT status FROM matches WHERE id=?", (existing["match_id"],)).fetchone()
                if current and current["status"] == "active":
                    raise HTTPException(status_code=409, detail=f"CTFd user {payload.user_id} is already assigned to an active match.")
        conn.execute("INSERT INTO matches(id,status,mode,created_at,updated_at) VALUES(?,?,?,?,?)",
                     (match_id, "active", "solo", created_at, created_at))
        conn.execute(
            """INSERT INTO participants(ctfd_user_id,username,match_id,role,updated_at)
               VALUES(?,?,?,?,?) ON CONFLICT(ctfd_user_id) DO UPDATE SET
               username=excluded.username,match_id=excluded.match_id,role=excluded.role,updated_at=excluded.updated_at""",
            (payload.user_id, username, match_id, "red", created_at),
        )
    return {"match_id": match_id, "status": "active", "mode": "solo", "created_at": created_at,
            "participant": {"ctfd_user_id": payload.user_id, "username": username}}


def _admin_match_payload(match_id: str) -> dict[str, Any]:
    with db_connection() as conn:
        match = conn.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
        if not match:
            raise HTTPException(status_code=404, detail="Arena match not found.")
        participants = conn.execute("SELECT ctfd_user_id,username,workspace_id,role,updated_at FROM participants WHERE match_id=? ORDER BY role", (match_id,)).fetchall()
        attacks = conn.execute("SELECT id,sequence_number,red_user_id,source_id,filename,sha256,attack_name,status,created_at,mlflow_run_id FROM attacks WHERE match_id=? ORDER BY sequence_number", (match_id,)).fetchall()
        responses = conn.execute("SELECT id,attack_id,sequence_number,blue_user_id,defense_parameters_json,protected_result_json,created_at,mlflow_run_id FROM blue_responses WHERE match_id=? ORDER BY sequence_number", (match_id,)).fetchall()
    return {"match": dict(match), "participants": [dict(row) for row in participants],
            "attacks": [dict(row) for row in attacks],
            "responses": [{**dict(row), "defense_parameters": json.loads(row["defense_parameters_json"]),
                           "protected_result": json.loads(row["protected_result_json"])} for row in responses]}


@app.get("/admin/matches")
def list_matches(x_arena_admin_key: str = Header(default="", alias="X-Arena-Admin-Key")):
    require_key(x_arena_admin_key, ARENA_ADMIN_KEY, "Invalid Arena administrator request.")
    with db_connection() as conn:
        ids = [row["id"] for row in conn.execute("SELECT id FROM matches ORDER BY created_at DESC").fetchall()]
    return {"matches": [_admin_match_payload(match_id) for match_id in ids]}


@app.get("/admin/matches/{match_id}")
def get_match(match_id: str, x_arena_admin_key: str = Header(default="", alias="X-Arena-Admin-Key")):
    require_key(x_arena_admin_key, ARENA_ADMIN_KEY, "Invalid Arena administrator request.")
    return _admin_match_payload(match_id)


@app.post("/admin/matches/{match_id}/reset")
def reset_match(match_id: str, x_arena_admin_key: str = Header(default="", alias="X-Arena-Admin-Key")):
    require_key(x_arena_admin_key, ARENA_ADMIN_KEY, "Invalid Arena administrator request.")
    root = ARTIFACTS_DIR.resolve()
    match_dir = (ARTIFACTS_DIR / match_id).resolve()
    if root not in match_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid match ID.")
    with db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        match = conn.execute("SELECT id FROM matches WHERE id=?", (match_id,)).fetchone()
        if not match:
            raise HTTPException(status_code=404, detail="Arena match not found.")
        conn.execute("DELETE FROM blue_responses WHERE match_id=?", (match_id,))
        conn.execute("DELETE FROM attacks WHERE match_id=?", (match_id,))
        conn.execute("UPDATE matches SET status='active',updated_at=? WHERE id=?", (utc_now(), match_id))
    if match_dir.exists():
        shutil.rmtree(match_dir)
    return {"match_id": match_id, "status": "reset", "sequence_number": 0}
