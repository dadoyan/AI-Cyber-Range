"""Authoritative scoring for the single-student LLM safety challenge."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, Header, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field

from .audit import log_evaluation
from .core import (FIXED_OUTPUT_TARGET, MODELS, SLOTS, prompt_hash,
                   score_outputs, validate_submission)
from .model import ModelEngine

logger = logging.getLogger(__name__)
app = FastAPI(title="LLM Safety Challenge Evaluator")
engine = ModelEngine()
evaluation_lock = threading.Lock()


class EvaluationRequest(BaseModel):
    slot: Literal["blackbox_1", "blackbox_2", "blackbox_3", "whitebox_fixed", "whitebox_custom"]
    prompt: str = Field(min_length=1, max_length=4096)
    strategy: str = Field(default="", max_length=120)
    output_target: str = Field(default="", max_length=200)


class TransferRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    direction: Literal["qwen3_to_qwen25", "qwen25_to_qwen3"]


class ProbeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)


def _state_path() -> Path:
    return Path(os.getenv("LLM_SAFETY_STATE_DIR", "/app/state")) / "progress.sqlite3"


@contextmanager
def _database():
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS accepted (
        user_id INTEGER NOT NULL,
        slot TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        strategy TEXT NOT NULL,
        output_target TEXT NOT NULL,
        successes INTEGER NOT NULL,
        recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, slot),
        UNIQUE (user_id, prompt_hash)
    )""")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _identity(authorization: str, request: FastAPIRequest) -> dict:
    if not authorization.startswith("Bearer ") or not authorization[7:].strip():
        raise HTTPException(401, "A managed workspace token is required")
    key = os.getenv("LLM_SAFETY_IDENTITY_RESOLVE_KEY", "")
    if not key:
        raise HTTPException(503, "Workspace identity is not configured")
    token = authorization[7:].strip()
    body = json.dumps({"token": token, "remote_addr": ""}).encode()
    url = os.getenv("WORKSPACE_LAUNCHER_URL", "http://launcher:7000").rstrip("/")
    lookup = Request(
        url + "/internal/identity/resolve",
        data=body,
        headers={"Content-Type": "application/json", "X-Workspace-Identity-Key": key},
        method="POST",
    )
    try:
        with urlopen(lookup, timeout=5) as response:
            identity = json.load(response)
    except HTTPError as exc:
        if exc.code in (401, 403, 404):
            raise HTTPException(401, "Workspace token was not recognized") from exc
        raise HTTPException(503, "Workspace identity service failed") from exc
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise HTTPException(503, "Workspace identity service unavailable") from exc
    try:
        user_id = int(identity["ctfd_user_id"])
        if user_id <= 0 or not identity["workspace_id"]:
            raise ValueError("Invalid identity")
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(503, "Workspace identity response was invalid") from exc
    allowed = os.getenv("LLM_SAFETY_ALLOWED_USER_ID", "").strip()
    if not allowed:
        raise HTTPException(503, "Instructor has not selected the student account")
    if not allowed.isdecimal() or user_id != int(allowed):
        raise HTTPException(403, "This single-student challenge is assigned to another account")
    return identity


def _accepted(conn: sqlite3.Connection, user_id: int) -> list[dict]:
    return [dict(row) for row in conn.execute(
        "SELECT slot, prompt_hash, strategy, output_target, successes, recorded_at "
        "FROM accepted WHERE user_id=? ORDER BY slot", (user_id,)
    )]


def _progress(conn: sqlite3.Connection, user_id: int) -> dict:
    accepted = _accepted(conn, user_id)
    completed = {row["slot"] for row in accepted} == set(SLOTS)
    response = {
        "accepted": accepted,
        "completed_slots": len(accepted),
        "required_slots": len(SLOTS),
        "complete": completed,
    }
    if completed:
        flag = os.getenv("LLM_SAFETY_FLAG", "")
        if not flag:
            raise HTTPException(503, "Challenge flag is not configured")
        response["flag"] = flag
    return response


@app.get("/health")
def health():
    return {"status": "ok", "models": MODELS, "model_loaded": list(engine._loaded),
            "configured": bool(os.getenv("LLM_SAFETY_ALLOWED_USER_ID") and os.getenv("LLM_SAFETY_FLAG"))}


@app.get("/status")
def status(request: FastAPIRequest, authorization: str = Header(default="")):
    identity = _identity(authorization, request)
    with _database() as conn:
        return _progress(conn, int(identity["ctfd_user_id"]))


@app.post("/evaluate")
def evaluate(submission: EvaluationRequest, request: FastAPIRequest,
             authorization: str = Header(default="")):
    identity = _identity(authorization, request)
    try:
        validate_submission(submission.slot, submission.prompt,
                            submission.strategy, submission.output_target)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    user_id = int(identity["ctfd_user_id"])
    prompt_digest = prompt_hash(submission.prompt)
    with evaluation_lock:
        with _database() as conn:
            existing = _accepted(conn, user_id)
            duplicate = next((row for row in existing if row["prompt_hash"] == prompt_digest
                              and row["slot"] != submission.slot), None)
            if duplicate:
                raise HTTPException(409, f"That prompt already passed {duplicate['slot']}; use a distinct prompt")
            if submission.slot.startswith("blackbox_"):
                label = submission.strategy.strip().casefold()
                if any(row["slot"] != submission.slot and row["slot"].startswith("blackbox_")
                       and row["strategy"].strip().casefold() == label for row in existing):
                    raise HTTPException(409, "Use a distinct black-box strategy label")
        start = time.monotonic()
        try:
            outputs = engine.evaluate(submission.prompt, "qwen25")
            result = score_outputs(outputs)
            result["peak_cuda_vram_mib"] = engine.last_peak_vram_mib
        except Exception as exc:
            duration = time.monotonic() - start
            log_evaluation(identity=identity, slot=submission.slot, model_key="qwen25",
                           prompt=submission.prompt, strategy=submission.strategy,
                           output_target=submission.output_target, result=None,
                           outputs=None, duration_s=duration, error=type(exc).__name__)
            logger.exception("LLM safety model evaluation failed")
            raise HTTPException(503, "Model evaluation failed; no progress was recorded") from exc
        duration = time.monotonic() - start
        if result["passed"]:
            with _database() as conn:
                conn.execute("""INSERT INTO accepted
                    (user_id, slot, prompt_hash, strategy, output_target, successes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, slot) DO UPDATE SET
                    prompt_hash=excluded.prompt_hash, strategy=excluded.strategy,
                    output_target=excluded.output_target, successes=excluded.successes""",
                    (user_id, submission.slot, prompt_digest, submission.strategy.strip(),
                     submission.output_target, result["successes"]))
                conn.commit()
        run_id = log_evaluation(identity=identity, slot=submission.slot, model_key="qwen25",
                                prompt=submission.prompt, strategy=submission.strategy,
                                output_target=submission.output_target, result=result,
                                outputs=outputs, duration_s=duration)
        with _database() as conn:
            progress = _progress(conn, user_id)
        return {"slot": submission.slot, "model_id": MODELS["qwen25"], **result,
                "outputs": outputs, "duration_s": duration, "mlflow_run_id": run_id,
                "progress": progress}


@app.post("/probe")
def probe(submission: ProbeRequest, request: FastAPIRequest,
          authorization: str = Header(default="")):
    """Ten-run black-box trial that never consumes a scoring slot."""
    identity = _identity(authorization, request)
    with evaluation_lock:
        start = time.monotonic()
        try:
            outputs = engine.evaluate(submission.prompt, "qwen25")
            result = score_outputs(outputs)
            result["peak_cuda_vram_mib"] = engine.last_peak_vram_mib
        except Exception as exc:
            log_evaluation(identity=identity, slot="probe", model_key="qwen25",
                           prompt=submission.prompt, strategy="probe", output_target="",
                           result=None, outputs=None, duration_s=time.monotonic()-start,
                           error=type(exc).__name__)
            logger.exception("LLM safety probe failed")
            raise HTTPException(503, "Model evaluation failed") from exc
        duration = time.monotonic() - start
        run_id = log_evaluation(identity=identity, slot="probe", model_key="qwen25",
                                prompt=submission.prompt, strategy="probe", output_target="",
                                result=result, outputs=outputs, duration_s=duration)
        return {"model_id": MODELS["qwen25"], **result, "outputs": outputs,
                "duration_s": duration, "mlflow_run_id": run_id}


@app.post("/transfer")
def transfer(submission: TransferRequest, request: FastAPIRequest,
             authorization: str = Header(default="")):
    identity = _identity(authorization, request)
    order = ("qwen3", "qwen25") if submission.direction == "qwen3_to_qwen25" else ("qwen25", "qwen3")
    evaluations = {}
    with evaluation_lock:
        for key in order:
            start = time.monotonic()
            try:
                outputs = engine.evaluate(submission.prompt, key)
                result = score_outputs(outputs)
                result["peak_cuda_vram_mib"] = engine.last_peak_vram_mib
            except Exception as exc:
                log_evaluation(identity=identity, slot="transfer_" + submission.direction,
                               model_key=key, prompt=submission.prompt, strategy="transfer",
                               output_target="", result=None, outputs=None,
                               duration_s=time.monotonic() - start, error=type(exc).__name__)
                logger.exception("LLM safety transfer evaluation failed")
                raise HTTPException(503, "Model evaluation failed") from exc
            duration = time.monotonic() - start
            run_id = log_evaluation(identity=identity, slot="transfer_" + submission.direction,
                                    model_key=key, prompt=submission.prompt, strategy="transfer",
                                    output_target="", result=result, outputs=outputs,
                                    duration_s=duration)
            evaluations[key] = {"model_id": MODELS[key], **result,
                                "outputs": outputs, "duration_s": duration,
                                "mlflow_run_id": run_id}
    return {"direction": submission.direction, "evaluations": evaluations,
            "transfer_success_required_for_flag": False}
