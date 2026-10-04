"""One MLflow run per completed or failed server-side evaluation."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import uuid

from .core import MAX_NEW_TOKENS, MODELS, MODEL_REVISIONS, NUM_RUNS, SYSTEM_INSTRUCTION, TEMPERATURE

logger = logging.getLogger(__name__)


def log_evaluation(*, identity: dict, slot: str, model_key: str, prompt: str,
                   strategy: str, output_target: str, result: dict | None,
                   outputs: list[str] | None, duration_s: float, error: str | None = None) -> str | None:
    """Log exact prompts and sampled outputs as restricted MLflow artifacts.

    MLflow is best effort: an unavailable tracking server never grants a flag
    and never alters the evaluator's independently stored scoring decision.
    """
    client = run_id = None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        name = os.getenv("MLFLOW_LLM_SAFETY_EXPERIMENT_NAME", "llm-safety")
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        tags = {
            "challenge": "llm-safety",
            "slot": slot,
            "model": model_key,
            "model_id": MODELS.get(model_key, model_key),
            "model_revision": MODEL_REVISIONS.get(model_key, "unknown"),
            "ctfd_user_id": str(identity["ctfd_user_id"]),
            "ctfd_username": str(identity["ctfd_username"]),
            "workspace_id": str(identity["workspace_id"]),
            "identity_source": str(identity.get("identity_source", "launcher")),
            "evaluation_status": "failed" if error else "complete",
            "passed": str(bool(result and result["passed"])).lower(),
        }
        run = client.create_run(experiment_id, tags=tags, run_name=f"llm-{slot}-{uuid.uuid4().hex[:8]}")
        run_id = run.info.run_id
        for key, value in {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "strategy": strategy,
            "output_target": output_target,
            "num_runs": NUM_RUNS,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
        }.items():
            client.log_param(run_id, key, str(value)[:500])
        client.log_metric(run_id, "duration_s", duration_s)
        if result:
            client.log_metric(run_id, "successes", result["successes"])
            client.log_metric(run_id, "success_rate", result["rate"])
            client.log_metric(run_id, "passed", int(result["passed"]))
            if result.get("peak_cuda_vram_mib") is not None:
                client.log_metric(run_id, "peak_cuda_vram_mib", result["peak_cuda_vram_mib"])
        with tempfile.TemporaryDirectory(prefix="llm-safety-audit-") as temporary:
            path = Path(temporary) / "evaluation.json"
            path.write_text(json.dumps({
                "prompt": prompt,
                "strategy": strategy,
                "output_target": output_target,
                "system_instruction": SYSTEM_INSTRUCTION,
                "model_id": MODELS.get(model_key, model_key),
                "model_revision": MODEL_REVISIONS.get(model_key, "unknown"),
                "outputs": outputs or [],
                "result": result,
                "error_type": error,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            client.log_artifact(run_id, str(path), artifact_path="evaluation")
        client.set_terminated(run_id, status="FAILED" if error else "FINISHED")
        return run_id
    except Exception:
        logger.warning("Could not record LLM safety MLflow evaluation", exc_info=True)
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:
                pass
        return None
