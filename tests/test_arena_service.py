from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TEST_DATA = Path(tempfile.mkdtemp(prefix="arena-test-data-"))
os.environ["ARENA_DATA_DIR"] = str(TEST_DATA)
os.environ["ARENA_INTERNAL_KEY"] = "internal-test-key"
os.environ["ARENA_LAUNCHER_KEY"] = "launcher-test-key"
os.environ["ARENA_ADMIN_KEY"] = "admin-test-key"
os.environ["XRAY_RED_FLAG"] = "RANGE{arena_red_test}"
os.environ["XRAY_BLUE_FLAG"] = "RANGE{arena_blue_test}"
sys.path.insert(0, str(Path(__file__).parents[1] / "arena_service"))

import app as arena  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def fake_response(payload: dict, status: int = 200):
    import requests

    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


class FakeMlflowClient:
    def __init__(self):
        self.tags = None
        self.artifacts = {}
        self.status = None

    def create_run(self, _experiment_id, *, tags, run_name):
        self.tags = dict(tags)
        self.tags["mlflow.runName"] = run_name
        return SimpleNamespace(info=SimpleNamespace(run_id="arena-test-run"))

    def log_batch(self, *_args, **_kwargs):
        pass

    def log_artifact(self, _run_id, path, artifact_path=None):
        key = f"{artifact_path}/{Path(path).name}" if artifact_path else Path(path).name
        self.artifacts[key] = Path(path).read_bytes()

    def set_terminated(self, _run_id, status):
        self.status = status


class ArenaServiceTests(unittest.TestCase):
    def setUp(self):
        with arena.db_connection() as conn:
            conn.execute("DELETE FROM blue_responses")
            conn.execute("DELETE FROM attacks")
            conn.execute("DELETE FROM participants")
            conn.execute("DELETE FROM matches")
        shutil.rmtree(arena.ARTIFACTS_DIR, ignore_errors=True)
        arena.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.client = TestClient(arena.app)
        self.red_token = "red-workspace-token-" + "r" * 24
        self.blue_token = "blue-workspace-token-" + "b" * 24
        self._register(11, "red_audit_user", self.red_token, "u11")
        self._register(12, "blue_audit_user", self.blue_token, "u12")
        created = self.client.post(
            "/admin/matches",
            headers={"X-Arena-Admin-Key": "admin-test-key"},
            json={"red_user_id": 11, "red_username": "red_audit_user",
                  "blue_user_id": 12, "blue_username": "blue_audit_user"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.match_id = created.json()["match_id"]
        self.red_headers = {"Authorization": f"Bearer {self.red_token}"}
        self.blue_headers = {"Authorization": f"Bearer {self.blue_token}"}
        self.png = b"\x89PNG\r\n\x1a\n" + b"bounded-test-png"
        self.validation = {
            "valid": True, "source_id": "4", "true_class_index": 0,
            "true_class_label": "NORMAL", "clean_probability": 0.98,
            "adversarial_class_index": 1, "adversarial_class_label": "PNEUMONIA",
            "adversarial_probability": 0.91, "measured_linf": 0.0196,
            "max_linf": 0.02, "linf_tolerance": 1e-8,
            "sha256": hashlib.sha256(self.png).hexdigest(),
            "width": 256, "height": 256,
        }
        self.protected = {
            "true_class_index": 0, "true_class_label": "NORMAL",
            "clean_prediction_index": 0, "clean_prediction_label": "NORMAL",
            "clean_vote_confidence": 0.9, "clean_utility_pass": True,
            "undefended_prediction_index": 1, "undefended_prediction_label": "PNEUMONIA",
            "defended_prediction_index": 0, "defended_prediction_label": "NORMAL",
            "defended_vote_confidence": 0.8, "attack_success": True,
            "defense_success": True, "attack_survived": False,
            "parameters": {"sigma": 0.05, "num_samples": 100, "abstain_threshold": 0.75},
        }

    def _register(self, user_id, username, token, workspace_id):
        response = self.client.post(
            "/internal/participants/register",
            headers={"X-Arena-Launcher-Key": "launcher-test-key"},
            json={"ctfd_user_id": user_id, "username": username,
                  "workspace_id": workspace_id, "token": token},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def _mock_evaluator(self, method, path, **kwargs):
        if path == "/internal/arena/validate-attack":
            return fake_response(self.validation)
        if path == "/internal/arena/evaluate-defense":
            return fake_response(self.protected)
        if path == "/internal/arena/sources":
            return fake_response({"sources": [{"source_id": "4", "class_index": 0,
                                                 "class_label": "NORMAL"}],
                                  "max_linf": 0.02, "class_names": ["NORMAL", "PNEUMONIA"]})
        if path == "/internal/arena/sources/4":
            response = fake_response({})
            response._content = self.png
            response.headers["Content-Type"] = "image/png"
            return response
        if path == "/internal/arena/model":
            response = fake_response({})
            response._content = b"checkpoint"
            return response
        raise AssertionError(f"Unexpected evaluator call: {method} {path}")

    def _submit_attack(self):
        return self.client.post(
            "/api/red/attacks",
            headers=self.red_headers,
            json={"source_id": "4", "image_b64": base64.b64encode(self.png).decode(),
                  "attack_name": "PGD", "epsilon": 0.02, "step_size": 0.0025,
                  "iterations": 8, "ctfd_user_id": 999, "role": "blue"},
        )

    def test_token_identity_role_checks_and_red_blue_exchange(self):
        self.assertEqual(self.client.get("/api/identity", headers=self.red_headers).json()["role"], "red")
        self.assertEqual(self.client.get("/api/identity", headers=self.blue_headers).json()["ctfd_user_id"], 12)
        self.assertEqual(self.client.get("/api/red/flag", headers=self.red_headers).status_code, 403)
        self.assertEqual(self.client.get("/api/blue/flag", headers=self.blue_headers).status_code, 403)
        self.assertEqual(self.client.get("/api/red/flag", headers=self.blue_headers).status_code, 403)
        self.assertEqual(self.client.get("/api/blue/flag", headers=self.red_headers).status_code, 403)
        self.assertEqual(self.client.get("/api/blue/pending", headers=self.red_headers).status_code, 403)
        self.assertEqual(self.client.post("/api/blue/responses", headers=self.red_headers,
                                         json={"attack_id": "f" * 32, "sigma": 0.05,
                                               "num_samples": 100}).status_code, 403)
        original = arena.evaluator_request
        logger = arena._log_mlflow_run
        arena.evaluator_request = self._mock_evaluator
        telemetry = []
        arena._log_mlflow_run = lambda **kwargs: telemetry.append(kwargs) or f"fake-run-{len(telemetry)}"
        try:
            self.assertEqual(self.client.get("/api/sources", headers=self.red_headers).status_code, 200)
            submitted = self._submit_attack()
            self.assertEqual(submitted.status_code, 200, submitted.text)
            info = submitted.json()
            self.assertEqual(info["flag"], arena.XRAY_RED_FLAG)
            self.assertEqual(self.client.get("/api/red/flag", headers=self.red_headers).json()["flag"], arena.XRAY_RED_FLAG)
            self.assertEqual(info["sequence_number"], 1)
            self.assertEqual(info["sha256"], hashlib.sha256(self.png).hexdigest())
            self.assertEqual(info["adversarial_prediction"], "PNEUMONIA")
            red_audit = telemetry[0]["artifact_json"]["decision_audit"]
            self.assertTrue(red_audit["success"])
            self.assertNotIn("flag_returned", red_audit)
            self.assertEqual(red_audit["failure_reasons"], [])
            self.assertIn("maximum_measured_linf", [item["name"] for item in red_audit["criteria"]])

            pending = self.client.get("/api/blue/pending", headers=self.blue_headers).json()
            self.assertTrue(pending["pending"])
            self.assertEqual(pending["attack_id"], info["attack_id"])
            image = self.client.get(pending["artifact_url"], headers=self.blue_headers)
            self.assertEqual(image.status_code, 200)
            self.assertEqual(hashlib.sha256(image.content).hexdigest(), pending["sha256"])
            self.assertEqual(image.headers["X-Artifact-SHA256"], pending["sha256"])

            rejected_second = self._submit_attack()
            self.assertEqual(rejected_second.status_code, 409)
            self.assertIn("waiting for Blue", rejected_second.json()["detail"])

            defense = self.client.post(
                "/api/blue/responses", headers=self.blue_headers,
                json={"attack_id": info["attack_id"], "sigma": 0.05,
                      "num_samples": 100, "abstain_threshold": 0.75},
            )
            self.assertEqual(defense.status_code, 200, defense.text)
            self.assertTrue(defense.json()["protected_result"]["defense_success"])
            self.assertEqual(defense.json()["flag"], arena.XRAY_BLUE_FLAG)
            self.assertEqual(self.client.get("/api/blue/flag", headers=self.blue_headers).json()["flag"], arena.XRAY_BLUE_FLAG)
            blue_audit = telemetry[1]["artifact_json"]["decision_audit"]
            self.assertTrue(blue_audit["success"])
            self.assertEqual(blue_audit["failure_reasons"], [])
            self.assertIn("adversarial_attack_blocked", [item["name"] for item in blue_audit["criteria"]])
            red_result = self.client.get("/api/red/latest-response", headers=self.red_headers).json()
            self.assertTrue(red_result["available"])
            self.assertEqual(red_result["defense_parameters"]["sigma"], 0.05)
            self.assertEqual(self.client.get("/api/match/history", headers=self.red_headers).json()["history"][0]["sequence_number"], 1)
            self.assertFalse(self.client.get("/api/blue/pending", headers=self.blue_headers).json()["pending"])

            second = self._submit_attack()
            self.assertEqual(second.status_code, 200, second.text)
            self.assertEqual(second.json()["sequence_number"], 2)
            status = self.client.get("/api/match/status", headers=self.red_headers).json()
            self.assertEqual(status["pending_attack_sequence"], 2)
            self.assertEqual(status["red_user"]["username"], "red_audit_user")
            self.assertEqual(status["blue_user"]["username"], "blue_audit_user")
            self.assertEqual(status["last_response_status"], "pending_blue")
            self.assertEqual(status["last_response_sequence"], 1)
            self.assertEqual(status["current_blue_defense"]["sigma"], 0.05)
            self.assertEqual(telemetry[0]["metrics"]["flag_awarded"], 1.0)
            self.assertEqual(telemetry[1]["metrics"]["flag_awarded"], 1.0)
            self.assertNotIn(arena.XRAY_RED_FLAG, repr(telemetry))
            self.assertNotIn(arena.XRAY_BLUE_FLAG, repr(telemetry))
        finally:
            arena.evaluator_request = original
            arena._log_mlflow_run = logger

    def test_solo_match_uses_one_workspace_for_both_roles(self):
        token = "solo-workspace-token-" + "s" * 24
        self._register(13, "learner", token, "u13")
        headers = {"Authorization": f"Bearer {token}"}
        admin = {"X-Arena-Admin-Key": "admin-test-key"}
        self.assertEqual(self.client.post(
            "/admin/matches/solo", headers={"X-Arena-Admin-Key": "wrong"},
            json={"user_id": 13, "username": "learner"},
        ).status_code, 403)
        created = self.client.post(
            "/admin/matches/solo", headers=admin,
            json={"user_id": 13, "username": "learner"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["mode"], "solo")
        match_id = created.json()["match_id"]
        self.assertEqual(self.client.get("/api/identity", headers=headers).json()["mode"], "solo")
        status = self.client.get("/api/match/status", headers=headers).json()
        self.assertEqual(status["mode"], "solo")
        self.assertEqual(status["role"], "solo")
        self.assertEqual(status["red_user"]["ctfd_user_id"], 13)
        self.assertEqual(status["blue_user"]["ctfd_user_id"], 13)
        self.assertEqual(self.client.get("/api/blue/pending", headers=headers).json(), {"pending": False})
        self.assertEqual(self.client.post(
            "/admin/matches/solo", headers=admin,
            json={"user_id": 13, "username": "learner"},
        ).status_code, 409)
        self.assertEqual(self.client.post(
            "/admin/matches", headers=admin,
            json={"red_user_id": 13, "red_username": "learner",
                  "blue_user_id": 12, "blue_username": "blue_audit_user"},
        ).status_code, 409)

        original_evaluator = arena.evaluator_request
        original_logger = arena._log_mlflow_run
        telemetry = []
        arena.evaluator_request = self._mock_evaluator
        arena._log_mlflow_run = lambda **kwargs: telemetry.append(kwargs) or f"solo-run-{len(telemetry)}"
        try:
            submitted = self.client.post(
                "/api/red/attacks", headers=headers,
                json={"source_id": "4", "image_b64": base64.b64encode(self.png).decode(),
                      "attack_name": "PGD", "epsilon": 0.02, "step_size": 0.0025,
                      "iterations": 8},
            )
            self.assertEqual(submitted.status_code, 200, submitted.text)
            self.assertEqual(submitted.json()["flag"], arena.XRAY_RED_FLAG)
            attack_id = submitted.json()["attack_id"]
            self.assertEqual(self.client.get("/api/red/flag", headers=headers).json()["flag"], arena.XRAY_RED_FLAG)
            self.assertEqual(self.client.get("/api/blue/pending", headers=headers).json()["attack_id"], attack_id)
            self.assertEqual(self.client.get(f"/api/attacks/{attack_id}/artifact", headers=headers).status_code, 200)
            defended = self.client.post(
                "/api/blue/responses", headers=headers,
                json={"attack_id": attack_id, "sigma": 0.05,
                      "num_samples": 100, "abstain_threshold": 0.75},
            )
            self.assertEqual(defended.status_code, 200, defended.text)
            self.assertEqual(defended.json()["flag"], arena.XRAY_BLUE_FLAG)
            self.assertEqual(self.client.get("/api/blue/flag", headers=headers).json()["flag"], arena.XRAY_BLUE_FLAG)
            self.assertTrue(self.client.get("/api/red/latest-response", headers=headers).json()["available"])
            self.assertEqual([item["role"] for item in telemetry], ["red", "blue"])
            self.assertTrue(all(item["match_id"] == match_id for item in telemetry))
            self.assertTrue(all(item["participant_info"]["ctfd_user_id"] == 13 for item in telemetry))
        finally:
            arena.evaluator_request = original_evaluator
            arena._log_mlflow_run = original_logger

    def test_existing_database_migrates_match_mode(self):
        with tempfile.TemporaryDirectory(prefix="arena-schema-") as temp_dir:
            data_dir = Path(temp_dir)
            with sqlite3.connect(data_dir / "arena.db") as conn:
                conn.execute("CREATE TABLE matches (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
                conn.execute("INSERT INTO matches VALUES ('old', 'active', 'before', 'before')")
            with patch.object(arena, "DATA_DIR", data_dir), \
                 patch.object(arena, "DB_PATH", data_dir / "arena.db"), \
                 patch.object(arena, "ARTIFACTS_DIR", data_dir / "artifacts"):
                arena.init_db()
                with arena.db_connection() as conn:
                    self.assertEqual(conn.execute("SELECT mode FROM matches WHERE id='old'").fetchone()["mode"], "paired")

    def test_unsuccessful_blue_response_does_not_award_flag(self):
        self.protected.update({"defense_success": False, "attack_survived": True,
                               "defended_prediction_index": 1, "defended_prediction_label": "PNEUMONIA"})
        original_evaluator = arena.evaluator_request
        original_logger = arena._log_mlflow_run
        arena.evaluator_request = self._mock_evaluator
        telemetry = []
        arena._log_mlflow_run = lambda **kwargs: telemetry.append(kwargs) or "test-run"
        try:
            attack = self._submit_attack()
            self.assertEqual(attack.status_code, 200, attack.text)
            response = self.client.post(
                "/api/blue/responses", headers=self.blue_headers,
                json={"attack_id": attack.json()["attack_id"], "sigma": 0.05,
                      "num_samples": 100, "abstain_threshold": 0.75},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("flag", response.json())
            self.assertEqual(self.client.get("/api/blue/flag", headers=self.blue_headers).status_code, 403)
            self.assertEqual(telemetry[-1]["metrics"]["flag_awarded"], 0.0)
        finally:
            arena.evaluator_request = original_evaluator
            arena._log_mlflow_run = original_logger

    def test_missing_flag_configuration_does_not_consume_a_turn(self):
        with patch.object(arena, "XRAY_RED_FLAG", ""):
            response = self._submit_attack()
            self.assertEqual(response.status_code, 503)
        self.assertEqual(self.client.get("/api/match/status", headers=self.red_headers).json()["latest_sequence"], 0)
        with patch.object(arena, "XRAY_BLUE_FLAG", ""):
            response = self.client.post(
                "/api/blue/responses", headers=self.blue_headers,
                json={"attack_id": "f" * 32, "sigma": 0.05, "num_samples": 100},
            )
            self.assertEqual(response.status_code, 503)

    def test_match_isolation_and_parameter_bounds(self):
        self._register(21, "red_other", "red-other-token-" + "x" * 24, "u21")
        self._register(22, "blue_other", "blue-other-token-" + "y" * 24, "u22")
        other = self.client.post(
            "/admin/matches", headers={"X-Arena-Admin-Key": "admin-test-key"},
            json={"red_user_id": 21, "red_username": "red_other",
                  "blue_user_id": 22, "blue_username": "blue_other"},
        ).json()
        other_red_headers = {"Authorization": "Bearer red-other-token-" + "x" * 24}
        other_blue_headers = {"Authorization": "Bearer blue-other-token-" + "y" * 24}
        self.assertEqual(self.client.get("/api/match/status", headers=other_blue_headers).json()["match_id"], other["match_id"])
        self.assertNotEqual(other["match_id"], self.match_id)
        original_evaluator = arena.evaluator_request
        original_logger = arena._log_mlflow_run
        arena.evaluator_request = self._mock_evaluator
        arena._log_mlflow_run = lambda **kwargs: None
        try:
            attack = self._submit_attack().json()
            self.assertTrue(self.client.get("/api/blue/pending", headers=self.blue_headers).json()["pending"])
            self.assertFalse(self.client.get("/api/blue/pending", headers=other_blue_headers).json()["pending"])
            self.assertEqual(
                self.client.get(f"/api/attacks/{attack['attack_id']}/artifact", headers=other_blue_headers).status_code,
                404,
            )
            self.assertEqual(self.client.get("/api/match/history", headers=other_red_headers).json()["history"], [])
        finally:
            arena.evaluator_request = original_evaluator
            arena._log_mlflow_run = original_logger
        self.assertEqual(self.client.post(
            "/api/blue/responses", headers=self.blue_headers,
            json={"attack_id": "a" * 32, "sigma": 0.11, "num_samples": 501,
                  "abstain_threshold": 0.75},
        ).status_code, 422)
        self.assertEqual(self.client.get("/admin/matches", headers={"X-Arena-Admin-Key": "wrong"}).status_code, 403)

    def test_model_evaluated_red_rejection_is_audited_without_changing_422(self):
        original_evaluator = arena.evaluator_request
        original_logger = arena._log_mlflow_run
        captured = []

        def reject_candidate(method, path, **_kwargs):
            self.assertEqual(path, "/internal/arena/validate-attack")
            return fake_response({
                "error": "Candidate exceeds the exercise L-infinity budget",
                "measured_linf": 0.03,
                "max_linf": 0.02,
                "linf_tolerance": 1e-8,
            }, status=422)

        def capture(**kwargs):
            kwargs["artifact_bytes"] = Path(kwargs["artifact"]).read_bytes()
            captured.append(kwargs)
            return "audit-run"

        arena.evaluator_request = reject_candidate
        arena._log_mlflow_run = capture
        try:
            response = self._submit_attack()
        finally:
            arena.evaluator_request = original_evaluator
            arena._log_mlflow_run = original_logger

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("flag", response.json())
        self.assertEqual(self.client.get("/api/red/flag", headers=self.red_headers).status_code, 403)
        self.assertEqual(response.json()["detail"], "Candidate exceeds the exercise L-infinity budget")
        self.assertEqual(len(captured), 1)
        audit = captured[0]["artifact_json"]["decision_audit"]
        self.assertFalse(audit["success"])
        self.assertNotIn("flag_returned", audit)
        self.assertEqual(audit["failure_reasons"], ["maximum_measured_linf"])
        self.assertEqual(audit["criteria"][1]["observed"], 0.03)
        self.assertEqual(audit["criteria"][1]["maximum"], 0.02)
        self.assertEqual(audit["criteria"][1]["tolerance"], 1e-8)
        self.assertAlmostEqual(audit["criteria"][1]["effective_maximum"], 0.02000001)
        self.assertEqual(captured[0]["metrics"]["success"], 0.0)
        self.assertEqual(captured[0]["artifact_bytes"], self.png)

    def test_non_flipping_red_image_earns_no_flag_and_creates_no_blue_turn(self):
        captured = []

        def reject_candidate(method, path, **_kwargs):
            self.assertEqual(path, "/internal/arena/validate-attack")
            return fake_response({
                "error": "Candidate does not fool the standard model",
                "clean_prediction": "NORMAL",
                "adversarial_prediction": "NORMAL",
                "measured_linf": 5 / 255,
                "max_linf": 0.02,
            }, status=422)

        with patch.object(arena, "evaluator_request", side_effect=reject_candidate), \
             patch.object(arena, "_log_mlflow_run", side_effect=lambda **kwargs: captured.append(kwargs) or "audit-run"):
            response = self._submit_attack()

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "Candidate does not fool the standard model")
        self.assertNotIn("flag", response.json())
        self.assertEqual(self.client.get("/api/red/flag", headers=self.red_headers).status_code, 403)
        status = self.client.get("/api/match/status", headers=self.red_headers).json()
        self.assertEqual(status["latest_sequence"], 0)
        self.assertIsNone(status["pending_attack_sequence"])
        self.assertFalse(self.client.get("/api/blue/pending", headers=self.blue_headers).json()["pending"])
        self.assertEqual(captured[0]["artifact_json"]["decision_audit"]["failure_reasons"],
                         ["standard_model_misclassification"])
        self.assertEqual(captured[0]["metrics"]["attack_success"], 0.0)

    def test_arena_mlflow_writes_audited_evaluation_artifact_and_reason_tags(self):
        client = FakeMlflowClient()
        audit = {"success": False, "failure_reasons": ["maximum_measured_linf"],
                 "criteria": [{"name": "maximum_measured_linf", "passed": False}]}
        with patch.object(arena, "_mlflow_client", return_value=client), \
             patch.object(arena, "_mlflow_experiment", return_value="experiment-test"):
            run_id = arena._log_mlflow_run(
                role="red", match_id="m" * 32, sequence=3,
                participant_info={"ctfd_user_id": 11, "username": "red_audit_user"},
                params={"epsilon": 0.02}, metrics={"success": 0.0},
                artifact_json={"result": {"success": False}, "decision_audit": audit},
            )
        self.assertEqual(run_id, "arena-test-run")
        self.assertEqual(client.tags["evaluation_success"], "false")
        self.assertEqual(client.tags["failure_reasons"], "maximum_measured_linf")
        self.assertEqual(client.tags["evaluation_audit_status"], "complete")
        logged = json.loads(client.artifacts["evaluation/evaluation.json"])
        self.assertEqual(logged["decision_audit"], audit)
        self.assertEqual(client.status, "FINISHED")

    def test_arena_missing_audit_is_not_mislabeled_as_no_failure(self):
        client = FakeMlflowClient()
        with patch.object(arena, "_mlflow_client", return_value=client), \
             patch.object(arena, "_mlflow_experiment", return_value="experiment-test"):
            arena._log_mlflow_run(
                role="blue", match_id="m" * 32, sequence=4,
                participant_info={"ctfd_user_id": 12, "username": "blue_audit_user"},
                params={}, metrics={"success": 0.0},
                artifact_json={"result": {"defense_success": False}},
            )
        self.assertEqual(client.tags["evaluation_success"], "false")
        self.assertEqual(client.tags["failure_reasons"], "evaluation_audit_missing")
        self.assertEqual(client.tags["evaluation_audit_status"], "missing")

    def test_instructor_reset_clears_arena_data_but_preserves_pairing(self):
        original_evaluator = arena.evaluator_request
        original_logger = arena._log_mlflow_run
        arena.evaluator_request = self._mock_evaluator
        arena._log_mlflow_run = lambda **kwargs: None
        try:
            self.assertEqual(self._submit_attack().status_code, 200)
        finally:
            arena.evaluator_request = original_evaluator
            arena._log_mlflow_run = original_logger

        artifact = Path(arena.ARTIFACTS_DIR) / self.match_id / "red" / "attack_001.png"
        self.assertTrue(artifact.is_file())
        reset = self.client.post(
            f"/admin/matches/{self.match_id}/reset",
            headers={"X-Arena-Admin-Key": "admin-test-key"},
        )
        self.assertEqual(reset.status_code, 200, reset.text)
        self.assertFalse(artifact.exists())
        self.assertFalse(self.client.get("/api/blue/pending", headers=self.blue_headers).json()["pending"])
        self.assertEqual(self.client.get("/api/match/status", headers=self.red_headers).json()["latest_sequence"], 0)
        match = self.client.get(
            f"/admin/matches/{self.match_id}",
            headers={"X-Arena-Admin-Key": "admin-test-key"},
        ).json()
        self.assertEqual(len(match["participants"]), 2)
        self.assertEqual(match["attacks"], [])
        self.assertEqual(match["responses"], [])

    def test_persistent_sqlite_state_and_workspace_unregister(self):
        match_id = self.match_id
        arena.init_db()
        with arena.db_connection() as conn:
            self.assertEqual(conn.execute("SELECT status FROM matches WHERE id=?", (match_id,)).fetchone()[0], "active")
        removed = self.client.post(
            "/internal/participants/unregister",
            headers={"X-Arena-Launcher-Key": "launcher-test-key"},
            json={"ctfd_user_id": 11, "workspace_id": "u11"},
        )
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(self.client.get("/api/identity", headers=self.red_headers).status_code, 401)


if __name__ == "__main__":
    unittest.main()
