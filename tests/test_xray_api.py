import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import importlib
import io
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "xray_red_blue"))
sys.modules.setdefault("xray_defenses", importlib.import_module("defenses"))
import redblue_app


class FakeExecutor:
    def submit(self, _function, *_args, **_kwargs):
        return object()


class ThresholdClassifier:
    def predict(self, images, batch_size=8):
        score = np.asarray(images).mean(axis=(1, 2, 3))
        class_one = score >= 0.995
        p1 = np.where(class_one, 0.9, 0.1).astype(np.float32)
        return np.stack((1.0 - p1, p1), axis=1)


class MeanClassifier:
    def predict(self, images, batch_size=8):
        score = np.asarray(images).mean(axis=(1, 2, 3))
        class_one = score >= 0.5
        p1 = np.where(class_one, 0.9, 0.1).astype(np.float32)
        return np.stack((1.0 - p1, p1), axis=1)


class SubtractSecondImageAttack:
    def __init__(self, estimator, eps, **_kwargs):
        self.eps = eps

    def generate(self, images):
        adversarial = np.asarray(images).copy()
        adversarial[1] -= self.eps
        return adversarial


class XRayApiTests(unittest.TestCase):
    def setUp(self):
        redblue_app.APP_ROOT = Path("/app")
        self.previous_state = redblue_app._state.copy()
        redblue_app._state.clear()
        redblue_app._state.update(redblue_app._default_state())
        redblue_app._jobs.clear()
        redblue_app.app.config["TESTING"] = True
        self.client = redblue_app.app.test_client()

    def tearDown(self):
        redblue_app._state.clear()
        redblue_app._state.update(self.previous_state)
        redblue_app._jobs.clear()

    def test_health_and_published_blue_options(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        config = self.client.get("/api/config").get_json()
        self.assertIn("robust", config["blue_models"])
        self.assertIn("disagreement", config["detectors"])
        self.assertEqual(config["scoring"]["blue_min_detection_rate"], 0.05)

    def test_blue_download_contains_only_successful_active_red_examples(self):
        run_id = "accepted-red"
        images = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
        labels = np.array([0, 0, 1, 1])
        clean_predictions = np.array([0, 1, 1, 1])
        adv_predictions = np.array([1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)):
            np.savez_compressed(
                Path(directory) / f"red-{run_id}.npz", x_clean=images,
                x_adv=images + 1, labels=labels,
                clean_predictions=clean_predictions, adv_predictions=adv_predictions,
            )
            redblue_app._state.update(phase="red", red_run_id=run_id, red_result={"success": True})
            self.assertEqual(self.client.get(f"/api/round/latest/successes?run_id={run_id}").status_code, 404)
            redblue_app._state["phase"] = "blue"
            self.assertEqual(self.client.get("/api/round/latest/successes?run_id=wrong").status_code, 404)
            response = self.client.get(f"/api/round/latest/successes?run_id={run_id}")
            self.assertEqual(response.status_code, 200)
            with np.load(io.BytesIO(response.data), allow_pickle=False) as data:
                np.testing.assert_array_equal(data["labels"], np.array([0, 1]))
                np.testing.assert_array_equal(data["x_adv"], images[[0, 3]] + 1)

    def test_red_rejects_bad_attack_and_only_queues_one_job(self):
        bad = self.client.post("/api/red/attack", json={"attack": "OTHER"})
        self.assertEqual(bad.status_code, 400)
        with patch.object(redblue_app, "_executor", FakeExecutor()):
            first = self.client.post("/api/red/attack", json={"attack": "FGSM"})
            duplicate = self.client.post("/api/red/attack", json={"attack": "PGD"})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(duplicate.status_code, 409)

    def test_red_submission_uses_launcher_identity_and_does_not_expose_it_in_job(self):
        identity = {
            "ctfd_user_id": 3,
            "ctfd_username": "student1",
            "workspace_id": "u3",
            "identity_source": "arena_token",
        }
        with patch.object(redblue_app, "_resolve_workspace_identity", return_value=identity), \
             patch.object(redblue_app, "_executor", FakeExecutor()):
            response = self.client.post(
                "/api/red/attack",
                json={"attack": "FGSM", "ctfd_user_id": 999, "ctfd_username": "spoofed"},
            )

        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["job_id"]
        self.assertEqual(redblue_app._jobs[job_id]["identity"], identity)
        self.assertNotIn("identity", redblue_app._job_view(job_id))

    def test_identity_endpoint_returns_launcher_mapping_for_workspace(self):
        identity = {
            "ctfd_user_id": 3,
            "ctfd_username": "student1",
            "workspace_id": "u3",
            "identity_source": "jupyter_token",
        }
        with patch.object(redblue_app, "_resolve_workspace_identity", return_value=identity):
            response = self.client.get("/api/identity")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), identity)

    def test_concurrent_red_requests_reserve_only_one_job(self):
        barrier = threading.Barrier(2)
        original_number = redblue_app._number

        def synchronized_number(body, key, default, minimum, maximum):
            value = original_number(body, key, default, minimum, maximum)
            if key == "epsilon":
                barrier.wait(timeout=5)
            return value

        clients = [redblue_app.app.test_client(), redblue_app.app.test_client()]
        with patch.object(redblue_app, "_number", side_effect=synchronized_number), \
             patch.object(redblue_app, "_executor", FakeExecutor()):
            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = list(pool.map(
                    lambda client: client.post("/api/red/attack", json={"attack": "FGSM"}),
                    clients,
                ))
        self.assertEqual(sorted(response.status_code for response in responses), [202, 409])

    def test_blue_rejects_unknown_model(self):
        redblue_app._state.update(phase="blue", red_run_id="fixture")
        response = self.client.post("/api/blue/defend", json={"model": "untrusted"})
        self.assertEqual(response.status_code, 400)


    def test_public_round_summaries_never_expose_flags(self):
        redblue_app._state.update(
            phase="complete",
            red_run_id="red-run",
            red_result={"run_id": "red-run", "success": True, "flag": "red-private-value"},
            blue_result={"run_id": "blue-run", "success": True, "flag": "blue-private-value"},
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)):
            (Path(directory) / "red-red-run.npz").touch()
            round_response = self.client.get("/api/round")
            latest_response = self.client.get("/api/round/latest")
        self.assertEqual(round_response.status_code, 200)
        self.assertEqual(latest_response.status_code, 200)
        public_json = json.dumps([round_response.get_json(), latest_response.get_json()])
        self.assertNotIn("red-private-value", public_json)
        self.assertNotIn("blue-private-value", public_json)
        self.assertNotIn('"flag"', public_json)

    def test_successful_red_flag_is_job_only_not_round_state(self):
        images = np.stack((np.zeros((3, 4, 4), dtype=np.float32), np.ones((3, 4, 4), dtype=np.float32)))
        labels = np.array([0, 1], dtype=np.int64)
        job_id = "red-job-id"
        redblue_app._jobs[job_id] = {"job_id": job_id, "status": "queued", "role": "red"}
        parameters = {
            "attack": "FGSM", "epsilon": 0.01, "step": 0.0025, "iterations": 1,
            "round_id": redblue_app._state["round_id"],
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)), \
             patch.object(redblue_app, "STATE_FILE", Path(directory) / "round.json"), \
             patch.object(redblue_app, "_challenge_batch", return_value=(images, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=ThresholdClassifier()), \
             patch.object(redblue_app, "FastGradientMethod", SubtractSecondImageAttack), \
             patch.object(redblue_app, "log_red_attempt") as mlflow_log, \
             patch.object(redblue_app, "RED_FLAG", "red-private-value"):
            redblue_app._run_red(job_id, parameters)

        job = redblue_app._job_view(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertTrue(job["result"]["success"])
        self.assertEqual(job["result"]["flag"], "red-private-value")
        self.assertNotIn("flag", redblue_app._state["red_result"])
        public_round_id = redblue_app._state["red_result"]["run_id"]
        self.assertNotEqual(public_round_id, job_id)
        mlflow_log.assert_called_once()
        self.assertTrue(mlflow_log.call_args.kwargs["metrics"]["success"])
        self.assertNotIn("flag", mlflow_log.call_args.kwargs["evaluation"])
        audit = mlflow_log.call_args.kwargs["evaluation"]["decision_audit"]
        self.assertTrue(audit["success"])
        self.assertTrue(audit["flag_returned"])
        self.assertEqual(audit["failure_reasons"], [])
        self.assertEqual(
            [criterion["name"] for criterion in audit["criteria"]],
            ["clean_correct_samples_available", "minimum_attack_success_rate",
             "maximum_measured_linf", "published_to_active_round"],
        )
        budget = next(c for c in audit["criteria"] if c["name"] == "maximum_measured_linf")
        self.assertEqual(budget["tolerance"], redblue_app.RED_LINF_TOLERANCE)
        self.assertEqual(budget["effective_maximum"], redblue_app.RED_MAX_EPSILON + redblue_app.RED_LINF_TOLERANCE)
        self.assertIn("x_adv", mlflow_log.call_args.kwargs["arrays"])
        self.assertEqual(self.client.get(f"/api/jobs/{public_round_id}").status_code, 404)
        self.assertNotIn("red-private-value", json.dumps(self.client.get("/api/round").get_json()))
        self.assertNotIn("red-private-value", json.dumps(self.client.get("/api/round/latest").get_json()))

    def test_later_red_success_earns_flag_without_replacing_shared_artifact(self):
        images = np.stack((np.zeros((3, 4, 4), dtype=np.float32), np.ones((3, 4, 4), dtype=np.float32)))
        labels = np.array([0, 1], dtype=np.int64)
        first_result = {"run_id": "first-red", "success": True, "adversarial_accuracy": 0.5}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)), \
             patch.object(redblue_app, "STATE_FILE", Path(directory) / "round.json"), \
             patch.object(redblue_app, "_challenge_batch", return_value=(images, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=ThresholdClassifier()), \
             patch.object(redblue_app, "FastGradientMethod", SubtractSecondImageAttack), \
             patch.object(redblue_app, "log_red_attempt") as mlflow_log, \
             patch.object(redblue_app, "RED_FLAG", "red-private-value"), \
             patch.object(redblue_app, "_executor", FakeExecutor()):
            artifact_path = Path(directory) / "red-first-red.npz"
            np.savez_compressed(artifact_path, x_clean=images, x_adv=images, labels=labels)
            frozen_bytes = artifact_path.read_bytes()
            for phase in ("blue", "complete"):
                with self.subTest(phase=phase):
                    redblue_app._state.update(
                        phase=phase, red_run_id="first-red", red_result=first_result,
                        blue_result={"run_id": "first-blue", "success": True} if phase == "complete" else None,
                    )
                    before = dict(redblue_app._state)
                    queued = self.client.post("/api/red/attack", json={"attack": "FGSM", "epsilon": 0.01})
                    self.assertEqual(queued.status_code, 202)
                    job_id = queued.get_json()["job_id"]
                    redblue_app._run_red(job_id, redblue_app._jobs[job_id]["parameters"])
                    result = redblue_app._job_view(job_id)["result"]
                    self.assertTrue(result["success"])
                    self.assertEqual(result["flag"], "red-private-value")
                    self.assertFalse(result["shared_artifact_published"])
                    self.assertEqual(redblue_app._state, before)
                    self.assertEqual(artifact_path.read_bytes(), frozen_bytes)
            self.assertEqual(mlflow_log.call_count, 2)
            for call in mlflow_log.call_args_list:
                self.assertTrue(call.kwargs["metrics"]["success"])
                self.assertFalse(call.kwargs["metrics"]["shared_artifact_published"])
                self.assertNotIn("flag", call.kwargs["evaluation"])
                audit = call.kwargs["evaluation"]["decision_audit"]
                self.assertTrue(audit["success"])
                self.assertIn("qualified_with_active_round", [item["name"] for item in audit["criteria"]])

    def test_red_job_from_reset_round_cannot_earn_flag(self):
        images = np.stack((np.zeros((3, 4, 4), dtype=np.float32), np.ones((3, 4, 4), dtype=np.float32)))
        labels = np.array([0, 1], dtype=np.int64)
        job_id = "stale-red"
        redblue_app._jobs[job_id] = {"job_id": job_id, "status": "queued", "role": "red"}
        parameters = {"attack": "FGSM", "epsilon": 0.01, "step": 0.0025,
                      "iterations": 1, "round_id": "previous-round"}
        with patch.object(redblue_app, "_challenge_batch", return_value=(images, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=ThresholdClassifier()), \
             patch.object(redblue_app, "FastGradientMethod", SubtractSecondImageAttack), \
             patch.object(redblue_app, "log_red_attempt") as mlflow_log, \
             patch.object(redblue_app, "RED_FLAG", "red-private-value"):
            redblue_app._run_red(job_id, parameters)
        result = redblue_app._job_view(job_id)["result"]
        self.assertFalse(result["success"])
        self.assertNotIn("flag", result)
        self.assertFalse(mlflow_log.call_args.kwargs["metrics"]["success"])

    def test_perfect_model_without_any_detection_does_not_pass_blue(self):
        images = np.zeros((4, 3, 4, 4), dtype=np.float32)
        labels = np.zeros(4, dtype=np.int64)
        job_id = "blue-no-detection"
        run_id = "red-run"
        redblue_app._state.update(
            phase="blue", red_run_id=run_id, red_result={"adversarial_accuracy": 0.4}
        )
        redblue_app._jobs[job_id] = {"job_id": job_id, "status": "queued", "role": "blue"}
        parameters = {
            "model": "robust", "preprocess": "none", "bits": 5, "detector": "none",
            "samples": 4, "sigma": 0.03, "threshold": 0.75, "confidence": 0.6,
            "round_id": redblue_app._state["round_id"],
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)), \
             patch.object(redblue_app, "STATE_FILE", Path(directory) / "round.json"), \
             patch.object(redblue_app, "_challenge_batch", return_value=(images, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=MeanClassifier()), \
             patch.object(redblue_app, "log_blue_attempt") as mlflow_log, \
             patch.object(redblue_app, "BLUE_MIN_DETECTION_RATE", 0.15), \
             patch.object(redblue_app, "BLUE_FLAG", "blue-private-value"):
            np.savez_compressed(Path(directory) / f"red-{run_id}.npz", x_clean=images, x_adv=images, labels=labels)
            redblue_app._run_blue(job_id, parameters, run_id)

        result = redblue_app._job_view(job_id)["result"]
        self.assertEqual(result["clean_usable_accuracy"], 1.0)
        self.assertEqual(result["adv_resolved_accuracy"], 1.0)
        self.assertEqual(result["detection_rate"], 0.0)
        self.assertFalse(result["success"])
        self.assertNotIn("flag", result)
        mlflow_log.assert_called_once()
        self.assertFalse(mlflow_log.call_args.kwargs["metrics"]["success"])
        audit = mlflow_log.call_args.kwargs["evaluation"]["decision_audit"]
        self.assertFalse(audit["success"])
        self.assertFalse(audit["flag_returned"])
        self.assertIn("minimum_detection_rate", audit["failure_reasons"])
        self.assertIn("minimum_adversarial_resolved_accuracy", [c["name"] for c in audit["criteria"]])

    def test_successful_blue_flag_is_job_only_not_round_state(self):
        clean = np.zeros((4, 3, 4, 4), dtype=np.float32)
        adversarial = np.ones_like(clean)
        labels = np.zeros(4, dtype=np.int64)
        job_id = "blue-job-id"
        red_run_id = "red-run"
        redblue_app._state.update(
            phase="blue", red_run_id=red_run_id, red_result={"adversarial_accuracy": 0.4}
        )
        redblue_app._jobs[job_id] = {"job_id": job_id, "status": "queued", "role": "blue"}
        parameters = {
            "model": "standard", "preprocess": "none", "bits": 5, "detector": "disagreement",
            "samples": 4, "sigma": 0.03, "threshold": 0.75, "confidence": 0.6,
            "round_id": redblue_app._state["round_id"],
        }
        clean_flags = np.array([True, False, False, False])
        adv_flags = np.array([True, True, True, False])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)), \
             patch.object(redblue_app, "STATE_FILE", Path(directory) / "round.json"), \
             patch.object(redblue_app, "_challenge_batch", return_value=(clean, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=MeanClassifier()), \
             patch.object(redblue_app, "_confidence_gated_disagreement", side_effect=[clean_flags, adv_flags]), \
             patch.object(redblue_app, "log_blue_attempt") as mlflow_log, \
             patch.object(redblue_app, "BLUE_MIN_DETECTION_RATE", 0.15), \
             patch.object(redblue_app, "BLUE_FLAG", "blue-private-value"):
            np.savez_compressed(Path(directory) / f"red-{red_run_id}.npz", x_clean=clean, x_adv=adversarial, labels=labels)
            redblue_app._run_blue(job_id, parameters, red_run_id)

        result = redblue_app._job_view(job_id)["result"]
        self.assertTrue(result["success"])
        self.assertEqual(result["flag"], "blue-private-value")
        mlflow_log.assert_called_once()
        self.assertTrue(mlflow_log.call_args.kwargs["metrics"]["success"])
        self.assertNotIn("flag", mlflow_log.call_args.kwargs["evaluation"])
        audit = mlflow_log.call_args.kwargs["evaluation"]["decision_audit"]
        self.assertTrue(audit["success"])
        self.assertTrue(audit["flag_returned"])
        self.assertEqual(audit["failure_reasons"], [])
        self.assertNotIn("flag", redblue_app._state["blue_result"])
        public_blue_run_id = redblue_app._state["blue_result"]["run_id"]
        self.assertNotEqual(public_blue_run_id, job_id)
        self.assertEqual(self.client.get(f"/api/jobs/{public_blue_run_id}").status_code, 404)
        self.assertNotIn("blue-private-value", json.dumps(self.client.get("/api/round").get_json()))
        self.assertNotIn("blue-private-value", json.dumps(self.client.get("/api/round/latest").get_json()))

    def test_later_blue_success_earns_flag_without_replacing_completed_result(self):
        clean = np.zeros((4, 3, 4, 4), dtype=np.float32)
        adversarial = np.ones_like(clean)
        labels = np.zeros(4, dtype=np.int64)
        red_run_id = "first-red"
        first_blue = {"run_id": "first-blue", "success": True}
        redblue_app._state.update(
            phase="complete", red_run_id=red_run_id,
            red_result={"adversarial_accuracy": 0.4, "success": True},
            blue_result=first_blue,
        )
        before = dict(redblue_app._state)
        clean_flags = np.array([True, False, False, False])
        adv_flags = np.array([True, True, True, False])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(redblue_app, "STATE_DIR", Path(directory)), \
             patch.object(redblue_app, "STATE_FILE", Path(directory) / "round.json"), \
             patch.object(redblue_app, "_challenge_batch", return_value=(clean, labels, ["NORMAL", "PNEUMONIA"], 2)), \
             patch.object(redblue_app.attack_core, "build_classifier", return_value=MeanClassifier()), \
             patch.object(redblue_app, "_confidence_gated_disagreement", side_effect=[clean_flags, adv_flags]), \
             patch.object(redblue_app, "log_blue_attempt") as mlflow_log, \
             patch.object(redblue_app, "BLUE_MIN_DETECTION_RATE", 0.15), \
             patch.object(redblue_app, "BLUE_FLAG", "blue-private-value"), \
             patch.object(redblue_app, "_executor", FakeExecutor()):
            np.savez_compressed(Path(directory) / f"red-{red_run_id}.npz",
                                x_clean=clean, x_adv=adversarial, labels=labels)
            queued = self.client.post("/api/blue/defend", json={
                "model": "standard", "detector": "disagreement", "confidence": 0.6,
            })
            self.assertEqual(queued.status_code, 202)
            job_id = queued.get_json()["job_id"]
            redblue_app._run_blue(job_id, redblue_app._jobs[job_id]["parameters"], red_run_id)
        result = redblue_app._job_view(job_id)["result"]
        self.assertTrue(result["success"])
        self.assertEqual(result["flag"], "blue-private-value")
        self.assertFalse(result["shared_result_published"])
        self.assertEqual(redblue_app._state, before)
        self.assertTrue(mlflow_log.call_args.kwargs["metrics"]["success"])
        self.assertFalse(mlflow_log.call_args.kwargs["metrics"]["shared_result_published"])
        self.assertNotIn("flag", mlflow_log.call_args.kwargs["evaluation"])
        audit = mlflow_log.call_args.kwargs["evaluation"]["decision_audit"]
        self.assertIn("qualified_with_completed_round", [item["name"] for item in audit["criteria"]])


if __name__ == "__main__":
    unittest.main()
