import sys
import unittest
from io import BytesIO
from pathlib import Path
import importlib

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "xray_red_blue"))
sys.modules.setdefault("xray_defenses", importlib.import_module("defenses"))
from defenses import prediction_consistency, preprocess_batch, randomized_smoothing
import redblue_app
from redblue_app import _confidence_gated_disagreement, _prediction_disagreement


class StableClassifier:
    def predict(self, images, batch_size=8):
        probabilities = np.zeros((len(images), 3), dtype=np.float32)
        probabilities[:, 1] = 0.9
        probabilities[:, 0] = 0.1
        return probabilities


class DefenseTests(unittest.TestCase):
    def setUp(self):
        self.images = np.linspace(0.0, 1.0, num=2 * 3 * 12 * 12, dtype=np.float32).reshape(2, 3, 12, 12)

    def test_preprocessors_preserve_shape_range_and_input(self):
        original = self.images.copy()
        for name in ("none", "median3", "gaussian", "quantize"):
            with self.subTest(name=name):
                result = preprocess_batch(self.images, name, bits=4)
                self.assertEqual(result.shape, self.images.shape)
                self.assertTrue(np.isfinite(result).all())
                self.assertGreaterEqual(float(result.min()), 0.0)
                self.assertLessEqual(float(result.max()), 1.0)
        np.testing.assert_array_equal(self.images, original)

    def test_invalid_preprocessor_and_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_batch(self.images, "arbitrary")
        with self.assertRaises(ValueError):
            preprocess_batch(np.zeros((1, 1, 8, 8), dtype=np.float32), "median3")

    def test_prediction_consistency_returns_majority_and_flags(self):
        labels, confidence, flagged = prediction_consistency(
            StableClassifier(), self.images, samples=4, sigma=0.01, threshold=0.75
        )
        np.testing.assert_array_equal(labels, np.ones(2, dtype=np.int64))
        np.testing.assert_array_equal(confidence, np.ones(2, dtype=np.float32))
        np.testing.assert_array_equal(flagged, np.zeros(2, dtype=bool))

    def test_prediction_disagreement_flags_only_different_labels(self):
        flagged = _prediction_disagreement(
            np.array([0, 1, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
        )
        np.testing.assert_array_equal(flagged, np.array([False, True, False]))

    def test_confidence_gated_detector_rejects_low_confidence_disagreements(self):
        primary = np.array([[0.8, 0.2], [0.55, 0.45], [0.4, 0.6]], dtype=np.float32)
        secondary = np.array([[0.1, 0.9], [0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        flagged = _confidence_gated_disagreement(primary, secondary, 0.6)
        np.testing.assert_array_equal(flagged, np.array([True, False, True]))


class ArenaEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.client = redblue_app.app.test_client()
        self.old_key = redblue_app.ARENA_INTERNAL_KEY
        redblue_app.ARENA_INTERNAL_KEY = "arena-test-key"
        self.source = np.full((3, 256, 256), 128 / 255.0, dtype=np.float32)
        image = np.rint(self.source.transpose(1, 2, 0) * 255).astype(np.uint8)
        png = BytesIO()
        Image.fromarray(image, mode="RGB").save(png, format="PNG")
        self.source_png = png.getvalue()
        self.previous_sources = redblue_app._arena_source_cache
        self.previous_metadata = redblue_app._arena_source_metadata
        self.previous_class_names = redblue_app._arena_class_names
        redblue_app._arena_source_cache = {"4": {
            "source_id": "4", "image": self.source, "label_index": 0,
            "label": "NORMAL", "png": self.source_png,
        }}
        redblue_app._arena_source_metadata = [{"source_id": "4", "class_index": 0, "class_label": "NORMAL"}]
        redblue_app._arena_class_names = ["NORMAL", "PNEUMONIA"]
        self.previous_model = redblue_app._arena_classifier
        redblue_app._arena_classifier = self.ThresholdClassifier()

    def tearDown(self):
        redblue_app.ARENA_INTERNAL_KEY = self.old_key
        redblue_app._arena_source_cache = self.previous_sources
        redblue_app._arena_source_metadata = self.previous_metadata
        redblue_app._arena_class_names = self.previous_class_names
        redblue_app._arena_classifier = self.previous_model

    class ThresholdClassifier:
        def predict(self, images, batch_size=8):
            probabilities = []
            for image in images:
                if float(np.mean(image)) > 129 / 255.0:
                    probabilities.append([0.1, 0.9])
                else:
                    probabilities.append([0.9, 0.1])
            return np.asarray(probabilities, dtype=np.float32)

    def _candidate_png(self, delta_pixels):
        candidate = np.clip(np.rint(self.source.transpose(1, 2, 0) * 255).astype(np.int16) + delta_pixels, 0, 255).astype(np.uint8)
        output = BytesIO()
        Image.fromarray(candidate, mode="RGB").save(output, format="PNG")
        return output.getvalue()

    def test_internal_source_requires_key_and_attack_is_checked_against_authoritative_source(self):
        denied = self.client.get("/internal/arena/sources")
        self.assertEqual(denied.status_code, 403)
        valid = self.client.post(
            "/internal/arena/validate-attack",
            headers={"X-Arena-Internal-Key": "arena-test-key"},
            data={"source_id": "4", "image": (BytesIO(self._candidate_png(2)), "candidate.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(valid.status_code, 200, valid.get_json())
        result = valid.get_json()
        self.assertTrue(result["valid"])
        self.assertEqual(result["true_class_label"], "NORMAL")
        self.assertEqual(result["adversarial_class_label"], "PNEUMONIA")
        self.assertLessEqual(result["measured_linf"], 0.02)
        self.assertEqual(result["linf_tolerance"], redblue_app.ARENA_LINF_TOLERANCE)

        over_budget = self.client.post(
            "/internal/arena/validate-attack",
            headers={"X-Arena-Internal-Key": "arena-test-key"},
            data={"source_id": "4", "image": (BytesIO(self._candidate_png(6)), "candidate.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(over_budget.status_code, 422)
        self.assertIn("L-infinity", over_budget.get_json()["error"])
        self.assertEqual(over_budget.get_json()["linf_tolerance"], redblue_app.ARENA_LINF_TOLERANCE)

    def test_randomized_smoothing_is_reproducible_and_bounded(self):
        images = np.zeros((1, 3, 8, 8), dtype=np.float32)
        first = randomized_smoothing(self.ThresholdClassifier(), images, samples=500,
                                     sigma=0.05, seed=14, batch_size=8)
        second = randomized_smoothing(self.ThresholdClassifier(), images, samples=500,
                                      sigma=0.05, seed=14, batch_size=8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        with self.assertRaises(ValueError):
            randomized_smoothing(self.ThresholdClassifier(), images, samples=501,
                                 sigma=0.05, seed=14)

    def test_arena_probability_metadata_uses_normalized_softmax_scores(self):
        probabilities = redblue_app._arena_probabilities(np.asarray([[2.0, 1.0], [4.0, 4.0]]))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), atol=1e-7)
        self.assertGreater(float(probabilities[0, 0]), 0.5)


if __name__ == "__main__":
    unittest.main()
