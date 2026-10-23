import itertools
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from poref.data import load_manifest
from poref.metrics import evaluate, summarize
from poref.pdm import decide, entropy_confidence, forward_backward
from poref.preprocessing import target_affine


def sample_manifest():
    return {"test_design": "fixed_holdout", "samples": [
        {"patient_id": f"p{i}", "label": i % 2, "images": {"QSM": f"p{i}.nii"}, "slice_axis": 2}
        for i in range(6)], "folds": [{"train": ["p0", "p1"], "val": ["p2", "p3"], "test": ["p4", "p5"]}]}


class DesignTests(unittest.TestCase):
    def test_forward_backward_matches_exhaustive_state_enumeration(self):
        p = np.array([0.2, 0.7, 0.85, 0.3])
        transition = np.array([[0.8, 0.2], [0.3, 0.7]])
        prior = np.array([0.6, 0.4])
        marginals, total = np.zeros(4), 0.0
        for states in itertools.product((0, 1), repeat=4):
            weight = prior[states[0]]
            for i, state in enumerate(states):
                weight *= p[i] if state else 1 - p[i]
                if i:
                    weight *= transition[states[i - 1], state]
            total += weight
            marginals += np.array(states) * weight
        np.testing.assert_allclose(forward_backward(p, transition, prior), marginals / total, atol=1e-12)

    def test_entropy_window_and_strict_threshold(self):
        h, c = entropy_confidence([0, 0.5, 1])
        np.testing.assert_allclose(h, [0, 1, 0])
        np.testing.assert_allclose(c, [1, 0, 1])
        with self.assertRaisesRegex(ValueError, "zero total confidence"):
            decide([0.5] * 96, window_length=5, window_threshold=0)
        result = decide([0.8] * 96, window_length=5, window_threshold=0)
        tied = decide([0.8] * 96, window_length=5, window_threshold=result["score"])
        self.assertEqual(tied["prediction"], 0)
        self.assertEqual(len(result["window_scores"]), 92)
        for length in (96, 128):
            output = decide([0.99] * length, window_length=4, window_threshold=0.5)
            self.assertEqual(output["prediction"], 1)
            self.assertTrue(np.isfinite(output["posterior"]).all())
        with self.assertRaises(ValueError):
            decide([0.2], window_length=2, window_threshold=0.5)
        extreme = forward_backward([0, 1] * 64, [[1, 0], [0, 1]], [1, 0])
        np.testing.assert_array_equal(extreme, np.zeros(128))

    def test_metrics_confusion_order_auc_ties_and_std(self):
        result = evaluate([1, 1, 0, 0], [0.8, 0.4, 0.4, 0.1], [1, 0, 1, 0])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["auc"], 0.875)
        self.assertEqual(result["acc"], 0.5)
        other = {**result, "acc": 1.0}
        self.assertAlmostEqual(summarize([result, other], 1)["acc"]["std"], np.std([0.5, 1], ddof=1))
        undefined = evaluate([0, 0], [0.1, 0.2], [0, 0])
        self.assertIsNone(undefined["auc"])

    def test_pdm_uses_continuous_disease_posterior_and_reports_slice_numbers(self):
        output = decide([0.8, 0.7, 0.1, 0.2], window_length=2, window_threshold=0.5,
                        transition=[[0.5, 0.5], [0.5, 0.5]], prior=[0.5, 0.5])
        q = np.asarray(output["posterior"])
        np.testing.assert_allclose(q, [0.8, 0.7, 0.1, 0.2])
        h = -(q * np.log2(q) + (1-q) * np.log2(1-q))
        self.assertNotIn("support", output)
        expected = np.dot(q[:2], 1-h[:2]) / (1-h[:2]).sum()
        self.assertAlmostEqual(output["score"], expected)
        self.assertEqual(output["best_window_slices"], [1, 2])
        self.assertEqual(output["refined_labels"], [1, 1, 0, 0])

    def test_zero_confidence_windows_are_excluded_and_json_safe(self):
        with patch("poref.pdm.forward_backward", return_value=np.array([.5, .5, .8, .8])):
            result = decide([.5]*4, window_length=2, window_threshold=.7)
        self.assertIsNone(result["window_scores"][0])
        self.assertEqual(result["valid_windows"], [False, True, True])
        self.assertAlmostEqual(result["score"], .8)
        self.assertEqual(result["prediction"], 1)
        json.dumps(result, allow_nan=False)

    def test_confidence_weighted_average_differs_from_dividing_by_length(self):
        result = decide([.8, .9], window_length=2, window_threshold=.7,
                        transition=[[.5, .5], [.5, .5]], prior=[.5, .5])
        self.assertAlmostEqual(result["score"], .8656309422050593)
        self.assertEqual(result["prediction"], 1)

    def test_manifest_split_and_image_leakage(self):
        Path("work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="work") as folder:
            file = Path(folder) / "splits.json"
            doc = sample_manifest()
            file.write_text(json.dumps(doc), encoding="utf-8")
            _, samples = load_manifest(file, ["QSM"])
            self.assertEqual(len(samples), 6)
            doc["folds"][0]["val"] = ["p0", "p3"]
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_manifest(file, ["QSM"])
            doc = sample_manifest()
            doc["samples"][1]["images"] = doc["samples"][0]["images"]
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different patients"):
                load_manifest(file, ["QSM"])

    def test_resampling_affine_preserves_world_endpoints(self):
        old = np.diag([2.0, 3.0, 4.0, 1.0])
        old[:3, 3] = [10, -5, 7]
        affine = target_affine((10, 20, 30), old, (20, 40, 60))
        np.testing.assert_allclose(old @ [9, 19, 29, 1], affine @ [19, 39, 59, 1])
        np.testing.assert_allclose(old @ [0, 0, 0, 1], affine @ [0, 0, 0, 1])

    def test_fixed_holdout_rotates_validation_but_never_test_patients(self):
        Path("work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="work") as folder:
            file = Path(folder) / "splits.json"
            doc = sample_manifest()
            doc["folds"].append({"train": ["p2", "p3"], "val": ["p0", "p1"], "test": ["p4", "p5"]})
            file.write_text(json.dumps(doc), encoding="utf-8")
            parsed, _ = load_manifest(file, ["QSM"])
            self.assertEqual(parsed["folds"][0]["test"], parsed["folds"][1]["test"])
            doc["folds"][1] = dict(doc["folds"][0])
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation folds must be disjoint"):
                load_manifest(file, ["QSM"])
            doc["folds"][1] = {"train": ["p4", "p5"], "val": ["p0", "p1"], "test": ["p2", "p3"]}
            file.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same external test"):
                load_manifest(file, ["QSM"])

    def test_pd_ms_weighted_accuracy_identity(self):
        for positives, negatives in ((31, 26), (20, 20)):
            labels = [1] * positives + [0] * negatives
            predictions = [1] * (positives - 3) + [0] * 3 + [1] * 2 + [0] * (negatives - 2)
            metrics = evaluate(labels, predictions, predictions)
            expected = (positives * metrics["sen"] + negatives * metrics["spe"]) / (positives + negatives)
            self.assertAlmostEqual(metrics["acc"], expected)


if __name__ == "__main__":
    unittest.main()
