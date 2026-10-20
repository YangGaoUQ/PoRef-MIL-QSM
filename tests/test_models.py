import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)

    def test_real_feature_extraction_shape(self):
        from feature_ex.feature import FeatureExtraction
        from feature_ex.upper import AttentionFeedForwardBlock
        self.assertIsInstance(AttentionFeedForwardBlock().norm_attn, torch.nn.LayerNorm)
        model = FeatureExtraction().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 256, 256))
        self.assertTrue(torch.isfinite(output).all())

    def test_chunked_axial_matches_unchunked_values_and_gradients(self):
        from relabel.selfAttention import ZAxisSelfAttention
        small = ZAxisSelfAttention(attn_dim=8, num_heads=2, chunk_size=2)
        large = ZAxisSelfAttention(attn_dim=8, num_heads=2, chunk_size=100)
        large.load_state_dict(small.state_dict())
        x = torch.randn(1, 4, 2, 3, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a, b = small(x), large(y)
        torch.testing.assert_close(a, b)
        a.sum().backward()
        b.sum().backward()
        torch.testing.assert_close(x.grad, y.grad)

    def test_raw_fe2_freeze_film_and_all_fusion_modes(self):
        from poref.model import PoRefNet
        from feature_ex.film import AgeFiLM

        class TinyFE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(1)
                self.project = torch.nn.Linear(1, 256)

            def forward(self, x):
                if x.ndim != 4 or x.shape[1] != 1:
                    raise ValueError("FE must receive raw images")
                return self.project(self.bn(x).mean((2, 3))).unsqueeze(1).expand(-1, 2, -1)

        with patch("poref.model.FeatureExtraction", TinyFE):
            for fusion in ("cross-attention", "concatenation", "mlp-level", "decision-level"):
                for age_position in ("none", "pre", "post"):
                    with self.subTest(fusion=fusion, age=age_position):
                        model = PoRefNet(2, fusion, age_position, axial_chunk_size=64, slice_batch_size=2)
                        model.train()
                        inputs = torch.randn(3, 2, 1, 8, 8)
                        age = torch.tensor(61.0) if age_position != "none" else None
                        out = model(inputs, age)
                        self.assertEqual(tuple(out.shape), (3,))
                        counter = model.fe1.branches[0].bn.num_batches_tracked.clone()
                        out.sum().backward()
                        torch.testing.assert_close(model.fe1.branches[0].bn.num_batches_tracked, counter)
                        model.start_round(2)
                        model.train()
                        self.assertFalse(model.fe1.training)
                        self.assertTrue(all(not p.requires_grad for p in model.fe1.parameters()))
                        before = model.fe1.branches[0].bn.running_mean.clone()
                        model(inputs, age).sum().backward()
                        torch.testing.assert_close(model.fe1.branches[0].bn.running_mean, before)
                        self.assertEqual(sum(isinstance(m, AgeFiLM) for m in model.modules()), int(age_position != "none"))

    def test_pseudo_targets_are_train_only_and_use_previous_predictions(self):
        from poref.experiment import make_targets
        samples = {"train": {"label": 0}, "test": {"label": 1}}
        first = make_targets({"train"}, samples, {}, 1, "cpu", 3, "train")
        second = make_targets({"train"}, samples, {"train": [1, 0, 1]}, 2, "cpu", 3, "train")
        self.assertEqual(first.tolist(), [0, 0, 0])
        self.assertEqual(second.tolist(), [1, 0, 1])
        with self.assertRaises(ValueError):
            make_targets({"train"}, samples, {}, 2, "cpu", 3, "test")

    def test_two_round_training_and_postprocess_with_synthetic_patients(self):
        from poref.cli import build_parser, resolve_config
        from poref.experiment import main
        from poref.postprocess import main as postprocess
        from test_design import sample_manifest

        class TinyPatientModel(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))
            def start_round(self, number):
                self.round = number
            def forward(self, images, age=None):
                return images * self.weight

        def synthetic_input(row, modalities, expected_slices=None):
            sign = 1 if int(row["patient_id"][1:]) % 2 else -1
            return torch.tensor([sign * 0.5, sign, sign * 0.7, sign * 0.9])

        Path("work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="work") as folder:
            root = Path(folder)
            manifest = sample_manifest()
            for sample in manifest["samples"]:
                (root / sample["images"]["QSM"]).touch()  # Empty sentinels, not MRI.
            split_file = root / "splits.json"
            split_file.write_text(json.dumps(manifest), encoding="utf-8")
            parser = build_parser()
            args = parser.parse_args(["train", "--mode", "poref", "--split-file", str(split_file), "--output", str(root / "run"), "--epochs", "1", "--rounds", "2", "--window-length", "2", "--window-threshold", "0.5", "--device", "cpu"])
            config = resolve_config(parser, args)
            with patch("poref.experiment.PoRefNet", TinyPatientModel), patch("poref.experiment.load_patient", synthetic_input):
                main(config)
            cached = json.loads((root / "run/fold_1/pseudo_labels_for_round_2.json").read_text())
            self.assertEqual(set(cached["labels"]), {"p0", "p1"})
            self.assertTrue((root / "run/summary.json").is_file())
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertEqual(summary["independent_test_cohorts"], 1)
            self.assertFalse(summary["ensemble_predictions"])
            self.assertTrue((root / "run/fold_1/best_round_2.pth").is_file())
            postprocess({**config, "input_dir": str(root / "run"), "save_dir": str(root / "pdm"), "n_folds": 1})
            self.assertTrue((root / "pdm/fold_1_pdm_results.csv").is_file())
            (root / "run/fold_1_test_results.csv").rename(root / "run/fold_1_test_slice_results.csv")
            postprocess({**config, "input_dir": str(root / "run"), "save_dir": str(root / "pdm_pp"), "n_folds": 1})
            from poref.inference import main as predict
            unlabelled = {"samples": [{key: value for key, value in row.items() if key != "label"} for row in manifest["samples"][-2:]]}
            patients_file = root / "unlabelled.json"
            patients_file.write_text(json.dumps(unlabelled), encoding="utf-8")
            with patch("poref.inference.PoRefNet", TinyPatientModel), patch("poref.inference.load_patient", synthetic_input):
                predict({"device": "cpu", "model_path": str(root / "run/fold_1/best_round_2.pth"), "patients_file": str(patients_file), "save_dir": str(root / "predictions")})
            self.assertTrue((root / "predictions/predictions.json").is_file())


if __name__ == "__main__":
    unittest.main()
