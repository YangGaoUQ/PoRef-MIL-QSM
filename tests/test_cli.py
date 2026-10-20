import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from poref.cli import TRAIN_MODULES, main


class CommandTests(unittest.TestCase):
    def test_gatefusenet_style_aliases(self):
        entry = Mock()
        with patch("poref.cli.importlib.import_module", return_value=SimpleNamespace(main=entry)):
            main(["train", "--mode", "single", "--train_root", "input/train", "--test_root", "input/test", "--save_dir", "output", "--batch_size", "4", "--learning_rate", "0.0002", "--num_epochs", "25", "--num_folds", "3", "--weight_decay", "0.01"])
        config = entry.call_args.args[0]
        self.assertEqual(config["batch_size"], 4)
        self.assertEqual(config["lr"], 0.0002)
        self.assertEqual(config["num_epochs"], 25)
        self.assertEqual(config["n_splits"], 3)
        self.assertEqual(config["weight_decay"], 0.01)

    def train_args(self, mode):
        args = ["train", "--mode", mode, "--output", "test-output", "--epochs", "2", "--lr", "0.002", "--device", "cpu"]
        if mode == "poref":
            args.extend(["--split-file", "splits.json", "--window-length", "3", "--window-threshold", "0.5"])
        elif mode.startswith("dual"):
            for key in ("qsm-train-root", "t1-train-root", "qsm-test-root", "t1-test-root"):
                args.extend(["--" + key, "example/" + key])
        else:
            args.extend(["--train-root", "example/train", "--test-root", "example/test"])
        if mode == "dual-age":
            args.extend(["--age-csv", "example/age.csv"])
        return args

    def test_training_dispatch_and_parameters(self):
        for mode, module_name in TRAIN_MODULES.items():
            with self.subTest(mode=mode):
                entry = Mock()
                with patch("poref.cli.importlib.import_module", return_value=SimpleNamespace(main=entry)) as load:
                    main(self.train_args(mode))
                load.assert_called_once_with(module_name)
                config = entry.call_args.args[0]
                self.assertEqual(config["num_epochs"], 2)
                self.assertEqual(config["lr"], 0.002)
                self.assertEqual(config["device"], "cpu")
                self.assertEqual(config["num_epochs_stage1"], 2)
                self.assertEqual(config["lr_stage1"], 0.002)

    def test_dry_run_does_not_import_models(self):
        with patch("poref.cli.importlib.import_module") as load, contextlib.redirect_stdout(io.StringIO()) as output:
            main(self.train_args("dual-age") + ["--dry-run"])
        load.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["mode"], "dual-age")

    def test_invalid_training_arguments(self):
        cases = [
            ["train", "--mode", "dual", "--output", "out"],
            self.train_args("single") + ["--folds", "1"],
            self.train_args("single") + ["--lr", "nan"],
            self.train_args("single") + ["--batch-size", "0"],
            self.train_args("single") + ["--age-csv", "age.csv"],
        ]
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)

    def test_postprocess_dispatch(self):
        entry = Mock()
        with patch("poref.cli.importlib.import_module", return_value=SimpleNamespace(main=entry)) as load:
            main(["postprocess", "--input-dir", "in", "--output", "out", "--folds", "3", "--window-length", "4", "--window-threshold", "0.6"])
        load.assert_called_once_with("classifier.postProcess")
        self.assertEqual(entry.call_args.args[0]["n_folds"], 3)
        self.assertEqual(entry.call_args.args[0]["window_length"], 4)

    def test_heatmap_dispatch(self):
        entry = Mock()
        with patch("poref.cli.importlib.import_module", return_value=SimpleNamespace(save_patient_heatmap=entry)):
            main(["heatmap", "--checkpoint", "model.pth", "--input", "sample.nii", "--output", "out", "--device", "cpu"])
        self.assertEqual(entry.call_args.kwargs["device"], "cpu")
        self.assertEqual(entry.call_args.kwargs["patient_name"], "patient")


if __name__ == "__main__":
    unittest.main()
