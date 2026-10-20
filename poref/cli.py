"""Parse commands without importing PyTorch or loading patient data."""

import argparse
import importlib
import json
from pathlib import Path

TRAIN_MODULES = {
    "poref": "poref.experiment",
    "single": "classifier.diagnosisTrain",
    "single-slice": "classifier.diagnosisTrainPP",
    "dual": "classifier.dualModalTrain",
    "dual-slice": "classifier.dualModalTrainPP",
    "dual-age": "classifier.dualModalAge",
    "frozen": "relabel.singleTrain",
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def positive_float(value):
    value = float(value)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def nonnegative_float(value):
    value = float(value)
    if not 0 <= value < float("inf"):
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return value


def unit_float(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be finite and in [0,1]")
    return value


def add_pdm_arguments(parser, required=False):
    parser.add_argument("--window-length", type=positive_int, required=required)
    parser.add_argument("--window-threshold", type=unit_float, required=required)
    parser.add_argument("--stay-hc", type=unit_float, default=0.9)
    parser.add_argument("--stay-disease", type=unit_float, default=0.9)
    parser.add_argument("--prior-hc", type=unit_float, default=0.5)
    parser.add_argument("--std-ddof", type=int, choices=(0, 1), default=1)


def build_parser():
    parser = argparse.ArgumentParser(prog="poref", description="PoRef-MIL training and postprocessing")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train an existing experiment")
    train.add_argument("--mode", choices=TRAIN_MODULES, required=True)
    for name in ("train-root", "test-root", "qsm-train-root", "t1-train-root", "qsm-test-root", "t1-test-root", "age-csv"):
        train.add_argument("--" + name, "--" + name.replace("-", "_"), type=Path)
    train.add_argument("--output", "--save_dir", dest="save_dir", required=True, type=Path)
    train.add_argument("--batch-size", "--batch_size", type=positive_int, default=8)
    train.add_argument("--epochs", "--num_epochs", dest="num_epochs", type=positive_int, default=300)
    train.add_argument("--patience", type=positive_int, default=50)
    train.add_argument("--folds", "--num_folds", dest="n_splits", type=positive_int)
    train.add_argument("--lr", "--learning_rate", type=positive_float, default=1e-4)
    train.add_argument("--weight-decay", "--weight_decay", type=nonnegative_float, default=1e-5)
    train.add_argument("--stage2-epochs", dest="num_epochs_stage2", type=positive_int, default=300)
    train.add_argument("--stage2-lr", dest="lr_stage2", type=positive_float, default=5e-5)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--dry-run", action="store_true", help="Print resolved parameters without loading data or models")
    train.add_argument("--split-file", "--data_split_file", type=Path)
    train.add_argument("--modalities", nargs="+", default=["QSM"])
    train.add_argument("--rounds", type=int, choices=range(1, 6), default=5)
    train.add_argument("--round-lrs", nargs="+", type=positive_float)
    train.add_argument("--fusion", choices=("cross-attention", "concatenation", "mlp-level", "decision-level"), default="cross-attention")
    train.add_argument("--age-position", choices=("none", "pre", "post"), default="none")
    train.add_argument("--supervision", choices=("iterative", "traditional-mil"), default="iterative")
    train.add_argument("--aggregation", choices=("pdm", "mean"), default="pdm")
    train.add_argument("--seed", type=int, default=2025)
    train.add_argument("--alpha", type=unit_float, default=0.5)
    train.add_argument("--gamma", type=nonnegative_float, default=2.0)
    train.add_argument("--relabel-threshold", type=unit_float, default=0.5)
    train.add_argument("--slice-batch-size", type=positive_int, default=4)
    train.add_argument("--axial-chunk-size", type=positive_int, default=256)
    train.add_argument("--expected-slices", type=positive_int)
    train.add_argument("--disease-name", choices=("PD", "MS"), default="PD")
    train.add_argument("--expected-test-counts", nargs=2, type=positive_int, metavar=("HC", "DISEASE"))
    add_pdm_arguments(train)

    post = commands.add_parser("postprocess", help="Apply single-pass forward-backward PDM to ordered slice probabilities")
    post.add_argument("--input-dir", type=Path, required=True)
    post.add_argument("--output", dest="save_dir", type=Path, required=True)
    post.add_argument("--folds", dest="n_folds", type=positive_int, default=5)
    add_pdm_arguments(post, required=True)
    post.add_argument("--dry-run", action="store_true")

    heat = commands.add_parser("heatmap", help="Export single-modal slice probabilities and volume masks")
    heat.add_argument("--checkpoint", dest="model_path", type=Path, required=True)
    heat.add_argument("--input", dest="nii_path", type=Path, required=True)
    heat.add_argument("--output", dest="save_dir", type=Path, required=True)
    heat.add_argument("--patient-name", default="patient")
    heat.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    heat.add_argument("--dry-run", action="store_true")
    prepare = commands.add_parser("preprocess", help="Resample pre-registered images to a physical grid; never overwrite input")
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--output", dest="save_dir", type=Path, required=True)
    prepare.add_argument("--shape", nargs=3, type=positive_int, required=True)
    prepare.add_argument("--reference", type=Path)
    prepare.add_argument("--dry-run", action="store_true")
    predict = commands.add_parser("predict", help="Predict unlabelled patients using a saved PoRef experiment checkpoint")
    predict.add_argument("--checkpoint", dest="model_path", type=Path, required=True)
    predict.add_argument("--patients-file", type=Path, required=True)
    predict.add_argument("--output", dest="save_dir", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict.add_argument("--dry-run", action="store_true")
    return parser


def resolve_config(parser, args):
    config = vars(args).copy()
    if args.command == "train":
        if args.mode == "dual-age" and args.age_position == "none":
            config["age_position"] = "post"
        if args.n_splits is not None and args.n_splits < 2:
            parser.error("--folds must be at least 2")
        if args.mode != "poref" and args.n_splits is None:
            config["n_splits"] = 5
        if args.seed < 0 or args.seed >= 2**32:
            parser.error("--seed must be in [0,2**32)")
        if args.mode == "poref":
            if args.split_file is None:
                parser.error("--mode poref requires --split-file with patient train/val/test assignments")
            if len(args.modalities) != len(set(args.modalities)):
                parser.error("--modalities must be unique ordered names")
            if args.supervision == "traditional-mil" and args.rounds != 1:
                parser.error("traditional-mil requires --rounds 1")
            if args.aggregation == "pdm" and (args.window_length is None or args.window_threshold is None):
                parser.error("PDM requires explicit --window-length and --window-threshold")
            if args.round_lrs is not None and len(args.round_lrs) != args.rounds:
                parser.error("--round-lrs must contain exactly --rounds rates")
            config["round_lrs"] = args.round_lrs or [rate * args.lr / 1e-4 for rate in [1e-4, 5e-5, 3e-5, 1e-5, 5e-6][:args.rounds]]
            if any(config[key] is not None for key in ("train_root", "test_root", "qsm_train_root", "t1_train_root", "qsm_test_root", "t1_test_root", "age_csv")):
                parser.error("Use image paths and age in the split manifest for --mode poref")
        dual = args.mode.startswith("dual")
        required = {"qsm_train_root", "t1_train_root", "qsm_test_root", "t1_test_root"} if dual else {"train_root", "test_root"}
        if args.mode == "dual-age":
            required.add("age_csv")
        supplied_paths = {key for key in ("train_root", "test_root", "qsm_train_root", "t1_train_root", "qsm_test_root", "t1_test_root", "age_csv") if config[key] is not None}
        missing = required - supplied_paths
        if missing and args.mode != "poref":
            parser.error("missing arguments for this mode: " + ", ".join("--" + key.replace("_", "-") for key in sorted(missing)))
        if supplied_paths - required and args.mode != "poref":
            parser.error("input paths do not match the selected mode")
        config["num_epochs_stage1"] = args.num_epochs
        config["lr_stage1"] = args.lr
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value.expanduser().resolve())
    if args.command == "heatmap" and (Path(args.patient_name).name != args.patient_name or args.patient_name in ("", ".", "..") or "/" in args.patient_name or "\\" in args.patient_name):
        parser.error("--patient-name must be a filename without a directory")
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(parser, args)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    if args.command == "train":
        module_name = TRAIN_MODULES[args.mode]
    elif args.command == "postprocess":
        module_name = "classifier.postProcess"
    elif args.command == "preprocess":
        module_name = "poref.preprocessing"
    elif args.command == "predict":
        module_name = "poref.inference"
    else:
        module_name = "classifier.heatmap"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        parser.exit(2, f"Missing module {exc.name!r}. Install the project dependencies in this Python environment.\n")
    if args.command == "heatmap":
        module.save_patient_heatmap(**{key: config[key] for key in ("model_path", "nii_path", "save_dir", "patient_name", "device")})
    else:
        module.main(config)
