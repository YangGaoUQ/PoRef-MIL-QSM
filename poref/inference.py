"""Unlabelled patient prediction using the exact saved experiment model."""
import json
from pathlib import Path

import torch

from poref.data import load_patient
from poref.model import PoRefNet
from poref.pdm import decide


def main(options):
    device_name = options["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(options["model_path"], map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("predict requires a new PoRef experiment checkpoint, not legacy slice weights")
    config = checkpoint["config"]
    model = PoRefNet(len(config["modalities"]), config["fusion"], config["age_position"], config["axial_chunk_size"], config["slice_batch_size"])
    model.start_round(checkpoint["round"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    source = Path(options["patients_file"]).resolve()
    patients = json.loads(source.read_text(encoding="utf-8-sig"))["samples"]
    ids = [row["patient_id"] for row in patients]
    if not ids or any(not isinstance(pid, str) or not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Require unique nonempty patient ids")
    output = Path(options["save_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty prediction directory")
    records = []
    with torch.no_grad():
        for original in patients:
            row = {**original, "images": dict(original["images"])}
            for name in config["modalities"]:
                path = Path(row["images"][name]).expanduser()
                row["images"][name] = str((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
            if row.get("slice_axis", 2) not in (0, 2):
                raise ValueError("slice_axis must be 0 or 2")
            images = load_patient(row, config["modalities"], config.get("expected_slices")).to(device)
            age = torch.tensor(float(row["age"]), device=device) if config["age_position"] != "none" else None
            if age is not None and (not torch.isfinite(age) or not 0 <= age.item() <= 120):
                raise ValueError("A finite age in [0,120] is required")
            probabilities = model(images, age).sigmoid().cpu().tolist()
            if config["aggregation"] == "pdm":
                result = decide(probabilities, window_length=config["window_length"], window_threshold=config["window_threshold"],
                                transition=[[config["stay_hc"], 1 - config["stay_hc"]], [1 - config["stay_disease"], config["stay_disease"]]],
                                prior=[config["prior_hc"], 1 - config["prior_hc"]])
            else:
                score = sum(probabilities) / len(probabilities)
                result = {"score": score, "prediction": int(score >= 0.5)}
            records.append({"patient_name": row["patient_id"], "slice_probs": probabilities, **result})
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps({"checkpoint": str(Path(options["model_path"]).resolve()), "model_config": config, "patients": records}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
