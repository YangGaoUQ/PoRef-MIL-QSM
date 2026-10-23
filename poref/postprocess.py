"""PDM over existing ordered slice CSVs, independent of any model family."""
import ast
import csv
import json
from pathlib import Path

from poref.metrics import evaluate, summarize
from poref.pdm import decide, WINDOW_SCORE_FORMULA


def main(config):
    root = Path(config["input_dir"])
    output = Path(config["save_dir"])
    if output.resolve() == root.resolve() or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose a new/empty output directory distinct from input")
    input_files = []
    for fold in range(1, config.get("n_folds", 5) + 1):
        candidates = [p for p in (root / f"fold_{fold}_test_results.csv", root / f"fold_{fold}_test_slice_results.csv") if p.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Fold {fold}: require exactly one results or slice_results CSV; found {len(candidates)}")
        input_files.append(candidates[0])
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold, source in enumerate(input_files, 1):
        records = []
        seen = set()
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                pid = row["patient_name"]
                if pid in seen:
                    raise ValueError("Duplicate patient rows")
                seen.add(pid)
                # Old CSVs contain Python list syntax; literal_eval never executes code.
                probabilities = ast.literal_eval(row["slice_probs"])
                if row.get("z_indices"):
                    if json.loads(row["z_indices"]) != list(range(len(probabilities))):
                        raise ValueError("Slice order must be the original contiguous Z order")
                result = decide(probabilities, window_length=config["window_length"], window_threshold=config["window_threshold"],
                                transition=[[config["stay_hc"], 1 - config["stay_hc"]], [1 - config["stay_disease"], config["stay_disease"]]],
                                prior=[config["prior_hc"], 1 - config["prior_hc"]])
                records.append({"patient_name": pid, "true_label": int(row["true_label"]), **result})
        metrics = evaluate([r["true_label"] for r in records], [r["score"] for r in records], [r["prediction"] for r in records])
        summaries.append(metrics)
        (output / f"fold_{fold}_pdm.json").write_text(json.dumps({"patients": records, "metrics": metrics}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        with (output / f"fold_{fold}_pdm_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("patient_name", "true_label", "score", "prediction"))
            writer.writeheader()
            writer.writerows({key: r[key] for key in writer.fieldnames} for r in records)
    (output / "summary.json").write_text(json.dumps({"config": config, "score_formula": WINDOW_SCORE_FORMULA, "metrics": summarize(summaries, config.get("std_ddof", 1))}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
