"""Patient-split PoRef-MIL training, train-only iterative labels, paired PDM."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from poref.data import load_manifest, load_patient
from poref.metrics import evaluate, summarize
from poref.model import PoRefNet
from poref.pdm import decide


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def focal_loss(logits, targets, alpha=0.5, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = logits.sigmoid()
    pt = torch.where(targets == 1, p, 1 - p)
    weight = torch.where(targets == 1, alpha, 1 - alpha)
    return (weight * (1 - pt).pow(gamma) * bce).mean()


def make_targets(train_ids, samples, pseudo_labels, round_number, device, length, patient_id):
    if patient_id not in train_ids:
        raise ValueError("Pseudo supervision is restricted to training patients")
    if round_number == 1:
        return torch.full((length,), float(samples[patient_id]["label"]), device=device)
    labels = pseudo_labels[patient_id]
    if len(labels) != length:
        raise ValueError("Pseudo-label sequence length/order changed")
    return torch.tensor(labels, dtype=torch.float32, device=device)


def patient_input(row, config, device):
    images = load_patient(row, config["modalities"], config.get("expected_slices")).to(device)
    age = torch.tensor(float(row["age"]), device=device) if config["age_position"] != "none" else None
    return images, age


def update_pseudo_labels(model, train_ids, samples, config, device):
    """The caller supplies TRAIN ids only; no labels are consulted here."""
    model.eval()
    result = {}
    with torch.no_grad():
        for pid in train_ids:
            images, age = patient_input(samples[pid], config, device)
            probabilities = model(images, age).sigmoid()
            result[pid] = (probabilities >= config["relabel_threshold"]).long().cpu().tolist()
    return result


def evaluate_patients(model, ids, samples, config, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for pid in ids:
            row = samples[pid]
            images, age = patient_input(row, config, device)
            probs = model(images, age).sigmoid().cpu().numpy()
            mean_score = float(probs.mean())
            result = {"patient_name": pid, "true_label": row["label"], "slice_probs": probs.tolist(),
                      "z_indices": list(range(len(probs))), "mean_score": mean_score, "mean_prediction": int(mean_score >= 0.5)}
            if config["aggregation"] == "pdm":
                pdm = decide(probs, window_length=config["window_length"], window_threshold=config["window_threshold"],
                             transition=[[config["stay_hc"], 1 - config["stay_hc"]], [1 - config["stay_disease"], config["stay_disease"]]],
                             prior=[config["prior_hc"], 1 - config["prior_hc"]])
                result.update({"pdm": pdm, "score": pdm["score"], "prediction": pdm["prediction"]})
            else:
                result.update(score=mean_score, prediction=result["mean_prediction"])
            rows.append(result)
    metrics = evaluate([r["true_label"] for r in rows], [r["score"] for r in rows], [r["prediction"] for r in rows])
    mean_metrics = evaluate([r["true_label"] for r in rows], [r["mean_score"] for r in rows], [r["mean_prediction"] for r in rows])
    return rows, metrics, mean_metrics


def save_predictions(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient_name", "true_label", "score", "prediction", "mean_score", "mean_prediction", "slice_probs", "z_indices"))
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in writer.fieldnames}
            for key in ("slice_probs", "z_indices"):
                record[key] = json.dumps(record[key])
            writer.writerow(record)


def main(config):
    doc, samples = load_manifest(config["split_file"], config["modalities"], require_age=config["age_position"] != "none")
    if config.get("n_splits") is not None and config["n_splits"] != len(doc["folds"]):
        raise ValueError("--folds does not match the explicit manifest fold count")
    config = {**config, "n_splits": len(doc["folds"])}
    if config["seed"] + len(doc["folds"]) - 1 >= 2**32:
        raise ValueError("Fold seeds exceed the NumPy seed range")
    for fold in doc["folds"]:
        counts = [sum(samples[p]["label"] == label for p in fold["test"]) for label in (0, 1)]
        if config.get("expected_test_counts") and counts != config["expected_test_counts"]:
            raise ValueError(f"Test counts HC/disease {counts} differ from the configured expectation")
    for row in samples.values():
        for name in config["modalities"]:
            if not Path(row["images"][name]).is_file():
                raise FileNotFoundError(row["images"][name])
    output = Path(config["save_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new/empty output directory for every independent experiment")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(device_name)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {**config, "resolved_device": device_name, "torch_version": torch.__version__, "score_formula": "posterior * (1 - binary_entropy_bits)" if config["aggregation"] == "pdm" else "mean(slice_probabilities)", "status": "started"})
    write_json(output / "resolved_splits.json", doc)
    metrics_all, mean_all = [], []
    for fold_index, fold in enumerate(doc["folds"], 1):
        fold_seed = config["seed"] + fold_index - 1
        seed_everything(fold_seed)
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir()
        write_json(fold_dir / "split.json", {**fold, "seed": fold_seed, "counts": {split: {"HC": sum(samples[p]["label"] == 0 for p in fold[split]), "disease": sum(samples[p]["label"] == 1 for p in fold[split])} for split in ("train", "val", "test")}})
        model = PoRefNet(len(config["modalities"]), config["fusion"], config["age_position"], config["axial_chunk_size"], config["slice_batch_size"]).to(device)
        pseudo = {}
        log = []
        for number in range(1, config["rounds"] + 1):
            model.start_round(number)
            model.to(device)
            lr = config["round_lrs"][number - 1]
            optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-7)
            best_auc, stale = -1.0, 0
            best_path = fold_dir / f"best_round_{number}.pth"
            for epoch in range(1, config["num_epochs"] + 1):
                model.train()
                train_ids = list(fold["train"])
                random.shuffle(train_ids)
                total_loss = 0.0
                # Accumulate one ordered patient sequence at a time.
                for start in range(0, len(train_ids), config["batch_size"]):
                    batch = train_ids[start:start + config["batch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    for pid in batch:
                        images, age = patient_input(samples[pid], config, device)
                        logits = model(images, age)
                        if config["supervision"] == "traditional-mil":
                            patient_logit = torch.logit(logits.sigmoid().mean().clamp(1e-6, 1 - 1e-6))
                            loss = focal_loss(patient_logit, torch.tensor(float(samples[pid]["label"]), device=device), config["alpha"], config["gamma"])
                        else:
                            targets = make_targets(set(fold["train"]), samples, pseudo, number, device, len(logits), pid)
                            loss = focal_loss(logits, targets, config["alpha"], config["gamma"])
                        (loss / len(batch)).backward()
                        total_loss += float(loss.detach())
                    optimizer.step()
                _, validation, _ = evaluate_patients(model, fold["val"], samples, config, device)
                auc = validation["auc"]
                scheduler.step(auc)
                improved = auc > best_auc + 1e-4
                if improved:
                    best_auc, stale = auc, 0
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "round": number, "epoch": epoch, "seed": fold_seed, "config": config}, best_path)
                else:
                    stale += 1
                log.append({"round": number, "epoch": epoch, "train_loss": total_loss / len(train_ids), "validation": validation, "lr": optimizer.param_groups[0]["lr"]})
                write_json(fold_dir / "training_log.json", log)
                print(f"Fold {fold_index} round {number} epoch {epoch}: loss={total_loss/len(train_ids):.6f} val_auc={auc:.6f}", flush=True)
                if stale >= config["patience"]:
                    break
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            if number < config["rounds"]:
                pseudo = update_pseudo_labels(model, fold["train"], samples, config, device)
                write_json(fold_dir / f"pseudo_labels_for_round_{number + 1}.json", {"source_round": number, "threshold": config["relabel_threshold"], "train_only": True, "labels": pseudo})
        # Test labels are used here only, for final metrics, never for pseudo updates.
        val_rows, val_metrics, _ = evaluate_patients(model, fold["val"], samples, config, device)
        rows, metrics, mean_metrics = evaluate_patients(model, fold["test"], samples, config, device)
        save_predictions(output / f"fold_{fold_index}_test_results.csv", rows)
        save_predictions(output / f"fold_{fold_index}_val_results.csv", val_rows)
        write_json(fold_dir / "patient_predictions.json", rows)
        write_json(fold_dir / "metrics.json", {"validation": val_metrics, "test": metrics, "same_model_without_pdm": mean_metrics})
        metrics_all.append(metrics)
        mean_all.append(mean_metrics)
    write_json(output / "summary.json", {"test_design": doc["test_design"],
               "model_evaluations": len(metrics_all),
               "independent_test_cohorts": 1 if doc["test_design"] == "fixed_holdout" else len(metrics_all),
               "ensemble_predictions": False,
               "test": summarize(metrics_all, config["std_ddof"]),
               "same_model_mean_aggregation": summarize(mean_all, config["std_ddof"]), "status": "complete"})
