"""Patient manifest validation and ordered, pre-aligned multimodal loading."""
import json
from pathlib import Path

import numpy as np


def load_manifest(path, modalities, *, require_age=False):
    path = Path(path).resolve()
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    if doc.get("test_design") not in ("outer_cv", "fixed_holdout"):
        raise ValueError("manifest.test_design must explicitly be outer_cv or fixed_holdout")
    samples = {}
    owners = {}
    for original in doc["samples"]:
        row = dict(original)
        pid = row["patient_id"]
        if not isinstance(pid, str) or not pid or pid in samples:
            raise ValueError("patient_id must be a unique nonempty string")
        if row["label"] not in (0, 1):
            raise ValueError(f"Invalid binary label: {pid}")
        if require_age and ("age" not in row or not np.isfinite(float(row["age"])) or not 0 <= float(row["age"]) <= 120):
            raise ValueError(f"Missing/invalid age: {pid}")
        row["images"] = dict(row["images"])
        for modality in modalities:
            image = Path(row["images"][modality]).expanduser()
            image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
            row["images"][modality] = str(image)
            identity = str(image).casefold()
            if identity in owners and owners[identity] != pid:
                raise ValueError(f"Same image assigned to different patients: {pid}, {owners[identity]}")
            owners[identity] = pid
        if row.get("slice_axis", 2) not in (0, 2):
            raise ValueError("slice_axis must explicitly be 0 or 2")
        samples[pid] = row
    if not samples or not doc["folds"]:
        raise ValueError("Require samples and nonempty folds")
    tested = set()
    reference_test = None
    validated = set()
    for index, fold in enumerate(doc["folds"], 1):
        sets = []
        for split in ("train", "val", "test"):
            ids = fold[split]
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= samples.keys():
                raise ValueError(f"Fold {index}: invalid {split} patient list")
            group = set(ids)
            if any(group & previous for previous in sets):
                raise ValueError(f"Fold {index}: patient leakage between splits")
            if {samples[pid]["label"] for pid in ids} != {0, 1}:
                raise ValueError(f"Fold {index}: {split} must contain both classes")
            sets.append(group)
        if set.union(*sets) != samples.keys():
            raise ValueError(f"Fold {index}: every patient must have an explicit split assignment")
        if doc["test_design"] == "outer_cv" and tested & sets[2]:
            raise ValueError("outer_cv test patients must be disjoint across folds")
        if doc["test_design"] == "fixed_holdout" and reference_test is not None and sets[2] != reference_test:
            raise ValueError("fixed_holdout must reuse exactly the same external test patients")
        if doc["test_design"] == "fixed_holdout" and len(doc["folds"]) > 1:
            if validated & sets[1]:
                raise ValueError("fixed_holdout validation folds must be disjoint")
            validated |= sets[1]
        reference_test = sets[2]
        tested |= sets[2]
    if doc["test_design"] == "outer_cv" and tested != samples.keys():
        raise ValueError("outer_cv must test every patient exactly once")
    if doc["test_design"] == "fixed_holdout" and len(doc["folds"]) > 1 and validated != samples.keys() - reference_test:
        raise ValueError("fixed_holdout must validate every development patient exactly once")
    return {**doc, "samples": list(samples.values())}, samples


def load_patient(row, modalities, expected_slices=None):
    import nibabel as nib
    import torch

    images = [nib.load(row["images"][name]) for name in modalities]
    reference = images[0]
    for image in images:
        if len(image.shape) != 3 or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=1e-5):
            raise ValueError(f"{row['patient_id']}: modalities require pre-aligned 3D shape AND affine")
    volumes = []
    for image in images:
        array = np.moveaxis(image.get_fdata(dtype=np.float32), row.get("slice_axis", 2), 0)
        if array.shape[1:] != (256, 256) or (expected_slices is not None and len(array) != expected_slices):
            raise ValueError(f"{row['patient_id']}: unexpected ordered [Z,H,W] shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{row['patient_id']}: nonfinite voxel values")
        array = (array - array.mean()) / (array.std() + 1e-8)
        volumes.append(torch.from_numpy(array.copy()).unsqueeze(1))
    # [Z, modalities, 1, H, W]; no sorting or shuffling along Z.
    return torch.stack(volumes, dim=1)
