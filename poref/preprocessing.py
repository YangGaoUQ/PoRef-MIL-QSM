"""Grid resampling for already registered NIfTI images; not registration."""
from pathlib import Path
import numpy as np


def target_affine(old_shape, old_affine, new_shape):
    old_shape, new_shape = np.asarray(old_shape), np.asarray(new_shape)
    if old_shape.shape != (3,) or new_shape.shape != (3,) or (old_shape < 2).any() or (new_shape < 2).any():
        raise ValueError("Resampling requires three dimensions, each at least 2")
    scale = np.eye(4)
    scale[:3, :3] = np.diag((old_shape - 1) / (new_shape - 1))
    return np.asarray(old_affine) @ scale


def main(config):
    import nibabel as nib
    from nibabel.processing import resample_from_to

    source, output = Path(config["input_dir"]).resolve(), Path(config["save_dir"]).resolve()
    if output == source or source in output.parents or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a new/empty output directory outside the input tree")
    paths = sorted(p for p in source.rglob("*") if p.name.endswith((".nii", ".nii.gz")))
    if not paths:
        raise ValueError("No NIfTI inputs found")
    reference = nib.load(config["reference"]) if config.get("reference") else None
    shape = tuple(config["shape"])
    for path in paths:
        image = nib.load(path)
        if len(image.shape) != 3:
            raise ValueError(f"Not a 3D image: {path}")
        grid = image if reference is None else reference
        affine = target_affine(grid.shape, grid.affine, shape)
        result = resample_from_to(image, (shape, affine), order=1)
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The resampler creates the new affine; never reuse the unscaled affine.
        nib.save(result, destination)
