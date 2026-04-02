"""Fine-tune StarDist 3D on ground truth data.

NOTE: Fine-tuning with simple spherical masks from GT centroids/diameters
gives poor results because the masks don't match real nuclear morphology.
Better approach: use the pretrained model's predictions as initial masks,
then refine using GT centroids to correct the labels before training.
This module implements both approaches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def create_training_masks(
    volumes: list[np.ndarray],
    ground_truth_nuclei: list[list[dict]],
) -> list[np.ndarray]:
    """Create instance segmentation masks from ground truth nuclei positions.

    Each nucleus is represented as a sphere in the mask with its GT diameter.

    Args:
        volumes: List of 3D volumes (Z, Y, X) for shape reference.
        ground_truth_nuclei: List of lists of dicts with keys: x, y, z, diameter.

    Returns:
        List of 3D label arrays (same shape as volumes).
    """
    masks = []
    for vol, nuclei_list in zip(volumes, ground_truth_nuclei):
        mask = np.zeros(vol.shape, dtype=np.uint16)
        for label_id, nuc in enumerate(nuclei_list, start=1):
            x, y, z = nuc["x"], nuc["y"], nuc["z"]
            radius = nuc["diameter"] / 2.0

            # Create spherical mask for this nucleus
            zz, yy, xx = np.ogrid[
                max(0, int(z - radius)):min(vol.shape[0], int(z + radius + 1)),
                max(0, int(y - radius)):min(vol.shape[1], int(y + radius + 1)),
                max(0, int(x - radius)):min(vol.shape[2], int(x + radius + 1)),
            ]
            dist_sq = (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
            sphere = dist_sq <= radius ** 2
            mask[
                max(0, int(z - radius)):min(vol.shape[0], int(z + radius + 1)),
                max(0, int(y - radius)):min(vol.shape[1], int(y + radius + 1)),
                max(0, int(x - radius)):min(vol.shape[2], int(x + radius + 1)),
            ][sphere] = label_id

        masks.append(mask)
    return masks


def prepare_training_data(
    image_dir: str | Path,
    nuclei_dir: str | Path,
    embryo_name: str,
    timepoints: list[int] | None = None,
    max_timepoints: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Prepare training data from images and ground truth.

    Args:
        image_dir: Directory with TIFF stacks.
        nuclei_dir: Directory with GT nuclei files.
        embryo_name: Embryo name prefix.
        timepoints: Specific timepoints to use (None = auto-select).
        max_timepoints: Max number of timepoints to use for training.

    Returns:
        Tuple of (images, masks) lists.
    """
    from starrynite_py.io.tiff_loader import discover_timepoints, load_timepoint
    from starrynite_py.io.ground_truth import load_ground_truth

    logger.info("Loading ground truth...")
    gt = load_ground_truth(nuclei_dir)

    logger.info("Discovering image files...")
    all_tp = discover_timepoints(image_dir, embryo_name)
    tp_map = {t: p for t, p in all_tp}

    # Select timepoints that have both images and GT
    available = sorted(set(tp_map.keys()) & set(gt.keys()))
    if timepoints is not None:
        available = [t for t in available if t in timepoints]

    # Subsample evenly across developmental stages
    if len(available) > max_timepoints:
        indices = np.linspace(0, len(available) - 1, max_timepoints, dtype=int)
        available = [available[i] for i in indices]

    logger.info(f"Using {len(available)} timepoints for training")

    images = []
    masks = []
    for t in available:
        vol = load_timepoint(tp_map[t])
        gt_nuclei = [
            {"x": n.x, "y": n.y, "z": n.z, "diameter": n.diameter}
            for n in gt[t].valid_nuclei
        ]

        if len(gt_nuclei) == 0:
            continue

        mask_list = create_training_masks([vol], [gt_nuclei])
        images.append(vol)
        masks.append(mask_list[0])

    return images, masks


def fine_tune_stardist(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    model_name: str = "3D_demo",
    output_dir: str | Path = "models/finetuned",
    epochs: int = 50,
    train_fraction: float = 0.8,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Fine-tune a StarDist 3D model on custom data.

    Args:
        images: List of 3D image volumes.
        masks: List of 3D label masks.
        model_name: Base pretrained model name.
        output_dir: Directory to save fine-tuned model.
        epochs: Number of training epochs.
        train_fraction: Fraction of data for training (rest for validation).
        anisotropy: (Z, Y, X) anisotropy for the data.

    Returns:
        Path to the fine-tuned model directory.
    """
    from stardist import fill_label_holes
    from stardist.models import StarDist3D, Config3D
    from csbdeep.utils import normalize

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize images
    logger.info("Normalizing images...")
    images_norm = [normalize(img, 1, 99.8) for img in images]
    masks_filled = [fill_label_holes(m) for m in masks]

    # Split train/val
    n_train = max(1, int(len(images_norm) * train_fraction))
    X_train, Y_train = images_norm[:n_train], masks_filled[:n_train]
    X_val, Y_val = images_norm[n_train:], masks_filled[n_train:]

    if len(X_val) == 0:
        X_val, Y_val = X_train[-1:], Y_train[-1:]

    logger.info(f"Training: {len(X_train)} images, Validation: {len(X_val)} images")

    # Configure model
    n_rays = 96  # Same as pretrained model
    grid = (2, 2, 2)

    conf = Config3D(
        n_rays=n_rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=False,  # Will auto-detect
        n_channel_in=1,
        train_patch_size=(48, 96, 96),
        train_batch_size=2,
    )

    model = StarDist3D(conf, name="finetuned", basedir=str(output_dir))

    # Load pretrained weights
    pretrained = StarDist3D.from_pretrained(model_name)
    model.keras_model.set_weights(pretrained.keras_model.get_weights())
    logger.info("Loaded pretrained weights")

    # Fine-tune
    logger.info(f"Fine-tuning for {epochs} epochs...")
    model.train(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
    )

    # Optimize thresholds
    logger.info("Optimizing thresholds...")
    model.optimize_thresholds(X_val, Y_val)

    logger.info(f"Model saved to {output_dir / 'finetuned'}")
    return str(output_dir / "finetuned")
