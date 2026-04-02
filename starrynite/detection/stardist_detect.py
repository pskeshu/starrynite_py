"""StarDist 3D nuclear detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrynite.config.schema import StarDistConfig, ImagingConfig


@dataclass
class DetectionResult:
    """Result of nuclear detection for one timepoint."""

    labels: np.ndarray  # (Z, Y, X) instance segmentation labels
    centroids: np.ndarray  # (N, 3) centroid positions [x, y, z]
    diameters: np.ndarray  # (N,) effective diameters
    ray_distances: np.ndarray | None = None  # (N, n_rays) star-convex distances
    probabilities: np.ndarray | None = None  # (N,) detection probabilities


def detect_nuclei(
    volume: np.ndarray,
    stardist_config: StarDistConfig,
    imaging_config: ImagingConfig,
) -> DetectionResult:
    """Detect nuclei in a 3D volume using StarDist.

    Args:
        volume: 3D numpy array (Z, Y, X), float32.
        stardist_config: StarDist configuration.
        imaging_config: Imaging parameters (for anisotropy).

    Returns:
        DetectionResult with labels, centroids, and diameters.
    """
    from stardist.models import StarDist3D

    model = StarDist3D.from_pretrained(stardist_config.model_name)

    # Build prediction kwargs
    predict_kwargs = {}
    if stardist_config.prob_thresh is not None:
        predict_kwargs["prob_thresh"] = stardist_config.prob_thresh
    if stardist_config.nms_thresh is not None:
        predict_kwargs["nms_thresh"] = stardist_config.nms_thresh
    if stardist_config.n_tiles is not None:
        predict_kwargs["n_tiles"] = tuple(stardist_config.n_tiles)

    # Set anisotropy from imaging config
    anisotropy = imaging_config.anisotropy
    if stardist_config.scale is not None:
        predict_kwargs["scale"] = tuple(stardist_config.scale)
    elif anisotropy != 1.0:
        predict_kwargs["scale"] = (anisotropy, 1.0, 1.0)

    # Normalize
    if stardist_config.normalize:
        from csbdeep.utils import normalize as csbd_normalize
        volume = csbd_normalize(
            volume,
            stardist_config.normalize_low,
            stardist_config.normalize_high,
        )

    # Run prediction
    labels, details = model.predict_instances(volume, **predict_kwargs)

    # Extract centroids (StarDist returns [z, y, x] order in details['points'])
    points_zyx = details["points"]
    # Convert to [x, y, z] for our convention
    centroids = np.column_stack([
        points_zyx[:, 2],  # x
        points_zyx[:, 1],  # y
        points_zyx[:, 0],  # z
    ])

    # Extract diameters from ray distances
    ray_distances = details.get("dist", None)
    if ray_distances is not None:
        # Effective diameter = 2 * mean ray distance
        diameters = 2.0 * np.mean(ray_distances, axis=1)
    else:
        # Fallback: estimate from label volumes
        diameters = _estimate_diameters_from_labels(labels)

    probabilities = details.get("prob", None)

    return DetectionResult(
        labels=labels,
        centroids=centroids,
        diameters=diameters,
        ray_distances=ray_distances,
        probabilities=probabilities,
    )


def _estimate_diameters_from_labels(labels: np.ndarray) -> np.ndarray:
    """Estimate nuclear diameters from segmentation label volumes."""
    from scipy import ndimage

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    diameters = np.zeros(len(unique_labels))
    for i, label_id in enumerate(unique_labels):
        volume = np.sum(labels == label_id)
        # Diameter of sphere with equivalent volume
        diameters[i] = 2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    return diameters
