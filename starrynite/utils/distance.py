"""Vectorized distance computations with anisotropy support."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def anisotropic_distance(
    a: np.ndarray,
    b: np.ndarray,
    anisotropy: float = 1.0,
) -> np.ndarray:
    """Compute anisotropic Euclidean distance between point sets.

    Scales the Z coordinate by the anisotropy factor before computing distance.

    Args:
        a: (N, 3) array of [x, y, z] positions.
        b: (M, 3) array of [x, y, z] positions.
        anisotropy: z_res / xy_res ratio.

    Returns:
        (N, M) distance matrix.
    """
    a_scaled = a.copy().astype(np.float64)
    b_scaled = b.copy().astype(np.float64)
    a_scaled[:, 2] *= anisotropy
    b_scaled[:, 2] *= anisotropy

    # Use broadcasting: (N,1,3) - (1,M,3) -> (N,M,3) -> (N,M)
    diff = a_scaled[:, np.newaxis, :] - b_scaled[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def match_detections(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float,
    anisotropy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Match detected nuclei to ground truth using greedy nearest-neighbor.

    Uses anisotropic distance and a maximum matching threshold.
    Mirrors the logic of compareDetectionWRadius_3.m.

    Args:
        detected: (N, 3) detected positions [x, y, z].
        ground_truth: (M, 3) ground truth positions [x, y, z].
        threshold: Maximum matching distance.
        anisotropy: z_res / xy_res ratio.

    Returns:
        Tuple of:
            - matches_det: (N,) array, GT index for each detection (-1 if unmatched)
            - matches_gt: (M,) array, detection index for each GT (-1 if unmatched)
            - tp: Number of true positives
            - fp: Number of false positives
            - fn: Number of false negatives
    """
    n_det = len(detected)
    n_gt = len(ground_truth)

    if n_det == 0 or n_gt == 0:
        return (
            np.full(n_det, -1, dtype=int),
            np.full(n_gt, -1, dtype=int),
            0,
            n_det,
            n_gt,
        )

    # Scale Z for anisotropy
    det_scaled = detected.copy().astype(np.float64)
    gt_scaled = ground_truth.copy().astype(np.float64)
    det_scaled[:, 2] *= anisotropy
    gt_scaled[:, 2] *= anisotropy

    # Build KDTree on GT points for efficient querying
    tree = KDTree(gt_scaled)

    matches_det = np.full(n_det, -1, dtype=int)
    matches_gt = np.full(n_gt, -1, dtype=int)

    # Query all detections at once
    distances, indices = tree.query(det_scaled, k=1)

    # Sort by distance — greedily assign closest matches first
    order = np.argsort(distances)

    for i in order:
        dist = distances[i]
        gt_idx = indices[i]
        if dist <= threshold and matches_gt[gt_idx] == -1:
            matches_det[i] = gt_idx
            matches_gt[gt_idx] = i

    tp = int(np.sum(matches_det >= 0))
    fp = n_det - tp
    fn = n_gt - tp

    return matches_det, matches_gt, tp, fp, fn
