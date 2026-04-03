"""Cellpose 3D nuclear detection wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrynite_py.config.schema import ImagingConfig


@dataclass
class CellposeResult:
    """Result of Cellpose detection for one timepoint."""

    labels: np.ndarray  # (Z, Y, X) instance segmentation labels
    centroids: np.ndarray  # (N, 3) centroid positions [x, y, z]
    diameters: np.ndarray  # (N,) effective diameters
    flows: list | None = None  # Optional flow fields


def detect_nuclei_cellpose(
    volume: np.ndarray,
    imaging_config: ImagingConfig,
    model_type: str = "nuclei",
    diameter: float | None = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    do_3D: bool = True,
    anisotropy: float | None = None,
    normalize: bool = True,
) -> CellposeResult:
    """Detect nuclei in a 3D volume using Cellpose.

    Args:
        volume: 3D numpy array (Z, Y, X).
        imaging_config: Imaging parameters.
        model_type: Cellpose model ("nuclei", "cyto", "cyto3", etc.).
        diameter: Expected nucleus diameter in pixels (None=auto).
        cellprob_threshold: Cell probability threshold (-6 to 6).
        flow_threshold: Flow error threshold for mask quality.
        do_3D: Use 3D segmentation (vs per-slice 2D).
        anisotropy: Z/XY anisotropy (None=from imaging config).
        normalize: Percentile-normalize input.

    Returns:
        CellposeResult with labels, centroids, and diameters.
    """
    from cellpose import models

    model = models.CellposeModel(gpu=True)

    if anisotropy is None:
        anisotropy = imaging_config.anisotropy

    # Run prediction (Cellpose v4.x API)
    labels = model.eval(
        volume,
        diameter=diameter,
        do_3D=do_3D,
        z_axis=0,
        anisotropy=anisotropy,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        normalize=normalize,
    )[0]

    # Extract centroids and diameters from labels
    centroids, diameters = _extract_centroids_diameters(labels)

    return CellposeResult(
        labels=labels,
        centroids=centroids,
        diameters=diameters,
        flows=flows,
    )


def _extract_centroids_diameters(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract centroid positions and equivalent diameters from label mask."""
    from scipy import ndimage

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    centroids = np.zeros((len(unique_labels), 3), dtype=np.float64)
    diameters = np.zeros(len(unique_labels), dtype=np.float64)

    for i, label_id in enumerate(unique_labels):
        mask = labels == label_id
        coords = np.argwhere(mask)  # (N, 3) in [z, y, x]
        centroid_zyx = coords.mean(axis=0)
        centroids[i] = [centroid_zyx[2], centroid_zyx[1], centroid_zyx[0]]  # x, y, z

        volume = coords.shape[0]
        diameters[i] = 2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    return centroids, diameters
