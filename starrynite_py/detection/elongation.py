"""Detect pre-mitotic nuclear elongation from StarDist ray distances.

Before division, nuclei elongate along the future division axis. StarDist's
96 star-convex ray distances provide a rich shape descriptor that can capture
this elongation. We compute the elongation ratio (max/min principal axis)
from the ray distances to flag nuclei that may be about to divide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ElongationResult:
    """Elongation analysis for detected nuclei."""

    elongation_ratios: np.ndarray  # (N,) ratio of max/min ray extent
    division_axis: np.ndarray  # (N, 3) estimated division axis direction
    is_elongated: np.ndarray  # (N,) boolean flag for pre-mitotic candidates


def compute_elongation_from_rays(
    ray_distances: np.ndarray,
    ray_vertices: np.ndarray | None = None,
    threshold: float = 1.5,
) -> ElongationResult:
    """Compute nuclear elongation from StarDist ray distances.

    For each nucleus, the ray distances describe the boundary in star-convex
    coordinates. PCA on the ray endpoints gives principal axes — the ratio
    of the largest to smallest eigenvalue indicates elongation.

    Args:
        ray_distances: (N, n_rays) ray distances from StarDist.
        ray_vertices: (n_rays, 3) unit vectors for ray directions.
            If None, uses StarDist's default 96-ray tessellation.
        threshold: Elongation ratio above which a nucleus is flagged
            as pre-mitotic (default 1.5 = 50% longer than wide).

    Returns:
        ElongationResult with ratios, axes, and flags.
    """
    n_nuclei, n_rays = ray_distances.shape

    if ray_vertices is None:
        ray_vertices = _get_stardist_ray_vertices(n_rays)

    elongation_ratios = np.zeros(n_nuclei)
    division_axes = np.zeros((n_nuclei, 3))

    for i in range(n_nuclei):
        # Convert ray distances to 3D endpoints
        endpoints = ray_distances[i, :, np.newaxis] * ray_vertices  # (n_rays, 3)

        # PCA on endpoints
        centered = endpoints - endpoints.mean(axis=0)
        cov = np.cov(centered.T)  # (3, 3)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (largest last)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Elongation ratio = sqrt(max_eigenvalue / min_eigenvalue)
        if eigenvalues[0] > 0:
            elongation_ratios[i] = np.sqrt(eigenvalues[2] / eigenvalues[0])
        else:
            elongation_ratios[i] = 1.0

        # Division axis = direction of maximum extent
        division_axes[i] = eigenvectors[:, 2]  # Largest eigenvalue direction

    is_elongated = elongation_ratios > threshold

    return ElongationResult(
        elongation_ratios=elongation_ratios,
        division_axis=division_axes,
        is_elongated=is_elongated,
    )


def _get_stardist_ray_vertices(n_rays: int) -> np.ndarray:
    """Get StarDist's ray direction unit vectors.

    StarDist uses a Fibonacci sphere tessellation for 3D ray directions.
    """
    # Fibonacci sphere sampling (same as StarDist)
    indices = np.arange(0, n_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_rays)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.column_stack([z, y, x])  # ZYX order to match StarDist


def flag_premitotic_nuclei(
    ray_distances: np.ndarray,
    centroids: np.ndarray,
    elongation_threshold: float = 1.5,
    size_threshold: float | None = None,
    mean_diameter: float | None = None,
) -> tuple[np.ndarray, ElongationResult]:
    """Flag nuclei likely to divide based on shape analysis.

    Combines elongation detection with optional size filtering
    (dividing cells tend to be larger than average).

    Args:
        ray_distances: (N, n_rays) from StarDist.
        centroids: (N, 3) positions [x, y, z].
        elongation_threshold: Min elongation ratio to flag.
        size_threshold: Min diameter relative to mean (e.g., 1.2 = 20% above mean).
        mean_diameter: Reference diameter (None = compute from data).

    Returns:
        Tuple of (premitotic_flags, elongation_result).
    """
    elong = compute_elongation_from_rays(ray_distances, threshold=elongation_threshold)

    flags = elong.is_elongated.copy()

    if size_threshold is not None:
        diameters = 2.0 * np.mean(ray_distances, axis=1)
        if mean_diameter is None:
            mean_diameter = np.mean(diameters)
        size_ok = diameters >= size_threshold * mean_diameter
        flags = flags & size_ok

    return flags, elong
