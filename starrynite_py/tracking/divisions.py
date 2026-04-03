"""Division detection from frame-to-frame detection data.

Detects cell divisions by identifying cases where one nucleus at time t
corresponds to two daughter nuclei at time t+1. Uses spatial proximity
and size constraints inspired by StarryNite's calculateCellTripleVector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


@dataclass
class DivisionCandidate:
    """A candidate cell division event."""

    parent_idx: int  # Index in parent timepoint
    daughter1_idx: int  # Index in daughter timepoint
    daughter2_idx: int  # Index in daughter timepoint
    parent_position: np.ndarray  # (3,) xyz
    daughter1_position: np.ndarray  # (3,) xyz
    daughter2_position: np.ndarray  # (3,) xyz
    score: float  # Lower is better (distance-based)
    midpoint_distance: float  # Distance from parent to daughters' midpoint
    separation: float  # Distance between daughters


def detect_divisions(
    positions_t: np.ndarray,
    positions_t1: np.ndarray,
    diameters_t: np.ndarray,
    diameters_t1: np.ndarray,
    max_distance: float = 30.0,
    min_separation: float = 5.0,
    max_separation: float | None = None,
    anisotropy: float = 1.0,
) -> list[DivisionCandidate]:
    """Detect division events between consecutive timepoints.

    A division is detected when:
    1. A parent nucleus at t has two nearby nuclei at t+1
    2. The daughters' midpoint is close to the parent position
    3. The daughters are separated by a reasonable distance
    4. No closer single-nucleus match exists (parent is "unmatched" in 1-1)

    Args:
        positions_t: (N, 3) parent positions [x, y, z].
        positions_t1: (M, 3) daughter positions [x, y, z].
        diameters_t: (N,) parent diameters.
        diameters_t1: (M,) daughter diameters.
        max_distance: Max distance from parent to daughters' midpoint.
        min_separation: Min distance between daughters.
        max_separation: Max distance between daughters (None = 3x mean diameter).
        anisotropy: Z/XY anisotropy ratio.

    Returns:
        List of DivisionCandidate sorted by score (best first).
    """
    if len(positions_t) == 0 or len(positions_t1) < 2:
        return []

    if max_separation is None:
        max_separation = 3.0 * np.mean(diameters_t)

    # Scale Z for anisotropy
    pos_t = positions_t.copy()
    pos_t1 = positions_t1.copy()
    pos_t[:, 2] *= anisotropy
    pos_t1[:, 2] *= anisotropy

    # Build KDTree on t+1 positions
    tree = KDTree(pos_t1)

    # For each parent, find K nearest daughters
    k = min(10, len(pos_t1))
    distances, indices = tree.query(pos_t, k=k)

    # First do 1-1 matching to find unmatched parents
    d1, i1 = tree.query(pos_t, k=1)
    tree_parent = KDTree(pos_t)
    d_back, i_back = tree_parent.query(pos_t1, k=1)

    # Mutual nearest neighbors are matched (not dividing)
    matched_parents = set()
    matched_daughters = set()
    for p_idx in range(len(pos_t)):
        nn_daughter = i1[p_idx]
        if i_back[nn_daughter] == p_idx and d1[p_idx] < max_distance:
            matched_parents.add(p_idx)
            matched_daughters.add(nn_daughter)

    # Look for divisions among unmatched parents
    candidates = []
    for p_idx in range(len(pos_t)):
        if p_idx in matched_parents:
            continue

        # Get nearby daughters
        nearby = [(distances[p_idx, j], indices[p_idx, j])
                  for j in range(k)
                  if distances[p_idx, j] < max_distance * 2]

        # Try all pairs of nearby daughters
        for i in range(len(nearby)):
            for j in range(i + 1, len(nearby)):
                d1_idx = nearby[i][1]
                d2_idx = nearby[j][1]

                # Daughters' midpoint
                midpoint = (pos_t1[d1_idx] + pos_t1[d2_idx]) / 2
                midpoint_dist = np.linalg.norm(pos_t[p_idx] - midpoint)

                if midpoint_dist > max_distance:
                    continue

                # Daughter separation
                separation = np.linalg.norm(pos_t1[d1_idx] - pos_t1[d2_idx])
                if separation < min_separation or separation > max_separation:
                    continue

                # Score: weighted combination of midpoint distance and separation
                score = midpoint_dist + 0.5 * abs(separation - np.mean(diameters_t))

                candidates.append(DivisionCandidate(
                    parent_idx=p_idx,
                    daughter1_idx=d1_idx,
                    daughter2_idx=d2_idx,
                    parent_position=positions_t[p_idx],
                    daughter1_position=positions_t1[d1_idx],
                    daughter2_position=positions_t1[d2_idx],
                    score=score,
                    midpoint_distance=midpoint_dist,
                    separation=separation,
                ))

    # Sort by score and remove conflicting assignments
    candidates.sort(key=lambda c: c.score)
    used_parents = set()
    used_daughters = set()
    filtered = []
    for c in candidates:
        if c.parent_idx in used_parents:
            continue
        if c.daughter1_idx in used_daughters or c.daughter2_idx in used_daughters:
            continue
        filtered.append(c)
        used_parents.add(c.parent_idx)
        used_daughters.add(c.daughter1_idx)
        used_daughters.add(c.daughter2_idx)

    return filtered
