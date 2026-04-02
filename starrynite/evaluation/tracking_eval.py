"""Evaluate tracking results against ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrynite.io.ground_truth import GroundTruthTimepoint, GroundTruthNucleus
from starrynite.tracking.adapter import TrackerResult
from starrynite.utils.distance import match_detections


@dataclass
class TrackingMetrics:
    """Tracking evaluation metrics."""

    total_links_gt: int
    total_links_predicted: int
    correct_links: int
    total_divisions_gt: int
    total_divisions_predicted: int
    correct_divisions: int
    max_lineage_depth: int

    @property
    def link_accuracy(self) -> float:
        if self.total_links_gt == 0:
            return 0.0
        return self.correct_links / self.total_links_gt

    @property
    def division_precision(self) -> float:
        if self.total_divisions_predicted == 0:
            return 0.0
        return self.correct_divisions / self.total_divisions_predicted

    @property
    def division_recall(self) -> float:
        if self.total_divisions_gt == 0:
            return 0.0
        return self.correct_divisions / self.total_divisions_gt


def evaluate_tracking(
    tracker_result: TrackerResult,
    ground_truth: dict[int, GroundTruthTimepoint],
    match_threshold: float = 15.0,
    anisotropy: float = 1.0,
) -> TrackingMetrics:
    """Evaluate tracking against ground truth lineage data.

    Compares predecessor/successor links and division events.

    Args:
        tracker_result: Tracking result to evaluate.
        ground_truth: Dict of timepoint → GroundTruthTimepoint.
        match_threshold: Max distance for matching nuclei.
        anisotropy: Z/XY resolution ratio.

    Returns:
        TrackingMetrics with link accuracy and division metrics.
    """
    total_links_gt = 0
    correct_links = 0
    total_divisions_gt = 0

    sorted_times = sorted(ground_truth.keys())

    for t_idx, t in enumerate(sorted_times[:-1]):
        t_next = sorted_times[t_idx + 1]
        gt_t = ground_truth[t]
        gt_next = ground_truth.get(t_next)
        if gt_next is None:
            continue

        # Count GT links (successor relationships)
        for nuc in gt_t.valid_nuclei:
            if nuc.successor_id > 0:
                total_links_gt += 1

        # Count GT divisions (two nuclei in next frame with same predecessor)
        pred_counts: dict[int, int] = {}
        for nuc in gt_next.valid_nuclei:
            if nuc.predecessor_id > 0:
                pred_counts[nuc.predecessor_id] = pred_counts.get(nuc.predecessor_id, 0) + 1
        for pid, count in pred_counts.items():
            if count == 2:
                total_divisions_gt += 1

    return TrackingMetrics(
        total_links_gt=total_links_gt,
        total_links_predicted=len(tracker_result.nodes),
        correct_links=correct_links,  # TODO: implement full link matching
        total_divisions_gt=total_divisions_gt,
        total_divisions_predicted=len(tracker_result.divisions),
        correct_divisions=0,  # TODO: implement division matching
        max_lineage_depth=_compute_max_depth(tracker_result),
    )


def _compute_max_depth(result: TrackerResult) -> int:
    """Compute maximum lineage tree depth."""
    if not result.divisions:
        return 0

    # Build parent → children map
    children_map: dict[int, list[int]] = {}
    for parent_id, d1, d2, _ in result.divisions:
        children_map[parent_id] = [d1, d2]

    # Find root tracks (no parent)
    all_children = set()
    for _, d1, d2, _ in result.divisions:
        all_children.add(d1)
        all_children.add(d2)
    roots = set(children_map.keys()) - all_children

    # BFS for max depth
    max_depth = 0
    queue = [(r, 0) for r in roots]
    while queue:
        track_id, depth = queue.pop(0)
        max_depth = max(max_depth, depth)
        for child in children_map.get(track_id, []):
            queue.append((child, depth + 1))

    return max_depth
