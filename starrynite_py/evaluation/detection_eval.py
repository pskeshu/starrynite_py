"""Evaluate detection results against ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrynite_py.io.ground_truth import GroundTruthTimepoint
from starrynite_py.detection.stardist_detect import DetectionResult
from starrynite_py.utils.distance import match_detections


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics for one timepoint."""

    timepoint: int
    n_detected: int
    n_ground_truth: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def evaluate_detection(
    detection: DetectionResult,
    ground_truth: GroundTruthTimepoint,
    threshold_factor: float = 0.75,
    anisotropy: float = 1.0,
) -> DetectionMetrics:
    """Evaluate detection against ground truth for one timepoint.

    Args:
        detection: StarDist detection result.
        ground_truth: Ground truth nuclei.
        threshold_factor: Matching threshold as fraction of mean GT diameter.
        anisotropy: Z/XY resolution ratio.

    Returns:
        DetectionMetrics with precision/recall/F1.
    """
    gt_nuclei = ground_truth.valid_nuclei
    if not gt_nuclei:
        return DetectionMetrics(
            timepoint=ground_truth.timepoint,
            n_detected=len(detection.centroids),
            n_ground_truth=0,
            true_positives=0,
            false_positives=len(detection.centroids),
            false_negatives=0,
        )

    gt_positions = np.array([[n.x, n.y, n.z] for n in gt_nuclei])
    gt_diameters = np.array([n.diameter for n in gt_nuclei])
    threshold = threshold_factor * np.mean(gt_diameters)

    _, _, tp, fp, fn = match_detections(
        detection.centroids, gt_positions, threshold, anisotropy
    )

    return DetectionMetrics(
        timepoint=ground_truth.timepoint,
        n_detected=len(detection.centroids),
        n_ground_truth=len(gt_nuclei),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def summarize_detection_metrics(metrics: list[DetectionMetrics]) -> dict:
    """Summarize detection metrics across timepoints."""
    total_tp = sum(m.true_positives for m in metrics)
    total_fp = sum(m.false_positives for m in metrics)
    total_fn = sum(m.false_negatives for m in metrics)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_timepoints": len(metrics),
        "mean_f1": np.mean([m.f1 for m in metrics]),
    }
