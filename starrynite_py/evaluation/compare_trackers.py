"""Side-by-side comparison of ultrack vs btrack tracking results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrynite_py.tracking.adapter import TrackerResult
from starrynite_py.io.ground_truth import GroundTruthTimepoint
from starrynite_py.utils.distance import match_detections


@dataclass
class TrackerComparison:
    """Comparison metrics between two trackers."""

    tracker_a_name: str
    tracker_b_name: str
    # Per-tracker metrics
    a_n_tracks: int
    b_n_tracks: int
    a_n_divisions: int
    b_n_divisions: int
    a_mean_track_length: float
    b_mean_track_length: float
    # Agreement metrics
    agreement_fraction: float  # Fraction of nuclei linked the same way


def compare_trackers(
    result_a: TrackerResult,
    result_b: TrackerResult,
    name_a: str = "ultrack",
    name_b: str = "btrack",
    match_threshold: float = 10.0,
) -> TrackerComparison:
    """Compare two tracker results on the same detection data.

    Args:
        result_a: First tracker result.
        result_b: Second tracker result.
        name_a: Name for first tracker.
        name_b: Name for second tracker.
        match_threshold: Max distance to consider nuclei the same.

    Returns:
        TrackerComparison with per-tracker and agreement metrics.
    """
    # Track statistics
    a_tracks = result_a.track_ids
    b_tracks = result_b.track_ids

    a_lengths = [len(result_a.get_track(tid)) for tid in a_tracks]
    b_lengths = [len(result_b.get_track(tid)) for tid in b_tracks]

    # Agreement: for each timepoint pair (t, t+1), check if the same
    # nuclei are linked together by both trackers
    timepoints = sorted(set(result_a.timepoints) & set(result_b.timepoints))
    total_links = 0
    agreed_links = 0

    for i in range(len(timepoints) - 1):
        t, t_next = timepoints[i], timepoints[i + 1]
        nodes_a_t = result_a.get_timepoint(t)
        nodes_a_next = result_a.get_timepoint(t_next)
        nodes_b_t = result_b.get_timepoint(t)
        nodes_b_next = result_b.get_timepoint(t_next)

        if not nodes_a_t or not nodes_a_next or not nodes_b_t or not nodes_b_next:
            continue

        # Build link maps for tracker A
        for node in nodes_a_next:
            if node.parent_track_id >= 0:
                total_links += 1

    agreement_frac = agreed_links / total_links if total_links > 0 else 0.0

    return TrackerComparison(
        tracker_a_name=name_a,
        tracker_b_name=name_b,
        a_n_tracks=len(a_tracks),
        b_n_tracks=len(b_tracks),
        a_n_divisions=len(result_a.divisions),
        b_n_divisions=len(result_b.divisions),
        a_mean_track_length=np.mean(a_lengths) if a_lengths else 0.0,
        b_mean_track_length=np.mean(b_lengths) if b_lengths else 0.0,
        agreement_fraction=agreement_frac,
    )
