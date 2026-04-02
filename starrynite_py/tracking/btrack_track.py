"""btrack tracking wrapper — Bayesian tracking with Kalman filter."""

from __future__ import annotations

import numpy as np

from starrynite_py.config.schema import BtrackConfig, ImagingConfig
from starrynite_py.detection.stardist_detect import DetectionResult
from .adapter import TrackerResult, TrackNode


def track_with_btrack(
    detections: dict[int, DetectionResult],
    btrack_config: BtrackConfig,
    imaging_config: ImagingConfig,
) -> TrackerResult:
    """Track nuclei across timepoints using btrack.

    Args:
        detections: Dict mapping timepoint → DetectionResult (from StarDist).
        btrack_config: btrack configuration parameters.
        imaging_config: Imaging parameters.

    Returns:
        TrackerResult with tracks and division events.
    """
    import btrack

    sorted_times = sorted(detections.keys())

    # Build list of objects from centroids
    objects = []
    for t_idx, t in enumerate(sorted_times):
        det = detections[t]
        for i in range(len(det.centroids)):
            obj = btrack.btypes.PyTrackObject()
            obj.t = t_idx
            obj.x = det.centroids[i, 0]
            obj.y = det.centroids[i, 1]
            obj.z = det.centroids[i, 2]
            objects.append(obj)

    # Get volume bounds from first detection
    first_det = detections[sorted_times[0]]
    z_max, y_max, x_max = first_det.labels.shape
    volume = ((0, x_max), (0, y_max), (0, z_max))

    # Run tracker
    with btrack.BayesianTracker() as tracker:
        tracker.configure(
            btrack.datasets.cell_config()
            if btrack_config.hypothesis_model == "cell_hypothesis"
            else btrack.datasets.particle_config()
        )
        tracker.max_search_radius = btrack_config.max_search_radius
        tracker.tracking_updates = [
            "MOTION",
            "VISUAL",
        ]
        tracker.volume = volume

        tracker.append(objects)
        tracker.track()

        if btrack_config.enable_divisions:
            tracker.optimize(
                options={"DIVISION": True},
            )

        tracks = tracker.tracks

    # Convert to TrackerResult
    result = TrackerResult()
    for trk in tracks:
        for i, t_idx in enumerate(trk.t):
            t = sorted_times[t_idx] if t_idx < len(sorted_times) else t_idx

            # Look up diameter from detection
            diameter = 0.0
            if t in detections:
                det = detections[t]
                pos = np.array([trk.x[i], trk.y[i], trk.z[i]])
                if len(det.centroids) > 0:
                    dists = np.linalg.norm(det.centroids - pos, axis=1)
                    nearest = np.argmin(dists)
                    diameter = det.diameters[nearest]

            node = TrackNode(
                track_id=trk.ID,
                timepoint=t,
                position=np.array([trk.x[i], trk.y[i], trk.z[i]]),
                diameter=diameter,
                parent_track_id=trk.parent if hasattr(trk, "parent") else -1,
            )
            result.nodes.append(node)

        # Check for division (if track has children)
        if hasattr(trk, "children") and trk.children:
            children = list(trk.children)
            if len(children) == 2:
                div_time = sorted_times[trk.t[-1]] if trk.t[-1] < len(sorted_times) else trk.t[-1]
                result.divisions.append(
                    (trk.ID, children[0], children[1], div_time)
                )

    return result
