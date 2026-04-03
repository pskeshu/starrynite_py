"""ultrack tracking wrapper — ILP-based global optimization."""

from __future__ import annotations

import numpy as np

from starrynite_py.config.schema import UltrackConfig, ImagingConfig
from starrynite_py.detection.stardist_detect import DetectionResult
from .adapter import TrackerResult, TrackNode


def track_with_ultrack(
    detections: dict[int, DetectionResult],
    ultrack_config: UltrackConfig,
    imaging_config: ImagingConfig,
) -> TrackerResult:
    """Track nuclei across timepoints using ultrack.

    Args:
        detections: Dict mapping timepoint → DetectionResult (from StarDist).
        ultrack_config: ultrack configuration parameters.
        imaging_config: Imaging parameters.

    Returns:
        TrackerResult with tracks and division events.
    """
    from ultrack import MainConfig, track, to_tracks_layer
    from ultrack.utils import labels_to_contours

    # Build ultrack config
    cfg = MainConfig()
    cfg.segmentation_config.min_area = ultrack_config.min_area
    cfg.segmentation_config.max_area = ultrack_config.max_area
    cfg.linking_config.max_distance = ultrack_config.max_distance
    cfg.tracking_config.division_weight = ultrack_config.division_weight
    cfg.tracking_config.appear_weight = ultrack_config.appear_weight
    cfg.tracking_config.disappear_weight = ultrack_config.disappear_weight

    # Stack all labels into a 4D array (T, Z, Y, X)
    sorted_times = sorted(detections.keys())
    label_stack = np.stack([detections[t].labels for t in sorted_times], axis=0)

    # Convert instance labels to foreground/edge contour maps
    detection_map, edge_map = labels_to_contours(label_stack)

    # Run ultrack pipeline
    track(
        cfg,
        foreground=detection_map,
        edges=edge_map,
        overwrite=True,
    )

    # Extract results
    tracks_df, graph = to_tracks_layer(cfg)

    # Convert to our TrackerResult format
    result = TrackerResult()
    for _, row in tracks_df.iterrows():
        t_idx = int(row["t"])
        t = sorted_times[t_idx] if t_idx < len(sorted_times) else t_idx

        node = TrackNode(
            track_id=int(row["track_id"]),
            timepoint=t,
            position=np.array([row["x"], row["y"], row["z"]]),
            diameter=0.0,
            parent_track_id=int(row.get("parent_track_id", -1)),
        )

        # Look up diameter from detection
        if t in detections:
            det = detections[t]
            if len(det.centroids) > 0:
                dists = np.linalg.norm(det.centroids - node.position, axis=1)
                nearest = np.argmin(dists)
                node.diameter = det.diameters[nearest]

        result.nodes.append(node)

    # Extract division events from graph
    # ultrack graph format: {child_track_id: parent_track_id} (child→parent)
    if graph is not None:
        # Invert to parent → [children]
        parent_to_children: dict[int, list[int]] = {}
        for child_id, parent_id in graph.items():
            parent_to_children.setdefault(parent_id, []).append(child_id)

        for parent_id, children in parent_to_children.items():
            if len(children) == 2:
                parent_nodes = result.get_track(parent_id)
                if parent_nodes:
                    div_time = parent_nodes[-1].timepoint
                    result.divisions.append(
                        (parent_id, children[0], children[1], div_time)
                    )

    return result
