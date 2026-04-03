"""End-to-end StarryNite pipeline: detect → track → export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from starrynite_py.config.schema import PipelineConfig
from starrynite_py.io.tiff_loader import discover_timepoints, load_timepoint
from starrynite_py.detection.stardist_detect import DetectionResult
from starrynite_py.tracking.adapter import TrackerResult
from starrynite_py.io.acetree_export import export_acetree_zip

logger = logging.getLogger(__name__)


def _detect_single_volume(volume, config: PipelineConfig) -> DetectionResult:
    """Run detection on a single volume using configured detector."""
    if config.detector == "cellpose":
        from starrynite_py.detection.cellpose_detect import detect_nuclei_cellpose
        result = detect_nuclei_cellpose(
            volume, config.imaging,
            diameter=config.cellpose.diameter,
            cellprob_threshold=config.cellpose.cellprob_threshold,
            flow_threshold=config.cellpose.flow_threshold,
            do_3D=config.cellpose.do_3D,
            anisotropy=config.cellpose.anisotropy,
        )
        # Convert CellposeResult to DetectionResult for pipeline compatibility
        return DetectionResult(
            labels=result.labels,
            centroids=result.centroids,
            diameters=result.diameters,
        )
    else:
        from starrynite_py.detection.stardist_detect import detect_nuclei
        return detect_nuclei(volume, config.stardist, config.imaging)


def run_detection(config: PipelineConfig) -> dict[int, DetectionResult]:
    """Run detection on all timepoints using configured detector.

    Args:
        config: Pipeline configuration.

    Returns:
        Dict mapping timepoint → DetectionResult.
    """
    timepoints = discover_timepoints(
        config.data.input_dir,
        config.data.embryo_name,
    )

    # Filter by time range
    start = config.data.start_time
    end = config.data.end_time
    if end is not None:
        timepoints = [(t, p) for t, p in timepoints if start <= t <= end]
    else:
        timepoints = [(t, p) for t, p in timepoints if t >= start]

    logger.info(f"Detecting nuclei in {len(timepoints)} timepoints ({config.detector})...")

    results = {}
    for i, (t, path) in enumerate(timepoints):
        logger.info(f"  [{i + 1}/{len(timepoints)}] t={t:03d}")
        volume = load_timepoint(path)
        results[t] = _detect_single_volume(volume, config)

    return results


def run_tracking(
    config: PipelineConfig,
    detections: dict[int, DetectionResult],
) -> TrackerResult:
    """Run tracking on detection results.

    Args:
        config: Pipeline configuration.
        detections: Detection results from run_detection.

    Returns:
        TrackerResult with tracks and divisions.
    """
    if config.tracker == "ultrack":
        from starrynite_py.tracking.ultrack_track import track_with_ultrack
        return track_with_ultrack(detections, config.ultrack, config.imaging)
    elif config.tracker == "btrack":
        from starrynite_py.tracking.btrack_track import track_with_btrack
        return track_with_btrack(detections, config.btrack, config.imaging)
    else:
        raise ValueError(f"Unknown tracker: {config.tracker}")


def run_export(
    config: PipelineConfig,
    detections: dict[int, DetectionResult],
    tracking: TrackerResult,
) -> Path:
    """Export results in acetree_py compatible format.

    Args:
        config: Pipeline configuration.
        detections: Detection results.
        tracking: Tracking results.

    Returns:
        Path to exported ZIP file.
    """
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build timepoint data from tracking results
    timepoint_data = {}
    for t in sorted(detections.keys()):
        nodes = tracking.get_timepoint(t)
        if nodes:
            positions = np.array([n.position for n in nodes])
            diameters = np.array([n.diameter for n in nodes])
        else:
            # Fallback to detection centroids if no tracking
            det = detections[t]
            positions = det.centroids
            diameters = det.diameters

        timepoint_data[t] = {
            "positions": positions,
            "diameters": diameters,
        }

    output_path = output_dir / f"{config.data.embryo_name}_tracked.zip"
    return export_acetree_zip(output_path, timepoint_data, config.data.embryo_name)


def run_pipeline(config: PipelineConfig) -> tuple[dict[int, DetectionResult], TrackerResult, Path]:
    """Run the full pipeline: detect → track → export.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (detections, tracking_result, export_path).
    """
    logger.info("=== StarryNite Pipeline ===")

    logger.info("Step 1: Detection")
    detections = run_detection(config)
    logger.info(f"  Detected nuclei in {len(detections)} timepoints")

    logger.info(f"Step 2: Tracking ({config.tracker})")
    tracking = run_tracking(config, detections)
    logger.info(f"  Found {len(tracking.track_ids)} tracks, {len(tracking.divisions)} divisions")

    logger.info("Step 3: Export")
    export_path = run_export(config, detections, tracking)
    logger.info(f"  Exported to {export_path}")

    return detections, tracking, export_path
