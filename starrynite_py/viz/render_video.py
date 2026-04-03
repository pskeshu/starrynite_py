"""Render annotated max-projection videos of detection and tracking results."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)


def render_frame(
    volume: np.ndarray,
    centroids: np.ndarray | None = None,
    diameters: np.ndarray | None = None,
    gt_positions: np.ndarray | None = None,
    gt_diameters: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    division_parents: list[int] | None = None,
    timepoint: int = 0,
    n_gt: int = 0,
    figsize: tuple[float, float] = (12, 5),
    title_extra: str = "",
) -> np.ndarray:
    """Render a single annotated max-projection frame.

    Args:
        volume: 3D array (Z, Y, X).
        centroids: (N, 3) detected positions [x, y, z].
        diameters: (N,) detected diameters.
        gt_positions: (M, 3) ground truth positions [x, y, z].
        gt_diameters: (M,) ground truth diameters.
        track_ids: (N,) track IDs for coloring.
        division_parents: List of centroid indices that are division parents.
        timepoint: Timepoint number for title.
        n_gt: Number of GT nuclei for title.
        figsize: Figure size in inches.
        title_extra: Extra text for title.

    Returns:
        RGB image as uint8 numpy array (H, W, 3).
    """
    # Max projection
    max_proj = np.max(volume, axis=0)  # (Y, X)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=100)

    # Left panel: raw max projection with detections
    ax = axes[0]
    ax.imshow(max_proj, cmap="gray", vmin=np.percentile(max_proj, 1), vmax=np.percentile(max_proj, 99.5))

    if centroids is not None and len(centroids) > 0:
        colors = _get_track_colors(track_ids, len(centroids))
        for i in range(len(centroids)):
            x, y = centroids[i, 0], centroids[i, 1]
            r = diameters[i] / 2 if diameters is not None else 5
            color = colors[i]
            linewidth = 2

            # Highlight division parents
            if division_parents and i in division_parents:
                color = "yellow"
                linewidth = 3

            circle = patches.Circle((x, y), r, linewidth=linewidth,
                                    edgecolor=color, facecolor="none", alpha=0.8)
            ax.add_patch(circle)

    n_det = len(centroids) if centroids is not None else 0
    ax.set_title(f"t={timepoint:03d} | Detected: {n_det} | GT: {n_gt}", fontsize=10)
    ax.axis("off")

    # Right panel: GT overlay comparison
    ax2 = axes[1]
    ax2.imshow(max_proj, cmap="gray", vmin=np.percentile(max_proj, 1), vmax=np.percentile(max_proj, 99.5))

    if gt_positions is not None and len(gt_positions) > 0:
        for i in range(len(gt_positions)):
            x, y = gt_positions[i, 0], gt_positions[i, 1]
            r = gt_diameters[i] / 2 if gt_diameters is not None else 5
            circle = patches.Circle((x, y), r, linewidth=1.5,
                                    edgecolor="lime", facecolor="none", alpha=0.7)
            ax2.add_patch(circle)

    if centroids is not None and len(centroids) > 0:
        for i in range(len(centroids)):
            x, y = centroids[i, 0], centroids[i, 1]
            ax2.plot(x, y, "r+", markersize=4, alpha=0.6)

    ax2.set_title(f"GT (green) vs Detected (red +) {title_extra}", fontsize=10)
    ax2.axis("off")

    fig.tight_layout()

    # Convert to image array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return image


def render_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: int = 5,
) -> Path:
    """Write frames to an MP4 video file using matplotlib animation.

    Args:
        frames: List of RGB image arrays (H, W, 3).
        output_path: Path for output video file.
        fps: Frames per second.

    Returns:
        Path to output video.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        writer.write(frame[:, :, ::-1])

    writer.release()
    logger.info(f"Video saved to {output_path} ({len(frames)} frames, {fps} fps)")
    return output_path


def _get_track_colors(track_ids: np.ndarray | None, n: int) -> list[str]:
    """Generate consistent colors for tracks."""
    if track_ids is None:
        return ["cyan"] * n

    cmap = plt.cm.get_cmap("tab20")
    colors = []
    for i in range(n):
        tid = track_ids[i] if i < len(track_ids) else i
        colors.append(cmap(tid % 20))
    return colors
