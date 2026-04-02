"""Read/write StarryNite nuclei format files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_nuclei_positions(path: str | Path) -> np.ndarray:
    """Read nuclei positions from a StarryNite-format nuclei file.

    Args:
        path: Path to a nuclei file.

    Returns:
        Array of shape (N, 3) with columns [x, y, z].
    """
    from .ground_truth import read_nuclei_file

    nuclei = read_nuclei_file(path)
    if not nuclei:
        return np.empty((0, 3), dtype=np.float64)
    return np.array([[n.x, n.y, n.z] for n in nuclei], dtype=np.float64)


def write_nuclei_file(
    path: str | Path,
    positions: np.ndarray,
    diameters: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    predecessors: np.ndarray | None = None,
    successors: np.ndarray | None = None,
    names: list[str] | None = None,
    intensities: np.ndarray | None = None,
) -> None:
    """Write nuclei data in StarryNite/AceTree CSV format.

    Args:
        path: Output file path.
        positions: (N, 3) array of [x, y, z] positions.
        diameters: (N,) array of nuclear diameters.
        valid: (N,) boolean array, default all True.
        predecessors: (N,) array of predecessor IDs, default -1.
        successors: (N,) array of successor IDs, default -1.
        names: List of cell names, default empty.
        intensities: (N,) array of intensities, default 0.
    """
    path = Path(path)
    n = len(positions)

    if valid is None:
        valid = np.ones(n, dtype=bool)
    if predecessors is None:
        predecessors = np.full(n, -1, dtype=int)
    if successors is None:
        successors = np.full(n, -1, dtype=int)
    if names is None:
        names = [""] * n
    if intensities is None:
        intensities = np.zeros(n, dtype=np.float64)

    with open(path, "w") as f:
        for i in range(n):
            x, y, z = positions[i]
            line = (
                f"{i + 1}, {int(valid[i])}, {predecessors[i]}, {successors[i]}, -1, "
                f"{x:.0f}, {y:.0f}, {z:.1f}, {diameters[i]:.0f}, {names[i]}, "
                f"{intensities[i]:.0f}, 0, 0, 0, , 0, 0, 0, 0, 0, "
            )
            f.write(line + "\n")
