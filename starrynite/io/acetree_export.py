"""Export tracking results in acetree_py compatible ZIP format."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np


def export_acetree_zip(
    output_path: str | Path,
    timepoints: dict[int, dict],
    embryo_name: str = "embryo",
) -> Path:
    """Export tracking results as a ZIP archive compatible with acetree_py.

    The ZIP contains nuclei/tNNN-nuclei files in StarryNite CSV format.

    Args:
        output_path: Path for the output ZIP file.
        timepoints: Dict mapping timepoint → dict with keys:
            - positions: (N, 3) array [x, y, z]
            - diameters: (N,) array
            - valid: (N,) bool array (optional)
            - predecessors: (N,) int array (optional)
            - successors: (N,) int array (optional)
            - names: list[str] (optional)
            - intensities: (N,) array (optional)
        embryo_name: Name prefix for the ZIP archive.

    Returns:
        Path to the created ZIP file.
    """
    output_path = Path(output_path)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in sorted(timepoints.keys()):
            data = timepoints[t]
            positions = data["positions"]
            diameters = data["diameters"]
            n = len(positions)

            valid = data.get("valid", np.ones(n, dtype=bool))
            preds = data.get("predecessors", np.full(n, -1, dtype=int))
            succs = data.get("successors", np.full(n, -1, dtype=int))
            names = data.get("names", [""] * n)
            intensities = data.get("intensities", np.zeros(n))

            lines = []
            for i in range(n):
                x, y, z = positions[i]
                line = (
                    f"{i + 1}, {int(valid[i])}, {preds[i]}, {succs[i]}, -1, "
                    f"{x:.0f}, {y:.0f}, {z:.1f}, {diameters[i]:.0f}, {names[i]}, "
                    f"{intensities[i]:.0f}, 0, 0, 0, , 0, 0, 0, 0, 0, "
                )
                lines.append(line)

            content = "\n".join(lines) + "\n"
            arcname = f"nuclei/t{t:03d}-nuclei"
            zf.writestr(arcname, content)

    return output_path
