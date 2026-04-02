"""Parse ground truth nuclei files from StarryNite/AceTree format."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundTruthNucleus:
    """A single nucleus from ground truth data."""

    id: int
    valid: bool
    predecessor_id: int  # -1 if no predecessor
    successor_id: int  # -1 if no successor
    x: float
    y: float
    z: float
    diameter: float
    name: str  # Cell name (Sulston nomenclature), empty if unnamed
    intensity: float


@dataclass
class GroundTruthTimepoint:
    """All nuclei at one timepoint."""

    timepoint: int
    nuclei: list[GroundTruthNucleus] = field(default_factory=list)

    @property
    def valid_nuclei(self) -> list[GroundTruthNucleus]:
        """Return only nuclei marked as valid."""
        return [n for n in self.nuclei if n.valid]

    @property
    def named_nuclei(self) -> list[GroundTruthNucleus]:
        """Return only nuclei with cell names."""
        return [n for n in self.nuclei if n.name]


def parse_nuclei_line(line: str) -> GroundTruthNucleus | None:
    """Parse a single line from a nuclei file.

    Format: id, valid, pred, succ, ?, x, y, z, diameter, name, intensity, ...
    """
    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 11:
        return None

    try:
        return GroundTruthNucleus(
            id=int(parts[0]),
            valid=bool(int(parts[1])),
            predecessor_id=int(parts[2]),
            successor_id=int(parts[3]),
            # parts[4] is unknown/unused field
            x=float(parts[5]),
            y=float(parts[6]),
            z=float(parts[7]),
            diameter=float(parts[8]),
            name=parts[9].strip(),
            intensity=float(parts[10]) if parts[10].strip() else 0.0,
        )
    except (ValueError, IndexError):
        return None


def read_nuclei_file(path: str | Path) -> list[GroundTruthNucleus]:
    """Read a single nuclei file (e.g., t001-nuclei).

    Args:
        path: Path to the nuclei file.

    Returns:
        List of parsed nuclei.
    """
    path = Path(path)
    nuclei = []
    with open(path) as f:
        for line in f:
            nucleus = parse_nuclei_line(line)
            if nucleus is not None:
                nuclei.append(nucleus)
    return nuclei


def discover_nuclei_files(nuclei_dir: str | Path) -> list[tuple[int, Path]]:
    """Discover all nuclei files in a directory.

    Matches files like: t001-nuclei, t200-nuclei, etc.

    Args:
        nuclei_dir: Directory containing nuclei files.

    Returns:
        Sorted list of (timepoint, file_path) tuples.
    """
    nuclei_dir = Path(nuclei_dir)
    pattern = re.compile(r"^t(\d+)-nuclei$")

    results = []
    for f in nuclei_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            results.append((int(m.group(1)), f))

    results.sort(key=lambda x: x[0])
    return results


def load_ground_truth(nuclei_dir: str | Path) -> dict[int, GroundTruthTimepoint]:
    """Load all ground truth data from a nuclei directory.

    Args:
        nuclei_dir: Directory containing t###-nuclei files.

    Returns:
        Dict mapping timepoint → GroundTruthTimepoint.
    """
    files = discover_nuclei_files(nuclei_dir)
    result = {}
    for t, path in files:
        nuclei = read_nuclei_file(path)
        result[t] = GroundTruthTimepoint(timepoint=t, nuclei=nuclei)
    return result
