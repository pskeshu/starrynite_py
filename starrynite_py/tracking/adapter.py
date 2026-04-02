"""Common tracking result interface for ultrack and btrack backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackNode:
    """A single nucleus in a track."""

    track_id: int
    timepoint: int
    position: np.ndarray  # (3,) [x, y, z]
    diameter: float
    parent_track_id: int = -1  # -1 if no parent (not from division)


@dataclass
class TrackerResult:
    """Unified tracking result from any backend."""

    # All track nodes indexed by (timepoint, nucleus_index)
    nodes: list[TrackNode] = field(default_factory=list)

    # Division events: (parent_track_id, daughter1_track_id, daughter2_track_id, timepoint)
    divisions: list[tuple[int, int, int, int]] = field(default_factory=list)

    def get_timepoint(self, t: int) -> list[TrackNode]:
        """Get all nodes at a given timepoint."""
        return [n for n in self.nodes if n.timepoint == t]

    def get_track(self, track_id: int) -> list[TrackNode]:
        """Get all nodes belonging to a track, sorted by time."""
        nodes = [n for n in self.nodes if n.track_id == track_id]
        return sorted(nodes, key=lambda n: n.timepoint)

    @property
    def track_ids(self) -> set[int]:
        """All unique track IDs."""
        return {n.track_id for n in self.nodes}

    @property
    def timepoints(self) -> list[int]:
        """Sorted list of all timepoints."""
        return sorted({n.timepoint for n in self.nodes})

    def to_positions_dict(self) -> dict[int, np.ndarray]:
        """Convert to dict of timepoint → (N, 3) position arrays."""
        result = {}
        for t in self.timepoints:
            nodes = self.get_timepoint(t)
            if nodes:
                result[t] = np.array([n.position for n in nodes])
            else:
                result[t] = np.empty((0, 3))
        return result
