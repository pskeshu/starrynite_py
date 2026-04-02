"""TIFF stack loading with optional dask lazy loading."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


def discover_timepoints(input_dir: str | Path, embryo_name: str) -> list[tuple[int, Path]]:
    """Discover all TIFF timepoint files in a directory.

    Matches files like: {embryo_name}_t000.tif, {embryo_name}_t001.tif, etc.
    Also handles patterns: {embryo_name}_t0000.tif, {embryo_name}_{digits}.tif

    Args:
        input_dir: Directory containing TIFF files.
        embryo_name: Embryo name prefix.

    Returns:
        Sorted list of (timepoint_index, file_path) tuples.
    """
    input_dir = Path(input_dir)
    pattern = re.compile(
        rf"^{re.escape(embryo_name)}[_.]?t?(\d+)\.tiff?$", re.IGNORECASE
    )

    results = []
    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            results.append((int(m.group(1)), f))

    results.sort(key=lambda x: x[0])
    return results


def load_timepoint(path: str | Path) -> np.ndarray:
    """Load a single 3D TIFF stack as a numpy array.

    Args:
        path: Path to a multi-page TIFF file.

    Returns:
        3D numpy array with shape (Z, Y, X), dtype float32.
    """
    stack = tifffile.imread(str(path))
    return stack.astype(np.float32)


def load_timepoint_lazy(path: str | Path) -> "dask.array.Array":
    """Lazily load a TIFF stack as a dask array.

    The data is only read from disk when compute() is called.

    Args:
        path: Path to a multi-page TIFF file.

    Returns:
        dask array with shape (Z, Y, X).
    """
    import dask.array as da

    store = tifffile.imread(str(path), aszarr=True)
    return da.from_zarr(store).astype(np.float32)


def load_sequence_lazy(
    input_dir: str | Path,
    embryo_name: str,
    start_time: int = 0,
    end_time: int | None = None,
) -> list[tuple[int, "dask.array.Array"]]:
    """Lazily load an entire time-lapse sequence.

    Args:
        input_dir: Directory with TIFF files.
        embryo_name: Embryo name prefix.
        start_time: First timepoint to include.
        end_time: Last timepoint to include (None=all).

    Returns:
        List of (timepoint, dask_array) tuples.
    """
    timepoints = discover_timepoints(input_dir, embryo_name)

    if end_time is not None:
        timepoints = [(t, p) for t, p in timepoints if start_time <= t <= end_time]
    else:
        timepoints = [(t, p) for t, p in timepoints if t >= start_time]

    return [(t, load_timepoint_lazy(p)) for t, p in timepoints]
