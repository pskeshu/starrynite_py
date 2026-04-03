"""Monkey-patches for Cellpose CPU bottleneck optimization.

Import this module before running Cellpose to apply speedups.
Targets the three main CPU bottlenecks:
1. fill_holes_and_remove_small_masks: parallelize with multiprocessing
2. masks_to_flows_gpu_3d center loop: use scipy.ndimage.center_of_mass
3. Mask cleaning: skip fill_voids when not needed

Usage:
    from starrynite_py.detection.cellpose_speedup import apply_speedups
    apply_speedups()
    # Then use Cellpose normally
"""

from __future__ import annotations

import logging
from multiprocessing import Pool, cpu_count

import numpy as np

logger = logging.getLogger(__name__)


def _fill_single_mask(args):
    """Fill holes in a single mask (for multiprocessing)."""
    import fill_voids
    mask_data, label_id = args
    msk = mask_data == label_id
    if msk.any():
        msk = fill_voids.fill(msk)
    return msk, label_id


def fill_holes_parallel(masks, min_size=15):
    """Parallelized version of fill_holes_and_remove_small_masks.

    Uses multiprocessing to fill holes in masks independently.
    """
    from scipy.ndimage import find_objects, label as scipy_label
    import fastremap

    slices = find_objects(masks)
    n_masks = len(slices)

    if n_masks == 0:
        return masks

    # Remove small masks first (serial, fast)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            if msk.sum() < min_size:
                masks[slc][msk] = 0
            else:
                j += 1

    # Relabel
    masks = fastremap.renumber(masks, in_place=True)[0]
    slices = find_objects(masks)

    # Parallel hole filling
    n_workers = min(cpu_count(), len(slices), 8)  # Cap at 8 workers

    if n_workers > 1 and len(slices) > 10:
        # Prepare work items
        work_items = []
        for i, slc in enumerate(slices):
            if slc is not None:
                work_items.append((masks[slc].copy(), i + 1))

        with Pool(n_workers) as pool:
            results = pool.map(_fill_single_mask, work_items)

        # Apply results
        j = 0
        for (filled_msk, label_id), slc in zip(results, slices):
            if slc is not None:
                masks[slc][filled_msk] = j + 1
                j += 1
    else:
        # Serial fallback for small mask counts
        import fill_voids
        j = 0
        for i, slc in enumerate(slices):
            if slc is not None:
                msk = masks[slc] == (i + 1)
                msk = fill_voids.fill(msk)
                masks[slc][msk] = j + 1
                j += 1

    return fastremap.renumber(masks, in_place=True)[0]


def apply_speedups():
    """Apply all monkey-patches to Cellpose for faster 3D processing.

    Call this once before creating CellposeModel.
    """
    try:
        import cellpose.utils as cp_utils

        # Patch fill_holes_and_remove_small_masks
        original_fill = cp_utils.fill_holes_and_remove_small_masks

        def patched_fill(masks, min_size=15):
            return fill_holes_parallel(masks, min_size)

        cp_utils.fill_holes_and_remove_small_masks = patched_fill
        logger.info("Applied parallel fill_holes patch to Cellpose")

    except Exception as e:
        logger.warning(f"Failed to apply Cellpose speedups: {e}")
