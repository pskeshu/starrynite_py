"""Compatibility patches for dependency version issues."""

import numpy as np


def patch_skimage_readonly_buffer():
    """Fix scikit-image read-only buffer error with ultrack.

    scikit-image 0.25+ uses Cython code that rejects read-only numpy arrays.
    ultrack passes pandas-backed arrays that are read-only, causing ValueError.
    This patch wraps map_array to copy inputs before passing to Cython.
    """
    try:
        import skimage.util._map_array as ma

        _original = ma.map_array

        def _patched(input_arr, input_vals, output_vals, out=None):
            return _original(
                np.asarray(input_arr).copy(),
                np.asarray(input_vals).copy(),
                np.asarray(output_vals).copy(),
                out=out,
            )

        ma.map_array = _patched
    except Exception:
        pass
