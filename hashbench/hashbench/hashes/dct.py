"""Explicit DCT-II basis matrices.

Both pHash and PDQ keep only the low-frequency corner of a 2-D DCT. Writing the
transform as a matrix product ``D @ X @ D.T`` makes that explicit: we only build
the rows of ``D`` for the frequencies we keep.
"""

from __future__ import annotations

import numpy as np


def dct2_matrix(n: int, k_start: int = 0, k_stop: int | None = None, scale: float = 2.0) -> np.ndarray:
    """Rows ``k_start..k_stop-1`` of the (unnormalised) DCT-II matrix of size ``n``.

    ``D[k, j] = scale * cos(pi * k * (2j + 1) / (2n))``

    With ``scale=2`` this is exactly ``scipy.fft.dct(x, type=2, norm=None)``.
    """
    k_stop = n if k_stop is None else k_stop
    k = np.arange(k_start, k_stop, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    return scale * np.cos(np.pi * k * (2 * j + 1) / (2 * n))


def dct2d_lowfreq(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Separable 2-D DCT restricted to the frequencies in ``d``: ``D @ X @ D.T``."""
    return d @ x @ d.T
