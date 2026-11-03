"""pHash: DCT-based perceptual hash (Zauner 2010; hackerfactor "Looks Like It").

Pipeline (defaults give the common 64-bit pHash):
    1. Convert to grayscale.
    2. Resize to (hash_size * highfreq_factor)^2 = 32x32, discarding high frequencies
       and making the hash independent of the input resolution / aspect ratio.
    3. 2-D DCT-II of the 32x32 image.
    4. Keep the top-left hash_size x hash_size (8x8) block: the lowest frequencies,
       which carry the coarse structure of the image.
    5. Each bit = coefficient > median of the block. Thresholding at the median gives
       a balanced hash (about half the bits set) regardless of image brightness.

The defaults reproduce ``imagehash.phash`` bit-for-bit; see
``scripts/validate_against_reference.py``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from hashbench.hashes.base import PerceptualHash
from hashbench.hashes.dct import dct2_matrix, dct2d_lowfreq


class PHash(PerceptualHash):
    name = "phash"

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4,
                 resample: Image.Resampling = Image.Resampling.LANCZOS):
        if hash_size < 2:
            raise ValueError("hash_size must be >= 2")
        self.hash_size = hash_size
        self.highfreq_factor = highfreq_factor
        self.img_size = hash_size * highfreq_factor
        self.resample = resample
        self.n_bits = hash_size * hash_size
        # Only the first hash_size DCT frequencies are ever used.
        self._dct = dct2_matrix(self.img_size, 0, hash_size)

    def lowfreq_dct(self, image: Image.Image) -> np.ndarray:
        """The hash_size x hash_size low-frequency DCT block (before thresholding)."""
        gray = image.convert("L").resize((self.img_size, self.img_size), self.resample)
        pixels = np.asarray(gray, dtype=np.float64)
        return dct2d_lowfreq(pixels, self._dct)

    def hash(self, image: Image.Image) -> np.ndarray:
        coeffs = self.lowfreq_dct(image)
        return (coeffs > np.median(coeffs)).flatten()
