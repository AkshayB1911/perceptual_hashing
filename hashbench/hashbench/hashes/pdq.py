"""PDQ: Meta's 256-bit perceptual hash (ThreatExchange, 2019).

PDQ is pHash with each step made more careful. Pipeline:
    1. RGB -> luma Y = 0.299 R + 0.587 G + 0.114 B (grayscale input is used as-is).
    2. Jarosz filter: two passes of a separable box blur. The window size is chosen
       so that downsampling to 64x64 does not alias (unlike a plain resize).
    3. Decimate to 64x64 by sampling the centre of each output cell.
    4. Quality score in [0, 100] from the sum of absolute gradients of the 64x64
       image. Flat / low-detail images get low quality; their hashes are unreliable
       (Meta recommends discarding hashes with quality < 50).
    5. 2-D DCT, keeping frequencies 1..16 in each direction (the DC term is skipped),
       giving a 16x16 block.
    6. Each bit = coefficient > median of the 256 coefficients.

This follows Meta's C++ reference (ThreatExchange ``pdq/cpp/hashing/pdqhashing.cpp``
and ``pdq/cpp/downscaling/downscaling.cpp``) operation by operation, including its
float32 arithmetic and summation order. That matters only for degenerate images
(flat, synthetic, perfectly symmetric) whose DCT coefficients are all ~0, where the
bits are decided by rounding; for natural images any careful implementation agrees.
The bit order and hex form match the reference (``pdq-photo-hasher``, ``pdqhash``).

Image loading is *not* part of the algorithm: Meta's C++ tool squashes images larger
than 512 px to 512x512, its Python reference shrinks them to fit in 512x512 keeping
the aspect ratio. Pass ``max_side=512`` to reproduce the latter.

PDQ can also produce the hashes of the 8 dihedral transforms (rotations by 90
degrees and flips) almost for free, by changing signs / transposing the DCT block
instead of re-hashing; see ``hash_dihedral``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from hashbench.hashes.base import PerceptualHash

F32 = np.float32
LUMA_R, LUMA_G, LUMA_B = F32(0.299), F32(0.587), F32(0.114)
DOWNSAMPLE_DIM = 64
DCT_DIM = 16
NUM_JAROSZ_PASSES = 2
MIN_HASHABLE_DIM = 5


def _pdq_dct_matrix() -> np.ndarray:
    """PDQ's 16x64 DCT matrix: frequencies 1..16 (no DC), orthonormal scaling sqrt(2/64).

    Built exactly as the C++ does: float scale factor times a double cosine, stored as float.
    """
    scale = float(F32(np.sqrt(2.0 / DOWNSAMPLE_DIM)))
    i = np.arange(1, DCT_DIM + 1, dtype=np.float64)[:, None]
    j = np.arange(DOWNSAMPLE_DIM, dtype=np.float64)[None, :]
    pi = 3.14159265358979323846
    return (scale * np.cos((pi / 2.0 / DOWNSAMPLE_DIM) * i * (2 * j + 1))).astype(F32)


_PDQ_DCT = _pdq_dct_matrix()

DIHEDRAL_NAMES = (
    "original", "rotate90", "rotate180", "rotate270",
    "flip_x", "flip_y", "flip_plus1", "flip_minus1",
)


def to_luma(image: Image.Image) -> np.ndarray:
    """Float32 luma, as ``fillFloatLumaFromRGB`` / ``fillFloatLumaFromGrey``."""
    if image.mode == "L":
        return np.asarray(image, dtype=F32)
    rgb = np.asarray(image.convert("RGB"), dtype=F32)
    return LUMA_R * rgb[..., 0] + LUMA_G * rgb[..., 1] + LUMA_B * rgb[..., 2]


def jarosz_window_size(old_dim: int, new_dim: int = DOWNSAMPLE_DIM) -> int:
    """Box-filter window for downsampling old_dim -> new_dim (about old / (2 * new))."""
    return (old_dim + 2 * new_dim - 1) // (2 * new_dim)


def box_filter_1d(x: np.ndarray, window: int, axis: int) -> np.ndarray:
    """Running-mean box filter along ``axis``, exactly as PDQ's ``box1DFloat``.

    Output i is the mean of inputs [i - (window - half), i + half - 1], clipped to
    the array, where half = (window + 2) // 2; near the borders the window shrinks
    (no padding). The running sum is updated in float32 in the same order as the
    C++ code; the loop runs along ``axis`` and is vectorised over the other axis.
    """
    v = np.ascontiguousarray(x if axis == 0 else x.T, dtype=F32)
    n = v.shape[0]
    half = (window + 2) // 2
    out = np.empty_like(v)
    total = np.zeros(v.shape[1:], dtype=F32)
    li = ri = oi = size = 0
    for _ in range(half - 1):                      # phase 1: fill, no output
        total += v[ri]; size += 1; ri += 1
    for _ in range(window - half + 1):             # phase 2: growing window
        total += v[ri]; size += 1; ri += 1
        out[oi] = total / F32(size); oi += 1
    for _ in range(n - window):                    # phase 3: full window
        total += v[ri]; total -= v[li]; ri += 1; li += 1
        out[oi] = total / F32(size); oi += 1
    for _ in range(half - 1):                      # phase 4: shrinking window
        total -= v[li]; size -= 1; li += 1
        out[oi] = total / F32(size); oi += 1
    return out if axis == 0 else out.T


def jarosz_filter(luma: np.ndarray, passes: int = NUM_JAROSZ_PASSES) -> np.ndarray:
    rows, cols = luma.shape
    win_along_rows = jarosz_window_size(cols)
    win_along_cols = jarosz_window_size(rows)
    out = luma.astype(F32)
    for _ in range(passes):
        out = box_filter_1d(out, win_along_rows, axis=1)
        out = box_filter_1d(out, win_along_cols, axis=0)
    return out


def decimate(x: np.ndarray, out_rows: int = DOWNSAMPLE_DIM, out_cols: int = DOWNSAMPLE_DIM) -> np.ndarray:
    rows, cols = x.shape
    ri = (((np.arange(out_rows) + 0.5) * rows) / out_rows).astype(int)
    ci = (((np.arange(out_cols) + 0.5) * cols) / out_cols).astype(int)
    return x[np.ix_(ri, ci)]


def downsample_64(luma: np.ndarray) -> np.ndarray:
    """Blur and decimate to 64x64. A 64x64 input is used directly, as in the reference."""
    if luma.shape == (DOWNSAMPLE_DIM, DOWNSAMPLE_DIM):
        return luma.astype(F32)
    return decimate(jarosz_filter(luma))


def quality_metric(img64: np.ndarray) -> int:
    """Gradient-based image quality in [0, 100], as in ``pdqImageDomainQualityMetric``."""
    img64 = img64.astype(F32)
    dv = np.trunc((img64[:-1, :] - img64[1:, :]) * F32(100) / F32(255))
    dh = np.trunc((img64[:, :-1] - img64[:, 1:]) * F32(100) / F32(255))
    gradient_sum = int(np.abs(dv).sum(dtype=np.int64) + np.abs(dh).sum(dtype=np.int64))
    return min(gradient_sum // 90, 100)


def dct_64_to_16(img64: np.ndarray) -> np.ndarray:
    """B = D A D^T with float32 accumulation in the same order as ``dct64To16``."""
    a = img64.astype(F32)
    d = _PDQ_DCT
    t = np.zeros((DCT_DIM, DOWNSAMPLE_DIM), dtype=F32)   # T = D A
    for k in range(DOWNSAMPLE_DIM):
        t += d[:, k, None] * a[None, k, :]
    b = np.zeros((DCT_DIM, DCT_DIM), dtype=F32)          # B = T D^T
    for k in range(DOWNSAMPLE_DIM):
        b += t[:, k, None] * d[None, :, k]
    return b


def lower_median(values: np.ndarray) -> float:
    """The ceil(n/2)-th smallest value, which is what PDQ's torben median returns."""
    flat = values.ravel()
    k = (flat.size + 1) // 2 - 1
    return np.partition(flat, k)[k]


def bits_from_dct(dct16: np.ndarray) -> np.ndarray:
    """Threshold the 16x16 DCT block into 256 bits, in reference (MSB-first) order.

    The reference sets bit i*16 + j for coefficient [i, j] and prints bit 255 first,
    so we reverse the row-major flattening.
    """
    return (dct16 > lower_median(dct16)).ravel()[::-1].copy()


class PDQHash(PerceptualHash):
    name = "pdq"
    n_bits = DCT_DIM * DCT_DIM

    def __init__(self, max_side: int | None = None):
        """``max_side``: if set, shrink larger images to fit in max_side x max_side
        (keeping the aspect ratio) before hashing, like Meta's Python reference does
        with 512. Default: hash the pixels as given, like the C++ core / ``pdqhash``.
        """
        self.max_side = max_side
        self.last_quality: int | None = None

    def luma(self, image: Image.Image) -> np.ndarray:
        if self.max_side and max(image.size) > self.max_side:
            image = image.copy()
            image.thumbnail((self.max_side, self.max_side))
        return to_luma(image)

    def dct_and_quality_from_luma(self, luma: np.ndarray) -> tuple[np.ndarray | None, int]:
        """16x16 DCT block and quality; (None, 0) if the image is too small to hash."""
        if min(luma.shape) < MIN_HASHABLE_DIM:
            return None, 0
        img64 = downsample_64(luma)
        return dct_64_to_16(img64), quality_metric(img64)

    def dct_and_quality(self, image: Image.Image) -> tuple[np.ndarray | None, int]:
        return self.dct_and_quality_from_luma(self.luma(image))

    def hash_with_quality(self, image: Image.Image) -> tuple[np.ndarray, int]:
        dct16, quality = self.dct_and_quality(image)
        if dct16 is None:  # reference returns the all-zero hash
            return np.zeros(self.n_bits, dtype=bool), quality
        return bits_from_dct(dct16), quality

    def hash(self, image: Image.Image) -> np.ndarray:
        bits, self.last_quality = self.hash_with_quality(image)
        return bits

    def hash_dihedral(self, image: Image.Image) -> tuple[dict[str, np.ndarray], int]:
        """Hashes of the 8 dihedral transforms of ``image``, computed from one DCT.

        Flipping the image negates the odd-frequency coefficients along that axis
        (our frequencies are 1..16, so odd frequency = even index), and transposing
        the image transposes the DCT block. Rotations are compositions of the two.
        """
        b, quality = self.dct_and_quality(image)
        if b is None:
            return {k: np.zeros(self.n_bits, dtype=bool) for k in DIHEDRAL_NAMES}, quality
        sign = np.where(np.arange(DCT_DIM) % 2 == 0, F32(-1), F32(1))  # frequency k = idx + 1
        flip_rows = sign[:, None] * b   # image flipped top <-> bottom
        flip_cols = b * sign[None, :]   # image flipped left <-> right
        both = sign[:, None] * b * sign[None, :]
        blocks = {
            "original": b,
            "rotate90": flip_cols.T,
            "rotate180": both,
            "rotate270": flip_rows.T,
            "flip_x": flip_rows,
            "flip_y": flip_cols,
            "flip_plus1": b.T,
            "flip_minus1": both.T,
        }
        return {k: bits_from_dct(v) for k, v in blocks.items()}, quality

    def dihedral_min_distance(self, query: Image.Image, reference_hash: np.ndarray) -> int:
        """Rotation/flip-invariant matching: min distance over the 8 dihedral hashes."""
        hashes, _ = self.hash_dihedral(query)
        return min(self.distance(h, reference_hash) for h in hashes.values())
