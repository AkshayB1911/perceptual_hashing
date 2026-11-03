"""PDQ against Meta's own reference outputs and against Meta's C++ core (via ``pdqhash``).

tests/data/meta_pdq holds test images and expected hashes from facebook/ThreatExchange
(BSD licence, see LICENSE-ThreatExchange there; provenance in expected.json).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hashbench.hashes import from_hex
from hashbench.hashes.pdq import PDQHash, bits_from_dct

META = Path(__file__).parent / "data" / "meta_pdq"
EXPECTED = json.loads((META / "expected.json").read_text())
pdqhash = pytest.importorskip("pdqhash")


def _hd(a, b):
    return int(np.count_nonzero(a != b))


@pytest.mark.parametrize("name, expected", sorted(EXPECTED["python_reference"]["hashes"].items()))
def test_matches_meta_python_reference(name, expected):
    """Same preprocessing as Meta's Python reference (thumbnail to 512, Pillow decoder).

    Exact for every image except near-flat ones (quality < 50), where the bits are
    decided by tiny pixel differences between JPEG decoder versions. Meta itself
    says such hashes should be discarded.
    """
    path = META / name
    if not path.exists():
        pytest.skip(f"{name} not vendored")
    bits, quality = PDQHash(max_side=512).hash_with_quality(Image.open(path))
    d = _hd(bits, from_hex(expected))
    if quality >= 50:
        assert d == 0
    else:
        assert d <= 31, f"quality {quality}"


@pytest.mark.parametrize("name, expected", sorted(EXPECTED["cpp_reference"]["hashes"].items()))
def test_close_to_meta_cpp_tool(name, expected):
    """Meta's C++ tool decodes with CImg and squashes to 512x512 (nearest neighbour).

    We can't reproduce its JPEG decoder bit-for-bit, so we only require a close match;
    Meta's own Python and C++ references differ by the same amount on these images.
    """
    img = Image.open(META / name).convert("RGB")
    if max(img.size) > 512:
        img = img.resize((512, 512), Image.Resampling.NEAREST)
    bits, quality = PDQHash().hash_with_quality(img)
    assert _hd(bits, from_hex(expected["hash"])) <= 10
    assert abs(quality - expected["quality"]) <= 2


def _wheel_luma(rgb):
    # Exactly what the pdqhash bindings compute before calling Meta's C++ core.
    return (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype("float32")


def _degenerate_images():
    rng = np.random.default_rng(0)
    yield "flat_0", np.zeros((100, 100, 3))
    yield "flat_128", np.full((100, 100, 3), 128)
    yield "flat_255", np.full((100, 100, 3), 255)
    for n in (64, 150, 333):
        yield f"h_gradient_{n}", np.tile(np.linspace(0, 255, n)[None, :, None], (150, 1, 3))
    yield "v_gradient", np.tile(np.linspace(0, 255, 150)[:, None, None], (1, 200, 3))
    for i in range(5):
        h, w = rng.integers(5, 300, 2)
        yield f"noise_{i}", rng.integers(0, 256, (h, w, 3))
        yield f"blocks_{i}", np.kron(rng.integers(0, 2, (8, 8)) * 255, np.ones((16, 16)))[:, :, None].repeat(3, 2)
    for h, w in [(5, 5), (5, 300), (63, 64), (64, 64), (65, 65), (700, 6)]:
        yield f"size_{h}x{w}", rng.integers(0, 256, (h, w, 3))


@pytest.mark.parametrize("name, arr", list(_degenerate_images()), ids=lambda x: x if isinstance(x, str) else "")
def test_core_bit_exact_with_meta_cpp(name, arr):
    """Given the same luma, our core reproduces Meta's C++ core bit-for-bit, even on
    images whose bits are decided purely by float rounding (flat, gradients, ...)."""
    rgb = np.ascontiguousarray(arr.astype(np.uint8))
    ref_bits, ref_q = pdqhash.compute(rgb)
    dct, q = PDQHash().dct_and_quality_from_luma(_wheel_luma(rgb))
    assert np.array_equal(bits_from_dct(dct), ref_bits.astype(bool))
    assert q == ref_q


@pytest.mark.parametrize("size", [(1, 1), (4, 4), (3, 7), (1, 500), (500, 4)])
def test_too_small_gives_zero_hash(size):
    rgb = np.random.default_rng(0).integers(0, 256, size + (3,)).astype(np.uint8)
    ref_bits, ref_q = pdqhash.compute(rgb)
    bits, q = PDQHash().hash_with_quality(Image.fromarray(rgb))
    assert not bits.any() and q == 0
    assert np.array_equal(bits, ref_bits.astype(bool)) and q == ref_q


def test_grayscale_uses_grey_values_directly():
    """Meta's C++ uses the grey value itself as luma for 1-channel images."""
    grey = Image.open(META / "q0004.jpg").convert("L")
    dct, _ = PDQHash().dct_and_quality(grey)
    ref, _ = PDQHash().dct_and_quality_from_luma(np.asarray(grey, dtype=np.float32))
    assert np.array_equal(dct, ref)
