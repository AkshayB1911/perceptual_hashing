from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hashbench.attacks import ATTACK_SUITE, apply_attack
from hashbench.hashes import PDQHash, PHash, from_hex, get_hasher, to_hex
from hashbench.hashes.pdq import DIHEDRAL_NAMES

SAMPLES = sorted((Path(__file__).parent.parent / "data" / "samples").glob("*"))
imagehash = pytest.importorskip("imagehash")
pdqhash = pytest.importorskip("pdqhash")


@pytest.fixture(params=SAMPLES, ids=lambda p: p.name)
def image(request):
    return Image.open(request.param).convert("RGB")


def test_phash_matches_imagehash(image):
    assert np.array_equal(PHash().hash(image), imagehash.phash(image).hash.flatten())


@pytest.mark.parametrize("hash_size", [8, 16])
def test_phash_matches_imagehash_other_sizes(image, hash_size):
    ours = PHash(hash_size=hash_size).hash(image)
    assert np.array_equal(ours, imagehash.phash(image, hash_size=hash_size).hash.flatten())


def test_pdq_matches_reference(image):
    ref_bits, ref_q = pdqhash.compute(np.asarray(image))
    bits, q = PDQHash().hash_with_quality(image)
    assert np.array_equal(bits, ref_bits.astype(bool))
    assert q == ref_q


def test_pdq_dihedral_matches_reference(image):
    ref, _ = pdqhash.compute_dihedral(np.asarray(image))
    ours, _ = PDQHash().hash_dihedral(image)
    assert list(ours) == list(DIHEDRAL_NAMES)
    for name, r in zip(DIHEDRAL_NAMES, ref):
        assert np.array_equal(ours[name], r.astype(bool)), name


def test_pdq_dihedral_flip_matches_hash_of_flipped_image(image):
    pdq = PDQHash()
    flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    dih, _ = pdq.hash_dihedral(flipped)
    # One of the dihedral variants of the flipped image matches the original. Not exactly:
    # PDQ's box filter window is off-centre by one pixel, so flipping is not perfectly
    # symmetric. 31 is PDQ's recommended match threshold.
    assert min(pdq.distance(h, pdq.hash(image)) for h in dih.values()) <= 31


@pytest.mark.parametrize("name, n_bits", [("phash", 64), ("pdq", 256)])
def test_hash_shape_and_balance(image, name, n_bits):
    h = get_hasher(name).hash(image)
    assert h.dtype == bool and h.shape == (n_bits,)
    assert abs(h.sum() - n_bits / 2) <= n_bits / 4  # median threshold => roughly balanced


def test_hex_roundtrip():
    bits = np.random.default_rng(0).random(256) > 0.5
    assert np.array_equal(from_hex(to_hex(bits)), bits)


@pytest.mark.parametrize("attack", sorted(ATTACK_SUITE))
def test_attacks_run(attack):
    img = Image.open(SAMPLES[0]).convert("RGB")
    for param in ATTACK_SUITE[attack]:
        out = apply_attack(img, attack, param)
        assert out.mode == "RGB" and out.width > 0 and out.height > 0
