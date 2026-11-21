"""Tests for the classical-hash (pHash / PDQ) adapter of the NES attack."""

import numpy as np
import pytest
import torch
from PIL import Image

from hashbench.blackbox import HashService, NESConfig, nes_attack
from hashbench.blackbox_classical import ClassicalAdapter
from hashbench.hashes import get_hasher


def _image(w=120, h=90, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)


@pytest.mark.parametrize("name", ["phash", "pdq"])
def test_adapter_hash_matches_hasher(name):
    hasher = get_hasher(name)
    adapter = ClassicalAdapter(hasher)
    img = _image()
    x = adapter.preprocess(img)
    assert x.shape == (3, 90, 120)
    h = adapter.hash_t(adapter.outputs(x[None]))[0].numpy()
    assert np.array_equal(h, hasher.hash(img))


def test_nes_on_phash_non_square():
    adapter = ClassicalAdapter(get_hasher("phash"))
    service = HashService(adapter, chunk=64)
    x0 = adapter.preprocess(_image(seed=3))[None]
    cfg = NESConfig(eps=16 / 255, max_queries=600, pairs=5, dim=8, seed=0)
    r = nes_attack(service, x0, service.hashes(x0), "evasion", threshold=2, cfg=cfg)[0]
    assert r.queries <= cfg.max_queries
    assert r.linf <= cfg.eps + 0.5 / 255 + 1e-6
    assert r.adv.shape == (3, 90, 120)
    assert r.distance >= 1                       # pHash moves easily under 16/255 noise
    assert torch.equal(service.hashes(x0), service.hashes(x0.clone()))


def test_separable_upsample_equals_interpolate():
    from hashbench.blackbox import _bilinear_matrix, _upsample
    z = torch.randn(2, 3, 32, 32)
    rh, rw = _bilinear_matrix(32, 375, "cpu"), _bilinear_matrix(32, 500, "cpu")
    assert torch.allclose(rh @ z @ rw.t(), _upsample(z, 375, 500), atol=1e-5)
