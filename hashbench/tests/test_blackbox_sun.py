"""Tests for the Sun et al. 2026 NES replication (hashbench/blackbox_sun.py)."""

import numpy as np
import torch
from PIL import Image

from hashbench.blackbox_sun import PILResize, SunNESConfig, sun_nes_evasion, to_tensor


def _random_image(seed, size=512):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)).astype(int)
    img = img + rng.integers(-40, 41, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_pil_resize_is_exact():
    for n_out, rs in [(224, Image.Resampling.BILINEAR), (360, Image.Resampling.BICUBIC)]:
        r = PILResize(512, 512, n_out, rs, "cpu")
        for s in range(2):
            im = _random_image(s)
            ref = np.asarray(im.resize((n_out, n_out), rs))
            got = (r(to_tensor(im)[None])[0] * 255).round().byte().permute(1, 2, 0).numpy()
            assert (ref == got).all()


class _ToyService:
    """96 'bits' = signs of fixed random projections of the (centred, downsampled) image."""

    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(96, 3 * 64 * 64, generator=g)

    def _h(self, x):
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode="area").flatten(1)
        return (x - x.mean(1, keepdim=True)) @ self.w.t() >= 0

    def hashes(self, x):
        return self._h(x)

    def distances(self, x, ref):
        return (self._h(x) != ref).sum(1).float()


def test_budget_and_query_limit():
    svc = _ToyService()
    x0 = torch.stack([to_tensor(_random_image(s)) for s in range(2)])
    ref = svc.hashes(x0)
    cfg = SunNESConfig(l2=20.0, max_queries=300, chunk=32, seed=0)
    for r in sun_nes_evasion(svc, x0, ref, threshold=1000.0, cfg=cfg):   # never succeeds
        assert r.queries <= 300 and r.steps == (300 - 1) // 65
        assert r.l2 <= 20.0 + 0.5 / 255 * (3 * 512 * 512) ** 0.5       # cap + 8-bit rounding
