"""Tests for the NES black-box attack (hashbench.blackbox) on a small dummy bit hash."""

import torch
import torch.nn as nn

from hashbench.adversarial import _BitAdapter
from hashbench.blackbox import HashService, NESConfig, nes_attack


class _Dummy(nn.Module):
    """16 random hyperplanes over a 4x4 average-pooled image: a smooth, easy bit hash."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = nn.Parameter(torch.randn(16, 3 * 4 * 4, generator=g), requires_grad=False)

    def forward(self, x):
        p = nn.functional.adaptive_avg_pool2d(x - 0.5, 4).flatten(1)
        return p @ self.w.t()


def _setup(b=3):
    adapter = _BitAdapter(_Dummy(), lambda im: im, 32)
    service = HashService(adapter, chunk=64)
    g = torch.Generator().manual_seed(1)
    x0 = (torch.rand(b, 3, 32, 32, generator=g) * 255).round() / 255
    return service, x0


def test_evasion_budget_and_ball():
    service, x0 = _setup()
    cfg = NESConfig(eps=16 / 255, max_queries=1500, pairs=10, dim=8, seed=0)
    res = nes_attack(service, x0, service.hashes(x0), "evasion", threshold=2, cfg=cfg)
    for r in res:
        assert r.queries <= cfg.max_queries
        assert r.linf <= cfg.eps + 0.5 / 255 + 1e-6          # 8-bit rounding only
        assert r.distance >= r.orig_distance == 0
        assert ((r.adv * 255).round() - r.adv * 255).abs().max() < 1e-4   # 8-bit image
    assert any(r.success for r in res)


def test_collision_already_matched_costs_one_query():
    service, x0 = _setup(2)
    ref = service.hashes(x0)                    # target = own hash -> distance 0 already
    cfg = NESConfig(eps=8 / 255, max_queries=500, pairs=5, dim=8)
    res = nes_attack(service, x0, ref, "collision", threshold=0, cfg=cfg)
    assert all(r.success and r.queries == 1 and r.steps == 0 for r in res)


def test_failure_spends_at_most_budget():
    service, x0 = _setup(2)
    cfg = NESConfig(eps=1 / 255, max_queries=250, pairs=5, dim=8)
    res = nes_attack(service, x0, service.hashes(x0), "evasion", threshold=16, cfg=cfg)  # impossible
    for r in res:
        assert not r.success
        assert r.queries == 1 + r.steps * cfg.queries_per_step <= cfg.max_queries
