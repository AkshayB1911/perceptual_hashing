"""White-box attack core: budget, convergence and success, on a tiny dummy model.

The dummy tests need only torch (no downloaded weights), so they run anywhere. The tests
marked ``real`` exercise the actual hashers and are skipped unless their model files exist.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hashbench.adversarial import (  # noqa: E402
    AttackConfig, _BitAdapter, _CertAdapter, _pgd, _apgd, _project, _make_objective, evade, collide,
)


# --- a tiny differentiable "bit hash": a fixed linear map [0,1]^(3x8x8) -> 16 logits ---

class _TinyBitModel(torch.nn.Module):
    def __init__(self, n_in=3 * 8 * 8, bits=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(bits, n_in, generator=g), requires_grad=False)

    def forward(self, x):
        return x.flatten(1) @ self.w.t()


def _pre_8(image):  # unused; adapter.preprocess not called in the dummy tests
    raise NotImplementedError


def _tiny_adapter(seed=0):
    return _BitAdapter(_TinyBitModel(seed=seed), _pre_8, input_size=8)


def _rand_batch(b=4, gen_seed=1):
    g = torch.Generator().manual_seed(gen_seed)
    return torch.rand(b, 3, 8, 8, generator=g)


@pytest.mark.parametrize("norm", ["linf", "l2"])
@pytest.mark.parametrize("method", ["pgd", "apgd"])
def test_respects_budget(norm, method):
    adapter = _tiny_adapter()
    x0 = _rand_batch()
    eps = 0.1
    cfg = AttackConfig(norm=norm, eps=eps, steps=30, method=method, device="cpu", seed=0)
    ref = adapter.reference(x0)
    loss_fn = lambda out: adapter.evasion_loss(out, ref, 0.0)
    gen = torch.Generator().manual_seed(0)
    adv = (_apgd if method == "apgd" else _pgd)(adapter, x0, loss_fn, cfg, gen)
    assert adv.shape == x0.shape
    assert adv.min() >= -1e-6 and adv.max() <= 1 + 1e-6          # valid image range
    delta = adv - x0
    if norm == "linf":
        assert delta.abs().max() <= eps + 1e-5                   # l_inf budget
    else:
        assert delta.flatten(1).norm(dim=1).max() <= eps + 1e-4  # l_2 budget


@pytest.mark.parametrize("method", ["pgd", "apgd"])
def test_evasion_flips_bits(method):
    """A large enough budget must flip a good fraction of the 16 bits."""
    adapter = _tiny_adapter()
    x0 = _rand_batch()
    clean = adapter.hash(x0)
    cfg = AttackConfig(norm="linf", eps=0.25, steps=60, method=method, device="cpu", seed=0)
    ref = adapter.reference(x0)
    loss_fn = lambda out: adapter.evasion_loss(out, ref, 0.0)
    gen = torch.Generator().manual_seed(0)
    adv = (_apgd if method == "apgd" else _pgd)(adapter, x0, loss_fn, cfg, gen)
    flipped = adapter.distance(adapter.hash(adv), clean)
    assert flipped.mean() >= 4                                   # >= 25% of 16 bits, per sample mean


@pytest.mark.parametrize("method", ["pgd", "apgd"])
def test_collision_reduces_distance(method):
    adapter = _tiny_adapter()
    x0 = _rand_batch(gen_seed=1)
    target = _rand_batch(gen_seed=99)
    tgt_ref = adapter.reference(target)
    tgt_hash = adapter.hash(target)
    before = adapter.distance(adapter.hash(x0), tgt_hash)
    cfg = AttackConfig(norm="linf", eps=0.3, steps=80, method=method, device="cpu", seed=0)
    loss_fn = lambda out: adapter.collision_loss(out, tgt_ref, 0.0)
    gen = torch.Generator().manual_seed(0)
    adv = (_apgd if method == "apgd" else _pgd)(adapter, x0, loss_fn, cfg, gen)
    after = adapter.distance(adapter.hash(adv), tgt_hash)
    assert after.mean() < before.mean()                          # moved toward the target


class _SkewedBitModel(_TinyBitModel):
    """Like _TinyBitModel, but half the bits have ~1000x larger logits (as trained DinoHash's
    +-500 logits do), so the lowest hinge loss is not the iterate with the most flipped bits."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        scale = torch.ones(self.w.shape[0], 1)
        scale[::2] = 1000.0
        self.w.data *= scale


@pytest.mark.parametrize("mode", ["evasion", "collision"])
def test_objective_selection_never_worse_than_loss_selection(mode):
    """PGD's trajectory does not depend on the selection rule, so picking the iterate by hash
    distance must give a distance at least as good as picking it by loss, for every sample."""
    adapter = _BitAdapter(_SkewedBitModel(), _pre_8, input_size=8)
    x0 = _rand_batch(b=6)
    if mode == "evasion":
        ref = adapter.reference(x0)
        ref_h = adapter.hash_t(adapter.outputs(x0))
        loss_fn = lambda out: adapter.evasion_loss(out, ref, 0.0)
    else:
        xt = _rand_batch(b=6, gen_seed=7)
        ref = adapter.reference(xt)
        ref_h = adapter.hash_t(adapter.outputs(xt))
        loss_fn = lambda out: adapter.collision_loss(out, ref, 0.0)
    obj = _make_objective(adapter, ref_h, mode, quantize=False)
    cfg = AttackConfig(norm="linf", eps=0.05, steps=40, method="pgd", device="cpu", seed=0)
    by_loss = _pgd(adapter, x0, loss_fn, cfg, torch.Generator().manual_seed(0))
    by_obj = _pgd(adapter, x0, loss_fn, cfg, torch.Generator().manual_seed(0), objective=obj)
    with torch.no_grad():
        assert (obj(by_obj) <= obj(by_loss)).all()


@pytest.mark.parametrize("method", ["pgd", "apgd"])
def test_evasion_never_worse_than_clean(method):
    adapter = _BitAdapter(_SkewedBitModel(), _pre_8, input_size=8)
    x0 = _rand_batch()
    ref = adapter.reference(x0)
    ref_h = adapter.hash_t(adapter.outputs(x0))
    obj = _make_objective(adapter, ref_h, "evasion", quantize=True)
    cfg = AttackConfig(norm="linf", eps=0.02, steps=20, method=method, device="cpu", seed=0)
    adv = (_apgd if method == "apgd" else _pgd)(adapter, x0, lambda o: adapter.evasion_loss(o, ref, 0.0),
                                                 cfg, torch.Generator().manual_seed(0), objective=obj)
    with torch.no_grad():
        assert (obj(adv) <= obj(x0)).all()


def test_project_l2_shrinks_only_when_outside():
    inside = torch.full((2, 3, 4, 4), 0.001)
    out = _project(inside, eps=1.0, norm="l2")
    assert torch.allclose(out, inside)                           # untouched inside the ball
    big = torch.full((2, 3, 4, 4), 1.0)
    out = _project(big, eps=1.0, norm="l2")
    assert torch.allclose(out.flatten(1).norm(dim=1), torch.ones(2), atol=1e-5)


def test_zero_budget_is_a_no_op():
    adapter = _tiny_adapter()
    x0 = _rand_batch()
    cfg = AttackConfig(norm="linf", eps=0.0, steps=10, method="pgd", device="cpu",
                       random_start=False, seed=0)
    ref = adapter.reference(x0)
    loss_fn = lambda out: adapter.evasion_loss(out, ref, 0.0)
    gen = torch.Generator().manual_seed(0)
    adv = _pgd(adapter, x0, loss_fn, cfg, gen)
    assert torch.allclose(adv, x0, atol=1e-6)


# --------------------------------------------------------------------------------------
# real-model smoke tests (skipped without weights)
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).parent
SAMPLES = sorted((ROOT.parent / "data" / "samples").glob("*"))


def _sample_images(n=2):
    from PIL import Image
    if len(SAMPLES) < n:
        pytest.skip("no sample images")
    return [Image.open(p).convert("RGB") for p in SAMPLES[:n]]


@pytest.mark.parametrize("name", ["dinohash", "neuralhash", "certphash_adv"])
def test_real_evasion_smoke(name):
    from hashbench.hashes import get_hasher
    try:
        hasher = get_hasher(name)
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"{name} unavailable: {e}")
    img = _sample_images(1)[0]
    # small, cheap attack just to confirm the pipeline runs and the budget holds
    cfg = AttackConfig(norm="linf", eps=8 / 255, steps=10, method="pgd", device="cpu", seed=0)
    res = evade(hasher, img, cfg, threshold=0.0)
    # linf is measured on the realizable 8-bit image, so allow half a quantisation level
    assert res.linf <= 8 / 255 + 0.5 / 255 + 1e-4
    assert res.distance >= res.orig_distance                      # evasion never gets closer
    assert 0.0 <= res.ssim <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", ["evasion", "collision"])
def test_warm_start_is_monotone_in_eps(mode):
    """Warm-starting eps_k from the eps_{k-1} result can never do worse (the start is a candidate)."""
    from hashbench.adversarial import _run
    adapter = _BitAdapter(_SkewedBitModel(), _pre_8, input_size=8)
    x0 = _rand_batch(b=6)
    src = x0 if mode == "evasion" else _rand_batch(b=6, gen_seed=7)
    ref = adapter.reference(src)
    ref_h = adapter.hash_t(adapter.outputs(src))
    lf = (lambda o: adapter.evasion_loss(o, ref, 0.0)) if mode == "evasion" else \
         (lambda o: adapter.collision_loss(o, ref, 0.0))
    obj = _make_objective(adapter, ref_h, mode, quantize=True)
    prev, prev_obj = None, None
    for eps in [0.01, 0.02, 0.04, 0.08]:
        cfg = AttackConfig(norm="linf", eps=eps, steps=15, method="ensemble", device="cpu", seed=0)
        adv = _run(adapter, x0, lf, cfg, torch.Generator().manual_seed(0), obj, prev)
        adv = adv.mul(255).round().div(255)          # what the sweep carries forward
        with torch.no_grad():
            o = obj(adv)
        assert (adv - x0).abs().max() <= eps + 0.5 / 255 + 1e-6
        if prev_obj is not None:
            assert (o <= prev_obj).all()
        prev, prev_obj = adv, o


def test_ensemble_default_step_is_small():
    assert AttackConfig(eps=8 / 255, method="ensemble").default_step_size() == pytest.approx(0.25 / 255)
    assert AttackConfig(eps=1 / 255, method="ensemble").default_step_size() == pytest.approx(0.1 / 255)
