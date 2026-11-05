"""White-box adversarial attacks on the differentiable perceptual hashes.

Model-agnostic PGD / APGD *evasion* and *collision* attacks for any hash whose network is
a differentiable ``nn.Module`` mapping [0, 1] RGB images to per-bit logits (DinoHash,
NeuralHash) or to CertPHash's 144 PhotoDNA-style outputs. Classical pHash / PDQ are not
differentiable and are handled later via soft surrogates (thesis_research_plan.md, Phase C).

Threat model and conventions
----------------------------
* The attacker perturbs the *model input* (the preprocessed [0, 1] tensor) inside an
  l_inf or l_2 budget ``eps`` measured in [0, 1] **pixel space**. Every network here applies
  its ImageNet / [-1, 1] normalisation *inside* ``forward``, so attacking the [0, 1] input
  is attacking pixel space directly. This is the same convention as the CertPHash authors'
  certified training (``certified.py`` divides eps by the ImageNet std before adding it to
  the normalised input) and their PGD attack (``adv2_evasion_attack.py`` perturbs [0, 1]
  pixels), so their eps = k/255 and ours are the same box.
* **Evasion**: make an image stop matching its own hash (push the hash away).
* **Collision**: make an image match a chosen target hash (pull the hash onto the target).
* The attack optimises a continuous perturbation; the returned image is then **quantised to
  8 bits** and success is re-checked on the quantised image, so a reported success survives
  being written out as a real PNG rather than living in sub-pixel float values. The reported
  ``linf``/``l2``/``psnr``/``ssim`` are of that realizable 8-bit image, so ``linf`` may exceed
  ``eps`` by up to half a quantisation level (~0.002 at eps = 8/255); the continuous
  perturbation the optimiser produces is strictly within the budget.
* Attacks run at the model's input resolution (224 / 360 / 64). Attacking the full-size
  original and letting the resize happen afterwards is a separate, harder setting; see
  ``AttackConfig`` notes.

GPU: everything honours ``device``; pass ``device="cuda"`` on Kaggle. The DinoHash ViT
backward pass is the expensive part, so batch images and keep ``eot_samples`` small on CPU.

Typical use
-----------
    from hashbench.hashes import get_hasher
    from hashbench.adversarial import AttackConfig, evade, collide

    h = get_hasher("dinohash")                 # or neuralhash, certphash, certphash_adv, ...
    cfg = AttackConfig(norm="linf", eps=8/255, steps=100, method="apgd", device="cuda")
    res = evade(h, image, cfg, threshold=0.18 * h.n_bits)   # threshold in *bits* (or L1 for cert)
    print(res.success, res.distance, res.linf, res.ssim)
    res.adv_image.save("evaded.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hashbench.hashes.base import PerceptualHash


# --------------------------------------------------------------------------------------
# configuration and results
# --------------------------------------------------------------------------------------

@dataclass
class AttackConfig:
    """Hyper-parameters for a white-box attack.

    norm         "linf" or "l2"; the budget geometry.
    eps          perturbation budget in [0, 1] pixel space (l_inf max per pixel, or l_2 norm
                 over the whole image). Report it next to the model input size.
    steps        optimisation iterations.
    method       "pgd" (fixed step), "apgd" (Croce & Hein 2020, auto step size), or
                 "ensemble": APGD *and* small-step PGD from the same start, keeping the better
                 result per sample. Trained DinoHash's loss surface is so rough that APGD's
                 large initial steps (2 eps) stall at ~20 flipped bits while PGD with a step of
                 a fraction of 1/255 flips ~90 (Kaggle diagnostic, 2026-11-04); smoother models
                 (CertPHash) favour APGD. The ensemble covers both.
    step_size    PGD step size; default eps/4 (l_inf) or eps/2 (l_2) for "pgd", and
                 min(eps/10, 0.25/255) for the PGD half of "ensemble". APGD ignores it.
    random_start begin from a random point in the budget ball.
    margin       confidence margin in logit units for the bit-hash hinge losses (0 = only
                 flip the sign). Logit scales differ by model (DinoHash ~+-500,
                 NeuralHash ~+-10), so a non-zero margin is model-specific; keep 0 unless
                 tuning one model.
    eot_samples  Expectation-over-Transformation samples; >1 with ``eot_transform`` set
                 averages the gradient over random transforms (needed to attack randomised
                 defences, Phase C ideas 2/5).
    eot_transform callable mapping a batch tensor -> batch tensor (e.g. add Gaussian noise).
    quantize     round the final image to 8-bit and re-evaluate success (honest reporting).
    device       "cpu", "cuda", or None to auto-select.
    seed         RNG seed for the random start / EOT noise (reproducibility).
    """

    norm: str = "linf"
    eps: float = 8.0 / 255.0
    steps: int = 100
    method: str = "apgd"
    step_size: float | None = None
    random_start: bool = True
    margin: float = 0.0
    eot_samples: int = 1
    eot_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    quantize: bool = True
    device: str | None = None
    seed: int | None = None

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def default_step_size(self) -> float:
        if self.step_size is not None:
            return self.step_size
        if self.method == "ensemble":
            return min(self.eps / 10.0, 0.25 / 255.0) if self.norm == "linf" else self.eps / 10.0
        return self.eps / 4.0 if self.norm == "linf" else self.eps / 2.0


@dataclass
class AttackResult:
    mode: str                       # "evasion" / "collision"
    success: bool                   # after quantisation, if cfg.quantize
    distance: float                 # evasion: d(adv, original); collision: d(adv, target)
    orig_distance: float            # the same distance before the attack (sanity baseline)
    steps_used: int
    linf: float                     # ||adv - clean||_inf in [0, 1]
    l2: float                       # ||adv - clean||_2 in [0, 1]
    psnr: float
    ssim: float
    adv: torch.Tensor | None = field(repr=False)        # (C, H, W) in [0, 1], the model input
    adv_image: Image.Image | None = field(repr=False)   # the quantised PIL image at input size


# --------------------------------------------------------------------------------------
# per-family adapters: expose a uniform differentiable interface
# --------------------------------------------------------------------------------------

class _Adapter:
    """Uniform view of a hasher's differentiable network for the attack core."""

    input_size: int

    def __init__(self, model: torch.nn.Module, preprocess: Callable[[Image.Image], torch.Tensor]):
        self.model = model
        self._preprocess = preprocess

    def to(self, device: torch.device) -> "_Adapter":
        self.model = self.model.to(device).eval()
        return self

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image)

    def outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable network outputs (logits or 144 values) for a batch x in [0, 1]."""
        return self.model(x)

    # --- discrete hash + distance (numpy), matching the PerceptualHash semantics ---
    def hash(self, x: torch.Tensor) -> np.ndarray: ...
    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    # --- differentiable per-sample losses (lower = better for the attacker) ---
    def evasion_loss(self, out: torch.Tensor, ref: torch.Tensor, margin: float) -> torch.Tensor: ...
    def collision_loss(self, out: torch.Tensor, tgt: torch.Tensor, margin: float) -> torch.Tensor: ...

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """Detached target tensor derived from a clean input (signs / rounded outputs)."""
        with torch.no_grad():
            return self._reference(self.outputs(x))

    def _reference(self, out: torch.Tensor) -> torch.Tensor: ...

    # --- the same discrete hash + distance on torch tensors (used to pick the best iterate) ---
    def hash_t(self, out: torch.Tensor) -> torch.Tensor: ...
    def dist_t(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


class _BitAdapter(_Adapter):
    """DinoHash / NeuralHash: outputs are per-bit logits; bit = logit >= 0; Hamming distance."""

    def __init__(self, model, preprocess, input_size):
        super().__init__(model, preprocess)
        self.input_size = input_size

    def hash(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return (self.outputs(x) >= 0).cpu().numpy()

    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.count_nonzero(a != b, axis=-1).astype(np.int64)

    def hash_t(self, out):
        return out >= 0

    def dist_t(self, a, b):
        return (a != b).sum(dim=1).float()

    def _reference(self, out: torch.Tensor) -> torch.Tensor:
        # signs in {-1, +1}; treat exactly-zero logits as +1 so they still get a gradient.
        return torch.where(out >= 0, 1.0, -1.0)

    def evasion_loss(self, out, ref, margin):
        # push logit * original_sign below -margin  =>  flip the bit. Mean over bits.
        return F.relu(margin + out * ref).mean(dim=1)

    def collision_loss(self, out, tgt, margin):
        # pull logit * target_sign above +margin  =>  match the target bit.
        return F.relu(margin - out * tgt).mean(dim=1)


class _CertAdapter(_Adapter):
    """CertPHash: 144 real outputs; hash = round & clip to [0, 255]; L1 distance; thr 1800."""

    input_size = 64

    def __init__(self, model, preprocess):
        super().__init__(model, preprocess)

    def hash(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return np.clip(np.round(self.outputs(x).cpu().numpy()), 0, 255).astype(np.uint8)

    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a.astype(np.int64) - b.astype(np.int64)).sum(axis=-1)

    def hash_t(self, out):
        return out.round().clamp(0, 255)

    def dist_t(self, a, b):
        return (a - b).abs().sum(dim=1)

    def _reference(self, out: torch.Tensor) -> torch.Tensor:
        # continuous target = the clean rounded hash, as float (the quantity L1 is measured to).
        return out.round().clamp(0, 255)

    def evasion_loss(self, out, ref, margin):
        # maximise L1 distance to the clean hash  =>  minimise its negative.
        return -(out - ref).abs().sum(dim=1)

    def collision_loss(self, out, tgt, margin):
        return (out - tgt).abs().sum(dim=1)


def build_adapter(hasher: PerceptualHash) -> _Adapter:
    """Wrap a hasher in the differentiable adapter the attack core expects.

    Works for DinoHash, NeuralHash and CertPHash. Raises for hashes with no differentiable
    network (pHash, PDQ) -- those need the soft surrogates from Phase C.
    """
    from hashbench.hashes.dinohash import DinoHash, preprocess as dino_pre, INPUT_SIZE as DINO_SIZE
    from hashbench.hashes.neuralhash import NeuralHash, preprocess as nh_pre, INPUT_SIZE as NH_SIZE
    from hashbench.hashes.certphash import CertPHash, preprocess as cert_pre

    if isinstance(hasher, DinoHash):
        return _BitAdapter(hasher.model, dino_pre, DINO_SIZE)
    if isinstance(hasher, NeuralHash):
        return _BitAdapter(hasher.model, nh_pre, NH_SIZE)
    if isinstance(hasher, CertPHash):
        return _CertAdapter(hasher.model, cert_pre)
    raise TypeError(
        f"{type(hasher).__name__} has no differentiable network for white-box attacks; "
        "pHash/PDQ need the soft surrogates (Phase C). Use a learned hash.")


def default_threshold(hasher: PerceptualHash) -> float:
    """A reasonable success threshold when the caller gives none.

    CertPHash: the paper's L1 match threshold (1800). Bit hashes: no single canonical value
    (it is calibrated per FPR in the benchmark), so fall back to 'the hash changed at all'
    for evasion / 'exact match' for collision by returning 0, and warn the caller to pass one.
    """
    from hashbench.hashes.certphash import CertPHash, MATCH_THRESHOLD
    if isinstance(hasher, CertPHash):
        return float(MATCH_THRESHOLD)
    return 0.0


# --------------------------------------------------------------------------------------
# image <-> tensor helpers
# --------------------------------------------------------------------------------------

def _to_image(x: torch.Tensor) -> Image.Image:
    """(C, H, W) float in [0, 1] -> 8-bit PIL image."""
    arr = (x.clamp(0, 1).mul(255).round().byte().cpu().permute(1, 2, 0).numpy())
    return Image.fromarray(arr)


def _quantize(x: torch.Tensor) -> torch.Tensor:
    """Round a [0, 1] tensor to the nearest 8-bit level (what saving a PNG would do)."""
    return x.clamp(0, 1).mul(255).round().div(255)


# --------------------------------------------------------------------------------------
# image-quality metrics (per sample, on [0, 1] tensors)
# --------------------------------------------------------------------------------------

def _psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = (a - b).pow(2).flatten(1).mean(dim=1)
    return torch.where(mse > 0, 10.0 * torch.log10(1.0 / mse), torch.full_like(mse, float("inf")))


def _gaussian_window(window: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float32, device=device) - window // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    kernel2d = (g @ g.t())
    return kernel2d.expand(channels, 1, window, window).contiguous()


def _ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM per sample for batches in [0, 1] (Wang et al. 2004; C1/C2 for data range 1)."""
    c = a.shape[1]
    w = _gaussian_window(window, sigma, c, a.device)
    pad = window // 2
    mu_a = F.conv2d(a, w, padding=pad, groups=c)
    mu_b = F.conv2d(b, w, padding=pad, groups=c)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, w, padding=pad, groups=c) - mu_a2
    sb = F.conv2d(b * b, w, padding=pad, groups=c) - mu_b2
    sab = F.conv2d(a * b, w, padding=pad, groups=c) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return ssim_map.flatten(1).mean(dim=1)


# --------------------------------------------------------------------------------------
# attack core (operates on a batch of model inputs x0, minimising a per-sample loss)
# --------------------------------------------------------------------------------------

def _project(delta: torch.Tensor, eps: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        return delta.clamp(-eps, eps)
    flat = delta.flatten(1)
    n = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    factor = (eps / n).clamp(max=1.0)
    return (flat * factor).view_as(delta)


def _random_start(x0: torch.Tensor, eps: float, norm: str, gen: torch.Generator) -> torch.Tensor:
    if norm == "linf":
        d = torch.empty_like(x0).uniform_(-eps, eps, generator=gen)
    else:
        d = torch.randn(x0.shape, generator=gen, device=x0.device, dtype=x0.dtype)
        d = _project(d * eps, eps, norm)
    return d


def _loss_vec(adapter: _Adapter, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor],
              cfg: AttackConfig) -> torch.Tensor:
    if cfg.eot_samples <= 1 or cfg.eot_transform is None:
        return loss_fn(adapter.outputs(x))
    total = None
    for _ in range(cfg.eot_samples):
        lv = loss_fn(adapter.outputs(cfg.eot_transform(x)))
        total = lv if total is None else total + lv
    return total / cfg.eot_samples


def _better(obj, lv, best_obj, best_lv):
    """Per-sample: is (obj, loss) better than the best so far? Objective first, loss breaks ties."""
    return (obj < best_obj) | ((obj == best_obj) & (lv < best_lv))


def _start(x0, cfg, gen, init):
    """Starting point: ``init`` (a warm start, projected into this budget) if given, else a
    random point in the ball (``random_start``) or the clean image."""
    if init is not None:
        return (x0 + _project(init.to(x0) - x0, cfg.eps, cfg.norm)).clamp(0, 1)
    if cfg.random_start:
        return (x0 + _random_start(x0, cfg.eps, cfg.norm, gen)).clamp(0, 1)
    return x0.clone()


def _pgd(adapter, x0, loss_fn, cfg, gen, objective=None, init=None):
    """Fixed-step PGD. Returns the iterate that is best by ``objective`` (lower = better for
    the attacker, e.g. minus the Hamming distance for evasion); the loss only breaks ties.
    Without an objective the loss itself is used (the pre-fix behaviour)."""
    eps, norm, alpha = cfg.eps, cfg.norm, cfg.default_step_size()
    objective = objective or (lambda x: _loss_vec(adapter, x, loss_fn, cfg).detach())
    delta = (_start(x0, cfg, gen, init) - x0).detach().requires_grad_(True)
    best = x0.detach().clone()
    with torch.no_grad():
        best_obj = objective(best)
    best_loss = torch.full((x0.shape[0],), float("inf"), device=x0.device)
    for it in range(cfg.steps + 1):         # the extra pass scores the final iterate
        x_cur = (x0 + delta).clamp(0, 1)
        lv = _loss_vec(adapter, x_cur, loss_fn, cfg)
        grad, = torch.autograd.grad(lv.sum(), delta)
        with torch.no_grad():
            x_cur = x_cur.detach()
            obj = objective(x_cur)
            improved = _better(obj, lv.detach(), best_obj, best_loss)
            best_obj = torch.where(improved, obj, best_obj)
            best_loss = torch.where(improved, lv.detach(), best_loss)
            best[improved] = x_cur[improved]
            if it == cfg.steps:
                break
            if norm == "linf":
                delta -= alpha * grad.sign()
            else:
                g = grad.flatten(1)
                g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta -= alpha * g.view_as(delta)
            delta.copy_(_project(delta, eps, norm))
            delta.copy_((x0 + delta).clamp(0, 1) - x0)
        delta.grad = None
    return best.detach()


def _apgd(adapter, x0, loss_fn, cfg, gen, objective=None, init=None):
    """Auto-PGD (Croce & Hein, ICML 2020), minimisation form with momentum + step halving.

    Simplified relative to the reference AutoAttack code: the step size is halved at a
    checkpoint only when fewer than 75% of the steps since the previous checkpoint improved
    the best loss (the paper's condition 1); condition 2 (step size and best loss both
    unchanged since the last checkpoint) is not checked. On halving, the iterate and its
    gradient restart from the best point, as in the reference code.

    Two "bests" are tracked. The step-size schedule counts improvements of the *loss*, as in
    the paper. The iterate that is returned (and restarted from) is the best by ``objective``
    -- the hash distance the attack is judged on -- with the loss breaking ties. Picking by
    loss alone returned iterates that flipped fewer bits than earlier ones on trained
    DinoHash, whose ~+-500 logits let a few bits dominate the hinge loss.
    """
    eps, norm = cfg.eps, cfg.norm
    n = cfg.steps
    alpha = (2.0 * eps) * torch.ones(x0.shape[0], device=x0.device)   # per-sample step size

    # checkpoints p_j (fractions of n), after Croce & Hein.
    checkpoints, p, p_prev = [], 0.22, 0.0
    while int(p * n) < n:
        checkpoints.append(int(p * n))
        p, p_prev = p + max(p - p_prev - 0.03, 0.06), p
    checkpoints = sorted(set(c for c in checkpoints if 0 < c < n))

    def clip(x):
        return (x0 + _project(x - x0, eps, norm)).clamp(0, 1)

    x = clip(_start(x0, cfg, gen, init)).detach().requires_grad_(True)
    lv = _loss_vec(adapter, x, loss_fn, cfg)
    grad, = torch.autograd.grad(lv.sum(), x)
    x = x.detach()
    objective = objective or (lambda z: _loss_vec(adapter, z, loss_fn, cfg).detach())
    with torch.no_grad():
        best_loss = lv.detach().clone()                       # for the step-size schedule
        best, best_grad = x.clone(), grad.clone()             # best by objective
        best_obj, best_sel_loss = objective(x), lv.detach().clone()
        clean_obj = objective(x0)
        use_clean = clean_obj < best_obj                      # never return worse than no attack
        best_obj = torch.where(use_clean, clean_obj, best_obj)
        best[use_clean] = x0[use_clean]
        best_sel_loss = torch.where(use_clean, torch.full_like(best_sel_loss, float("inf")), best_sel_loss)
    x_prev = x.clone()
    momentum = 0.75
    succ_since = torch.zeros(x0.shape[0], device=x0.device)
    ci = 0

    for k in range(1, n):
        with torch.no_grad():
            a = alpha.view(-1, 1, 1, 1)
            if norm == "linf":
                z = clip(x - a * grad.sign())
            else:
                g = grad.flatten(1)
                g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                z = clip(x - a * g.view_as(grad))
            x_next = clip(x + momentum * (z - x) + (1 - momentum) * (x - x_prev))
            x_prev = x.clone()
        x = x_next.detach().requires_grad_(True)
        lv = _loss_vec(adapter, x, loss_fn, cfg)
        grad, = torch.autograd.grad(lv.sum(), x)
        x = x.detach()
        with torch.no_grad():
            lv = lv.detach()
            improved = lv < best_loss
            succ_since += improved.float()
            best_loss = torch.where(improved, lv, best_loss)
            obj = objective(x)
            better = _better(obj, lv, best_obj, best_sel_loss)
            best_obj = torch.where(better, obj, best_obj)
            best_sel_loss = torch.where(better, lv, best_sel_loss)
            best[better] = x[better]
            best_grad[better] = grad[better]
            if ci < len(checkpoints) and k == checkpoints[ci]:
                period = checkpoints[ci] - (checkpoints[ci - 1] if ci else 0)
                halve = succ_since < 0.75 * period
                alpha = torch.where(halve, alpha / 2.0, alpha)
                x = torch.where(halve.view(-1, 1, 1, 1), best, x)   # restart from best
                grad = torch.where(halve.view(-1, 1, 1, 1), best_grad, grad)
                succ_since.zero_()
                ci += 1
    return best.detach()


# --------------------------------------------------------------------------------------
# high-level API
# --------------------------------------------------------------------------------------

def _run(adapter, x0, loss_fn, cfg, gen, objective=None, init=None):
    if cfg.method != "ensemble":
        return (_apgd if cfg.method == "apgd" else _pgd)(adapter, x0, loss_fn, cfg, gen, objective, init)
    objective = objective or (lambda x: _loss_vec(adapter, x, loss_fn, cfg).detach())
    if init is None:   # both halves start from the same random point
        init = _start(x0, cfg, gen, None).detach()
    a = _apgd(adapter, x0, loss_fn, cfg, gen, objective, init)
    b = _pgd(adapter, x0, loss_fn, cfg, gen, objective, init)
    with torch.no_grad():
        take_b = objective(b) < objective(a)
    return torch.where(take_b.view(-1, *[1] * (x0.dim() - 1)), b, a)


def _make_objective(adapter, ref_hash: torch.Tensor, mode: str, quantize: bool):
    """Per-sample attack objective, lower = better: minus the distance to the clean hash
    (evasion) or the distance to the target hash (collision), on the image that will
    actually be reported -- the 8-bit quantised one when ``quantize``."""
    sign = -1.0 if mode == "evasion" else 1.0

    def objective(x):
        xq = _quantize(x) if quantize else x
        return sign * adapter.dist_t(adapter.hash_t(adapter.outputs(xq)), ref_hash)
    return objective


def _results(mode, adapter, clean, adv, threshold, cfg, target_hash=None,
             keep_images=True) -> list[AttackResult]:
    """One AttackResult per sample of the batch."""
    if cfg.quantize:
        adv = _quantize(adv)
    clean_h, adv_h = adapter.hash(clean), adapter.hash(adv)
    ref_h = target_hash if mode == "collision" else clean_h
    dist = adapter.distance(adv_h, ref_h)
    orig = adapter.distance(clean_h, ref_h)
    success = dist > threshold if mode == "evasion" else dist <= threshold
    delta = adv - clean
    linf = delta.abs().flatten(1).max(dim=1).values
    l2 = delta.flatten(1).norm(dim=1)
    psnr, ssim = _psnr(clean, adv), _ssim(clean, adv)
    out = []
    for i in range(clean.shape[0]):
        out.append(AttackResult(
            mode=mode, success=bool(success[i]), distance=float(dist[i]), orig_distance=float(orig[i]),
            steps_used=cfg.steps, linf=float(linf[i]), l2=float(l2[i]),
            psnr=float(psnr[i]), ssim=float(ssim[i]),
            adv=adv[i].cpu() if keep_images else None,
            adv_image=_to_image(adv[i]) if keep_images else None))
    return out


def attack_batch(hasher: PerceptualHash, images: Sequence[Image.Image], cfg: AttackConfig | None = None,
                 threshold: float | None = None, targets: Sequence[Image.Image] | None = None,
                 adapter: _Adapter | None = None, keep_images: bool = True,
                 init: Sequence[torch.Tensor] | torch.Tensor | None = None) -> list[AttackResult]:
    """Attack a batch of images in one optimisation (the GPU-efficient entry point).

    ``targets=None`` runs evasion; otherwise ``targets[i]`` is the collision target of
    ``images[i]``. ``threshold`` is in the hash's own units (Hamming bits, or L1 for
    CertPHash); ``None`` uses ``default_threshold``. Pass a prebuilt ``adapter`` (from
    ``build_adapter(h).to(device)``) to avoid moving the model for every batch, and
    ``keep_images=False`` to drop the adversarial tensors when only the numbers are needed.
    Samples are optimised independently: the summed loss has per-sample gradients, and APGD
    keeps a per-sample step size. ``init`` (one model-input tensor per image, e.g. the
    adversarial images of a smaller budget) warm-starts the attack instead of a random start;
    since the start is a candidate for the best iterate, the result is never worse than it.
    """
    cfg = cfg or AttackConfig()
    thr = threshold if threshold is not None else default_threshold(hasher)
    device = cfg.resolved_device()
    gen = torch.Generator(device=device)
    if cfg.seed is not None:
        gen.manual_seed(cfg.seed)
    adapter = adapter or build_adapter(hasher).to(device)
    x0 = torch.stack([adapter.preprocess(im) for im in images]).to(device)
    if targets is None:
        mode, target_hash = "evasion", None
        ref = adapter.reference(x0)
        loss_fn = lambda out: adapter.evasion_loss(out, ref, cfg.margin)
        with torch.no_grad():
            ref_hash_t = adapter.hash_t(adapter.outputs(x0))
    else:
        if len(targets) != len(images):
            raise ValueError("need exactly one target per image")
        mode = "collision"
        xt = torch.stack([adapter.preprocess(im) for im in targets]).to(device)
        ref = adapter.reference(xt)                     # signs / rounded hash of each target
        target_hash = adapter.hash(xt)
        loss_fn = lambda out: adapter.collision_loss(out, ref, cfg.margin)
        with torch.no_grad():
            ref_hash_t = adapter.hash_t(adapter.outputs(xt))
    objective = _make_objective(adapter, ref_hash_t, mode, cfg.quantize)
    if init is not None and not torch.is_tensor(init):
        init = torch.stack(list(init))
    adv = _run(adapter, x0, loss_fn, cfg, gen, objective, None if init is None else init.to(device))
    return _results(mode, adapter, x0, adv, thr, cfg, target_hash, keep_images)


def evade(hasher: PerceptualHash, image: Image.Image, cfg: AttackConfig | None = None,
          threshold: float | None = None) -> AttackResult:
    """Perturb ``image`` so its hash no longer matches the original within ``threshold``.

    ``threshold`` is in the hash's own distance units: Hamming *bits* for DinoHash/NeuralHash,
    L1 for CertPHash. If None, CertPHash uses 1800 and bit hashes use 0 (any change counts).
    """
    return attack_batch(hasher, [image], cfg, threshold)[0]


def collide(hasher: PerceptualHash, source_image: Image.Image, target_image: Image.Image,
            cfg: AttackConfig | None = None, threshold: float | None = None) -> AttackResult:
    """Perturb ``source_image`` so its hash matches ``target_image``'s within ``threshold``."""
    return attack_batch(hasher, [source_image], cfg, threshold, targets=[target_image])[0]


__all__ = ["AttackConfig", "AttackResult", "attack_batch", "evade", "collide", "build_adapter",
           "default_threshold"]
