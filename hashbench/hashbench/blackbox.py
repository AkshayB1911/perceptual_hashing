"""Score-based black-box attack (NES) on the learned perceptual hashes.

Threat model
------------
The attacker has **no weights and no gradients**. They can submit an 8-bit image to the
hashing service and read back its **hash** (96 bits for DinoHash / NeuralHash, the 144-byte
vector for CertPHash). From the hash they compute the distance to the original image's hash
(evasion) or to a target hash (collision) -- nothing more. Every submitted image is one
**query**, and the attack has a fixed query budget per image.

As in the white-box attacks (``adversarial.py``), the perturbation lives at the model's input
size (DinoHash 224, NeuralHash 360, CertPHash 64) inside an l_inf ball of radius ``eps`` in
[0, 1] pixel space, and every query is rounded to 8 bits, as a real upload would be.

Algorithm (Ilyas et al. 2018, "Black-box adversarial attacks with limited queries")
-------------------------------------------------------------------------------------
* score(x) = d(hash(x), hash(original))  for evasion   (to increase)
             -d(hash(x), target hash)     for collision (to increase)
* The perturbation is searched in a low-dimensional space: z has shape (3, dim, dim) and is
  upsampled bilinearly to the input size. Bilinear upsampling is a convex combination, so
  |z| <= eps keeps the image perturbation inside the ball.
* Each step draws ``pairs`` Gaussian directions u_i and queries x(z + sigma u_i) and
  x(z - sigma u_i) (antithetic pairs). The gradient estimate is
      g = sum_i [score(z + sigma u_i) - score(z - sigma u_i)] u_i / (2 pairs sigma),
  smoothed with momentum, and the step is z <- clip_eps(z + lr * sign(g)).
* After each step the new iterate is queried once more; the attack stops for an image as
  soon as it succeeds at the calibrated threshold, or when the next step would exceed the
  budget. The step size halves when the best score has not improved for ``plateau`` steps.
* The score is a bit count (or an L1 distance of rounded bytes), so it is piecewise
  constant; ``sigma`` must be large enough for the probes to actually change some bits.

Queries per image = 1 (hash of the original) + steps * (2 * pairs + 1).

The adapters of ``adversarial.py`` are reused only as forward-only "services": the attack
calls ``outputs`` under ``torch.no_grad`` and reads the discrete hash, never a logit value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from hashbench.adversarial import _Adapter, _quantize, _ssim


@dataclass
class NESConfig:
    """Hyper-parameters of the NES attack.

    eps          l_inf budget in [0, 1] pixel space.
    max_queries  query budget per image (the clean-hash query included).
    pairs        antithetic pairs per gradient estimate (2 * pairs queries per step).
    sigma_frac   probe scale sigma = sigma_frac * eps (in pixel units of the low-dim z).
    lr_frac      initial sign-step size = lr_frac * eps.
    momentum     momentum of the gradient estimate (Ilyas et al. use 0.9).
    dim          side of the low-dimensional search space (capped at the input size).
    plateau      steps without improvement of the best score before the step size halves.
    min_lr_frac  the step size never drops below min_lr_frac * eps.
    chunk        images per forward pass (GPU memory).
    device       torch device string.
    seed         RNG seed (the random directions).
    separable    upsample z with two small matrix products instead of F.interpolate; the same
                 bilinear map (up to float rounding), much faster on CPU for large images.
                 Used by the classical-hash runner; the GPU runs keep the default False.
    """

    eps: float = 8.0 / 255.0
    max_queries: int = 10_000
    pairs: int = 20
    sigma_frac: float = 1.0
    lr_frac: float = 0.2
    momentum: float = 0.9
    dim: int = 32
    plateau: int = 5
    min_lr_frac: float = 0.02
    chunk: int = 256
    device: str = "cpu"
    seed: int = 0
    separable: bool = False

    @property
    def queries_per_step(self) -> int:
        return 2 * self.pairs + 1


@dataclass
class NESResult:
    success: bool
    queries: int            # queries spent until success, or all queries spent if it failed
    distance: float         # best distance reached (evasion: to original; collision: to target)
    orig_distance: float    # the same distance before the attack
    steps: int
    linf: float             # of the best 8-bit image
    ssim: float
    adv: torch.Tensor       # (C, H, W) best 8-bit image in [0, 1], at the model input size


class HashService:
    """Forward-only view of a learned hash: images in, discrete hashes / distances out."""

    def __init__(self, adapter: _Adapter, chunk: int = 256):
        self.adapter = adapter
        self.chunk = chunk

    @torch.no_grad()
    def hashes(self, x: torch.Tensor) -> torch.Tensor:
        out = [self.adapter.hash_t(self.adapter.outputs(x[i:i + self.chunk]))
               for i in range(0, len(x), self.chunk)]
        return torch.cat(out)

    @torch.no_grad()
    def distances(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """d(hash(x[k]), ref[k]) for every k; ref holds discrete hashes."""
        out = []
        for i in range(0, len(x), self.chunk):
            h = self.adapter.hash_t(self.adapter.outputs(x[i:i + self.chunk]))
            out.append(self.adapter.dist_t(h, ref[i:i + self.chunk]).float())
        return torch.cat(out)


def _upsample(z: torch.Tensor, h: int, w: int | None = None) -> torch.Tensor:
    w = h if w is None else w
    if z.shape[-2:] == (h, w):
        return z
    return F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)


def _bilinear_matrix(n_in: int, n_out: int, device) -> torch.Tensor:
    """(n_out, n_in) matrix of 1-D bilinear upsampling (align_corners=False), as F.interpolate."""
    eye = torch.eye(n_in, device=device)[:, None, :]                           # (n_in, 1, n_in)
    return F.interpolate(eye, size=n_out, mode="linear", align_corners=False)[:, 0, :].t()


def nes_attack(service: HashService, x0: torch.Tensor, ref: torch.Tensor, mode: str,
               threshold: float, cfg: NESConfig) -> list[NESResult]:
    """NES evasion / collision on a batch.

    x0   (B, C, H, W) clean 8-bit model inputs in [0, 1]. Non-square inputs (the classical
         hashes attack the whole image) keep a square (dim x dim) search space.
    ref  (B, ...) discrete reference hashes: the clean hashes for evasion, the target hashes
         for collision.
    Success: evasion d > threshold; collision d <= threshold.
    """
    if mode not in ("evasion", "collision"):
        raise ValueError(mode)
    dev = torch.device(cfg.device)
    x0, ref = x0.to(dev), ref.to(dev)
    b, c, h, w = x0.shape
    d = min(cfg.dim, h, w)
    n, eps = cfg.pairs, cfg.eps
    sigma = cfg.sigma_frac * eps
    sign = 1.0 if mode == "evasion" else -1.0
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    def success(dist: torch.Tensor) -> torch.Tensor:
        return dist > threshold if mode == "evasion" else dist <= threshold

    if cfg.separable:
        rh, rw = _bilinear_matrix(d, h, dev), _bilinear_matrix(d, w, dev)

        def image(z: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            return _quantize(base + rh @ z @ rw.t())
    else:
        def image(z: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
            return _quantize(base + _upsample(z, h, w))

    orig = service.distances(x0, ref)
    queries = torch.ones(b, dtype=torch.long, device=dev)        # the clean query
    best_dist, best_x = orig.clone(), x0.clone()
    done = success(orig)
    succ_at = torch.where(done, queries, torch.zeros_like(queries))
    steps = torch.zeros(b, dtype=torch.long, device=dev)
    z = torch.zeros(b, c, d, d, device=dev)
    mom = torch.zeros_like(z)
    lr = torch.full((b,), cfg.lr_frac * eps, device=dev)
    stale = torch.zeros(b, dtype=torch.long, device=dev)

    while True:
        active = (~done) & (queries + cfg.queries_per_step <= cfg.max_queries)
        idx = active.nonzero().flatten()
        if len(idx) == 0:
            break
        k = len(idx)
        za, base = z[idx], x0[idx]
        u = torch.randn((k, n, c, d, d), generator=gen, device=dev)
        probes = torch.cat([za[:, None] + sigma * u, za[:, None] - sigma * u], dim=1)  # (k, 2n, ...)
        xp = image(probes.flatten(0, 1), base.repeat_interleave(2 * n, 0))
        dp = service.distances(xp, ref[idx].repeat_interleave(2 * n, 0)).view(k, 2 * n)
        diff = sign * (dp[:, :n] - dp[:, n:])                                         # (k, n)
        g = (diff[:, :, None, None, None] * u).sum(dim=1) / (2 * n * sigma)
        mom[idx] = cfg.momentum * mom[idx] + (1 - cfg.momentum) * g
        z[idx] = (za + lr[idx, None, None, None] * mom[idx].sign()).clamp(-eps, eps)

        xc = image(z[idx], base)
        dc = service.distances(xc, ref[idx])
        queries[idx] += cfg.queries_per_step
        steps[idx] += 1

        better = sign * (dc - best_dist[idx]) > 0
        bi = idx[better]
        best_dist[bi], best_x[bi] = dc[better], xc[better]
        stale[idx] = torch.where(better, torch.zeros_like(stale[idx]), stale[idx] + 1)
        decay = idx[stale[idx] >= cfg.plateau]
        lr[decay] = (lr[decay] * 0.5).clamp_min(cfg.min_lr_frac * eps)
        stale[decay] = 0

        hit = success(dc)
        hi = idx[hit]
        done[hi] = True
        succ_at[hi] = queries[hi]

    linf = (best_x - x0).abs().flatten(1).amax(dim=1)
    t_ssim = _ssim(best_x, x0)
    out = []
    for i in range(b):
        ok = bool(done[i])
        out.append(NESResult(
            success=ok, queries=int(succ_at[i] if ok else queries[i]),
            distance=float(best_dist[i]), orig_distance=float(orig[i]), steps=int(steps[i]),
            linf=float(linf[i]), ssim=float(t_ssim[i]), adv=best_x[i].cpu()))
    return out


def reference_hashes(service: HashService, x: torch.Tensor) -> torch.Tensor:
    """Discrete hashes of a batch (the attacker's one query per clean / target image)."""
    return service.hashes(x)


def budget_success(queries: np.ndarray, success: np.ndarray, budget: int) -> np.ndarray:
    """Success within ``budget`` queries, from the per-image (queries, success) record."""
    return success & (queries <= budget)


__all__ = ["NESConfig", "NESResult", "HashService", "nes_attack", "reference_hashes",
           "budget_success"]
