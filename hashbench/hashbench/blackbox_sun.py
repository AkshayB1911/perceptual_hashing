"""NES black-box attack in the setting of Sun et al. 2026 (arXiv 2608.03101), Appendix "adaptive attacks".

This is a separate replication; ``blackbox.py`` (our own NES setting) is not changed.

Their setting, as stated in the paper
-------------------------------------
* "All images are converted to RGB, resized to 512x512, represented in [0,1], and clipped to
  this range after modification."
* Budgets: l2 caps of 40, 90 and 180 (on the 512x512x3 image in [0, 1]).
* "Natural Evolution Strategies with the Hamming score returned by target queries."
* "The perturbation is optimized in a 64x64 grayscale space and mapped to the RGB query."
* "a clean start, at most 500 optimization steps, a 1,500-query limit, 32 antithetic directions
  per estimate, step size 2/255, NES scale 4/255, and early stopping after successful evasion."
* "matching threshold 0.2 for all eight hashes (i.e., bit error rate <= 20%)": an evasion
  succeeds when more than 20% of the 96 bits differ (>= 20 bits).

Choices for details the paper does not state (documented in the results file)
-----------------------------------------------------------------------------
* 512x512 resize of the original image: PIL bilinear.
* 64x64 -> 512x512: bilinear (align_corners=False), the same value copied to R, G and B.
* "32 antithetic directions": 32 Gaussian directions u_i, each queried at z + s*u_i and
  z - s*u_i (64 probe queries), plus one query of the new iterate per step for the early
  stop: 65 queries per step, so the 1,500-query limit allows 23 steps (the 500-step cap never
  binds). The clean hash costs 1 query.
* Update: z <- z + (2/255) * sign(g) (plain NES sign step, no momentum), followed by the
  projection onto the l2 ball. Probes are projected too, so no query leaves the budget.
* Queries are rounded to 8 bits (a real upload), as in all our attacks.

The hash service
----------------
The attacker sends the 512x512 image; the hash's own pipeline resizes it. To run this on a
GPU, the resize is reproduced exactly:
* DinoHash (PIL bilinear to 224) and NeuralHash (PIL bicubic to 360) use Pillow's 8-bit
  resampling: separable, horizontal pass first, fixed-point coefficients with 22 fractional
  bits, rounding to uint8 after each pass. ``PILResize`` computes the same integer
  coefficients and evaluates both passes with exact integer arithmetic (fp32 matmuls on
  split coefficients, every partial sum < 2^24), so its output equals ``Image.resize``.
* CertPHash resizes a float tensor with torchvision (bilinear, antialias); the same call is
  used on the GPU.
Each final image is also re-hashed with the plain hasher on CPU (``verified_*`` columns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hashbench.adversarial import _Adapter, _quantize, _ssim
from hashbench.blackbox import _bilinear_matrix

PRECISION_BITS = 22          # Pillow's Resample.c (32 - 8 - 2)


# --------------------------------------------------------------------------------------
# exact Pillow resize on torch tensors
# --------------------------------------------------------------------------------------

def _bilinear_filter(x: float) -> float:
    x = abs(x)
    return 1.0 - x if x < 1.0 else 0.0


def _bicubic_filter(x: float, a: float = -0.5) -> float:
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


_FILTERS = {Image.Resampling.BILINEAR: (_bilinear_filter, 1.0),
            Image.Resampling.BICUBIC: (_bicubic_filter, 2.0)}


def pil_coefficients(n_in: int, n_out: int, resample) -> np.ndarray:
    """(n_out, n_in) int64 fixed-point weights of Pillow's 8-bit resize (precompute_coeffs +
    normalize_coeffs_8bpc in Resample.c)."""
    filt, fsupport = _FILTERS[resample]
    scale = n_in / n_out
    filterscale = max(scale, 1.0)
    support = fsupport * filterscale
    out = np.zeros((n_out, n_in), dtype=np.int64)
    for xx in range(n_out):
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in) - xmin
        k = [filt((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        ww = sum(k)
        if ww != 0.0:
            k = [w / ww for w in k]
        for x, w in enumerate(k):
            v = w * (1 << PRECISION_BITS)
            out[xx, xmin + x] = int(-0.5 + v) if w < 0 else int(0.5 + v)   # C truncation
    return out


class PILResize:
    """Exact Pillow ``Image.resize`` (8-bit RGB, square output) for batches on any device.

    Input: (B, C, H, W) float tensor holding 8-bit values / 255. Output: the same for the
    resized image (values k / 255, k integer).
    """

    def __init__(self, n_in_h: int, n_in_w: int, n_out: int, resample, device):
        self.n_out = n_out
        self.h = self._split(pil_coefficients(n_in_w, n_out, resample), device)   # horizontal
        self.v = self._split(pil_coefficients(n_in_h, n_out, resample), device)   # vertical

    @staticmethod
    def _split(k: np.ndarray, device):
        hi = np.floor_divide(k, 2048)
        lo = k - hi * 2048                            # 0 <= lo < 2048
        return (torch.tensor(hi, dtype=torch.float32, device=device),
                torch.tensor(lo, dtype=torch.float32, device=device))

    @staticmethod
    def _round(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        tot = hi.double() * 2048.0 + lo.double() + float(1 << (PRECISION_BITS - 1))
        return torch.floor(tot / float(1 << PRECISION_BITS)).clamp_(0, 255).float()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        u8 = (x * 255.0).round()
        hh, hl = self.h
        tmp = self._round(u8 @ hh.t(), u8 @ hl.t())                 # (B, C, H, n_out)
        vh, vl = self.v
        out = self._round(vh @ tmp, vl @ tmp)                       # (B, C, n_out, n_out)
        return out / 255.0


# --------------------------------------------------------------------------------------
# the hash as a service on 512x512 images
# --------------------------------------------------------------------------------------

class ResizingAdapter:
    """Wraps a model adapter so that it takes the full (512x512) image and applies the hash's
    own resize first. Only the forward-only parts used by ``blackbox.HashService`` exist."""

    def __init__(self, inner: _Adapter, hasher, size: int, device):
        from hashbench.hashes.certphash import CertPHash, INPUT_SIZE as CERT_SIZE
        from hashbench.hashes.dinohash import DinoHash, INPUT_SIZE as DINO_SIZE
        from hashbench.hashes.neuralhash import NeuralHash, INPUT_SIZE as NH_SIZE

        self.inner, self.size = inner, size
        if isinstance(hasher, DinoHash):
            self.resize = PILResize(size, size, DINO_SIZE, Image.Resampling.BILINEAR, device)
        elif isinstance(hasher, NeuralHash):
            self.resize = PILResize(size, size, NH_SIZE, Image.Resampling.BICUBIC, device)
        elif isinstance(hasher, CertPHash):
            from torchvision.transforms import functional as TF
            self.resize = lambda x: TF.resize(x, [CERT_SIZE, CERT_SIZE],
                                              interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        else:
            raise TypeError(f"{type(hasher).__name__} is not supported")

    def outputs(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.outputs(self.resize(x))

    def hash_t(self, out):
        return self.inner.hash_t(out)

    def dist_t(self, a, b):
        return self.inner.dist_t(a, b)


def load_512(path, size: int = 512) -> Image.Image:
    """The attacked image: RGB, resized to size x size (PIL bilinear)."""
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------------------
# the attack
# --------------------------------------------------------------------------------------

@dataclass
class SunNESConfig:
    l2: float = 40.0                  # l2 cap on the [0, 1] image
    max_queries: int = 1500
    max_steps: int = 500
    directions: int = 32              # antithetic directions (2 queries each)
    sigma: float = 4.0 / 255.0        # NES scale
    step: float = 2.0 / 255.0         # sign-step size
    dim: int = 64                     # side of the grayscale search space
    chunk: int = 64
    device: str = "cpu"
    seed: int = 2026

    @property
    def queries_per_step(self) -> int:
        return 2 * self.directions + 1


@dataclass
class SunNESResult:
    success: bool
    queries: int
    distance: float                   # best distance to the clean hash over all iterates
    orig_distance: float
    steps: int
    l2: float                         # of the returned 8-bit image
    linf: float
    ssim: float
    adv: torch.Tensor                 # (3, H, W) returned 8-bit image (first success, else best)


def sun_nes_evasion(service, x0: torch.Tensor, ref: torch.Tensor, threshold: float,
                    cfg: SunNESConfig) -> list[SunNESResult]:
    """Evasion on a batch x0 (B, 3, H, W) of 8-bit images in [0, 1]; success: d > threshold."""
    dev = torch.device(cfg.device)
    x0, ref = x0.to(dev), ref.to(dev)
    b, c, h, w = x0.shape
    d, n = cfg.dim, cfg.directions
    rh, rw = _bilinear_matrix(d, h, dev), _bilinear_matrix(d, w, dev)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    def delta(z):                                     # (k, 1, d, d) -> (k, 1, h, w)
        return rh @ z @ rw.t()

    def project(z):                                   # l2 of the RGB perturbation <= cap
        norm = delta(z).flatten(1).norm(dim=1) * math.sqrt(c)
        f = (cfg.l2 / norm.clamp_min(1e-12)).clamp(max=1.0)
        return z * f.view(-1, *([1] * (z.dim() - 1)))

    def image(z, base):
        return _quantize(base + delta(z))             # broadcast grayscale over RGB

    orig = service.distances(x0, ref)
    queries = torch.ones(b, dtype=torch.long, device=dev)
    best_dist, best_x = orig.clone(), x0.clone()
    done = orig > threshold
    succ_at = torch.where(done, queries, torch.zeros_like(queries))
    steps = torch.zeros(b, dtype=torch.long, device=dev)
    z = torch.zeros(b, 1, d, d, device=dev)

    while True:
        active = (~done) & (queries + cfg.queries_per_step <= cfg.max_queries) & (steps < cfg.max_steps)
        idx = active.nonzero().flatten()
        if len(idx) == 0:
            break
        k = len(idx)
        za, base = z[idx], x0[idx]
        u = torch.randn((k, n, 1, d, d), generator=gen, device=dev)
        probes = torch.cat([za[:, None] + cfg.sigma * u, za[:, None] - cfg.sigma * u], dim=1)
        probes = project(probes.flatten(0, 1))                                       # (k*2n, 1, d, d)
        dp = torch.empty(k * 2 * n, device=dev)
        owner = torch.arange(k * 2 * n, device=dev) // (2 * n)       # probe -> image of the batch
        refs = ref[idx][owner]
        for s in range(0, k * 2 * n, cfg.chunk):                     # chunks keep 512x512 memory low
            sl = slice(s, s + cfg.chunk)
            dp[sl] = service.distances(image(probes[sl], base[owner[sl]]), refs[sl])
        dp = dp.view(k, 2 * n)
        diff = dp[:, :n] - dp[:, n:]
        g = (diff[:, :, None, None, None] * u).sum(dim=1) / (2 * n * cfg.sigma)
        z[idx] = project(za + cfg.step * g.sign())

        xc = image(z[idx], base)
        dc = service.distances(xc, ref[idx])
        queries[idx] += cfg.queries_per_step
        steps[idx] += 1

        hit = dc > threshold
        better = (dc > best_dist[idx]) | hit
        bi = idx[better]
        best_dist[bi], best_x[bi] = dc[better], xc[better]
        hi = idx[hit]
        done[hi] = True
        succ_at[hi] = queries[hi]

    diff = best_x - x0
    l2 = diff.flatten(1).norm(dim=1)
    linf = diff.abs().flatten(1).amax(dim=1)
    ssim = torch.cat([_ssim(best_x[i:i + 4], x0[i:i + 4]) for i in range(0, b, 4)])
    return [SunNESResult(success=bool(done[i]), queries=int(succ_at[i] if done[i] else queries[i]),
                         distance=float(best_dist[i]), orig_distance=float(orig[i]), steps=int(steps[i]),
                         l2=float(l2[i]), linf=float(linf[i]), ssim=float(ssim[i]), adv=best_x[i].cpu())
            for i in range(b)]


__all__ = ["PILResize", "pil_coefficients", "ResizingAdapter", "load_512", "to_tensor",
           "SunNESConfig", "SunNESResult", "sun_nes_evasion"]
