"""Forward-only "hash service" adapter for the classical hashes (pHash, PDQ) in the NES attack.

pHash and PDQ have no network, so the white-box adapters of ``adversarial.py`` do not cover
them. The NES attack (``hashbench.blackbox``) only needs the discrete hash of each query, so
this adapter turns every queried tensor back into an 8-bit PIL image and calls the normal
``PerceptualHash.hash`` on it -- exactly what a hashing service would see after an upload.

The attacked image is the whole picture as the benchmark loads it (``load_image`` with
max_side 512, aspect ratio kept), not a resized model input: the perturbation lives at that
resolution and the hash does its own resizing.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from hashbench.hashes.base import PerceptualHash


class ClassicalAdapter:
    """Implements the part of ``adversarial._Adapter`` that ``blackbox.HashService`` uses."""

    def __init__(self, hasher: PerceptualHash):
        self.hasher = hasher

    def to(self, device) -> "ClassicalAdapter":
        return self                                          # CPU only

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Discrete hashes (bool, (B, n_bits)) of a batch of [0, 1] images."""
        arr = x.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        hashes = [self.hasher.hash(Image.fromarray(a)) for a in arr]
        return torch.from_numpy(np.stack(hashes).astype(bool))

    def hash_t(self, out: torch.Tensor) -> torch.Tensor:
        return out

    def dist_t(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a != b).sum(dim=-1)


__all__ = ["ClassicalAdapter"]
