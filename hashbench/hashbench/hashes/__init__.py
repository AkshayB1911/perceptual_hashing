"""Registry of hash algorithms.

To add an algorithm (NeuralHash, CertPHash, ...): subclass ``PerceptualHash`` in a new
module and add a factory to ``_REGISTRY``. Neural hashes are imported lazily so the
classical hashes work without torch or downloaded weights.
"""

from __future__ import annotations

from typing import Callable

from hashbench.hashes.base import PerceptualHash, from_hex, hamming, to_hex
from hashbench.hashes.pdq import PDQHash
from hashbench.hashes.phash import PHash


def _dinohash(**fixed) -> Callable[..., PerceptualHash]:
    def make(**kwargs):
        from hashbench.hashes.dinohash import DinoHash
        return DinoHash(**{**fixed, **kwargs})
    return make


def _certphash(variant: str) -> Callable[..., PerceptualHash]:
    def make(**kwargs):
        from hashbench.hashes.certphash import CertPHash
        return CertPHash(variant=variant, **kwargs)
    return make


def _neuralhash(**kwargs) -> PerceptualHash:
    from hashbench.hashes.neuralhash import NeuralHash
    return NeuralHash(**kwargs)


_REGISTRY: dict[str, Callable[..., PerceptualHash]] = {
    "phash": PHash,
    "pdq": PDQHash,
    # Apple NeuralHash; needs the user-extracted model files in models/neuralhash/
    "neuralhash": _neuralhash,
    # DinoHash, released adversarially fine-tuned models
    "dinohash": _dinohash(variant="vits14_96", trained=True),
    "dinohash_vitb14_512": _dinohash(variant="vitb14_512", trained=True),
    # same architectures with the original DINOv2 weights (the paper's "untrained" ablation)
    "dinohash_untrained": _dinohash(variant="vits14_96", trained=False),
    "dinohash_vitb14_512_untrained": _dinohash(variant="vitb14_512", trained=False),
    # CertPHash (C-PDNA, COCO 64x64): certified training at eps_target = k/255 (ep1 = the
    # paper's default), plus the paper's normal (base) and adversarial (adv) baselines
    "certphash": _certphash("ep1"),
    **{f"certphash_{v}": _certphash(v) for v in ("ep2", "ep4", "ep8", "base", "adv")},
}


def get_hasher(name: str, **kwargs) -> PerceptualHash:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown hash {name!r}; available: {list_hashers()}") from None
    return factory(**kwargs)


def list_hashers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["PerceptualHash", "PHash", "PDQHash", "get_hasher", "list_hashers",
           "hamming", "to_hex", "from_hex"]
