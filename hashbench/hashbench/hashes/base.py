"""Common interface that every perceptual hash in this project implements.

A hash is a 1-D numpy array of length ``n_bits``: bool for bit hashes (pHash, PDQ,
NeuralHash, DinoHash), compared by Hamming distance. Hashes with other value types
(CertPHash's 144 PhotoDNA-style bytes, compared by L1) override ``distances``,
``max_distance`` and the string encoding, so the benchmark, metrics and storage code
never need to know which algorithm produced a hash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class PerceptualHash(ABC):
    #: Short identifier used in result files, e.g. "phash".
    name: str = ""
    #: Number of bits (or values) in the hash.
    n_bits: int = 0

    @property
    def max_distance(self) -> int:
        """Largest possible distance; normalized distances are divided by it."""
        return self.n_bits

    @abstractmethod
    def hash(self, image: Image.Image) -> np.ndarray:
        """Return the hash of ``image`` as a bool array of shape (n_bits,)."""

    def hash_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Hashes of several images, shape (len(images), n_bits). Neural hashes override
        this to run the network on whole batches."""
        return np.stack([self.hash(im) for im in images])

    def distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two hashes."""
        return hamming(a, b)

    def distances(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Distances between hashes along the last axis (broadcasting), as int64."""
        return np.count_nonzero(a != b, axis=-1).astype(np.int64)

    def normalized_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance divided by ``max_distance``, in [0, 1].

        Use this when comparing algorithms with different hash lengths.
        """
        return self.distance(a, b) / self.max_distance

    def to_string(self, h: np.ndarray) -> str:
        """Storage encoding (hex, most significant bit first)."""
        return to_hex(h)

    def from_string(self, s: str) -> np.ndarray:
        return from_hex(s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, n_bits={self.n_bits})"


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"hash shapes differ: {a.shape} vs {b.shape}")
    return int(np.count_nonzero(a != b))


def to_hex(bits: np.ndarray) -> str:
    """Hex string of a hash, most significant bit first (the usual pHash/PDQ convention)."""
    bits = np.asarray(bits, dtype=bool)
    if bits.size % 8:
        raise ValueError("hash length must be a multiple of 8")
    return np.packbits(bits).tobytes().hex()


def from_hex(hex_str: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)).astype(bool)
