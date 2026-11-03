"""Check our pHash and PDQ implementations against the reference libraries.

    pHash  vs imagehash.phash          (the de-facto reference implementation)
    PDQ    vs pdqhash.compute(_dihedral) (Python bindings to Meta's C++ code)

Prints the per-image Hamming distance between our hash and the reference hash;
every value should be 0.

Usage:
    python scripts/validate_against_reference.py data/samples
"""

from __future__ import annotations

import sys
from pathlib import Path

import imagehash
import numpy as np
import pdqhash
from PIL import Image

from hashbench.benchmark import IMAGE_EXTENSIONS
from hashbench.hashes import PDQHash, PHash


def main(folder: Path) -> int:
    ph, pq = PHash(), PDQHash()
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    worst = 0
    print(f"{'image':30s} {'size':>11s}  phash  pdq  pdq_q(ours/ref)  dihedral(8)")
    for p in paths:
        img = Image.open(p).convert("RGB")
        rgb = np.asarray(img)

        d_phash = int((ph.hash(img) != imagehash.phash(img).hash.flatten()).sum())

        ref_bits, ref_q = pdqhash.compute(rgb)
        bits, q = pq.hash_with_quality(img)
        d_pdq = int((bits != ref_bits.astype(bool)).sum())

        ref_dih, _ = pdqhash.compute_dihedral(rgb)
        dih, _ = pq.hash_dihedral(img)
        d_dih = [int((a != b.astype(bool)).sum()) for a, b in zip(dih.values(), ref_dih)]

        worst = max(worst, d_phash, d_pdq, *d_dih, abs(q - ref_q))
        size = f"{img.width}x{img.height}"
        print(f"{p.name:30s} {size:>11s}  {d_phash:5d} {d_pdq:4d}  {q:6d}/{ref_q:<6d}   {d_dih}")

    print(f"\n{len(paths)} images; largest difference from reference: {worst}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/samples")))
