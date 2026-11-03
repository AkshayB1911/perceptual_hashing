"""Write the fixed evaluation subsets of COCO val2017 used by every experiment.

    V1k.txt   1000 images: benign attack suite (Phase B)
    V200.txt   200 images, a subset of V1k: white-box attacks, certification (Phase C+)

All 5000 images (V5k) are used for impostor pairs, so it needs no file. Each file lists
image names relative to the image folder, one per line, sorted. The draw is seeded, so
rerunning this script reproduces the same files.

Usage:
    python scripts/make_splits.py [--images data/coco/val2017] [--out data/splits] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(names) < 1000:
        raise SystemExit(f"expected at least 1000 images in {args.images}, found {len(names)}")
    rng = np.random.default_rng(args.seed)
    v1k = rng.choice(names, size=1000, replace=False).tolist()
    v200 = rng.choice(v1k, size=200, replace=False).tolist()  # nested: V200 is a subset of V1k

    args.out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("V1k", v1k), ("V200", v200)):
        (args.out / f"{name}.txt").write_text("\n".join(sorted(subset)) + "\n", newline="\n")
        print(f"wrote {len(subset)} names to {args.out / f'{name}.txt'}")


if __name__ == "__main__":
    main()
