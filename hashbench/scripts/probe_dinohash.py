"""Targeted probes that check explanations of the DinoHash benchmark results (CPU, a few minutes).

1. Patch grid. In the benchmark a small shift changes many bits at once, and a 2% shift changes
   more bits than a 5% shift. DinoHash cuts its 224x224 input into 14x14 patches, so a shift by
   whole patches might matter less than a shift by part of a patch. Probe: shift the 224x224
   input by k pixels on both axes, filling with black (as the benchmark's translate does) or
   wrapping around (so no new content appears), and count the bits changed.
2. Closest impostor pairs. Saves a picture of the closest different-image pairs, to look at
   whether they share a scene, and tables of how close the same pairs are under other hashes.

Writes <run>/analysis/probe_patch_grid.csv, closest_pairs_dinohash.png and
closest_pairs_other_hashes.csv.

Usage:
    python scripts/probe_dinohash.py --run results/coco_run2 --other-run results/coco_run1 \
        --images data/coco/val2017 --list data/splits/V200.txt --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops

from hashbench.benchmark import load_image
from hashbench.hashes import get_hasher
from hashbench.hashes.dinohash import INPUT_SIZE

SHIFTS = [1, 2, 4, 7, 10, 14, 18, 21, 25, 28]
PATCH = 14
OTHER_HASHES = ["dinohash", "dinohash_untrained", "neuralhash", "certphash", "certphash_adv", "phash", "pdq"]


def shifted(image: Image.Image, k: int, fill: str) -> Image.Image:
    if fill == "wrap":
        return ImageChops.offset(image, k, k)
    return image.transform(image.size, Image.Transform.AFFINE, (1, 0, -k, 0, 1, -k))


def patch_grid(images: list[Image.Image]) -> pd.DataFrame:
    inputs = [im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR) for im in images]
    rows = []
    for name in ("dinohash", "dinohash_untrained"):
        hasher = get_hasher(name)
        ref = hasher.hash_batch(inputs)
        for fill in ("black", "wrap"):
            for k in SHIFTS:
                bits = (hasher.hash_batch([shifted(im, k, fill) for im in inputs]) != ref).sum(axis=1)
                rows.append({"algorithm": name, "fill": fill, "shift_px": k, "whole_patches": k % PATCH == 0,
                             "mean_bits_changed": bits.mean(), "std_bits_changed": bits.std(),
                             "se_bits_changed": bits.std() / np.sqrt(len(bits))})
    return pd.DataFrame(rows)


def closest_pairs_picture(run: Path, images_dir: Path, out: Path, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    closest = (pd.read_csv(run / "impostor.csv").query("algorithm == 'dinohash'")
               .nsmallest(n, "distance"))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 1.9 * rows), facecolor="#fcfcfb")
    for ax, r in zip(axes.ravel(), closest.itertuples()):
        a, b = (Image.open(images_dir / f).convert("RGB") for f in (r.image_a, r.image_b))
        h = 200
        a, b = (im.resize((round(im.width * h / im.height), h)) for im in (a, b))
        canvas = Image.new("RGB", (a.width + b.width + 6, h), (252, 252, 251))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 6, 0))
        ax.imshow(canvas)
        ax.set_title(f"{r.distance} bits", fontsize=9, color="#3d3d3a")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def closest_pairs_elsewhere(run: Path, other: Path, n: int = 100) -> pd.DataFrame:
    """Where do DinoHash's closest pairs rank among all 200,000 pairs under each other hash?"""
    imp = pd.concat([pd.read_csv(r / "impostor.csv") for r in (run, other)])
    wide = imp[imp["algorithm"].isin(OTHER_HASHES)].pivot_table(
        index=["image_a", "image_b"], columns="algorithm", values="norm_distance")
    ranks = wide.rank(pct=True)
    rows = []
    for source in ("dinohash", "dinohash_untrained"):
        chosen = wide[source].nsmallest(n).index
        for target in OTHER_HASHES:
            rows.append({"closest_under": source, "hash": target, "pairs": n,
                         "median_percentile": 100 * ranks.loc[chosen, target].median(),
                         "share_in_closest_1pct": (ranks.loc[chosen, target] <= 0.01).mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--other-run", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    out = args.run / "analysis"

    names = args.list.read_text().split()[:args.n]
    images = [load_image(args.images / n, max_side=512) for n in names]
    grid = patch_grid(images)
    grid.to_csv(out / "probe_patch_grid.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(grid.pivot_table(index="shift_px", columns=["algorithm", "fill"], values="mean_bits_changed").to_string())
        print("largest standard error:", grid["se_bits_changed"].max())

    closest_pairs_picture(args.run, args.images, out / "closest_pairs_dinohash.png")
    elsewhere = closest_pairs_elsewhere(args.run, args.other_run)
    elsewhere.to_csv(out / "closest_pairs_other_hashes.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(elsewhere.to_string(index=False))


if __name__ == "__main__":
    main()
