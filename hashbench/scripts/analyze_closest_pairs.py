"""Which kinds of different images do the hashes confuse? (closest impostor pairs)

For every algorithm, looks at the 100 closest impostor pairs and asks whether both images
are black-framed (all four 2% edge strips with mean luma < 20) or among the darkest 10% of
the dataset. Also counts false matches at the CertPHash paper's threshold (L1 <= 1800), and
splits CertPHash ep8's brightness-0.5 match rate by the original image's brightness.

Writes <run>/analysis/{black_frame_images, image_mean_luma, black_frame_impostors,
dark_image_impostors}.csv.

Usage:
    python scripts/analyze_closest_pairs.py --run results/coco_run1 --images data/coco/val2017
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAPER_THRESHOLD = 1800 / (144 * 255)


def both(flags: pd.Series, a: pd.Series, b: pd.Series) -> np.ndarray:
    return flags.reindex(a).to_numpy() & flags.reindex(b).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.001)
    args = ap.parse_args()
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)

    framed, luma = {}, {}
    for p in sorted(args.images.glob("*.jpg")):
        a = np.asarray(Image.open(p).convert("L").reduce(4), dtype=np.float32)
        k = max(1, round(0.02 * min(a.shape)))
        framed[p.name] = all(s < 20 for s in (a[:k].mean(), a[-k:].mean(), a[:, :k].mean(), a[:, -k:].mean()))
        luma[p.name] = float(a.mean())
    framed, luma = pd.Series(framed, name="framed"), pd.Series(luma, name="mean_luma")
    framed.to_csv(out / "black_frame_images.csv")
    luma.to_csv(out / "image_mean_luma.csv")
    dark = luma < luma.quantile(0.10)
    print(f"black-framed images: {int(framed.sum())} of {len(framed)} ({framed.mean():.1%})")

    impostor = pd.read_csv(args.run / "impostor.csv")
    frame_rows, dark_rows = [], []
    for alg, g in impostor.groupby("algorithm"):
        close = g.nsmallest(100, "norm_distance")
        fm = g[g["norm_distance"] <= PAPER_THRESHOLD + 1e-12] if alg.startswith("certphash") else g.iloc[:0]
        frame_rows.append({
            "algorithm": alg,
            "share_pairs_both_framed": both(framed, g["image_a"], g["image_b"]).mean(),
            "share_of_100_closest_both_framed": both(framed, close["image_a"], close["image_b"]).mean(),
            "paper_threshold_false_matches": len(fm),
            "of_which_both_framed": int(both(framed, fm["image_a"], fm["image_b"]).sum()) if len(fm) else 0})
        dark_rows.append({
            "algorithm": alg,
            "mean_luma_100_closest": (luma.reindex(close["image_a"]).to_numpy()
                                      + luma.reindex(close["image_b"]).to_numpy()).mean() / 2,
            "share_100_closest_both_darkest10pct": both(dark, close["image_a"], close["image_b"]).mean()})
    pd.DataFrame(frame_rows).to_csv(out / "black_frame_impostors.csv", index=False)
    pd.DataFrame(dark_rows).to_csv(out / "dark_image_impostors.csv", index=False)

    genuine = pd.read_csv(args.run / "genuine.csv")
    overall = pd.read_csv(args.run / "overall.csv").set_index("algorithm")
    if "certphash_ep8" in overall.index:
        g = genuine[(genuine["algorithm"] == "certphash_ep8") & (genuine["attack"] == "brightness")
                    & (genuine["param"] == 0.5)].copy()
        g["match"] = g["norm_distance"] <= overall.loc["certphash_ep8", f"threshold_at_fpr_{args.fpr:g}"]
        g["luma_quartile"] = pd.qcut(luma.reindex(g["image"]).to_numpy(), 4,
                                     labels=["darkest", "q2", "q3", "brightest"])
        print("ep8, brightness 0.5, match rate by original brightness quartile:")
        print(g.groupby("luma_quartile", observed=True)["match"].mean().round(3).to_string())
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(frame_rows).to_string(index=False))
        print(pd.DataFrame(dark_rows).to_string(index=False))


if __name__ == "__main__":
    main()
