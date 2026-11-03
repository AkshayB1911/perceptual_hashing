"""Compare two benchmark runs of the same images (e.g. the same run on two machines).

For every algorithm present in both runs, reports how many original hashes are identical,
how many bits differ on average, and how many genuine distances (image x attack x param)
are identical. Differences of a bit or two on a handful of images come from floating-point
rounding (bits whose logit is near zero) and do not affect the benchmark's conclusions.

Usage:
    python scripts/compare_runs.py results/crosscheck_laptop_dinohash results/crosscheck_linux_dinohash
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hashbench.hashes import get_hasher


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    args = ap.parse_args()

    ha, hb = (pd.read_csv(r / "hashes.csv").set_index("image") for r in (args.run_a, args.run_b))
    ga, gb = (pd.read_csv(r / "genuine.csv") for r in (args.run_a, args.run_b))
    common_images = ha.index.intersection(hb.index)
    if len(common_images) == 0:
        raise SystemExit("the two runs share no images")
    algs = sorted({c[:-5] for c in ha.columns if c.endswith("_hash")} & {c[:-5] for c in hb.columns if c.endswith("_hash")})

    rows = []
    for alg in algs:
        hasher = get_hasher("pdq" if alg == "pdq_dihedral" else alg)
        a = np.stack([hasher.from_string(s) for s in ha.loc[common_images, f"{alg}_hash"]])
        b = np.stack([hasher.from_string(s) for s in hb.loc[common_images, f"{alg}_hash"]])
        diff = hasher.distances(a[:, None, :], b[:, None, :])[:, 0] if hasattr(hasher, "distances") \
            else (a != b).sum(axis=1)
        key = ["image", "attack", "param"]
        m = ga[ga["algorithm"] == alg].merge(gb[gb["algorithm"] == alg], on=key, suffixes=("_a", "_b"))
        rows.append({
            "algorithm": alg, "images": len(common_images),
            "identical_original_hashes": float((diff == 0).mean()),
            "mean_original_distance": float(diff.mean()), "max_original_distance": int(diff.max()),
            "genuine_pairs": len(m),
            "identical_genuine_distances": float((m["distance_a"] == m["distance_b"]).mean()) if len(m) else np.nan,
            "max_abs_genuine_difference": int((m["distance_a"] - m["distance_b"]).abs().max()) if len(m) else 0,
        })
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
