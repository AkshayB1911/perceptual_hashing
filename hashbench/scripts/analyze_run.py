"""Deeper analysis and sanity checks of a benchmark run (after hashbench.report).

Writes CSV tables to <run>/analysis/:
    family_match_rates.csv    match rate per attack family at each calibrated FPR, with a
                              95% bootstrap interval over images (primary FPR)
    per_param.csv             mean distance, distance in impostor standard deviations, and
                              match rate for every attack x param (primary FPR)
    zero_fpr.csv              match rate at the largest threshold with no false match among
                              all impostor pairs
    exact_invariances.csv     share of attacked copies with distance exactly 0, per attack
    closest_impostors.csv     the closest impostor pairs per algorithm (near-duplicates?)
    bit_balance.csv           per-algorithm bit statistics of the original hashes
    pdq_quality.csv           PDQ match rate for low- vs normal-quality images
    image_difficulty_corr.csv Spearman correlation of per-image match rates between hashes

Usage:
    python scripts/analyze_run.py --run results/coco_run1 --fpr 0.001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hashbench.hashes import get_hasher

FAMILIES = {
    "compression/resampling": ["jpeg", "rescale", "aspect_ratio"],
    "filtering/noise": ["gaussian_blur", "median_filter", "gaussian_noise", "salt_and_pepper"],
    "photometric": ["brightness", "contrast", "gamma", "saturation", "hue_shift", "grayscale"],
    "geometric": ["rotate", "rotate_crop", "center_crop", "corner_crop", "translate", "border"],
    "flips": ["flip_horizontal", "flip_vertical"],
    "overlays": ["text_overlay", "occlusion"],
}


def bootstrap_ci(per_image: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_image), size=(n_boot, len(per_image)))
    means = per_image[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.001, help="primary calibrated FPR")
    args = ap.parse_args()
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)

    genuine = pd.read_csv(args.run / "genuine.csv")
    genuine = genuine[genuine["error"].isna()] if "error" in genuine else genuine
    impostor = pd.read_csv(args.run / "impostor.csv")
    overall = pd.read_csv(args.run / "overall.csv").set_index("algorithm")
    hashes = pd.read_csv(args.run / "hashes.csv")
    fpr_cols = sorted([c for c in overall.columns if c.startswith("threshold_at_fpr_")],
                      key=lambda c: -float(c.rsplit("_", 1)[1]))
    t_col = f"threshold_at_fpr_{args.fpr:g}"
    algs = list(overall.index)
    family_of = {a: f for f, attacks in FAMILIES.items() for a in attacks}
    genuine["family"] = genuine["attack"].map(family_of)

    # 1. Attack families: mean over images of the per-image mean over (attack, param).
    rows = []
    for alg in algs:
        g = genuine[genuine["algorithm"] == alg]
        for fam, gf in g.groupby("family"):
            row = {"algorithm": alg, "family": fam}
            for col in fpr_cols:
                matched = gf["norm_distance"] <= overall.loc[alg, col]
                row[f"match_{col.replace('threshold_at_', '')}"] = matched.mean()
            per_image = (gf["norm_distance"] <= overall.loc[alg, t_col]).groupby(gf["image"]).mean().to_numpy()
            row["ci_low"], row["ci_high"] = bootstrap_ci(per_image)
            rows.append(row)
    fam = pd.DataFrame(rows)
    fam.to_csv(out / "family_match_rates.csv", index=False)

    # 2. Per attack x param, with the genuine distance expressed in impostor std devs below
    #    the impostor mean (comparable across hashes with different distance scales).
    imp_stats = impostor.groupby("algorithm")["norm_distance"].agg(["mean", "std", "min"])
    per = genuine.groupby(["algorithm", "attack", "param"])["norm_distance"].agg(
        mean_distance="mean", median_distance="median").reset_index()
    per["z_below_impostor_mean"] = [(imp_stats.loc[a, "mean"] - d) / imp_stats.loc[a, "std"]
                                    for a, d in zip(per["algorithm"], per["mean_distance"])]
    per["match_rate"] = [
        (genuine.loc[(genuine["algorithm"] == a) & (genuine["attack"] == at) & (genuine["param"] == p),
                     "norm_distance"] <= overall.loc[a, t_col]).mean()
        for a, at, p in zip(per["algorithm"], per["attack"], per["param"])]
    per.to_csv(out / "per_param.csv", index=False)

    # 3. Zero false matches among all impostor pairs.
    zero = []
    for alg in algs:
        t0 = imp_stats.loc[alg, "min"]
        d = genuine.loc[genuine["algorithm"] == alg, "norm_distance"]
        zero.append({"algorithm": alg, "min_impostor_distance": t0,
                     "match_rate_below_min_impostor": float((d < t0).mean())})
    pd.DataFrame(zero).to_csv(out / "zero_fpr.csv", index=False)

    # 4. Exact invariances (distance exactly 0).
    exact = (genuine.assign(zero=genuine["distance"] == 0)
             .groupby(["attack", "algorithm"])["zero"].mean().unstack())
    exact.to_csv(out / "exact_invariances.csv")

    # 5. Closest impostor pairs: do several hashes agree that some COCO pairs are duplicates?
    closest = (impostor.sort_values("norm_distance").groupby("algorithm").head(10)
               .sort_values(["algorithm", "norm_distance"]))
    closest.to_csv(out / "closest_impostors.csv", index=False)

    # 6. Bit balance of the binary hashes (fraction of ones per bit; effective bits from the
    #    per-bit entropy, an upper bound that ignores correlations between bits).
    bits_rows = []
    for alg in algs:
        col = f"{alg}_hash"
        if col not in hashes or alg.startswith("certphash"):
            continue
        hasher = get_hasher("pdq" if alg == "pdq_dihedral" else alg)
        H = np.stack([hasher.from_string(s) for s in hashes[col]]).astype(np.float64)
        p = H.mean(axis=0).clip(1e-12, 1 - 1e-12)
        ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        corr = np.corrcoef(H.T)
        off = np.abs(corr[~np.eye(len(p), dtype=bool)])
        bits_rows.append({"algorithm": alg, "bits": H.shape[1], "mean_p1": p.mean(),
                          "min_p1": p.min(), "max_p1": p.max(), "sum_bit_entropy": ent.sum(),
                          "mean_abs_bit_corr": np.nanmean(off), "max_abs_bit_corr": np.nanmax(off)})
    pd.DataFrame(bits_rows).to_csv(out / "bit_balance.csv", index=False)

    # 7. PDQ quality: Meta recommends discarding hashes with quality < 50.
    if "pdq_quality" in hashes and "pdq" in algs:
        q = hashes.set_index("image")["pdq_quality"]
        g = genuine[genuine["algorithm"] == "pdq"].copy()
        g["low_quality"] = g["image"].map(q) < 50
        g["match"] = g["norm_distance"] <= overall.loc["pdq", t_col]
        pq = g.groupby("low_quality").agg(images=("image", "nunique"), match_rate=("match", "mean"))
        pq.to_csv(out / "pdq_quality.csv")

    # 8. Are the same images hard for every hash?
    per_img = pd.DataFrame({
        alg: (genuine.loc[genuine["algorithm"] == alg].set_index("image")["norm_distance"]
              <= overall.loc[alg, t_col]).groupby(level=0).mean()
        for alg in algs})
    per_img.corr(method="spearman").to_csv(out / "image_difficulty_corr.csv")

    # 9. Heatmap of family match rates (thesis figure).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = fam.pivot(index="algorithm", columns="family", values=f"match_fpr_{args.fpr:g}")
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index, list(FAMILIES)]
    fig, ax = plt.subplots(figsize=(1.3 * len(FAMILIES) + 2, 0.45 * len(table) + 1.2))
    im = ax.imshow(table.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label=f"match rate (FPR={args.fpr:g})")
    ax.set_title("Benign robustness by attack family")
    fig.tight_layout()
    fig.savefig(out / "family_heatmap.png", dpi=150)
    plt.close(fig)

    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(fam.pivot(index="family", columns="algorithm", values=f"match_fpr_{args.fpr:g}").to_string())
    print(f"\nwrote analysis tables to {out}")


if __name__ == "__main__":
    main()
