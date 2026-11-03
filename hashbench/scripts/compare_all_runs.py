"""Combine benchmark runs into one comparison of all hashes.

The runs must share images, attacks and impostor pairs (this is checked). Writes to --out:
    headline.csv              mean match rate at each FPR, AUC, FPR-exact and equal-family means
    family.csv                match rate per attack family at FPR 1e-3
    per_setting.csv           match rate per attack x strength at FPR 1e-3
    achieved_fpr.csv          the false-match rate each calibrated threshold really gives
    binomial_vs_observed.csv  false-match rate predicted by Binomial(bits, 0.5) vs measured
    bootstrap.csv             95% intervals, resampling images (thresholds recalibrated each time)
    bootstrap_differences.csv the same for chosen pairs of hashes
    bits_trained_vs_untrained.csv  mean bits changed per setting, DinoHash trained vs untrained
    impostor_corr.csv         do two hashes find the same different-image pairs close? (Spearman)
    image_difficulty_corr.csv do two hashes fail on the same images? (Spearman)
    ensemble.csv              "match if any hash matches", with false matches measured on the pairs
    family_heatmap.png, roc_low_fpr.png

Usage:
    python scripts/compare_all_runs.py --runs results/coco_run1 results/coco_run2 --out results/comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from hashbench.metrics import threshold_at_fpr

FPRS = [0.01, 0.001, 0.0001]
FAMILIES = {
    "compression/resampling": ["jpeg", "rescale", "aspect_ratio"],
    "filtering/noise": ["gaussian_blur", "median_filter", "gaussian_noise", "salt_and_pepper"],
    "photometric": ["brightness", "contrast", "gamma", "saturation", "hue_shift", "grayscale"],
    "geometric": ["rotate", "rotate_crop", "center_crop", "corner_crop", "translate", "border"],
    "flips": ["flip_horizontal", "flip_vertical"],
    "overlays": ["text_overlay", "occlusion"],
}
FAMILY_LABELS = ["Compression", "Filtering/noise", "Photometric", "Geometric", "Flips", "Overlays"]
BITS = {"phash": 64, "pdq": 256, "pdq_dihedral": 256, "neuralhash": 96, "dinohash": 96, "dinohash_untrained": 96}
LABELS = {
    "dinohash": "DinoHash trained", "dinohash_untrained": "DinoHash untrained", "neuralhash": "NeuralHash",
    "certphash": "CertPHash ε=1", "certphash_ep2": "CertPHash ε=2", "certphash_ep4": "CertPHash ε=4",
    "certphash_ep8": "CertPHash ε=8", "certphash_base": "CertPHash base", "certphash_adv": "CertPHash adv",
    "phash": "pHash", "pdq": "PDQ", "pdq_dihedral": "PDQ-dihedral",
}
DIFFERENCES = [("dinohash_untrained", "dinohash"), ("dinohash", "neuralhash"), ("dinohash", "certphash_adv"),
               ("dinohash_untrained", "certphash_adv"), ("certphash_adv", "certphash"), ("dinohash", "phash")]
ENSEMBLES = [("dinohash", "pdq_dihedral"), ("dinohash", "certphash_adv"), ("dinohash_untrained", "pdq_dihedral"),
             ("dinohash_untrained", "certphash_adv"), ("dinohash", "certphash_adv", "pdq_dihedral"),
             ("dinohash_untrained", "certphash_adv", "pdq_dihedral")]
PRIMARY = 0.001
N_BOOT = 400


def load(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    configs = [json.loads((r / "config.json").read_text()) for r in runs]
    for key in ("attack_suite", "max_side", "max_pairs", "seed", "n_images", "n_attacked"):
        if any(c[key] != configs[0][key] for c in configs):
            raise SystemExit(f"the runs differ in {key}")
    g = pd.concat([pd.read_csv(r / "genuine.csv", usecols=["image", "algorithm", "attack", "param", "distance", "norm_distance"])
                   for r in runs], ignore_index=True)
    i = pd.concat([pd.read_csv(r / "impostor.csv") for r in runs], ignore_index=True)
    pair_sets = {a: set(zip(d["image_a"], d["image_b"])) for a, d in i.groupby("algorithm")}
    first = next(iter(pair_sets.values()))
    if any(s != first for s in pair_sets.values()):
        raise SystemExit("the runs do not share the same impostor pairs")
    return g, i, sorted(i["algorithm"].unique())


def exact_fpr_match(imp: np.ndarray, gen: np.ndarray, target: float) -> float:
    """Match rate at exactly ``target`` FPR, interpolating between the two nearest thresholds.

    Equivalent to a randomised decision between them, so whole-bit hashes can be compared at the
    same FPR as hashes with continuous distances.
    """
    values, counts = np.unique(imp, return_counts=True)
    fmr = np.cumsum(counts) / imp.size
    above = np.searchsorted(fmr, target, side="right")
    lo_f = fmr[above - 1] if above > 0 else 0.0
    lo_m = (gen <= values[above - 1]).mean() if above > 0 else 0.0
    if above >= len(values):
        return float(lo_m)
    hi_f, hi_m = fmr[above], (gen <= values[above]).mean()
    return float(lo_m + (hi_m - lo_m) * (target - lo_f) / (hi_f - lo_f))


def bootstrap(g: pd.DataFrame, i: pd.DataFrame, algs: list[str]) -> pd.DataFrame:
    """Resample images, recalibrating every threshold on the pairs the resample keeps.

    Impostor pairs share images, so a pair is weighted by how often its two images were drawn.
    The same draw is used for every hash, which makes differences between hashes paired.
    """
    images = np.array(sorted(set(i["image_a"]) | set(i["image_b"])))
    index = pd.Series(np.arange(len(images)), index=images)
    attacked = np.array(sorted(g["image"].unique()))
    attacked_idx = index.loc[attacked].to_numpy()
    prepared = {}
    for a in algs:
        im = i[i["algorithm"] == a]
        order = np.argsort(im["norm_distance"].to_numpy())
        dist = im["norm_distance"].to_numpy()[order]
        ia = index.loc[im["image_a"]].to_numpy()[order]
        ib = index.loc[im["image_b"]].to_numpy()[order]
        wide = (g[g["algorithm"] == a].pivot_table(index="image", columns=["attack", "param"], values="norm_distance")
                .reindex(attacked).to_numpy())
        prepared[a] = (dist, ia, ib, wide)
    rng = np.random.default_rng(0)
    rows = np.empty((N_BOOT, len(algs)))
    for b in range(N_BOOT):
        counts = rng.multinomial(len(images), np.full(len(images), 1 / len(images))).astype(np.float64)
        w_gen = counts[attacked_idx]
        for j, a in enumerate(algs):
            dist, ia, ib, wide = prepared[a]
            w = counts[ia] * counts[ib]
            ends = np.r_[np.flatnonzero(dist[1:] != dist[:-1]), len(dist) - 1]
            fmr = np.cumsum(w)[ends] / w.sum()
            ok = np.flatnonzero(fmr <= PRIMARY)
            t = dist[ends[ok[-1]]] if len(ok) else -1.0
            rows[b, j] = ((wide <= t).mean(axis=1) * w_gen).sum() / w_gen.sum()
    return pd.DataFrame(rows, columns=algs)


def rate(d: pd.DataFrame, t: float) -> pd.Series:
    return d["norm_distance"] <= t


def heatmap(family: pd.DataFrame, order: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    table = family.loc[order, FAMILY_LABELS]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    mesh = ax.pcolormesh(table.to_numpy(), cmap=cmap, vmin=0, vmax=1, edgecolors="#fcfcfb", linewidth=2)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(FAMILY_LABELS)) + 0.5, FAMILY_LABELS, fontsize=9, color="#3d3d3a")
    ax.set_yticks(np.arange(len(order)) + 0.5, [LABELS[a] for a in order], fontsize=9, color="#3d3d3a")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for (r, c), v in np.ndenumerate(table.to_numpy()):
        ax.text(c + 0.5, r + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="#ffffff" if v >= 0.45 else "#1f1f1e")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Share of copies matched (FPR 1e-3)", fontsize=9, color="#3d3d3a")
    cb.ax.tick_params(labelsize=8, length=0, colors="#3d3d3a")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def roc_low_fpr(g: pd.DataFrame, i: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("dinohash_untrained", "#2a78d6"), ("dinohash", "#eb6834"), ("certphash_adv", "#1baf7a"),
              ("neuralhash", "#eda100"), ("pdq_dihedral", "#e87ba4"), ("phash", "#008300")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for a, color in series:
        imp = i.loc[i["algorithm"] == a, "norm_distance"].to_numpy()
        gen = np.sort(g.loc[g["algorithm"] == a, "norm_distance"].to_numpy())
        values, counts = np.unique(imp, return_counts=True)
        fmr = np.cumsum(counts) / imp.size
        tpr = np.searchsorted(gen, values, side="right") / gen.size
        keep = fmr <= 0.1
        ax.plot(fmr[keep], 100 * tpr[keep], color=color, lw=2, label=LABELS[a], solid_capstyle="round")
    ax.axvline(PRIMARY, color="#b4b3ae", lw=1, ls=(0, (3, 3)), zorder=0)
    ax.text(PRIMARY * 1.06, 52.5, "FPR 1e-3", fontsize=8, color="#6b6a65")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.1)
    ax.set_ylim(50, 100)
    ax.set_xlabel("False-match rate (different images matched)", fontsize=9, color="#3d3d3a")
    ax.set_ylabel("Copies matched (%)", fontsize=9, color="#3d3d3a")
    ax.grid(axis="y", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=0, colors="#3d3d3a")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b4b3ae")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncols=2, labelcolor="#3d3d3a")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    g, i, algs = load(args.runs)
    fam_of = {a: f for f, attacks in FAMILIES.items() for a in attacks}
    g["family"] = g["attack"].map(fam_of)
    overall = pd.concat([pd.read_csv(r / "overall.csv") for r in args.runs]).set_index("algorithm")

    imp = {a: d["norm_distance"].to_numpy() for a, d in i.groupby("algorithm")}
    gen = {a: d for a, d in g.groupby("algorithm")}
    thr = {a: {f: threshold_at_fpr(imp[a], f) for f in FPRS} for a in algs}

    # Thresholds must reproduce the per-run reports.
    for a in algs:
        for f in FPRS:
            assert abs(thr[a][f] - overall.loc[a, f"threshold_at_fpr_{f:g}"]) < 1e-12, (a, f)

    head, achieved = [], []
    for a in algs:
        d = gen[a]
        fams = [rate(d[d["family"] == f], thr[a][PRIMARY]).mean() for f in FAMILIES]
        row = {"algorithm": a, **{f"match_fpr_{f:g}": rate(d, thr[a][f]).mean() for f in FPRS},
               "match_exact_fpr_0.001": exact_fpr_match(imp[a], d["norm_distance"].to_numpy(), PRIMARY),
               "equal_family_mean": float(np.mean(fams)),
               "lowest_setting": d.assign(m=rate(d, thr[a][PRIMARY])).groupby(["attack", "param"])["m"].mean().min(),
               "settings_at_least_95pct": int((d.assign(m=rate(d, thr[a][PRIMARY])).groupby(["attack", "param"])["m"].mean() >= 0.95).sum()),
               "auc": overall.loc[a, "auc_all_attacks"], "eer": overall.loc[a, "eer_all_attacks"],
               "threshold_fpr_0.001": thr[a][PRIMARY]}
        head.append(row)
        achieved.append({"algorithm": a, **{f"achieved_fpr_{f:g}": float((imp[a] <= thr[a][f]).mean()) for f in FPRS},
                         **{f"false_matches_{f:g}": int((imp[a] <= thr[a][f]).sum()) for f in FPRS}})
    head = pd.DataFrame(head).sort_values(f"match_fpr_{PRIMARY:g}", ascending=False)
    head.to_csv(args.out / "headline.csv", index=False)
    pd.DataFrame(achieved).to_csv(args.out / "achieved_fpr.csv", index=False)
    order = list(head["algorithm"])

    family = pd.DataFrame({a: {lab: rate(gen[a][gen[a]["family"] == f], thr[a][PRIMARY]).mean()
                               for lab, f in zip(FAMILY_LABELS, FAMILIES)} for a in algs}).T
    family.loc[order].to_csv(args.out / "family.csv", index_label="algorithm")
    setting = pd.DataFrame({a: gen[a].assign(m=rate(gen[a], thr[a][PRIMARY])).groupby(["attack", "param"])["m"].mean()
                            for a in order})
    setting.to_csv(args.out / "per_setting.csv")

    rows = []
    for a, k in BITS.items():
        if a not in imp:
            continue
        hamming = np.rint(imp[a] * k)
        t = np.rint(thr[a][PRIMARY] * k)
        binomial_thr = max(b for b in range(k + 1) if binom.cdf(b, k, 0.5) <= PRIMARY)
        rows.append({"algorithm": a, "bits": k, "threshold_fpr_0.001_bits": int(t),
                     "observed_fpr_at_threshold": float((hamming <= t).mean()),
                     "binomial_fpr_at_threshold": float(binom.cdf(t, k, 0.5)),
                     "times_higher_than_binomial": float((hamming <= t).mean() / binom.cdf(t, k, 0.5)),
                     "binomial_threshold_for_fpr_0.001_bits": binomial_thr,
                     "observed_fpr_at_binomial_threshold": float((hamming <= binomial_thr).mean()),
                     "binomial_std_bits": float(np.sqrt(k) / 2), "observed_std_bits": float(hamming.std())})
    binomial = pd.DataFrame(rows)
    binomial.to_csv(args.out / "binomial_vs_observed.csv", index=False)

    boot = bootstrap(g, i, algs)
    point = head.set_index("algorithm")[f"match_fpr_{PRIMARY:g}"]
    ci = pd.DataFrame({"match_fpr_0.001": point, "ci_low": boot.quantile(0.025), "ci_high": boot.quantile(0.975)}).loc[order]
    ci.to_csv(args.out / "bootstrap.csv", index_label="algorithm")
    diff = pd.DataFrame([{"a": a, "b": b, "difference": point[a] - point[b],
                          "ci_low": (boot[a] - boot[b]).quantile(0.025), "ci_high": (boot[a] - boot[b]).quantile(0.975)}
                         for a, b in DIFFERENCES])
    diff.to_csv(args.out / "bootstrap_differences.csv", index=False)

    d2 = gen["dinohash"].groupby(["attack", "param"])["distance"].mean()
    d3 = gen["dinohash_untrained"].groupby(["attack", "param"])["distance"].mean()
    bits = pd.DataFrame({"trained": d2, "untrained": d3})
    bits["family"] = bits.index.get_level_values("attack").map(fam_of)
    bits.to_csv(args.out / "bits_trained_vs_untrained.csv")

    wide_imp = i.pivot_table(index=["image_a", "image_b"], columns="algorithm", values="norm_distance")
    wide_imp.corr(method="spearman").loc[order, order].to_csv(args.out / "impostor_corr.csv")
    per_image = pd.DataFrame({a: rate(gen[a], thr[a][PRIMARY]).groupby(gen[a]["image"]).mean() for a in order})
    per_image.corr(method="spearman").to_csv(args.out / "image_difficulty_corr.csv")

    keyed = {a: gen[a].set_index(["image", "attack", "param"])["norm_distance"] for a in algs}
    wide_gen = pd.DataFrame(keyed)
    fam_series = pd.Series(wide_gen.index.get_level_values("attack")).map(fam_of).to_numpy()
    rows = []
    for combo in ENSEMBLES:
        budget = PRIMARY / len(combo)
        ts = {a: threshold_at_fpr(imp[a], budget) for a in combo}
        false = np.logical_or.reduce([wide_imp[a].to_numpy() <= ts[a] for a in combo])
        match = np.logical_or.reduce([wide_gen[a].to_numpy() <= ts[a] for a in combo])
        rows.append({"hashes": " + ".join(combo), "budget_each": budget, "measured_fpr": false.mean(),
                     "mean_match": match.mean(),
                     **{lab: match[fam_series == f].mean() for lab, f in zip(FAMILY_LABELS, FAMILIES)},
                     "equal_family_mean": float(np.mean([match[fam_series == f].mean() for f in FAMILIES]))})
    ensemble = pd.DataFrame(rows)
    ensemble.to_csv(args.out / "ensemble.csv", index=False)

    heatmap(family, order, args.out / "family_heatmap.png")
    roc_low_fpr(g, i, args.out / "roc_low_fpr.png")

    with pd.option_context("display.width", 230, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        for name, t in [("headline", head), ("bootstrap", ci), ("differences", diff), ("binomial", binomial),
                        ("ensemble", ensemble), ("impostor correlation", wide_imp.corr(method="spearman").loc[order, order]),
                        ("image difficulty", per_image.corr(method="spearman"))]:
            print(f"\n=== {name}\n{t.round(4).to_string()}")
        fewer = bits[bits["trained"] < bits["untrained"]]
        print(f"\ntrained flips fewer bits on {len(fewer)} of {len(bits)} settings; mean {bits['trained'].mean():.2f} vs {bits['untrained'].mean():.2f}")
        print(bits.groupby("family")[["trained", "untrained"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
