"""Summary tables and plots for a benchmark run.

Two kinds of thresholds are reported for each algorithm:
    calibrated  - chosen from this run's impostor distances so that every algorithm
                  has the same false match rate (--fpr). This is the fair comparison.
                  Several rates can be given (e.g. --fpr 0.01 0.001 0.0001); the first is
                  the primary one used for ``match_rate_calibrated`` and the plots.
    recommended - the threshold commonly used in practice for that algorithm.

Match rates come with a 95% Wilson interval over images (``match_rate_ci_low/high``).

Outputs (in the run folder):
    overall.csv          per algorithm: AUC, EER, thresholds, impostor statistics
    summary.csv          per algorithm x attack x param: mean distance and match rates
                         (with 95% intervals), one match-rate column per --fpr
    attack_curves.png    match rate vs attack strength, one panel per attack
    distance_hist.png    genuine vs impostor distance distributions
    roc.png              ROC curves over all attacks

Usage:
    python -m hashbench.report --run results/smoke
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from hashbench import metrics  # noqa: E402

#: Normalized thresholds used in practice: pHash <= 10/64 bits is a common choice,
#: PDQ <= 31/256 is the value recommended by Meta, CertPHash (PhotoDNA-style) L1 <= 1800
#: of 144 * 255 is the paper's.
RECOMMENDED_THRESHOLDS = {"phash": 10 / 64, "pdq": 31 / 256, "pdq_dihedral": 31 / 256,
                          **{f"certphash{v}": 1800 / (144 * 255)
                             for v in ("", "_ep2", "_ep4", "_ep8", "_base", "_adv")}}


def summarize(run: Path, target_fprs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    genuine = pd.read_csv(run / "genuine.csv")
    if "error" in genuine:
        genuine = genuine[genuine["error"].isna()]
    impostor = pd.read_csv(run / "impostor.csv")

    overall_rows, thresholds = [], {}
    for alg, imp in impostor.groupby("algorithm"):
        gen = genuine.loc[genuine["algorithm"] == alg, "norm_distance"].to_numpy()
        imp_d = imp["norm_distance"].to_numpy()
        for fpr in target_fprs:
            if len(imp_d) < 10 / fpr:
                print(f"warning: only {len(imp_d)} impostor pairs for {alg}; the threshold at "
                      f"FPR={fpr:g} is unreliable (use more images)")
        t_cal = {fpr: metrics.threshold_at_fpr(imp_d, fpr) for fpr in target_fprs}
        t_rec = RECOMMENDED_THRESHOLDS.get(alg)
        thresholds[alg] = (t_cal, t_rec)
        eer, _ = metrics.equal_error_rate(gen, imp_d)
        overall_rows.append({
            "algorithm": alg,
            "auc_all_attacks": metrics.auc(gen, imp_d),
            "eer_all_attacks": eer,
            "impostor_pairs": len(imp_d),
            "impostor_mean": imp_d.mean(),
            "impostor_std": imp_d.std(),
            "impostor_min": imp_d.min(),
            **{f"threshold_at_fpr_{fpr:g}": t for fpr, t in t_cal.items()},
            "threshold_recommended": t_rec,
            "fpr_at_recommended": metrics.false_match_rate(imp_d, t_rec) if t_rec is not None else None,
        })
    overall = pd.DataFrame(overall_rows)

    summary_rows = []
    for (alg, attack, param), g in genuine.groupby(["algorithm", "attack", "param"]):
        d = g["norm_distance"].to_numpy()
        t_cal, t_rec = thresholds[alg]
        t_primary = t_cal[target_fprs[0]]
        lo, hi = metrics.wilson_interval(int((d <= t_primary).sum()), len(d))
        summary_rows.append({
            "algorithm": alg, "attack": attack, "param": param, "n": len(d),
            "mean_norm_distance": d.mean(), "std_norm_distance": d.std(),
            "match_rate_calibrated": metrics.match_rate(d, t_primary),
            "match_rate_ci_low": lo, "match_rate_ci_high": hi,
            **{f"match_rate_fpr_{fpr:g}": metrics.match_rate(d, t) for fpr, t in t_cal.items()},
            "match_rate_recommended": metrics.match_rate(d, t_rec) if t_rec is not None else None,
        })
    summary = pd.DataFrame(summary_rows)
    return overall, summary


def plot_attack_curves(summary: pd.DataFrame, path: Path, column: str) -> None:
    attacks = sorted(summary["attack"].unique())
    cols = 5
    rows = math.ceil(len(attacks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.8), squeeze=False)
    for ax, attack in zip(axes.flat, attacks):
        sub = summary[summary["attack"] == attack]
        for alg, g in sub.groupby("algorithm"):
            g = g.sort_values("param")
            x = g["param"].astype(str) if g["param"].nunique() == 1 else g["param"]
            ax.plot(x, g[column], marker="o", label=alg)
        ax.set_title(attack, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(attacks):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{column.replace('_', ' ')} vs attack parameter", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_distance_hist(run: Path, path: Path) -> None:
    genuine = pd.read_csv(run / "genuine.csv")
    impostor = pd.read_csv(run / "impostor.csv")
    algs = sorted(impostor["algorithm"].unique())
    fig, axes = plt.subplots(1, len(algs), figsize=(4.5 * len(algs), 3.2), squeeze=False)
    bins = [i / 64 for i in range(65)]
    for ax, alg in zip(axes.flat, algs):
        ax.hist(genuine.loc[genuine["algorithm"] == alg, "norm_distance"], bins=bins,
                density=True, alpha=0.6, label="genuine (all attacks)")
        ax.hist(impostor.loc[impostor["algorithm"] == alg, "norm_distance"], bins=bins,
                density=True, alpha=0.6, label="impostor")
        t = RECOMMENDED_THRESHOLDS.get(alg)
        if t is not None:
            ax.axvline(t, color="k", ls="--", lw=1, label="recommended threshold")
        ax.set_title(alg)
        ax.set_xlabel("normalized Hamming distance")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(run: Path, path: Path) -> None:
    genuine = pd.read_csv(run / "genuine.csv")
    impostor = pd.read_csv(run / "impostor.csv")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for alg, imp in impostor.groupby("algorithm"):
        gen = genuine.loc[genuine["algorithm"] == alg, "norm_distance"]
        fpr, tpr, _ = metrics.roc_curve(gen, imp["norm_distance"])
        ax.plot(fpr, tpr, label=f"{alg} (AUC={metrics.auc(gen, imp['norm_distance']):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("false match rate")
    ax.set_ylabel("true match rate")
    ax.set_title("ROC over all attacks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="folder written by hashbench.benchmark")
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.01],
                    help="target false match rate(s) for calibration; the first is the primary one")
    args = ap.parse_args(argv)

    overall, summary = summarize(args.run, args.fpr)
    overall.to_csv(args.run / "overall.csv", index=False)
    summary.to_csv(args.run / "summary.csv", index=False)
    plot_attack_curves(summary, args.run / "attack_curves.png", "match_rate_calibrated")
    plot_attack_curves(summary, args.run / "mean_distance_curves.png", "mean_norm_distance")
    plot_distance_hist(args.run, args.run / "distance_hist.png")
    plot_roc(args.run, args.run / "roc.png")

    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(overall.to_string(index=False))
        print()
        per_attack = summary.groupby(["attack", "algorithm"])["match_rate_calibrated"].mean().unstack()
        print(f"mean match rate per attack (threshold calibrated at FPR={args.fpr[0]:g}):")
        print(per_attack.to_string())
    print(f"\nwrote overall.csv, summary.csv and plots to {args.run}")


if __name__ == "__main__":
    main()
