"""Write a benchmark run's headline tables as Markdown, in the format of benchmark_findings_run1.md.

Needs hashbench.report and scripts/analyze_run.py to have been run first. With --reference,
also checks that the run used the same settings as that run's config.json, so the two can be
compared.

Usage:
    python scripts/write_run_summary.py --run results/coco_run2 \
        --reference results/coco_run1_config.json --out results/coco_run2/RESULTS_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FPRS = ["0.01", "0.001", "0.0001"]
FAMILIES = {
    "Compression": "compression/resampling", "Filtering/noise": "filtering/noise",
    "Photometric": "photometric", "Geometric": "geometric", "Flips": "flips", "Overlays": "overlays",
}
COMPARED_SETTINGS = ["attack_suite", "max_side", "max_pairs", "seed", "n_images", "n_attacked"]


def opt(value: float, fmt: str) -> str:
    return "–" if pd.isna(value) else format(value, fmt)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None, help="config.json of the run to compare with")
    args = ap.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    overall = pd.read_csv(args.run / "overall.csv").set_index("algorithm")
    summary = pd.read_csv(args.run / "summary.csv")
    family = pd.read_csv(args.run / "analysis" / "family_match_rates.csv")
    zero = pd.read_csv(args.run / "analysis" / "zero_fpr.csv").set_index("algorithm")
    bits = pd.read_csv(args.run / "analysis" / "bit_balance.csv")
    genuine = pd.read_csv(args.run / "genuine.csv")
    if "error" not in genuine:  # runs made before the checkpointed benchmark omit the column when empty
        genuine["error"] = pd.NA
    hashes = pd.read_csv(args.run / "hashes.csv", usecols=["image"])
    impostor_rows = pd.read_csv(args.run / "impostor.csv", usecols=["algorithm"])["algorithm"].value_counts()

    means = summary.groupby("algorithm")[[f"match_rate_fpr_{f}" for f in FPRS] + ["match_rate_recommended"]].mean()
    order = means[f"match_rate_fpr_{FPRS[1]}"].sort_values(ascending=False).index

    out = [f"# Results summary: {args.run.name}", ""]
    out += ["## Row counts", ""]
    n_settings = summary.groupby("algorithm").size()
    out.append(md_table(
        ["Hash", "Genuine rows", "Failed attacks", "Attacked images", "Settings", "Impostor pairs"],
        [[a, f"{(genuine['algorithm'] == a).sum():,}",
          f"{genuine.loc[genuine['algorithm'] == a, 'error'].notna().sum():,}",
          f"{genuine.loc[genuine['algorithm'] == a, 'image'].nunique():,}",
          str(n_settings[a]), f"{impostor_rows[a]:,}"] for a in order]))
    out += ["", f"Images hashed: {len(hashes):,}", ""]

    out += ["## Headline numbers", "", "**Mean match rate over all settings (%)**", ""]
    out.append(md_table(
        ["Hash", "FPR 1e-2", "FPR 1e-3", "FPR 1e-4", "Published threshold", "AUC"],
        [[a, f"{100 * means.loc[a, 'match_rate_fpr_0.01']:.1f}",
          f"**{100 * means.loc[a, 'match_rate_fpr_0.001']:.1f}**",
          f"{100 * means.loc[a, 'match_rate_fpr_0.0001']:.1f}",
          opt(100 * means.loc[a, 'match_rate_recommended'], ".1f"),
          f"{overall.loc[a, 'auc_all_attacks']:.3f}"] for a in order]))

    out += ["", "**Match rate by attack family at FPR 1e-3**", ""]
    pivot = family.pivot(index="algorithm", columns="family", values="match_fpr_0.001")
    out.append(md_table(["Hash", *FAMILIES],
                        [[a, *[f"{pivot.loc[a, f]:.2f}" for f in FAMILIES.values()]] for a in order]))

    out += ["", "**95% bootstrap intervals by family at FPR 1e-3**", ""]
    family = family.set_index(["algorithm", "family"])
    out.append(md_table(["Hash", *FAMILIES],
                        [[a, *[f"{family.loc[(a, f), 'ci_low']:.3f}-{family.loc[(a, f), 'ci_high']:.3f}"
                               for f in FAMILIES.values()]] for a in order]))

    out += ["", "## Thresholds and impostor distances", ""]
    out.append(md_table(
        ["Hash", "Threshold, FPR 1e-3", "Published threshold", "Impostor mean", "Impostor std", "Impostor min",
         "Match rate below impostor min"],
        [[a, f"{overall.loc[a, 'threshold_at_fpr_0.001']:.4f}", opt(overall.loc[a, 'threshold_recommended'], ".4f"),
          f"{overall.loc[a, 'impostor_mean']:.4f}", f"{overall.loc[a, 'impostor_std']:.4f}",
          f"{overall.loc[a, 'impostor_min']:.4f}",
          f"{100 * zero.loc[a, 'match_rate_below_min_impostor']:.1f}%"] for a in order]))

    out += ["", "## Bit statistics", ""]
    out.append(md_table(
        ["Hash", "Bits", "Mean share of ones", "Lowest bit", "Highest bit", "Sum of bit entropy", "Mean abs. correlation"],
        [[r.algorithm, str(r.bits), f"{r.mean_p1:.3f}", f"{r.min_p1:.3f}", f"{r.max_p1:.3f}",
          f"{r.sum_bit_entropy:.1f}", f"{r.mean_abs_bit_corr:.3f}"] for r in bits.itertuples()]))

    if args.reference:
        ref = json.loads(args.reference.read_text())
        out += ["", f"## Settings compared with {args.reference.name}", ""]
        out.append(md_table(["Setting", "Same?"], [
            [k, "yes" if config.get(k) == ref.get(k) else f"NO: {config.get(k)} vs {ref.get(k)}"]
            for k in COMPARED_SETTINGS]))

    args.out.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
