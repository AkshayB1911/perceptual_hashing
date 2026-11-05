"""Plot benign-transformation match rates from the COCO benchmark runs.

    python scripts/plot_benign.py [--out results/adv_v200_plots]

Reads results/coco_run1/summary.csv and results/coco_run2/summary.csv and writes:
  benign_by_model.png           - one panel per transformation, main models
  benign_certphash_variants.png - the same panels for the CertPHash training variants
Match rate is at each model's FPR 1e-3 threshold (the threshold the adversarial
sweep uses); bands are 95% Wilson CIs, n=1000 images per point.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_adversarial import CERT, GRID, INK, MAIN, MUTED, ROOT

PRETTY = {
    "aspect_ratio": "Aspect ratio", "border": "Border (frac.)", "brightness": "Brightness",
    "center_crop": "Center crop (kept)", "contrast": "Contrast", "corner_crop": "Corner crop (kept)",
    "flip_horizontal": "Flip horizontal", "flip_vertical": "Flip vertical", "gamma": "Gamma",
    "gaussian_blur": "Gaussian blur (σ)", "gaussian_noise": "Gaussian noise (σ)",
    "grayscale": "Grayscale", "hue_shift": "Hue shift (°)", "jpeg": "JPEG quality",
    "median_filter": "Median filter (k)", "occlusion": "Occlusion (area)",
    "rescale": "Rescale (factor)", "rotate": "Rotate (°)", "rotate_crop": "Rotate + crop (°)",
    "salt_and_pepper": "Salt & pepper (frac.)", "saturation": "Saturation",
    "text_overlay": "Text overlay (size)", "translate": "Translate (frac.)",
}


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load():
    d = pd.concat([pd.read_csv(ROOT / "results" / r / "summary.csv")
                   for r in ("coco_run1", "coco_run2")], ignore_index=True)
    d["rate"] = d["match_rate_fpr_0.001"]
    d["lo"], d["hi"] = wilson(d.rate, d.n)
    return d


def fmt(v):
    return f"{v:g}"


def grid_figure(df, series, path, suptitle):
    attacks = sorted(df.attack.unique(), key=lambda a: PRETTY.get(a, a))
    ncol = 6
    nrow = math.ceil(len(attacks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.0 * nrow), sharey=True)
    axes = axes.ravel()
    for ax, att in zip(axes, attacks):
        params = sorted(df[df.attack == att].param.unique())
        pos = {p: i for i, p in enumerate(params)}
        if len(params) == 1:  # one setting (flips, grayscale): grouped bars, one per model
            k = len(series)
            w = 0.8 / k
            for i, (key, label, color, ls, mk) in enumerate(series):
                d = df[(df.algorithm == key) & (df.attack == att)]
                if d.empty:
                    continue
                v = 100 * d.rate.iloc[0]
                ax.bar(-0.4 + w * (i + 0.5), v, width=w * 0.9, color=color,
                       hatch="//" if ls == "--" else None, edgecolor="white", linewidth=0)
                ax.text(-0.4 + w * (i + 0.5), v + 2, f"{v:.0f}", ha="center",
                        fontsize=7, color=MUTED)
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.5)
        for key, label, color, ls, mk in (series if len(params) > 1 else []):
            d = df[(df.algorithm == key) & (df.attack == att)].sort_values("param")
            if d.empty:
                continue
            x = [pos[p] for p in d.param]
            ax.fill_between(x, 100 * d.lo, 100 * d.hi, color=color, alpha=0.10, linewidth=0)
            ax.plot(x, 100 * d.rate, color=color, linestyle=ls, linewidth=2, marker=mk,
                    markersize=5, markeredgecolor="white", markeredgewidth=1, label=label)
        ax.set_title(PRETTY.get(att, att), color=INK, fontsize=10, loc="left")
        if len(params) > 1:
            ax.set_xticks(range(len(params)), [fmt(p) for p in params])
            ax.set_xlim(-0.4, len(params) - 1 + 0.4)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[len(attacks):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("Match rate (%)", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "adv_v200_plots"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    n = "COCO, n=1000 images per point; match at FPR 1e-3 threshold; bands = 95% Wilson CI"
    grid_figure(df, MAIN, out / "benign_by_model.png",
                f"Benign transformations: match rate by model ({n})")
    grid_figure(df, CERT, out / "benign_certphash_variants.png",
                f"Benign transformations: CertPHash training variants ({n})")


if __name__ == "__main__":
    main()
