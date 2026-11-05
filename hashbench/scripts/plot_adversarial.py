"""Plot adversarial success rates from the V200 sweep summaries.

    python scripts/plot_adversarial.py [--out results/adv_v200_plots]

Reads results/adv_v200_kaggle/summary.csv (DinoHash, CertPHash variants) and
results/adv_v200_neuralhash/summary.csv, and writes:
  success_by_model.png    - evasion / random collision / closest collision, main models
  certphash_variants.png  - the same three panels for the CertPHash training variants
  certphash_paper_thr.png - CertPHash variants scored at the paper's L1 threshold 1800
Success is at the calibrated FPR 1e-3 threshold unless stated; bands are 95% Wilson CIs.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODES = [("evasion", "Evasion"),
         ("collision", "Collision: random target"),
         ("collision_closest", "Collision: closest target")]

# fixed categorical order; light-mode palette (validated for CVD separation)
MAIN = [  # key, label, color, linestyle, marker
    ("dinohash", "DinoHash (trained)", "#2a78d6", "-", "o"),
    ("dinohash_untrained", "DinoHash (untrained)", "#2a78d6", "--", "o"),
    ("neuralhash", "NeuralHash", "#eb6834", "-", "s"),
    ("certphash", "CertPHash (ep1, certified)", "#1baf7a", "-", "D"),
    ("certphash_base", "CertPHash base (not certified)", "#1baf7a", "--", "D"),
]
# ordinal blue ramp for certification strength, the "adv" checkpoint in orange
CERT = [
    ("certphash_base", "base (ep0)", "#9ec5f4", "-", "o"),
    ("certphash", "ep1", "#6da7ec", "-", "o"),
    ("certphash_ep2", "ep2", "#3987e5", "-", "o"),
    ("certphash_ep4", "ep4", "#256abf", "-", "o"),
    ("certphash_ep8", "ep8", "#104281", "-", "o"),
    ("certphash_adv", "adv", "#eb6834", "--", "s"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load():
    frames = [pd.read_csv(ROOT / "results" / d / "summary.csv")
              for d in ("adv_v200_kaggle", "adv_v200_neuralhash")]
    return pd.concat(frames, ignore_index=True)


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("ε (/255, ℓ∞)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def panel_figure(df, series, path, suptitle, rate="success_rate", lo="ci_low", hi="ci_high"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (mode, title) in zip(axes, MODES):
        for key, label, color, ls, mk in series:
            d = df[(df.hash == key) & (df["mode"] == mode)].sort_values("eps_255")
            if d.empty or d[rate].isna().all():
                continue
            x, y = d.eps_255, 100 * d[rate]
            ax.fill_between(x, 100 * d[lo], 100 * d[hi], color=color, alpha=0.10, linewidth=0)
            ax.plot(x, y, color=color, linestyle=ls, linewidth=2, marker=mk, markersize=6,
                    markeredgecolor="white", markeredgewidth=1, label=label)
        style(ax, title)
    axes[0].set_ylabel("Attack success (%)", color=MUTED)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "adv_v200_plots"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    n = "V200, n=200 images per point; success at calibrated FPR 1e-3 threshold; bands = 95% Wilson CI"
    panel_figure(df, MAIN, out / "success_by_model.png", f"White-box attack success by model ({n})")
    panel_figure(df, CERT, out / "certphash_variants.png",
                 f"CertPHash training variants ({n})")
    panel_figure(df, CERT, out / "certphash_paper_thr.png",
                 "CertPHash variants at the paper's L1 threshold 1800 (V200, n=200; bands = 95% Wilson CI)",
                 rate="success_rate_paper_thr", lo="ci_low_paper", hi="ci_high_paper")


if __name__ == "__main__":
    main()
