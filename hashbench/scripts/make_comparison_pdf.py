"""Assemble the benign and adversarial plots into one PDF (headings + figures only).

    python scripts/make_comparison_pdf.py
Run plot_benign.py and plot_adversarial.py first.
"""
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "adv_v200_plots"
OUT = ROOT / "results" / "benign_vs_adversarial.pdf"
A4L = (11.69, 8.27)

SECTIONS = [
    ("Benign", ["benign_by_model.png", "benign_certphash_variants.png"]),
    ("Adversarial", ["success_by_model.png", "certphash_variants.png",
                     "certphash_paper_thr.png"]),
]


def page(pdf, heading, images):
    fig = plt.figure(figsize=A4L)
    top = 0.97
    if heading:
        fig.text(0.04, 0.95, heading, fontsize=22, fontweight="bold", color="#1f1f1e",
                 va="top")
        top = 0.89
    slot = (top - 0.02) / len(images)
    for i, name in enumerate(images):
        img = mpimg.imread(PLOTS / name)
        ax = fig.add_axes([0.03, top - slot * (i + 1) + 0.01, 0.94, slot - 0.02])
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def main():
    with PdfPages(OUT) as pdf:
        for heading, imgs in SECTIONS:
            if heading == "Benign":  # tall grids: one per page
                page(pdf, heading, imgs[:1])
                for im in imgs[1:]:
                    page(pdf, None, [im])
            else:  # wide strips: two per page
                page(pdf, heading, imgs[:2])
                for j in range(2, len(imgs), 2):
                    page(pdf, None, imgs[j:j + 2])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
