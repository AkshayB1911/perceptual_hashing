"""Transfer attack, step 2: re-hash saved adversarial images with every target hash.

Reads one or more folders written by ``generate_transfer_images.py`` and, for every target
hash B and every saved image made against a source hash A, decides whether the attack
transfers. No attacking happens here, only hashing, so it runs on the CPU.

Comparisons are made at the source's input size, so only the perturbation differs:

    evasion    d_B( B(adv_A), B(clean_A) )     >  thr_B      the copy is no longer found by B
    collision  d_B( B(adv_A), B(clean_A of the target image) )  <=  thr_B   a false match on B
    noise      like evasion, but adv_A is replaced by the clean image plus random +-eps
               (sign) noise, 8-bit. The control: a transfer only means something if it beats
               random noise of the same size.

Thresholds are the benign benchmark's calibrated FPR thresholds (1e-3 by default), the same
operating point as ``run_adversarial.py``. The diagonal (A == B) must reproduce the
white-box success in generation.csv; ``summary.md`` reports any mismatch.

Collision rates are reported over the pairs that did not already match before the attack
(the "already" column gives that share).

Outputs (``--out``): raw.csv (one row per source x target x mode x eps x image), summary.csv,
summary.md (source x target matrices) and heatmaps transfer_<mode>_eps<k>.png.
Resumable per (source, target): pairs already in raw.csv are skipped.

Example (from hashbench/):
    python scripts/transfer_matrix.py --gen results/transfer_v200 results/transfer_v200_neuralhash \
        --out results/transfer_v200_matrix
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashbench.hashes import get_hasher  # noqa: E402
from run_adversarial import load_thresholds  # noqa: E402
from generate_transfer_images import png_path  # noqa: E402

DEFAULT_TARGETS = ["dinohash", "dinohash_untrained", "neuralhash", "certphash_base", "certphash_adv",
                   "certphash", "certphash_ep2", "certphash_ep4", "certphash_ep8", "phash", "pdq"]
SHORT = {"dinohash": "Dino-T", "dinohash_untrained": "Dino-U", "neuralhash": "NH",
         "certphash_base": "C-base", "certphash_adv": "C-adv", "certphash": "C-ε1", "certphash_ep2": "C-ε2",
         "certphash_ep4": "C-ε4", "certphash_ep8": "C-ε8", "phash": "pHash", "pdq": "PDQ"}
COLUMNS = ["source", "target", "mode", "eps_255", "image", "target_image", "threshold", "max_distance",
           "distance", "orig_distance", "already", "success", "wb_success"]


def noise_image(clean: Image.Image, eps: float, key: str) -> Image.Image:
    """clean + random +-eps/255 per pixel and channel, clipped, 8-bit. Deterministic per key."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    x = np.asarray(clean, dtype=np.int16)
    x = x + rng.choice(np.array([-1, 1], dtype=np.int16), size=x.shape) * int(round(eps))
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))


def load_jobs(gen_dirs: list[Path], noise_eps: list[float] | None) -> tuple[pd.DataFrame, dict]:
    """All (source, mode, eps, image, target_image, wb_success) to evaluate, and source -> folder."""
    jobs, root_of = [], {}
    for g in gen_dirs:
        df = pd.read_csv(g / "generation.csv")
        for src in df["hash"].unique():
            if src in root_of:
                raise SystemExit(f"source {src} appears in two folders")
            root_of[src] = g
        jobs.append(df.assign(source=df["hash"], target_image=df["target"].fillna(""),
                              wb_success=df["success"])[["source", "mode", "eps_255", "image",
                                                         "target_image", "wb_success"]])
        cfg = json.loads((g / "config.json").read_text())
        eps_list = noise_eps if noise_eps is not None else cfg["eps_evasion"]
        names = df["image"].unique()
        for src in df["hash"].unique():
            for eps in eps_list:
                jobs.append(pd.DataFrame({"source": src, "mode": "noise", "eps_255": float(eps), "image": names,
                                          "target_image": "", "wb_success": np.nan}))
    return pd.concat(jobs, ignore_index=True), root_of


def hash_all(hasher, imgs: list[Image.Image], chunk: int = 64) -> np.ndarray:
    return np.concatenate([hasher.hash_batch(imgs[i:i + chunk]) for i in range(0, len(imgs), chunk)])


def evaluate(target: str, hasher, thr: int, jobs: pd.DataFrame, root_of: dict) -> pd.DataFrame:
    out = []
    for src, sj in jobs.groupby("source", sort=False):
        root = root_of[src]
        names = sorted(set(sj["image"]) | set(sj["target_image"]) - {""})
        clean_imgs = {nm: Image.open(png_path(root, src, "clean", None, nm)).convert("RGB") for nm in names}
        clean_h = dict(zip(names, hash_all(hasher, [clean_imgs[nm] for nm in names])))
        imgs = []
        for r in sj.itertuples():
            if r.mode == "noise":
                imgs.append(noise_image(clean_imgs[r.image], r.eps_255, f"{src}|{r.image}|{r.eps_255:g}"))
            else:
                imgs.append(Image.open(png_path(root, src, r.mode, r.eps_255, r.image)).convert("RGB"))
        adv_h = hash_all(hasher, imgs)
        ref_h = np.stack([clean_h[t] if m == "collision" else clean_h[i]
                          for m, i, t in zip(sj["mode"], sj["image"], sj["target_image"])])
        own_h = np.stack([clean_h[i] for i in sj["image"]])
        dist = hasher.distances(adv_h, ref_h)
        orig = hasher.distances(own_h, ref_h)
        coll = (sj["mode"] == "collision").to_numpy()
        success = np.where(coll, dist <= thr, dist > thr)
        already = np.where(coll, orig <= thr, False)
        out.append(sj.assign(target=target, threshold=thr, max_distance=hasher.max_distance, distance=dist,
                             orig_distance=orig, already=already, success=success))
    return pd.concat(out, ignore_index=True)[COLUMNS]


def summarize(raw: pd.DataFrame, out: Path, sources: list[str], targets: list[str]) -> None:
    raw = raw.copy()
    raw["norm_distance"] = raw["distance"] / raw["max_distance"]
    rows = []
    for (src, tgt, mode, eps), g in raw.groupby(["source", "target", "mode", "eps_255"]):
        valid = g[~g["already"]] if mode == "collision" else g
        rows.append({"source": src, "target": tgt, "mode": mode, "eps_255": eps, "n": len(valid),
                     "already_pct": 100 * g["already"].mean(),
                     "success_pct": 100 * valid["success"].mean() if len(valid) else np.nan,
                     "mean_norm_distance": g["norm_distance"].mean(),
                     "success_pct_given_wb": 100 * valid.loc[valid["wb_success"] == True, "success"].mean()
                     if (valid["wb_success"] == True).any() else np.nan})
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)

    src_order = [x for x in sources if x in set(s["source"])]
    tgt_order = [x for x in targets if x in set(s["target"])]
    lines = ["# Transfer attack: source x target matrices", "",
             "Rows: the hash that was attacked (white-box). Columns: the hash the attacked image is "
             "re-hashed with. Cells: success % at the target's calibrated FPR 1e-3 threshold. "
             "Diagonal = white-box. `noise` = random ±ε sign noise of the same size (control).", ""]
    for mode in ["evasion", "noise", "collision"]:
        for eps in sorted(s.loc[s["mode"] == mode, "eps_255"].unique()):
            m = s[(s["mode"] == mode) & (s["eps_255"] == eps)].pivot(index="source", columns="target",
                                                                   values="success_pct")
            m = m.reindex(index=src_order, columns=tgt_order)
            title = {"evasion": "Evasion", "noise": "Random-noise control (evasion)",
                     "collision": "Collision, random target (pairs not matching before)"}[mode]
            lines += [f"## {title}, ε = {eps:g}/255", "",
                      "| source \\ target | " + " | ".join(SHORT.get(t, t) for t in m.columns) + " |",
                      "|---|" + "---|" * len(m.columns)]
            for src, row in m.iterrows():
                cells = []
                for t, v in row.items():
                    txt = "–" if pd.isna(v) else f"{v:.1f}"
                    cells.append(f"**{txt}**" if t == src else txt)
                lines.append(f"| {SHORT.get(src, src)} | " + " | ".join(cells) + " |")
            lines.append("")
            _heatmap(m, f"{title}, ε = {eps:g}/255", out / f"transfer_{mode}_eps{eps:g}.png")

    # diagonal check against the white-box numbers from generation.csv
    diag = raw[(raw["source"] == raw["target"]) & (raw["mode"] != "noise")]
    bad = diag[diag["success"] != diag["wb_success"].astype(bool)]
    lines += ["## Diagonal check", "",
              f"Re-hashing the saved PNGs with the source hash itself reproduces the white-box result for "
              f"{len(diag) - len(bad)} of {len(diag)} images."]
    if len(bad):
        lines.append(f"Mismatches ({len(bad)}), e.g.: " + ", ".join(
            f"{r.source} {r.mode} ε{r.eps_255:g} {r.image}" for r in bad.head(5).itertuples()))
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _heatmap(m: pd.DataFrame, title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * m.shape[1], 1.0 + 0.55 * m.shape[0]))
    im = ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(m.shape[1]), [SHORT.get(t, t) for t in m.columns], rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]), [SHORT.get(t, t) for t in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 50 else "black")
    ax.set_xlabel("target (re-hashed)")
    ax.set_ylabel("source (attacked)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="success %")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", nargs="+", type=Path, required=True, help="generate_transfer_images.py folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--noise-eps", nargs="+", type=float, default=None,
                    help="budgets for the noise control (default: the evasion budgets)")
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--thresholds-from", nargs="+", type=Path,
                    default=[Path("results/coco_run1"), Path("results/coco_run2")])
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "raw.csv"
    jobs, root_of = load_jobs(args.gen, args.noise_eps)
    sources = [s for s in DEFAULT_TARGETS if s in root_of] + [s for s in root_of if s not in DEFAULT_TARGETS]
    if not args.summary_only:
        thresholds = load_thresholds(args.thresholds_from, args.fpr)
        missing = [t for t in args.targets if t not in thresholds]
        if missing:
            raise SystemExit(f"no FPR {args.fpr:g} threshold for {missing}")
        done = set()
        if raw_path.exists():
            d = pd.read_csv(raw_path, usecols=["source", "target"])
            done = set(zip(d["source"], d["target"]))
        t0 = time.time()
        for tgt in args.targets:
            todo = jobs[[(s, tgt) not in done for s in jobs["source"]]]
            if todo.empty:
                continue
            hasher = get_hasher(tgt)
            thr = math.floor(thresholds[tgt] * hasher.max_distance + 1e-6)
            for src, sj in todo.groupby("source", sort=False):
                res = evaluate(tgt, hasher, thr, sj, root_of)
                res.to_csv(raw_path, mode="a", header=not raw_path.exists(), index=False)
                ev = res[res["mode"] == "evasion"]
                print(f"[{time.time() - t0:6.0f}s] {src:18s} -> {tgt:18s} thr {thr:5d}  "
                      f"evasion transfer {100 * ev['success'].mean():5.1f}%  ({len(res)} images)", flush=True)
            del hasher
    raw = pd.read_csv(raw_path)
    summarize(raw, args.out, sources, args.targets)
    print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
