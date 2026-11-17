"""Reproduce Table 1 of the DinoHash paper with the authors' own attack.

Singhi, Gupta, Struppek, "DINOHash", ICML 2025 CODEML (arXiv 2503.11195), Table 1:
bit accuracy (share of hash bits unchanged) after a 100-step white-box APGD attack,
eps = 4/255 and 8/255:  DINOHash 72.2 / 64.8, untrained 0.3 / 0.0, NeuralHash 0.1 / 0.0.

What is reproduced exactly (see adversarial_testing_report.md, section 4):
  * the attack: the authors' APGD @ c7008888b7, vendored unmodified (scripts/third_party/
    dinohash_apgd_c700888.py): BCE towards the flipped hash on 20 * unit(logits), random
    start in the eps-ball, step 2*eps halved on stalls, best-loss iterate, 100 steps;
  * the protocol of their first evaluation script (attack_results.py @ f8c2fd5d65):
    224x224 bilinear input in [0, 1], normalisation inside the hash, bit accuracy
    = mean(adv bits == clean bits), no 8-bit rounding;
  * the model: their eval code uses DINOv2 ViT-S/14-reg + PCA_96 head; the released
    ONNX head equals that PCA_96 file exactly (checked 2026-11-11), so `dinohash`
    below is the paper's trained model.
What differs: images (200 COCO val2017 from V200 instead of DiffusionDB), CPU instead of GPU.

Models (``--models``):
  dinohash                  released fine-tuned ViT-S/96 (= paper's trained model)
  dinohash_untrained_paper  Meta's DINOv2 + the PCA_96 head: what their eval code computes
                            without load_model(), i.e. the paper's "untrained" column
  dinohash_untrained        Meta's DINOv2 + the repo's older PCA file (our benchmark's variant)
  neuralhash                NeuralHash attacked with the same APGD (what the column should be)
  neuralhash_as_committed   what neuralhash_attack_results.py @ f8c2fd5d65 computes: the attack
                            optimises untrained DinoHash, reference bits come from NeuralHash,
                            and the result is scored with DinoHash (resized to 224 so it runs)

Usage (from hashbench/):
    python scripts/reproduce_dinohash_table1.py --limit 2 --steps 5 --out results/_t1_timing
    python scripts/reproduce_dinohash_table1.py --out results/dinohash_table1_repro
Resumable: rows already in raw.csv are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))

from dinohash_apgd_c700888 import APGDAttack  # noqa: E402  (authors' code, unmodified)
from hashes.dinohash import set_model  # noqa: E402  (shim for their hard-coded import)
from hashbench.benchmark import load_image  # noqa: E402
from hashbench.hashes import dinohash as dh  # noqa: E402
from hashbench.hashes import neuralhash as nh  # noqa: E402

DEVICE = "cpu"

PAPER = {  # Table 1, bit accuracy in %
    "dinohash": {4: 72.2, 8: 64.8},
    "dinohash_untrained_paper": {4: 0.3, 8: 0.0},
    "neuralhash_as_committed": {4: 0.1, 8: 0.0},
    "neuralhash": {4: 0.1, 8: 0.0},
}


def dino(trained: bool, paper_head: bool = False) -> torch.nn.Module:
    m = dh.load_dinohash_model("vits14_96", trained=trained)
    if paper_head:  # Meta backbone + the PCA_96 head (= the released head)
        t = dh.load_dinohash_model("vits14_96", trained=True)
        m.head.load_state_dict(t.head.state_dict())
    return m


def build(name: str):
    """(attacked model, model giving the reference bits, model scoring the result, preprocess)."""
    a, r, s, pre = _build(name)
    return a.to(DEVICE), r.to(DEVICE), s.to(DEVICE), pre


def _build(name: str):
    if name == "dinohash":
        m = dino(True)
        return m, m, m, dh.preprocess
    if name == "dinohash_untrained_paper":
        m = dino(False, paper_head=True)
        return m, m, m, dh.preprocess
    if name == "dinohash_untrained":
        m = dino(False)
        return m, m, m, dh.preprocess
    if name == "neuralhash":
        m = nh.load_neuralhash_model()
        return m, m, m, nh.preprocess
    if name == "neuralhash_as_committed":
        d, n = dino(False, paper_head=True), nh.load_neuralhash_model()
        return d, n, d, None  # special-cased below (two input sizes)
    raise ValueError(name)


def run_batch(name, models, images, eps, steps, seed):
    attacked, ref_model, scorer, pre = models
    if name == "neuralhash_as_committed":
        x = torch.stack([dh.preprocess(im) for im in images]).to(DEVICE)
        with torch.no_grad():
            ref = ref_model(torch.stack([nh.preprocess(im) for im in images]).to(DEVICE))
    else:
        x = torch.stack([pre(im) for im in images]).to(DEVICE)
        with torch.no_grad():
            ref = ref_model(x)
    torch.manual_seed(seed)
    set_model(attacked)
    apgd = APGDAttack(eps=eps / 255, device=DEVICE)
    adv, _ = apgd.attack_single_run(x, ref, n_iter=steps)
    with torch.no_grad():
        clean_bits = ref >= 0
        adv_bits = scorer(adv) >= 0
        adv_q = torch.round(adv * 255) / 255
        adv_q_bits = scorer(adv_q) >= 0
    linf = (adv - x).abs().flatten(1).max(1).values * 255
    return [{
        "bitacc": float((adv_bits[i] == clean_bits[i]).float().mean()) * 100,
        "bitacc_8bit": float((adv_q_bits[i] == clean_bits[i]).float().mean()) * 100,
        "linf_255": float(linf[i]),
    } for i in range(len(images))]


def summarize(out: Path) -> pd.DataFrame:
    df = pd.read_csv(out / "raw.csv")
    s = (df.groupby(["model", "eps_255"])
         .agg(n=("image", "size"), bitacc=("bitacc", "mean"), bitacc_8bit=("bitacc_8bit", "mean"),
              bitacc_median=("bitacc", "median"), linf_255=("linf_255", "mean"))
         .reset_index())
    s["paper"] = [PAPER.get(m, {}).get(int(e), float("nan")) for m, e in zip(s.model, s.eps_255)]
    s.to_csv(out / "summary.csv", index=False)
    lines = ["# DinoHash Table 1 reproduction (authors' APGD, bit accuracy %)", "",
             "| Model | eps | n | Paper | Ours (float) | Ours (8-bit) | Median | mean linf (/255) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.model} | {int(r.eps_255)}/255 | {r.n} | {r.paper:.1f} | {r.bitacc:.1f} | "
                     f"{r.bitacc_8bit:.1f} | {r.bitacc_median:.1f} | {r.linf_255:.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=ROOT / "data/splits/V200.txt")
    ap.add_argument("--images", type=Path, default=ROOT / "data/coco/val2017")
    ap.add_argument("--out", type=Path, default=ROOT / "results/dinohash_table1_repro")
    ap.add_argument("--models", nargs="+", default=["dinohash", "dinohash_untrained_paper",
                                                    "neuralhash", "neuralhash_as_committed",
                                                    "dinohash_untrained"])
    ap.add_argument("--eps", nargs="+", type=int, default=[4, 8], help="in /255")
    ap.add_argument("--steps", type=int, default=100, help="paper: 100-step APGD")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device

    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.csv"
    if args.summary_only:
        print(summarize(args.out).to_string(index=False))
        return
    names = [l.strip() for l in args.list.read_text().splitlines() if l.strip()][:args.limit]
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["attack"] = "third_party/dinohash_apgd_c700888.py (authors' APGD @ c7008888b7, unmodified)"
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    done = set()
    if raw.exists():
        d = pd.read_csv(raw)
        done = set(zip(d.model, d.eps_255, d.image))

    images = {n: load_image(args.images / n, args.max_side) for n in names}
    for name in args.models:
        models = build(name)
        for eps in args.eps:
            todo = [n for n in names if (name, eps, n) not in done]
            for b in range(0, len(todo), args.batch):
                chunk = todo[b:b + args.batch]
                t0 = time.time()
                # seed tied to the batch's first image, so a resumed run draws the same starts
                res = run_batch(name, models, [images[n] for n in chunk], eps, args.steps,
                                args.seed + names.index(chunk[0]))
                rows = [{"model": name, "eps_255": eps, "image": n, "steps": args.steps, **r}
                        for n, r in zip(chunk, res)]
                pd.DataFrame(rows).to_csv(raw, mode="a", header=not raw.exists(), index=False)
                acc = np.mean([r["bitacc"] for r in res])
                print(f"{name} eps={eps}/255 [{b + len(chunk)}/{len(todo)}] bitacc {acc:.1f}% "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del models
    print(summarize(args.out).to_string(index=False))


if __name__ == "__main__":
    main()
