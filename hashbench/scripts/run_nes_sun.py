"""NES black-box evasion in the setting of Sun et al. 2026 (arXiv 2608.03101): runner and summary.

Separate from run_nes.py (our own NES setting), which is unchanged. See hashbench/blackbox_sun.py
for the setting and the choices made where the paper is silent.

Per hash and l2 budget, every image of the list is attacked; one row per image goes to
<out>/nes_sun.csv. Success: more than ``--ber`` (0.2) of the bits differ from the clean hash
(DinoHash / NeuralHash: > 19.2 of 96, i.e. >= 20 bits). CertPHash has no bits; it uses its
calibrated FPR 1e-3 L1 threshold from the benign runs. Each returned image is re-hashed with
the plain hasher on CPU; ``verified_success`` is the headline number.

Resumable (rows already in the csv are skipped); ``--time-limit`` stops starting new batches.

Example (from hashbench/):
    python scripts/run_nes_sun.py --list data/splits/V200.txt --out results/nes_sun_v200 \
        --device cuda --hashes dinohash neuralhash
    python scripts/run_nes_sun.py --out results/nes_sun_v200 --summary-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashbench.adversarial import _to_image, build_adapter  # noqa: E402
from hashbench.blackbox import HashService  # noqa: E402
from hashbench.blackbox_sun import (PILResize, ResizingAdapter, SunNESConfig, load_512,  # noqa: E402
                                    sun_nes_evasion, to_tensor)
from hashbench.hashes import get_hasher  # noqa: E402
from hashbench.metrics import wilson_interval  # noqa: E402
from run_adversarial import image_list, load_thresholds  # noqa: E402

COLUMNS = ["hash", "l2_budget", "image", "threshold", "max_distance", "orig_distance", "distance",
           "success", "queries", "steps", "l2", "linf", "ssim", "verified_distance", "verified_success",
           "seconds_per_image"]
CONFIG_KEYS = ["list", "limit", "size", "max_queries", "max_steps", "directions", "sigma_255", "step_255",
               "dim", "ber", "fpr", "seed"]
BIT_HASHES = {"dinohash", "dinohash_untrained", "neuralhash"}


def check_config(out: Path, config: dict) -> None:
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), config.get(k)) for k in CONFIG_KEYS if old.get(k) != config.get(k)}
        if diff:
            raise SystemExit(f"{out} holds a run with different settings {diff}; use another --out")
        config["hashes"] = sorted(set(old.get("hashes", [])) | set(config["hashes"]))
    path.write_text(json.dumps(config, indent=2))


def summarize(out: Path) -> pd.DataFrame:
    csvs = [out / "nes_sun.csv"] if (out / "nes_sun.csv").exists() else sorted(out.glob("*/nes_sun.csv"))
    if not csvs:
        raise SystemExit(f"no nes_sun.csv under {out}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df = df.drop_duplicates(["hash", "l2_budget", "image"], keep="last")
    rows = []
    for (h, bud), g in df.groupby(["hash", "l2_budget"], sort=False):
        n, k = len(g), int(g["verified_success"].sum())
        lo, hi = wilson_interval(k, n)
        s = g[g["success"]]
        rows.append({"hash": h, "l2_budget": bud, "n": n, "asr": round(100 * k / n, 1),
                     "ci": f"{100 * lo:.1f}-{100 * hi:.1f}",
                     "asr_attack_side": round(100 * g["success"].mean(), 1),
                     "agree": round(100 * (g["success"] == g["verified_success"]).mean(), 1),
                     "median_queries": float(s["queries"].median()) if len(s) else float("nan"),
                     "mean_bits_or_l1": round(float(g["verified_distance"].mean()), 1),
                     "threshold": float(g["threshold"].iloc[0]),
                     "mean_l2": round(float(g["l2"].mean()), 1),
                     "mean_linf_255": round(float(g["linf"].mean() * 255), 1),
                     "mean_ssim": round(float(g["ssim"].mean()), 3),
                     "seconds_per_image": round(float(g["seconds_per_image"].mean()), 1)})
    summ = pd.DataFrame(rows).sort_values(["hash", "l2_budget"])
    summ.to_csv(out / "summary.csv", index=False)

    piv = summ.pivot(index="hash", columns="l2_budget", values="asr")
    piv["mean"] = piv.mean(axis=1).round(1)
    lines = ["# NES black-box evasion, Sun et al. 2026 setting: summary", "",
             "ASR (%) = verified evasions (plain hasher on the returned 8-bit image). Success: BER > 0.2 "
             "(>= 20 of 96 bits) for DinoHash / NeuralHash; CertPHash: its FPR 1e-3 L1 threshold. "
             "'mean' = the average over the three budgets (Sun et al. Table 1).", "",
             "| hash | " + " | ".join(f"l2 {b:g}" for b in piv.columns[:-1]) + " | mean |",
             "|---|" + "---|" * len(piv.columns)]
    for h, r in piv.iterrows():
        lines.append(f"| {h} | " + " | ".join(f"{v}" for v in r.values) + " |")
    lines += ["", "| hash | l2 | n | ASR (95% CI) | attack-side ASR | agree % | median q | mean dist | "
              "l2 | l_inf ×255 | SSIM | s/img |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r['hash']} | {r['l2_budget']:g} | {r['n']} | {r['asr']} ({r['ci']}) | "
                     f"{r['asr_attack_side']} | {r['agree']} | {r['median_queries']:g} | "
                     f"{r['mean_bits_or_l1']} | {r['mean_l2']} | {r['mean_linf_255']} | {r['mean_ssim']} | "
                     f"{r['seconds_per_image']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summ


def resize_selfcheck(imgs: list, device: str) -> None:
    """Fail fast if this Pillow resizes differently from the reproduced resize."""
    from PIL import Image
    for n_out, rs in [(224, Image.Resampling.BILINEAR), (360, Image.Resampling.BICUBIC)]:
        r = PILResize(512, 512, n_out, rs, device)
        for im in imgs:
            ref = np.asarray(im.resize((n_out, n_out), rs))
            got = (r(to_tensor(im)[None].to(device))[0] * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            if (ref != got).any():
                raise SystemExit(f"resize self-check failed ({n_out}): {(ref != got).sum()} values differ")
    print("resize self-check passed (GPU resize == Pillow)", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=["dinohash", "dinohash_untrained", "neuralhash"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[40.0, 90.0, 180.0])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-queries", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--directions", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=4.0, help="NES scale, /255")
    ap.add_argument("--step", type=float, default=2.0, help="step size, /255")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ber", type=float, default=0.2)
    ap.add_argument("--fpr", type=float, default=0.001, help="CertPHash threshold only")
    ap.add_argument("--thresholds-from", nargs="+", type=Path,
                    default=[Path("results/coco_run1"), Path("results/coco_run2")])
    ap.add_argument("--batch-size", type=int, default=8, help="images attacked together")
    ap.add_argument("--chunk", type=int, default=64, help="512x512 images per forward pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--time-limit", type=float, default=None, help="hours; no new batch after this")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)

    if args.summary_only:
        summarize(args.out)
        return

    torch.backends.cuda.matmul.allow_tf32 = False      # the exact resize needs true fp32 matmuls
    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "paper": "Sun et al. 2026, arXiv 2608.03101 (black-box NES setting)",
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": len(paths),
        "size": args.size, "max_queries": args.max_queries, "max_steps": args.max_steps,
        "directions": args.directions, "sigma_255": args.sigma, "step_255": args.step, "dim": args.dim,
        "ber": args.ber, "fpr": args.fpr, "seed": args.seed, "budgets": args.budgets, "hashes": args.hashes,
        "queries_per_step": 2 * args.directions + 1,
        "choices": {"resize_512": "PIL bilinear", "upsample": "bilinear 64->512, gray copied to RGB",
                    "update": "z += step * sign(g), then l2 projection (probes projected too)",
                    "queries": "8-bit rounded", "cert_threshold": f"FPR {args.fpr:g} L1 (no bits)"},
        "device": device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    check_config(args.out, config)
    csv = args.out / "nes_sun.csv"
    done = set()
    if csv.exists():
        old = pd.read_csv(csv, usecols=["hash", "l2_budget", "image"])
        done = set(zip(old["hash"], old["l2_budget"].astype(float), old["image"]))
    images = {p.name: load_512(p, args.size) for p in paths}
    resize_selfcheck([images[nm] for nm in names[:3]], device)
    print(f"{len(names)} images, hashes {args.hashes}, l2 budgets {args.budgets}, device {device}; "
          f"{len(done)} rows already done", flush=True)

    t_start = time.time()
    deadline = t_start + args.time_limit * 3600 if args.time_limit else None
    for name in args.hashes:
        hasher = get_hasher(name)
        if name in BIT_HASHES:
            thr = args.ber * hasher.max_distance
        else:
            thr = math.floor(thresholds[name] * hasher.max_distance + 1e-6)
        inner = build_adapter(hasher).to(torch.device(device))
        adapter = ResizingAdapter(inner, hasher, args.size, device)
        service = HashService(adapter, chunk=args.chunk)
        x_clean = {nm: to_tensor(images[nm]) for nm in names}
        clean_hash = {}

        @torch.no_grad()
        def plain_hash(img):
            """The hasher's own pipeline: its PIL / torchvision preprocessing on CPU, then the
            network (on the run's device; the hasher's model object was moved there)."""
            return inner.hash_t(inner.outputs(inner.preprocess(img)[None].to(device)))

        for bud in args.budgets:
            todo = [nm for nm in names if (name, bud, nm) not in done]
            for b in range(0, len(todo), args.batch_size):
                if deadline and time.time() > deadline:
                    print(f"time limit reached after {time.time() - t_start:.0f}s; rerun to resume", flush=True)
                    summarize(args.out)
                    return
                batch = todo[b:b + args.batch_size]
                x0 = torch.stack([x_clean[nm] for nm in batch]).to(device)
                ref = service.hashes(x0)
                cfg = SunNESConfig(l2=bud, max_queries=args.max_queries, max_steps=args.max_steps,
                                   directions=args.directions, sigma=args.sigma / 255, step=args.step / 255,
                                   dim=args.dim, chunk=args.chunk, device=device,
                                   seed=args.seed + 1000 * int(bud) + names.index(batch[0]))
                t0 = time.time()
                res = sun_nes_evasion(service, x0, ref, thr, cfg)
                per_img = (time.time() - t0) / len(batch)
                rows = []
                for nm, r in zip(batch, res):
                    adv = _to_image(r.adv)
                    if nm not in clean_hash:
                        clean_hash[nm] = plain_hash(images[nm])
                    vd = float(inner.dist_t(plain_hash(adv), clean_hash[nm])[0])
                    if args.save_images:
                        dd = args.out / "images" / name / f"l2_{bud:g}"
                        dd.mkdir(parents=True, exist_ok=True)
                        adv.save(dd / (Path(nm).stem + ".png"))
                    rows.append({"hash": name, "l2_budget": bud, "image": nm, "threshold": thr,
                                 "max_distance": hasher.max_distance, "orig_distance": r.orig_distance,
                                 "distance": r.distance, "success": r.success, "queries": r.queries,
                                 "steps": r.steps, "l2": r.l2, "linf": r.linf, "ssim": r.ssim,
                                 "verified_distance": vd, "verified_success": vd > thr,
                                 "seconds_per_image": per_img})
                pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                ok = sum(r["verified_success"] for r in rows)
                agree = sum(r["verified_success"] == r["success"] for r in rows)
                print(f"[{time.time() - t_start:7.0f}s] {name:18s} l2={bud:g} {b + len(batch):4d}/{len(todo)}  "
                      f"success {ok}/{len(batch)}  (agree {agree}/{len(batch)})  {per_img:.1f} s/img", flush=True)
        del service, adapter, hasher
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"done in {time.time() - t_start:.0f}s")
    summarize(args.out)


if __name__ == "__main__":
    main()
