"""NES black-box attack (attack #2 of future_attacks.md): runner and summary.

For each hash and each setting (mode + eps), attacks every image of a list with the
score-based NES attack of ``hashbench.blackbox`` and appends one row per image to
``<out>/nes.csv``: whether it succeeded, the number of queries it took (or spent), the
best distance reached, and the l_inf / SSIM of the best 8-bit image. One run with a
budget of N queries gives the whole success-vs-queries curve, because each image stops
at its first success.

Success uses the same calibrated thresholds as every other Phase C result (FPR 1e-3,
from the benign runs' overall.csv):
    evasion    success if  d(adv, original) >  threshold
    collision  success if  d(adv, target)   <= threshold   (target: image (i + n/2) mod n)

Settings are run in the order given (default: evasion 8, collision 16, evasion 4), so the
headline setting finishes first if a Kaggle session runs out of time. The run is resumable:
rows already in nes.csv are skipped. ``--time-limit`` stops starting new batches after the
given number of hours so the output can still be saved.

``run_adversarial.py`` and ``generate_transfer_images.py`` are not modified; this script
only imports their helpers.

Example (from hashbench/):
    python scripts/run_nes.py --list data/splits/V200.txt --out results/nes_v200 \
        --device cuda --hashes dinohash
    python scripts/run_nes.py --out results/nes_v200 --summary-only
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

from hashbench.adversarial import _quantize, _to_image, build_adapter  # noqa: E402
from hashbench.benchmark import load_image  # noqa: E402
from hashbench.blackbox import HashService, NESConfig, nes_attack  # noqa: E402
from hashbench.hashes import get_hasher  # noqa: E402
from hashbench.metrics import wilson_interval  # noqa: E402
from run_adversarial import DEFAULT_HASHES, image_list, load_thresholds  # noqa: E402

COLUMNS = ["hash", "mode", "eps_255", "image", "target", "threshold", "max_distance", "orig_distance",
           "distance", "success", "queries", "steps", "linf", "ssim", "seconds_per_image"]
CONFIG_KEYS = ["list", "limit", "max_queries", "pairs", "sigma_frac", "lr_frac", "momentum", "dim",
               "plateau", "min_lr_frac", "fpr", "max_side", "seed", "thresholds_from"]
BUDGETS = [1000, 2000, 5000, 10000]


def parse_setting(s: str) -> tuple[str, float]:
    mode, eps = s.split(":")
    if mode not in ("evasion", "collision"):
        raise argparse.ArgumentTypeError(f"bad mode in {s!r}")
    return mode, float(eps)


def check_config(out: Path, config: dict) -> None:
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), config.get(k)) for k in CONFIG_KEYS if old.get(k) != config.get(k)}
        if diff:
            raise SystemExit(f"{out} holds a run with different settings {diff}; use another --out")
        config["hashes"] = sorted(set(old.get("hashes", [])) | set(config["hashes"]))
    path.write_text(json.dumps(config, indent=2))


def done_keys(csv: Path) -> set[tuple]:
    if not csv.exists():
        return set()
    df = pd.read_csv(csv, usecols=["hash", "mode", "eps_255", "image"])
    return set(zip(df["hash"], df["mode"], df["eps_255"].astype(float), df["image"]))


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------

def summarize(out: Path, budgets: list[int]) -> pd.DataFrame:
    csvs = [out / "nes.csv"] if (out / "nes.csv").exists() else sorted(out.glob("*/nes.csv"))
    if not csvs:
        raise SystemExit(f"no nes.csv under {out}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df = df.drop_duplicates(["hash", "mode", "eps_255", "image"], keep="last")
    rows = []
    for (h, mode, eps), g in df.groupby(["hash", "mode", "eps_255"], sort=False):
        already = (g["mode"] == "collision") & g["success"] & (g["queries"] <= 1)
        g = g[~already]
        n = len(g)
        row = {"hash": h, "mode": mode, "eps_255": eps, "n": n, "already_matched": int(already.sum())}
        for bud in budgets:
            k = int((g["success"] & (g["queries"] <= bud)).sum())
            lo, hi = wilson_interval(k, n)
            row[f"succ_{bud}"] = round(100 * k / n, 1) if n else float("nan")
            row[f"ci_{bud}"] = f"{100 * lo:.1f}-{100 * hi:.1f}"
        s = g[g["success"]]
        row["median_queries"] = float(s["queries"].median()) if len(s) else float("nan")
        row["mean_norm_distance"] = round(float((g["distance"] / g["max_distance"]).mean()), 3)
        row["mean_orig_norm_distance"] = round(float((g["orig_distance"] / g["max_distance"]).mean()), 3)
        row["mean_linf_255"] = round(float(g["linf"].mean() * 255), 2)
        row["mean_ssim"] = round(float(g["ssim"].mean()), 3)
        row["seconds_per_image"] = round(float(g["seconds_per_image"].mean()), 1)
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values(["mode", "eps_255", "hash"])
    summ.to_csv(out / "summary.csv", index=False)

    lines = ["# NES black-box attack: summary", "",
             "Success (%) within each query budget at the calibrated FPR 1e-3 threshold. "
             "Collision rows exclude pairs that already matched before the attack.", ""]
    for (mode, eps), g in summ.groupby(["mode", "eps_255"], sort=False):
        lines += [f"## {mode}, eps = {eps:g}/255", "",
                  "| hash | n | " + " | ".join(f"<={b // 1000}k" for b in budgets)
                  + " | median queries | mean dist (norm) | l_inf ×255 | SSIM |",
                  "|---|---|" + "---|" * len(budgets) + "---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r['hash']} | {r['n']} | " + " | ".join(f"{r[f'succ_{b}']}" for b in budgets)
                         + f" | {r['median_queries']:g} | {r['mean_norm_distance']} | "
                           f"{r['mean_linf_255']} | {r['mean_ssim']} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return summ


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=DEFAULT_HASHES)
    ap.add_argument("--settings", nargs="+", type=parse_setting,
                    default=[("evasion", 8.0), ("collision", 16.0), ("evasion", 4.0)],
                    help="mode:eps pairs, run in this order (default evasion:8 collision:16 evasion:4)")
    ap.add_argument("--max-queries", type=int, default=10_000)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--sigma-frac", type=float, default=1.0)
    ap.add_argument("--lr-frac", type=float, default=0.2)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--plateau", type=int, default=5)
    ap.add_argument("--min-lr-frac", type=float, default=0.02)
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--thresholds-from", nargs="+", type=Path,
                    default=[Path("results/coco_run1"), Path("results/coco_run2")])
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16, help="images attacked together")
    ap.add_argument("--chunk", type=int, default=256, help="images per forward pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=None, help="hours; no new batch after this")
    ap.add_argument("--save-images", action="store_true", help="save the best adversarial PNGs")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)

    if args.summary_only:
        summarize(args.out, [b for b in BUDGETS if b <= args.max_queries])
        return

    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    missing = [h for h in args.hashes if h not in thresholds]
    if missing:
        raise SystemExit(f"no FPR {args.fpr:g} threshold for {missing}")
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    n = len(paths)
    target_of = {names[i]: names[(i + n // 2) % n] for i in range(n)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": n,
        "max_queries": args.max_queries, "pairs": args.pairs, "sigma_frac": args.sigma_frac,
        "lr_frac": args.lr_frac, "momentum": args.momentum, "dim": args.dim, "plateau": args.plateau,
        "min_lr_frac": args.min_lr_frac, "fpr": args.fpr, "max_side": args.max_side, "seed": args.seed,
        "thresholds_from": [str(p) for p in args.thresholds_from], "hashes": args.hashes,
        "settings": [f"{m}:{e:g}" for m, e in args.settings],
        "collision_targets": "image i -> image (i + n//2) mod n of the list",
        "threat_model": "score-based: the attacker reads the discrete hash of each 8-bit query",
        "device": device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    check_config(args.out, config)
    csv = args.out / "nes.csv"
    done = done_keys(csv)
    images = {p.name: load_image(p, args.max_side) for p in paths}
    print(f"{n} images, hashes {args.hashes}, settings {config['settings']}, "
          f"budget {args.max_queries} queries, device {device}; {len(done)} rows already done", flush=True)

    t_start = time.time()
    deadline = t_start + args.time_limit * 3600 if args.time_limit else None
    for name in args.hashes:
        hasher = get_hasher(name)
        thr = math.floor(thresholds[name] * hasher.max_distance + 1e-6)
        adapter = build_adapter(hasher).to(torch.device(device))
        service = HashService(adapter, chunk=args.chunk)
        x_clean = {nm: _quantize(adapter.preprocess(images[nm])) for nm in names}

        for mode, eps in args.settings:
            todo = [nm for nm in names if (name, mode, eps, nm) not in done]
            for b in range(0, len(todo), args.batch_size):
                if deadline and time.time() > deadline:
                    print(f"time limit reached after {time.time() - t_start:.0f}s; rerun to resume", flush=True)
                    return
                batch = todo[b:b + args.batch_size]
                x0 = torch.stack([x_clean[nm] for nm in batch]).to(device)
                if mode == "evasion":
                    ref = service.hashes(x0)
                else:
                    ref = service.hashes(torch.stack([x_clean[target_of[nm]] for nm in batch]).to(device))
                cfg = NESConfig(eps=eps / 255.0, max_queries=args.max_queries, pairs=args.pairs,
                                sigma_frac=args.sigma_frac, lr_frac=args.lr_frac, momentum=args.momentum,
                                dim=args.dim, plateau=args.plateau, min_lr_frac=args.min_lr_frac,
                                chunk=args.chunk, device=device, seed=args.seed + names.index(batch[0]))
                t0 = time.time()
                res = nes_attack(service, x0, ref, mode, thr, cfg)
                per_img = (time.time() - t0) / len(batch)
                if args.save_images:
                    d = args.out / "images" / name / mode / f"eps{eps:g}"
                    d.mkdir(parents=True, exist_ok=True)
                    for nm, r in zip(batch, res):
                        _to_image(r.adv).save(d / (Path(nm).stem + ".png"))
                rows = [{
                    "hash": name, "mode": mode, "eps_255": eps, "image": nm,
                    "target": target_of[nm] if mode == "collision" else "", "threshold": thr,
                    "max_distance": hasher.max_distance, "orig_distance": r.orig_distance,
                    "distance": r.distance, "success": r.success, "queries": r.queries, "steps": r.steps,
                    "linf": r.linf, "ssim": r.ssim, "seconds_per_image": per_img} for nm, r in zip(batch, res)]
                pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                ok = sum(r.success for r in res)
                qs = [r.queries for r in res if r.success]
                print(f"[{time.time() - t_start:7.0f}s] {name:18s} {mode:9s} eps={eps:g}/255 "
                      f"{b + len(batch):4d}/{len(todo)}  success {ok}/{len(batch)}"
                      f"{f'  median q {int(np.median(qs))}' if qs else ''}  {per_img:.1f} s/img", flush=True)
        del service, adapter, hasher
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"done in {time.time() - t_start:.0f}s")
    summarize(args.out, [b for b in BUDGETS if b <= args.max_queries])


if __name__ == "__main__":
    main()
