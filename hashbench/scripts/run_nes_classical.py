"""NES black-box attack on the classical hashes (pHash, PDQ), CPU, parallel over images.

Same attack, settings, thresholds, output columns and summary as ``run_nes.py`` (which runs
the learned hashes on a GPU), so the two nes.csv files can simply be concatenated. The only
differences:

* the hash is computed by ``hashbench.blackbox_classical.ClassicalAdapter`` (8-bit PIL image
  -> ``PerceptualHash.hash``), and the attacked image is the whole picture as the benchmark
  loads it (max side 512, aspect ratio kept) instead of a square model input;
* images are attacked one at a time, spread over ``--workers`` processes (one CPU thread
  each); the per-image random seed is ``seed + index of the image in the list``.

Rows are appended to ``<out>/nes.csv`` as images finish (in completion order), so an
interrupted run resumes where it stopped when rerun with the same command.

Example (from hashbench/):
    python scripts/run_nes_classical.py --list data/splits/V200.txt \
        --out results/nes_v200_classical --workers 8
    python scripts/run_nes.py --out results/nes_v200_classical --summary-only
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashbench.benchmark import load_image  # noqa: E402
from hashbench.blackbox import HashService, NESConfig, nes_attack  # noqa: E402
from hashbench.blackbox_classical import ClassicalAdapter  # noqa: E402
from hashbench.hashes import get_hasher  # noqa: E402
from run_adversarial import image_list, load_thresholds  # noqa: E402
from run_nes import BUDGETS, COLUMNS, check_config, done_keys, parse_setting, summarize  # noqa: E402

CLASSICAL = ["phash", "pdq"]

_W: dict = {}          # per-worker state: images, cfg, thresholds, adapters


def _init_worker(paths: list[str], max_side: int, cfg: dict, thresholds: dict) -> None:
    torch.set_num_threads(1)
    _W["images"] = {Path(p).name: load_image(Path(p), max_side) for p in paths}
    _W["cfg"], _W["thr"], _W["services"] = cfg, thresholds, {}


def _attack_one(task: tuple) -> dict:
    name, mode, eps, img, target, seed = task
    if name not in _W["services"]:
        hasher = get_hasher(name)
        _W["services"][name] = (hasher, ClassicalAdapter(hasher))
    hasher, adapter = _W["services"][name]
    service = HashService(adapter, chunk=_W["cfg"]["chunk"])
    thr = math.floor(_W["thr"][name] * hasher.max_distance + 1e-6)
    x0 = adapter.preprocess(_W["images"][img])[None]
    ref = service.hashes(x0 if mode == "evasion" else adapter.preprocess(_W["images"][target])[None])
    cfg = NESConfig(eps=eps / 255.0, seed=seed, device="cpu", **_W["cfg"])
    t0 = time.time()
    r = nes_attack(service, x0, ref, mode, thr, cfg)[0]
    return {"hash": name, "mode": mode, "eps_255": eps, "image": img,
            "target": target if mode == "collision" else "", "threshold": thr,
            "max_distance": hasher.max_distance, "orig_distance": r.orig_distance,
            "distance": r.distance, "success": r.success, "queries": r.queries, "steps": r.steps,
            "linf": r.linf, "ssim": r.ssim, "seconds_per_image": time.time() - t0}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=CLASSICAL, choices=CLASSICAL)
    ap.add_argument("--settings", nargs="+", type=parse_setting,
                    default=[("evasion", 8.0), ("collision", 16.0), ("evasion", 4.0)])
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
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    n = len(paths)
    target_of = {names[i]: names[(i + n // 2) % n] for i in range(n)}

    config = {
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": n,
        "max_queries": args.max_queries, "pairs": args.pairs, "sigma_frac": args.sigma_frac,
        "lr_frac": args.lr_frac, "momentum": args.momentum, "dim": args.dim, "plateau": args.plateau,
        "min_lr_frac": args.min_lr_frac, "fpr": args.fpr, "max_side": args.max_side, "seed": args.seed,
        "thresholds_from": [str(p) for p in args.thresholds_from], "hashes": args.hashes,
        "settings": [f"{m}:{e:g}" for m, e in args.settings],
        "collision_targets": "image i -> image (i + n//2) mod n of the list",
        "threat_model": "score-based: the attacker reads the discrete hash of each 8-bit query",
        "attacked_image": "whole image as loaded by the benchmark (max side 512), not a model input",
        "runner": "run_nes_classical.py", "device": "cpu", "workers": args.workers,
        "torch": torch.__version__, "cpu_count": os.cpu_count(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    check_config(args.out, config)
    csv = args.out / "nes.csv"
    done = done_keys(csv)
    tasks = [(h, m, e, nm, target_of[nm], args.seed + i)
             for h in args.hashes for m, e in args.settings for i, nm in enumerate(names)
             if (h, m, e, nm) not in done]
    print(f"{n} images, hashes {args.hashes}, settings {config['settings']}, budget {args.max_queries} "
          f"queries, {args.workers} workers; {len(done)} rows already done, {len(tasks)} to run", flush=True)

    cfg = {"max_queries": args.max_queries, "pairs": args.pairs, "sigma_frac": args.sigma_frac,
           "lr_frac": args.lr_frac, "momentum": args.momentum, "dim": args.dim, "plateau": args.plateau,
           "min_lr_frac": args.min_lr_frac, "chunk": 64, "separable": True}
    t_start = time.time()
    counts: dict[tuple, list[int]] = {}
    with mp.get_context("spawn").Pool(args.workers, _init_worker,
                                      ([str(p) for p in paths], args.max_side, cfg, thresholds)) as pool:
        for k, row in enumerate(pool.imap_unordered(_attack_one, tasks), 1):
            pd.DataFrame([row], columns=COLUMNS).to_csv(csv, mode="a", header=not csv.exists(), index=False)
            c = counts.setdefault((row["hash"], row["mode"], row["eps_255"]), [0, 0])
            c[0] += 1
            c[1] += bool(row["success"])
            print(f"[{time.time() - t_start:7.0f}s] {k:4d}/{len(tasks)}  {row['hash']:5s} {row['mode']:9s} "
                  f"eps={row['eps_255']:g}/255 {row['image']}  success={row['success']!s:5s} "
                  f"q={row['queries']:5d}  dist={row['distance']:g}/thr {row['threshold']}  "
                  f"({c[1]}/{c[0]} so far)", flush=True)

    print(f"done in {time.time() - t_start:.0f}s")
    summarize(args.out, [b for b in BUDGETS if b <= args.max_queries])


if __name__ == "__main__":
    main()
