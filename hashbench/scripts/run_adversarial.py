"""Adversarial sweep: white-box evasion and collision for every learned hash and budget.

For each hash, each budget eps (in /255 units) and each mode, attacks every image of a
fixed list with ``hashbench.adversarial.attack_batch`` and appends one row per image to
``<out>/raw.csv``. Success uses the same calibrated thresholds as the benign benchmark
(FPR 1e-3 by default, read from the runs' ``overall.csv``), so the adversarial and benign
numbers are on the same operating point:

    evasion    success if  d(adv, original) >  threshold   (the copy is no longer found)
    collision  success if  d(adv, target)   <= threshold   (a false match is forced)

Collision targets are fixed: image i of the (sorted) list is pushed onto image
(i + n/2) mod n, so every model and budget attacks the same pairs.

The run is resumable: rows already in ``raw.csv`` are skipped, so after a Kaggle session
timeout, rerun the same command. A rerun with different attack settings (steps, method,
norm, ...) into the same folder is refused rather than mixing results.

When the sweep is done (or with ``--summary-only``), ``summary.csv`` and ``summary.md``
give the success rate with a 95% Wilson interval, mean distance and image quality per
hash x mode x eps.

Example (from hashbench/):
    python scripts/run_adversarial.py --list data/splits/V200.txt --out results/adv_v200 \
        --device cuda --batch-size 16
    # quick CPU pilot:
    python scripts/run_adversarial.py --limit 10 --out results/adv_pilot --steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hashbench.adversarial import AttackConfig, attack_batch, build_adapter  # noqa: E402
from hashbench.benchmark import load_image  # noqa: E402
from hashbench.hashes import get_hasher  # noqa: E402
from hashbench.metrics import wilson_interval  # noqa: E402

DEFAULT_HASHES = ["dinohash", "dinohash_untrained", "neuralhash", "certphash", "certphash_ep2",
                  "certphash_ep4", "certphash_ep8", "certphash_base", "certphash_adv"]
RAW_COLUMNS = ["hash", "mode", "norm", "eps_255", "image", "target", "threshold", "distance",
               "orig_distance", "success", "linf", "l2", "psnr", "ssim", "seconds_per_image"]
# settings that change the numbers; a rerun into the same folder must match them
CONFIG_KEYS = ["list", "limit", "norm", "steps", "method", "step_size", "random_start",
               "margin", "fpr", "max_side", "seed", "thresholds_from"]


def load_thresholds(runs: list[Path], fpr: float) -> dict[str, float]:
    """Normalized FPR thresholds per hash from the benign runs' overall.csv files."""
    col = f"threshold_at_fpr_{fpr:g}"
    out = {}
    for run in runs:
        df = pd.read_csv(run / "overall.csv")
        if col not in df.columns:
            raise SystemExit(f"{run / 'overall.csv'} has no column {col}")
        out.update(dict(zip(df["algorithm"], df[col])))
    return out


def image_list(images: Path, list_file: Path | None, limit: int | None) -> list[Path]:
    if list_file is not None:
        names = [l.strip() for l in list_file.read_text().splitlines() if l.strip()]
        paths = [images / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} listed images missing, e.g. {missing[0]}")
    else:
        paths = sorted(p for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return paths[:limit] if limit else paths


def check_config(out: Path, config: dict) -> None:
    path = out / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), config.get(k)) for k in CONFIG_KEYS if old.get(k) != config.get(k)}
        if diff:
            raise SystemExit(f"{out} holds a run with different settings {diff}; use another --out")
        # keep the original record, but remember every hash / eps / mode ever requested
        for k in ("hashes", "eps_255", "modes"):
            config[k] = sorted(set(old.get(k, [])) | set(config[k]), key=str)
    path.write_text(json.dumps(config, indent=2))


def done_keys(raw: Path) -> set[tuple]:
    if not raw.exists():
        return set()
    df = pd.read_csv(raw, usecols=["hash", "mode", "eps_255", "image"])
    return set(zip(df["hash"], df["mode"], df["eps_255"].astype(float), df["image"]))


def append_rows(raw: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.to_csv(raw, mode="a", header=not raw.exists(), index=False)


def summarize(out: Path) -> pd.DataFrame | None:
    raw = out / "raw.csv"
    if not raw.exists():
        return None
    df = pd.read_csv(raw)
    rows = []
    for (h, mode, eps), g in df.groupby(["hash", "mode", "eps_255"], sort=True):
        k, n = int(g["success"].sum()), len(g)
        lo, hi = wilson_interval(k, n)
        rows.append({
            "hash": h, "mode": mode, "eps_255": eps, "n": n, "success_rate": k / n,
            "ci_low": lo, "ci_high": hi, "threshold": g["threshold"].iloc[0],
            "mean_distance": g["distance"].mean(), "mean_orig_distance": g["orig_distance"].mean(),
            "already_success_before_attack": float(
                ((g["orig_distance"] > g["threshold"]) if mode == "evasion"
                 else (g["orig_distance"] <= g["threshold"])).mean()),
            "mean_linf": g["linf"].mean(), "mean_psnr": g["psnr"].replace(math.inf, float("nan")).mean(),
            "mean_ssim": g["ssim"].mean(), "seconds_per_image": g["seconds_per_image"].mean(),
        })
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    write_markdown(s, out / "summary.md")
    return s


def write_markdown(s: pd.DataFrame, path: Path) -> None:
    lines = ["# Adversarial sweep summary", "",
             "Success rate (%) at the benign FPR threshold, with 95% Wilson interval. "
             "Evasion: the attacked copy is no longer matched. Collision: a false match is forced.", ""]
    for mode in ("evasion", "collision"):
        m = s[s["mode"] == mode]
        if m.empty:
            continue
        eps_list = sorted(m["eps_255"].unique())
        lines += [f"## {mode.capitalize()}", "",
                  "| Hash | n | " + " | ".join(f"ε={e:g}/255" for e in eps_list) + " |",
                  "|---|---|" + "---|" * len(eps_list)]
        for h, g in m.groupby("hash", sort=False):
            cells = []
            for e in eps_list:
                r = g[g["eps_255"] == e]
                cells.append("–" if r.empty else
                             f"{100 * r.success_rate.iloc[0]:.1f} ({100 * r.ci_low.iloc[0]:.0f}–"
                             f"{100 * r.ci_high.iloc[0]:.0f})")
            lines.append(f"| {h} | {int(g['n'].max())} | " + " | ".join(cells) + " |")
        lines += ["", f"Mean SSIM of the attacked images ({mode}):", "",
                  "| Hash | " + " | ".join(f"ε={e:g}/255" for e in eps_list) + " |",
                  "|---|" + "---|" * len(eps_list)]
        for h, g in m.groupby("hash", sort=False):
            cells = [("–" if g[g["eps_255"] == e].empty else f"{g[g['eps_255'] == e].mean_ssim.iloc[0]:.3f}")
                     for e in eps_list]
            lines.append(f"| {h} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None, help="image list, e.g. data/splits/V200.txt")
    ap.add_argument("--limit", type=int, default=None, help="first N images of the list (pilots)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=DEFAULT_HASHES)
    ap.add_argument("--eps", nargs="+", type=float, default=[1, 2, 4, 8], help="budgets in /255 units")
    ap.add_argument("--modes", nargs="+", default=["evasion", "collision"], choices=["evasion", "collision"])
    ap.add_argument("--norm", default="linf", choices=["linf", "l2"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--method", default="apgd", choices=["apgd", "pgd"])
    ap.add_argument("--step-size", type=float, default=None, help="PGD step (default eps/4); APGD ignores it")
    ap.add_argument("--no-random-start", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--fpr", type=float, default=0.001, help="benign FPR whose threshold defines success")
    ap.add_argument("--thresholds-from", nargs="+", type=Path,
                    default=[Path("results/coco_run1"), Path("results/coco_run2")])
    ap.add_argument("--max-side", type=int, default=512, help="downscale originals as the benchmark does")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true", help="only rebuild summary.csv / summary.md")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles can't print ε otherwise
        sys.stdout.reconfigure(errors="replace")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(args.out)
        print(f"wrote {args.out / 'summary.md'}")
        return

    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    unknown = [h for h in args.hashes if h not in thresholds]
    if unknown:
        raise SystemExit(f"no FPR {args.fpr:g} threshold for {unknown} in {args.thresholds_from}")
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    n = len(paths)
    if n < 2 and "collision" in args.modes:
        raise SystemExit("collision needs at least 2 images")
    target_of = {names[i]: names[(i + n // 2) % n] for i in range(n)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": n,
        "norm": args.norm, "steps": args.steps, "method": args.method, "step_size": args.step_size,
        "random_start": not args.no_random_start, "margin": args.margin, "fpr": args.fpr,
        "max_side": args.max_side, "seed": args.seed,
        "thresholds_from": [str(p) for p in args.thresholds_from],
        "hashes": args.hashes, "eps_255": args.eps, "modes": args.modes,
        "collision_targets": "image i -> image (i + n//2) mod n of the sorted list",
        "device": device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    check_config(args.out, config)
    raw = args.out / "raw.csv"
    done = done_keys(raw)
    images = {p.name: load_image(p, args.max_side) for p in paths}
    print(f"{n} images, {len(args.hashes)} hashes, eps {args.eps}, modes {args.modes}, device {device}; "
          f"{len(done)} rows already done", flush=True)

    t_start = time.time()
    for name in args.hashes:
        hasher = get_hasher(name)
        thr = math.floor(thresholds[name] * hasher.max_distance + 1e-6)  # in bits / L1 units
        adapter = build_adapter(hasher).to(torch.device(device))
        for eps in args.eps:
            for mode in args.modes:
                todo = [nm for nm in names if (name, mode, float(eps), nm) not in done]
                if not todo:
                    continue
                cfg = AttackConfig(norm=args.norm, eps=eps / 255.0, steps=args.steps, method=args.method,
                                   step_size=None if args.step_size is None else args.step_size / 255.0,
                                   random_start=not args.no_random_start, margin=args.margin,
                                   device=device)
                for b in range(0, len(todo), args.batch_size):
                    batch = todo[b:b + args.batch_size]
                    # seed depends on the batch content only, so resuming reproduces it
                    cfg.seed = args.seed + names.index(batch[0])
                    targets = [images[target_of[nm]] for nm in batch] if mode == "collision" else None
                    t0 = time.time()
                    res = attack_batch(hasher, [images[nm] for nm in batch], cfg, thr, targets,
                                       adapter=adapter, keep_images=False)
                    per_img = (time.time() - t0) / len(batch)
                    append_rows(raw, [{
                        "hash": name, "mode": mode, "norm": args.norm, "eps_255": float(eps), "image": nm,
                        "target": target_of[nm] if mode == "collision" else "", "threshold": thr,
                        "distance": r.distance, "orig_distance": r.orig_distance, "success": r.success,
                        "linf": r.linf, "l2": r.l2, "psnr": r.psnr, "ssim": r.ssim,
                        "seconds_per_image": per_img} for nm, r in zip(batch, res)])
                    ok = sum(r.success for r in res)
                    print(f"[{time.time() - t_start:7.0f}s] {name:18s} {mode:9s} eps={eps:g}/255 "
                          f"{b + len(batch):4d}/{len(todo)}  success {ok}/{len(batch)}  "
                          f"{per_img:.2f} s/img", flush=True)
        del adapter, hasher
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    s = summarize(args.out)
    print(f"done in {time.time() - t_start:.0f}s; wrote {args.out / 'summary.md'}")
    if s is not None:
        print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
