"""Adversarial sweep: white-box evasion and collision for every learned hash and budget.

For each hash, each budget eps (in /255 units) and each mode, attacks every image of a
fixed list with ``hashbench.adversarial.attack_batch`` and appends one row per image to
``<out>/raw.csv``. Success uses the same calibrated thresholds as the benign benchmark
(FPR 1e-3 by default, read from the runs' ``overall.csv``), so the adversarial and benign
numbers are on the same operating point:

    evasion    success if  d(adv, original) >  threshold   (the copy is no longer found)
    collision  success if  d(adv, target)   <= threshold   (a false match is forced)

Two collision settings:

    collision          random fixed pairs: image i of the list is pushed onto image
                       (i + n/2) mod n, so every model and budget attacks the same pairs.
    collision_closest  each image is pushed onto the *other* list image whose clean hash is
                       already closest under that model (the setting of Struppek et al. and
                       CertPHash's Table 5; their pool is the whole test set, ours is the list).

Collision usually needs bigger budgets than evasion (CertPHash's successful collisions sit
near l_inf 0.10, about 25/255), so ``--eps-collision`` can add budgets such as 16 and 32.

Attack: by default "ensemble" (APGD and small-step PGD from the same start, better result
kept per image) with a warm start across budgets: each image's attack at eps_k starts from
its best adversarial image at the next smaller eps of the same mode, which lies inside the
larger ball. Success and progress are therefore monotone in eps by construction, and each
number is the strongest attack found at that budget. A resumed run loses the warm starts of
the interrupted budget (those images start randomly); ``--no-warm-start`` turns it off.

Reported next to the calibrated-threshold success rate:
    * bit accuracy for the 96-bit hashes (DinoHash's paper metric): share of bits that still
      equal the original (evasion) or already equal the target (collision);
    * CertPHash success at the paper's L1 threshold 1800 as well as at the calibrated one;
    * an epsilon-monotonicity check: a larger budget contains every smaller one, so success
      and progress should never fall as eps grows. Drops are listed in summary.md and
      monotonicity.csv; small drops are optimiser noise, large ones mean the attack is not
      converging (the pre-fix best-iterate bug showed up exactly this way).

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
RAW_COLUMNS = ["hash", "mode", "norm", "eps_255", "image", "target", "threshold", "max_distance",
               "distance", "orig_distance", "success", "linf", "l2", "psnr", "ssim", "seconds_per_image"]
MODES = ["evasion", "collision", "collision_closest"]
CERT_PAPER_THRESHOLD = 1800       # CertPHash paper, L1 match threshold (Sec. 5.2)
BIT_HASHES = ("dinohash", "neuralhash")   # prefixes of the 96-bit hashes
# settings that change the numbers; a rerun into the same folder must match them
CONFIG_KEYS = ["list", "limit", "norm", "steps", "method", "step_size", "random_start",
               "margin", "fpr", "max_side", "seed", "thresholds_from", "warm_start"]


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


def _is_cert(h: str) -> bool:
    return h.startswith("certphash")


def _is_bit(h: str) -> bool:
    return h.startswith(BIT_HASHES)


def _success_at(mode: str, dist: pd.Series, thr: float) -> pd.Series:
    return dist > thr if mode == "evasion" else dist <= thr


def summarize(out: Path) -> pd.DataFrame | None:
    raw = out / "raw.csv"
    if not raw.exists():
        return None
    df = pd.read_csv(raw)
    rows = []
    for (h, mode, eps), g in df.groupby(["hash", "mode", "eps_255"], sort=True):
        k, n = int(g["success"].sum()), len(g)
        lo, hi = wilson_interval(k, n)
        row = {
            "hash": h, "mode": mode, "eps_255": eps, "n": n, "success_rate": k / n,
            "ci_low": lo, "ci_high": hi, "threshold": g["threshold"].iloc[0],
            "mean_distance": g["distance"].mean(), "mean_orig_distance": g["orig_distance"].mean(),
            "already_success_before_attack": float(_success_at(mode, g["orig_distance"], g["threshold"].iloc[0]).mean()),
            "bit_accuracy": float("nan"), "success_rate_paper_thr": float("nan"),
            "ci_low_paper": float("nan"), "ci_high_paper": float("nan"),
            "mean_linf": g["linf"].mean(), "mean_psnr": g["psnr"].replace(math.inf, float("nan")).mean(),
            "mean_ssim": g["ssim"].mean(), "seconds_per_image": g["seconds_per_image"].mean(),
        }
        if _is_bit(h) and "max_distance" in g:
            # evasion: bits still equal to the original; collision: bits equal to the target
            row["bit_accuracy"] = float(1 - (g["distance"] / g["max_distance"]).mean())
        if _is_cert(h):
            kp = int(_success_at(mode, g["distance"], CERT_PAPER_THRESHOLD).sum())
            row["success_rate_paper_thr"] = kp / n
            row["ci_low_paper"], row["ci_high_paper"] = wilson_interval(kp, n)
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    mono = monotonicity(df)
    mono.to_csv(out / "monotonicity.csv", index=False)
    write_markdown(s, mono, out / "summary.md")
    return s


def monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """Every place where a larger eps did *worse* than the next smaller one.

    level "rate":  the success rate of a hash x mode fell between consecutive budgets.
    level "image": one image's attack progress fell (evasion: fewer bits / smaller L1 from the
                   original; collision: further from the target). ``drop`` is in distance units.
    """
    out = []
    for (h, mode), g in df.groupby(["hash", "mode"], sort=True):
        eps = sorted(g["eps_255"].unique())
        rate = g.groupby("eps_255")["success"].mean()
        for a, b in zip(eps, eps[1:]):
            if rate[b] < rate[a] - 1e-12:
                out.append({"hash": h, "mode": mode, "level": "rate", "image": "", "eps_from": a,
                            "eps_to": b, "value_from": rate[a], "value_to": rate[b], "drop": rate[a] - rate[b]})
        # progress: larger is better
        prog = g.assign(p=g["distance"] if mode == "evasion" else -g["distance"])
        piv = prog.pivot_table(index="image", columns="eps_255", values="p", aggfunc="first")
        for a, b in zip(eps, eps[1:]):
            if a not in piv or b not in piv:
                continue
            d = (piv[a] - piv[b]).dropna()
            for img, drop in d[d > 0].items():
                out.append({"hash": h, "mode": mode, "level": "image", "image": img, "eps_from": a,
                            "eps_to": b, "value_from": abs(piv.at[img, a]), "value_to": abs(piv.at[img, b]),
                            "drop": float(drop)})
    return pd.DataFrame(out, columns=["hash", "mode", "level", "image", "eps_from", "eps_to",
                                      "value_from", "value_to", "drop"])


def _rate_cell(rate, lo, hi) -> str:
    return f"{100 * rate:.1f} ({100 * lo:.0f}–{100 * hi:.0f})"


def _table(m: pd.DataFrame, eps_list, fmt) -> list[str]:
    lines = ["| Hash | n | " + " | ".join(f"ε={e:g}/255" for e in eps_list) + " |",
             "|---|---|" + "---|" * len(eps_list)]
    for h, g in m.groupby("hash", sort=False):
        cells = []
        for e in eps_list:
            r = g[g["eps_255"] == e]
            cells.append("–" if r.empty else fmt(r.iloc[0]))
        lines.append(f"| {h} | {int(g['n'].max())} | " + " | ".join(cells) + " |")
    return lines


def write_markdown(s: pd.DataFrame, mono: pd.DataFrame, path: Path) -> None:
    lines = ["# Adversarial sweep summary", "",
             "Success rate (%) at the benign FPR threshold, with 95% Wilson interval. "
             "Evasion: the attacked copy is no longer matched. Collision: a false match is forced "
             "(random pairs, or the closest other image for `collision_closest`).", ""]
    titles = {"evasion": "Evasion", "collision": "Collision (random pairs)",
              "collision_closest": "Collision (closest target)"}
    for mode in MODES:
        m = s[s["mode"] == mode]
        if m.empty:
            continue
        eps_list = sorted(m["eps_255"].unique())
        lines += [f"## {titles[mode]}", ""]
        lines += _table(m, eps_list, lambda r: _rate_cell(r.success_rate, r.ci_low, r.ci_high))
        bits = m[m["hash"].map(_is_bit)]
        if not bits.empty:
            what = "bits still equal to the original" if mode == "evasion" else "bits equal to the target"
            lines += ["", f"Bit accuracy (%, {what}; DinoHash paper metric):", ""]
            lines += _table(bits, eps_list, lambda r: f"{100 * r.bit_accuracy:.1f}")
        cert = m[m["hash"].map(_is_cert)]
        if not cert.empty:
            lines += ["", f"CertPHash success (%) at the paper's L1 threshold {CERT_PAPER_THRESHOLD} "
                          "(compare with the calibrated-threshold table above):", ""]
            lines += _table(cert, eps_list, lambda r: _rate_cell(r.success_rate_paper_thr, r.ci_low_paper,
                                                                 r.ci_high_paper))
        lines += ["", "Mean SSIM of the attacked images:", ""]
        lines += _table(m, eps_list, lambda r: f"{r.mean_ssim:.3f}")
        lines.append("")
    lines += ["## ε-monotonicity check", ""]
    if mono.empty:
        lines.append("No drops: success and per-image progress never fell as ε grew.")
    else:
        rate = mono[mono["level"] == "rate"]
        img = mono[mono["level"] == "image"]
        lines.append("A larger budget contains every smaller one, so drops point to optimiser noise "
                     "(small) or a non-converging attack (large).")
        lines.append("")
        if not rate.empty:
            lines += ["Success-rate drops:", "", "| Hash | Mode | ε from → to | rate from → to |", "|---|---|---|---|"]
            for r in rate.itertuples():
                lines.append(f"| {r.hash} | {r.mode} | {r.eps_from:g} → {r.eps_to:g} | "
                             f"{100 * r.value_from:.1f} → {100 * r.value_to:.1f} |")
            lines.append("")
        if not img.empty:
            lines += ["Per-image progress drops (distance units; evasion = distance from the original, "
                      "collision = distance to the target):", "",
                      "| Hash | Mode | images with a drop | largest drop |", "|---|---|---|---|"]
            for (h, mode), g in img.groupby(["hash", "mode"]):
                lines.append(f"| {h} | {mode} | {g['image'].nunique()} | {g['drop'].max():g} |")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def report_monotonicity(mono: pd.DataFrame) -> None:
    rate = mono[mono["level"] == "rate"] if not mono.empty else mono
    if mono.empty:
        print("monotonicity: OK (no drops as eps grows)")
        return
    for r in rate.itertuples():
        print(f"WARNING monotonicity: {r.hash} {r.mode} success fell {100 * r.value_from:.1f}% -> "
              f"{100 * r.value_to:.1f}% from eps {r.eps_from:g} to {r.eps_to:g}")
    img = mono[mono["level"] == "image"]
    if not img.empty:
        print(f"monotonicity: {len(img)} per-image progress drops (see monotonicity.csv)")


def closest_targets(adapter, images: dict, names: list[str], device, batch: int = 32) -> dict[str, str]:
    """For each image, the other list image whose clean hash is closest (ties: list order)."""
    hs = []
    for b in range(0, len(names), batch):
        x = torch.stack([adapter.preprocess(images[nm]) for nm in names[b:b + batch]]).to(device)
        hs.append(adapter.hash(x))
    import numpy as np
    H = np.concatenate(hs)
    out = {}
    for i, nm in enumerate(names):
        d = adapter.distance(H, H[i][None, :]).astype(float)
        d[i] = np.inf
        out[nm] = names[int(np.argmin(d))]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None, help="image list, e.g. data/splits/V200.txt")
    ap.add_argument("--limit", type=int, default=None, help="first N images of the list (pilots)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=DEFAULT_HASHES)
    ap.add_argument("--eps", nargs="+", type=float, default=[1, 2, 4, 8], help="budgets in /255 units")
    ap.add_argument("--eps-collision", nargs="+", type=float, default=None,
                    help="budgets for the collision modes (default: --eps), e.g. 1 2 4 8 16 32")
    ap.add_argument("--modes", nargs="+", default=["evasion", "collision"], choices=MODES)
    ap.add_argument("--norm", default="linf", choices=["linf", "l2"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--method", default="ensemble", choices=["ensemble", "apgd", "pgd"])
    ap.add_argument("--step-size", type=float, default=None,
                    help="PGD step in /255 (default: eps/4 for pgd, min(eps/10, 0.25) for ensemble)")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="start every budget from a random point instead of the previous eps's result")
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
        report_monotonicity(pd.read_csv(args.out / "monotonicity.csv"))
        print(f"wrote {args.out / 'summary.md'}")
        return

    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    unknown = [h for h in args.hashes if h not in thresholds]
    if unknown:
        raise SystemExit(f"no FPR {args.fpr:g} threshold for {unknown} in {args.thresholds_from}")
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    n = len(paths)
    eps_coll = args.eps_collision if args.eps_collision is not None else args.eps
    if n < 2 and any(m.startswith("collision") for m in args.modes):
        raise SystemExit("collision needs at least 2 images")
    target_of = {names[i]: names[(i + n // 2) % n] for i in range(n)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": n,
        "norm": args.norm, "steps": args.steps, "method": args.method, "step_size": args.step_size,
        "random_start": not args.no_random_start, "margin": args.margin, "fpr": args.fpr,
        "max_side": args.max_side, "seed": args.seed, "warm_start": not args.no_warm_start,
        "thresholds_from": [str(p) for p in args.thresholds_from],
        "hashes": args.hashes, "eps_255": sorted(set(args.eps) | set(eps_coll)), "modes": args.modes,
        "eps_255_evasion": args.eps, "eps_255_collision": eps_coll,
        "collision_targets": "collision: image i -> image (i + n//2) mod n of the list; "
                             "collision_closest: the other list image with the closest clean hash",
        "best_iterate": "by hash distance on the 8-bit image, loss breaks ties",
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
        warm = {}     # (mode, image) -> best adversarial input at the previous eps
        closest = None
        if "collision_closest" in args.modes:
            closest = closest_targets(adapter, images, names, torch.device(device))
        for eps in sorted(set(args.eps) | set(eps_coll)):
            for mode in args.modes:
                if eps not in (args.eps if mode == "evasion" else eps_coll):
                    continue
                tgt_map = closest if mode == "collision_closest" else target_of
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
                    targets = [images[tgt_map[nm]] for nm in batch] if mode != "evasion" else None
                    init = None
                    if not args.no_warm_start and all((mode, nm) in warm for nm in batch):
                        init = [warm[(mode, nm)] for nm in batch]
                    t0 = time.time()
                    res = attack_batch(hasher, [images[nm] for nm in batch], cfg, thr, targets,
                                       adapter=adapter, keep_images=True, init=init)
                    for nm, r in zip(batch, res):
                        warm[(mode, nm)] = r.adv
                        r.adv_image = None
                    per_img = (time.time() - t0) / len(batch)
                    append_rows(raw, [{
                        "hash": name, "mode": mode, "norm": args.norm, "eps_255": float(eps), "image": nm,
                        "target": tgt_map[nm] if mode != "evasion" else "", "threshold": thr,
                        "max_distance": hasher.max_distance,
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
    report_monotonicity(pd.read_csv(args.out / "monotonicity.csv"))
    print(f"done in {time.time() - t_start:.0f}s; wrote {args.out / 'summary.md'}")
    if s is not None:
        print((args.out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
