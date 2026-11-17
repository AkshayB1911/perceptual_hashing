"""Transfer attack, step 1: generate and save white-box adversarial images.

Runs the same white-box attack as ``run_adversarial.py`` (APGD + small-step PGD ensemble,
100 steps, warm start across budgets, best iterate by hash distance on the 8-bit image) on
each *source* hash, and saves every attacked image as a PNG so that
``transfer_matrix.py`` can re-hash it with every other (*target*) hash.

``run_adversarial.py`` is not modified; this script only imports its helpers.

Saved layout (under ``--out``):

    images/<source>/clean/<stem>.png               clean model input at the source's size
    images/<source>/evasion/eps<k>/<stem>.png      attacked image (8-bit) at budget k/255
    images/<source>/collision/eps<k>/<stem>.png    pushed towards image (i + n/2) mod n
    generation.csv                                 one row per attacked image: white-box
                                                   distance / success on the source itself
    config.json                                    settings (a rerun must match them)

Images are saved at the source model's input size (DinoHash 224, NeuralHash 360,
CertPHash 64), because that is where the perturbation lives. The clean PNG is saved too, so
a target hash compares B(clean at source size) with B(adversarial at source size): only the
perturbation differs, not the resizing. For DinoHash / NeuralHash the clean input is already
8-bit (PIL resize); CertPHash resizes in float, so its clean PNG is the input rounded to 8
bits (at most 0.5/255 away from what the attack started from).

Resumable: images already listed in generation.csv are skipped. Warm starts are read back
from the PNGs of the previous budget, so a resumed run continues with the same chain.

Example (from hashbench/):
    python scripts/generate_transfer_images.py --list data/splits/V200.txt \
        --out results/transfer_v200 --device cuda --batch-size 16 \
        --hashes dinohash dinohash_untrained certphash certphash_ep2 certphash_ep4 \
                 certphash_ep8 certphash_base certphash_adv
    # NeuralHash locally (Apple's files stay on the laptop):
    python scripts/generate_transfer_images.py --list data/splits/V200.txt \
        --out results/transfer_v200_neuralhash --device cpu --hashes neuralhash
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hashbench.adversarial import AttackConfig, attack_batch, build_adapter, _to_image  # noqa: E402
from hashbench.benchmark import load_image  # noqa: E402
from hashbench.hashes import get_hasher  # noqa: E402
from run_adversarial import DEFAULT_HASHES, image_list, load_thresholds  # noqa: E402

COLUMNS = ["hash", "mode", "eps_255", "image", "target", "threshold", "max_distance", "distance",
           "orig_distance", "success", "linf", "ssim", "seconds_per_image"]
CONFIG_KEYS = ["list", "limit", "norm", "steps", "method", "random_start", "margin", "fpr",
               "max_side", "seed", "thresholds_from", "eps_evasion", "eps_collision"]


def eps_dir(eps: float) -> str:
    return f"eps{eps:g}"


def png_path(root: Path, source: str, mode: str, eps: float | None, name: str) -> Path:
    stem = Path(name).stem + ".png"
    if mode == "clean":
        return root / "images" / source / "clean" / stem
    return root / "images" / source / mode / eps_dir(eps) / stem


def load_png(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/coco/val2017"))
    ap.add_argument("--list", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hashes", nargs="+", default=DEFAULT_HASHES, help="source hashes to attack")
    ap.add_argument("--modes", nargs="+", default=["evasion", "collision"], choices=["evasion", "collision"])
    ap.add_argument("--eps-evasion", nargs="+", type=float, default=[2, 4, 8])
    ap.add_argument("--eps-collision", nargs="+", type=float, default=[8, 16])
    ap.add_argument("--norm", default="linf", choices=["linf"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--method", default="ensemble", choices=["ensemble", "apgd", "pgd"])
    ap.add_argument("--no-random-start", action="store_true")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--thresholds-from", nargs="+", type=Path,
                    default=[Path("results/coco_run1"), Path("results/coco_run2")])
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds_from, args.fpr)
    missing = [h for h in args.hashes if h not in thresholds]
    if missing:
        raise SystemExit(f"no FPR {args.fpr:g} threshold for {missing}")
    paths = image_list(args.images, args.list, args.limit)
    names = [p.name for p in paths]
    n = len(paths)
    if n < 2 and "collision" in args.modes:
        raise SystemExit("collision needs at least 2 images")
    target_of = {names[i]: names[(i + n // 2) % n] for i in range(n)}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    eps_of = {"evasion": sorted(args.eps_evasion), "collision": sorted(args.eps_collision)}

    config = {
        "list": str(args.list) if args.list else None, "limit": args.limit, "n_images": n,
        "norm": args.norm, "steps": args.steps, "method": args.method,
        "random_start": not args.no_random_start, "margin": args.margin, "fpr": args.fpr,
        "max_side": args.max_side, "seed": args.seed,
        "thresholds_from": [str(p) for p in args.thresholds_from],
        "eps_evasion": eps_of["evasion"], "eps_collision": eps_of["collision"],
        "hashes": args.hashes, "modes": args.modes,
        "collision_targets": "image i -> image (i + n//2) mod n of the list",
        "warm_start": "from the saved 8-bit PNG of the previous budget of the same mode",
        "device": device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    check_config(args.out, config)
    csv = args.out / "generation.csv"
    done = done_keys(csv)
    images = {p.name: load_image(p, args.max_side) for p in paths}
    print(f"{n} images, sources {args.hashes}, evasion eps {eps_of['evasion']}, "
          f"collision eps {eps_of['collision']}, device {device}; {len(done)} already done", flush=True)

    t_start = time.time()
    for name in args.hashes:
        hasher = get_hasher(name)
        thr = math.floor(thresholds[name] * hasher.max_distance + 1e-6)
        adapter = build_adapter(hasher).to(torch.device(device))

        # clean model inputs, saved once per source
        for nm in names:
            p = png_path(args.out, name, "clean", None, nm)
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                _to_image(adapter.preprocess(images[nm])).save(p)

        for mode in args.modes:
            prev_eps = None
            for eps in eps_of[mode]:
                todo = [nm for nm in names if (name, mode, float(eps), nm) not in done]
                if todo:
                    cfg = AttackConfig(norm=args.norm, eps=eps / 255.0, steps=args.steps, method=args.method,
                                       random_start=not args.no_random_start, margin=args.margin,
                                       device=device)
                    out_dir = args.out / "images" / name / mode / eps_dir(eps)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for b in range(0, len(todo), args.batch_size):
                        batch = todo[b:b + args.batch_size]
                        cfg.seed = args.seed + names.index(batch[0])
                        targets = [images[target_of[nm]] for nm in batch] if mode == "collision" else None
                        init = None
                        if prev_eps is not None:
                            prev = [png_path(args.out, name, mode, prev_eps, nm) for nm in batch]
                            if all(p.exists() for p in prev):
                                init = [load_png(p) for p in prev]
                        t0 = time.time()
                        res = attack_batch(hasher, [images[nm] for nm in batch], cfg, thr, targets,
                                           adapter=adapter, keep_images=True, init=init)
                        for nm, r in zip(batch, res):
                            r.adv_image.save(png_path(args.out, name, mode, eps, nm))
                        per_img = (time.time() - t0) / len(batch)
                        rows = [{
                            "hash": name, "mode": mode, "eps_255": float(eps), "image": nm,
                            "target": target_of[nm] if mode == "collision" else "", "threshold": thr,
                            "max_distance": hasher.max_distance, "distance": r.distance,
                            "orig_distance": r.orig_distance, "success": r.success, "linf": r.linf,
                            "ssim": r.ssim, "seconds_per_image": per_img} for nm, r in zip(batch, res)]
                        pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, mode="a", header=not csv.exists(),
                                                                   index=False)
                        ok = sum(r.success for r in res)
                        print(f"[{time.time() - t_start:7.0f}s] {name:18s} {mode:9s} eps={eps:g}/255 "
                              f"{b + len(batch):4d}/{len(todo)}  white-box success {ok}/{len(batch)}  "
                              f"{per_img:.2f} s/img", flush=True)
                prev_eps = eps
        del adapter, hasher
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.read_csv(csv)
    s = (df.groupby(["hash", "mode", "eps_255"])["success"].mean() * 100).round(1)
    print(f"done in {time.time() - t_start:.0f}s. White-box success (%) on the source itself:")
    print(s.to_string())


if __name__ == "__main__":
    main()
