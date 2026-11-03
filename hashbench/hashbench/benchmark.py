"""Robustness / discrimination benchmark.

For every image in a folder and every algorithm:
    * hash the original;
    * hash each transformed copy from the attack suite -> *genuine* distances;
    * compare originals of different images           -> *impostor* distances.

Outputs (in --out):
    hashes.csv    original hashes (hex) and PDQ quality per image
    genuine.csv   one row per (image, algorithm, attack, param)
    impostor.csv  one row per (algorithm, image pair)
    config.json   the exact run configuration

Then run ``python -m hashbench.report --run <out>`` for summary tables and plots.

With ``--attack-limit N`` the attack suite runs on a random subset of N images only,
while impostor pairs still use the originals of all images (hashing an original is cheap,
and more impostor pairs give a reliable threshold at a low false match rate).
``--attack-images FILE`` does the same with a fixed list of image names instead of a random
draw (e.g. ``data/splits/V1k.txt`` from ``scripts/make_splits.py``), so that later
experiments can reuse exactly the same images.

Long runs are checkpointed: every ``--checkpoint-every`` images (default 100) the finished
results are saved to ``<out>/checkpoints/``. Rerunning the same command after an
interruption skips the saved images and continues; the final outputs are identical to an
uninterrupted run. The checkpoints are deleted once the outputs are written.

Usage:
    python -m hashbench.benchmark --images data/samples --out results/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from hashbench.attacks import ATTACK_SUITE, apply_attack
from hashbench.hashes import PDQHash, get_hasher, list_hashers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Matcher:
    """How an algorithm is used for matching.

    Most algorithms hash the query once. PDQ in dihedral mode hashes the query in all
    8 rotations/flips and keeps the minimum distance, which is how PDQ is meant to be
    deployed when rotated/flipped copies must still match.
    """

    name: str
    algorithm: str
    dihedral: bool = False

    def __post_init__(self):
        self.hasher = get_hasher(self.algorithm)
        if self.dihedral and not isinstance(self.hasher, PDQHash):
            raise ValueError("dihedral matching is only implemented for PDQ")

    @property
    def n_bits(self) -> int:
        return self.hasher.n_bits

    @property
    def max_distance(self) -> int:
        return self.hasher.max_distance

    def reference_hash(self, image: Image.Image) -> np.ndarray:
        return self.hasher.hash(image)

    def query_hashes(self, image: Image.Image) -> np.ndarray:
        """Shape (k, n_bits): k=1 normally, k=8 in dihedral mode."""
        return self.query_hashes_batch([image])[0]

    def query_hashes_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Shape (len(images), k, n_bits)."""
        if self.dihedral:
            return np.stack([np.stack(list(self.hasher.hash_dihedral(im)[0].values())) for im in images])
        return self.hasher.hash_batch(images)[:, None, :]


def build_matchers(algorithms: list[str]) -> list[Matcher]:
    matchers = []
    for alg in algorithms:
        if alg == "pdq_dihedral":
            matchers.append(Matcher("pdq_dihedral", "pdq", dihedral=True))
        else:
            matchers.append(Matcher(alg, alg))
    return matchers


# Matchers are built once per process: neural hashes take seconds to load.
_MATCHER_CACHE: dict[tuple[str, ...], list[Matcher]] = {}


def get_matchers(algorithms: list[str]) -> list[Matcher]:
    key = tuple(algorithms)
    if key not in _MATCHER_CACHE:
        _MATCHER_CACHE[key] = build_matchers(algorithms)
    return _MATCHER_CACHE[key]


def _init_worker(torch_threads: int) -> None:
    """Split the CPU between worker processes instead of letting each torch use all cores."""
    try:
        import torch
        torch.set_num_threads(torch_threads)
    except ImportError:
        pass


def find_images(folder: Path, limit: int | None, seed: int) -> list[Path]:
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None and limit < len(paths):
        rng = np.random.default_rng(seed)
        paths = sorted(rng.choice(paths, size=limit, replace=False).tolist())
    return paths


def image_id(path: Path, folder: Path) -> str:
    """Name of an image relative to the image folder, with forward slashes."""
    return str(path.relative_to(folder)).replace("\\", "/")


def load_image(path: Path, max_side: int | None) -> Image.Image:
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _process_image(args) -> tuple[dict, list[dict]]:
    """Hash one image and all its transformed copies. Runs in a worker process."""
    path, image_id, algorithms, suite, max_side = args
    matchers = get_matchers(algorithms)
    img = load_image(Path(path), max_side)

    info = {"image": image_id, "width": img.width, "height": img.height}
    refs = {}
    for m in matchers:
        refs[m.name] = m.reference_hash(img)
        info[f"{m.name}_hash"] = m.hasher.to_string(refs[m.name])
        info[f"{m.name}_query"] = m.query_hashes(img)  # kept in memory for impostor pairs
    pdq = next((m.hasher for m in matchers if m.algorithm == "pdq"), None)
    if pdq is not None:
        info["pdq_quality"] = pdq.hash_with_quality(img)[1]

    rows, attacked, keys = [], [], []
    for attack, params in suite.items():
        for param in params:
            try:
                attacked.append(apply_attack(img, attack, param))
                keys.append((attack, param))
            except Exception as exc:  # keep going; record the failure
                rows.append({"image": image_id, "attack": attack, "param": param, "error": repr(exc)})
    for m in matchers:
        if not attacked:
            break
        q = m.query_hashes_batch(attacked)                         # (n, k, bits)
        dists = m.hasher.distances(q, refs[m.name][None, None, :]).min(axis=1)
        for (attack, param), d in zip(keys, dists):
            rows.append({
                "image": image_id, "algorithm": m.name, "attack": attack, "param": param,
                "distance": int(d), "norm_distance": int(d) / m.max_distance,
            })
    return info, rows


def impostor_distances(infos: list[dict], matchers: list[Matcher], max_pairs: int, seed: int) -> pd.DataFrame:
    """Distances between originals of *different* images (reference i vs query j)."""
    n = len(infos)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = pairs[rng.choice(len(pairs), size=max_pairs, replace=False)]
    frames = []
    for m in matchers:
        refs = np.stack([m.hasher.from_string(info[f"{m.name}_hash"]) for info in infos])  # (n, bits)
        queries = np.stack([info[f"{m.name}_query"] for info in infos])           # (n, k, bits)
        d = np.empty(len(pairs), dtype=np.int64)
        for start in range(0, len(pairs), 50_000):
            p = pairs[start:start + 50_000]
            d[start:start + 50_000] = m.hasher.distances(queries[p[:, 1]], refs[p[:, 0]][:, None, :]).min(axis=1)
        frames.append(pd.DataFrame({
            "algorithm": m.name,
            "image_a": [infos[i]["image"] for i in pairs[:, 0]],
            "image_b": [infos[j]["image"] for j in pairs[:, 1]],
            "distance": d,
            "norm_distance": d / m.max_distance,
        }))
    return pd.concat(frames, ignore_index=True)


def run(images: Path, out: Path, algorithms: list[str], attacks: list[str] | None,
        limit: int | None, max_side: int | None, max_pairs: int, workers: int, seed: int,
        attack_limit: int | None = None, attack_images: Path | None = None,
        checkpoint_every: int = 100) -> None:
    suite = {a: p for a, p in ATTACK_SUITE.items() if attacks is None or a in attacks}
    paths = find_images(images, limit, seed)
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images in {images}, found {len(paths)}")
    attacked = set(paths)
    if attack_images is not None:
        wanted = set(attack_images.read_text().split())
        attacked = {p for p in paths if image_id(p, images) in wanted}
        missing = wanted - {image_id(p, images) for p in paths}
        if missing:
            raise SystemExit(f"{len(missing)} images listed in {attack_images} are not in {images} "
                             f"(or were dropped by --limit), e.g. {sorted(missing)[0]}")
    elif attack_limit is not None and attack_limit < len(paths):
        rng = np.random.default_rng(seed + 1)
        attacked = set(rng.choice(paths, size=attack_limit, replace=False).tolist())
    out.mkdir(parents=True, exist_ok=True)
    matchers = get_matchers(algorithms)

    config = {
        "images": str(images), "n_images": len(paths), "algorithms": algorithms,
        "n_attacked": len(attacked), "attack_limit": attack_limit,
        "attack_images": str(attack_images) if attack_images else None,
        "attack_suite": suite, "limit": limit, "max_side": max_side,
        "max_pairs": max_pairs, "seed": seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    jobs = [(str(p), image_id(p, images), algorithms,
             suite if p in attacked else {}, max_side) for p in paths]
    chunks = [jobs[i:i + checkpoint_every] for i in range(0, len(jobs), checkpoint_every)]
    ckpt_dir = _prepare_checkpoints(out, config)
    todo = [k for k in range(len(chunks)) if not _chunk_path(ckpt_dir, k).exists()]
    if len(todo) < len(chunks):
        print(f"resuming: {len(chunks) - len(todo)} of {len(chunks)} chunks already done")

    # Each finished chunk goes straight to disk, so the main process holds no results.
    done = sum(len(c) for k, c in enumerate(chunks) if k not in todo)
    progress = tqdm(total=len(jobs), initial=done, desc="hashing")
    if workers > 1 and todo:
        torch_threads = max(1, (os.cpu_count() or 2) // workers)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(torch_threads,)) as pool:
            for k in todo:
                results = []
                for res in pool.map(_process_image, chunks[k], chunksize=4):
                    results.append(res)
                    progress.update()
                _save_chunk(ckpt_dir, k, chunks[k], results)
    else:
        for k in todo:
            results = []
            for job in chunks[k]:
                results.append(_process_image(job))
                progress.update()
            _save_chunk(ckpt_dir, k, chunks[k], results)
    progress.close()

    infos, n_genuine, n_err = [], 0, 0
    with open(out / "genuine.csv", "w", newline="") as f:
        for k in range(len(chunks)):
            chunk_infos, genuine = _load_chunk(ckpt_dir, k, chunks[k])
            infos.extend(chunk_infos)
            genuine.to_csv(f, index=False, header=(k == 0))
            n_genuine += len(genuine)
            n_err += int(genuine["error"].notna().sum())
    if n_err:
        print(f"warning: {n_err} attack applications failed (see 'error' column in genuine.csv)")

    impostor = impostor_distances(infos, matchers, max_pairs, seed)
    impostor.to_csv(out / "impostor.csv", index=False)

    pd.DataFrame([{k: v for k, v in info.items() if not k.endswith("_query")} for info in infos]) \
        .to_csv(out / "hashes.csv", index=False)
    shutil.rmtree(ckpt_dir)
    print(f"wrote {n_genuine} genuine and {len(impostor)} impostor distances to {out}")


#: Columns of genuine.csv, fixed so that every chunk is written with the same header.
GENUINE_COLUMNS = ["image", "algorithm", "attack", "param", "distance", "norm_distance", "error"]


def _prepare_checkpoints(out: Path, config: dict) -> Path:
    """Checkpoint folder for this run; refuses to mix in checkpoints of a different config."""
    ckpt_dir = out / "checkpoints"
    fingerprint = json.loads(json.dumps({k: v for k, v in config.items() if k != "started"}))
    fp_file = ckpt_dir / "config.json"
    if fp_file.exists():
        if json.loads(fp_file.read_text()) != fingerprint:
            raise SystemExit(f"{ckpt_dir} holds checkpoints of a run with a different configuration; "
                             f"delete it or use another --out")
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fp_file.write_text(json.dumps(fingerprint, indent=2))
    return ckpt_dir


def _chunk_path(ckpt_dir: Path, k: int) -> Path:
    return ckpt_dir / f"chunk_{k:05d}.pkl"


def _save_chunk(ckpt_dir: Path, k: int, jobs: list, results: list) -> None:
    genuine = pd.DataFrame([row for _, rows in results for row in rows], columns=GENUINE_COLUMNS)
    payload = {"images": [job[1] for job in jobs], "infos": [info for info, _ in results],
               "genuine": genuine}
    tmp = _chunk_path(ckpt_dir, k).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(_chunk_path(ckpt_dir, k))  # atomic: a kill never leaves a half-written chunk


def _load_chunk(ckpt_dir: Path, k: int, jobs: list) -> tuple[list[dict], pd.DataFrame]:
    with open(_chunk_path(ckpt_dir, k), "rb") as f:
        payload = pickle.load(f)
    if payload["images"] != [job[1] for job in jobs]:
        raise SystemExit(f"checkpoint {k} does not match this run's images; delete {ckpt_dir}")
    return payload["infos"], payload["genuine"]


def main(argv: list[str] | None = None) -> None:
    choices = list_hashers() + ["pdq_dihedral"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="folder of images (searched recursively)")
    ap.add_argument("--out", type=Path, required=True, help="output folder for this run")
    ap.add_argument("--algorithms", nargs="+", default=["phash", "pdq", "pdq_dihedral"], choices=choices)
    ap.add_argument("--attacks", nargs="+", default=None, choices=sorted(ATTACK_SUITE),
                    help="subset of attacks (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="random subset of N images")
    ap.add_argument("--attack-limit", type=int, default=None,
                    help="run the attack suite on a random subset of N of those images only")
    ap.add_argument("--attack-images", type=Path, default=None,
                    help="run the attack suite only on the images named in this file "
                         "(one name per line, relative to --images), e.g. data/splits/V1k.txt")
    ap.add_argument("--max-side", type=int, default=512,
                    help="downscale originals so the longest side is at most this (0 = keep size)")
    ap.add_argument("--max-pairs", type=int, default=200_000, help="max impostor pairs per algorithm")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="save finished results to <out>/checkpoints every N images (resumable)")
    args = ap.parse_args(argv)
    run(args.images, args.out, args.algorithms, args.attacks, args.limit,
        args.max_side or None, args.max_pairs, args.workers, args.seed, args.attack_limit,
        args.attack_images, args.checkpoint_every)


if __name__ == "__main__":
    main()
