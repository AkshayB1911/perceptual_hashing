"""Targeted probes that check explanations of benchmark results (CPU, a few minutes).

1. NeuralHash and global brightness/contrast. Its first layer is conv -> instance norm, so a
   positive affine change of the input (a * x + b) is removed exactly, apart from the
   convolution's zero padding and uint8 rounding/clipping. Probe: scale the [0, 1] image by
   a in float (no rounding, no clipping), compare logits and bits with the original.
2. CertPHash and the image border. The benchmark showed ep2/ep4 failing on a 2% black border
   (about 1 px at the 64x64 input). Probe: at the 64x64 input, overwrite only the outer k
   pixels with black / grey / white / the replicated edge, and measure the L1 distance
   relative to the run's calibrated threshold. No geometric change at all.
3. Flips for pHash/PDQ: a flip negates every DCT coefficient with an odd frequency along
   that axis, so roughly half the bits should change (distance ~ impostor level).

Writes <run>/analysis/probe_*.csv.

Usage:
    python scripts/probe_invariances.py --run results/coco_run1 --images data/coco/val2017 \
        --list data/splits/V200.txt --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from hashbench.benchmark import load_image
from hashbench.hashes import get_hasher
from hashbench.hashes.certphash import load_certphash_model, preprocess
from hashbench.hashes.neuralhash import INPUT_SIZE as NH_SIZE


def neuralhash_affine(images: list[Image.Image]) -> pd.DataFrame:
    nh = get_hasher("neuralhash")
    model = nh.model if hasattr(nh, "model") else None
    if model is None:
        from hashbench.hashes.neuralhash import load_neuralhash_model
        model = load_neuralhash_model()
    x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB").resize((NH_SIZE, NH_SIZE), Image.Resampling.BICUBIC),
                                                 dtype=np.float32) / 255.0).permute(2, 0, 1) for im in images])
    rows = []
    with torch.no_grad():
        ref = model(x)
        for name, fn in [("scale 0.5", lambda t: 0.5 * t), ("scale 0.7", lambda t: 0.7 * t),
                         ("scale 0.5 + shift 0.25", lambda t: 0.5 * t + 0.25),
                         ("gamma 2.0 (nonlinear)", lambda t: t ** 2.0),
                         ("scale 1.6 + clip (as brightness 1.6)", lambda t: (1.6 * t).clamp(0, 1))]:
            out = model(fn(x))
            bits = ((out > 0) != (ref > 0)).sum(dim=1).float()
            rel = ((out - ref).norm(dim=1) / ref.norm(dim=1))
            rows.append({"transform": name, "mean_bits_changed_of_96": bits.mean().item(),
                         "max_bits_changed": bits.max().item(), "mean_relative_logit_change": rel.mean().item()})
    return pd.DataFrame(rows)


def certphash_edges(images: list[Image.Image], thresholds: dict[str, float]) -> pd.DataFrame:
    x = torch.stack([preprocess(im) for im in images])  # (n, 3, 64, 64) in [0, 1]
    fills = {"black": 0.0, "grey": 0.5, "white": 1.0}
    rows = []
    for variant in ("ep1", "ep2", "ep4", "ep8", "base", "adv"):
        name = "certphash" if variant == "ep1" else f"certphash_{variant}"
        model = load_certphash_model(variant)
        with torch.no_grad():
            ref = model(x).round().clamp(0, 255)
            for k in (1, 2, 4):
                for fill in ("black", "grey", "white", "replicate"):
                    y = x.clone()
                    if fill == "replicate":
                        inner = x[:, :, k:-k, k:-k]
                        y = torch.nn.functional.pad(inner, (k, k, k, k), mode="replicate")
                    else:
                        v = fills[fill]
                        y[:, :, :k, :] = v; y[:, :, -k:, :] = v; y[:, :, :, :k] = v; y[:, :, :, -k:] = v
                    out = model(y).round().clamp(0, 255)
                    d = (out - ref).abs().sum(dim=1) / (144 * 255)
                    rows.append({"algorithm": name, "edge_px": k, "fill": fill,
                                 "mean_norm_distance": d.mean().item(),
                                 "match_rate_at_calibrated": (d <= thresholds[name]).float().mean().item()})
    return pd.DataFrame(rows)


def flips_dct(images: list[Image.Image]) -> pd.DataFrame:
    rows = []
    for alg in ("phash", "pdq"):
        h = get_hasher(alg)
        ref = h.hash_batch(images)
        for name, t in (("flip_horizontal", Image.Transpose.FLIP_LEFT_RIGHT),
                        ("flip_vertical", Image.Transpose.FLIP_TOP_BOTTOM)):
            q = h.hash_batch([im.transpose(t) for im in images])
            d = (q != ref).mean(axis=1)
            rows.append({"algorithm": alg, "flip": name, "mean_norm_distance": d.mean(),
                         "min": d.min(), "max": d.max()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--fpr", type=float, default=0.001)
    args = ap.parse_args()
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)

    names = args.list.read_text().split()[: args.n]
    images = [load_image(args.images / n, 512) for n in names]  # same downscaling as the benchmark
    overall = pd.read_csv(args.run / "overall.csv").set_index("algorithm")
    thresholds = overall[f"threshold_at_fpr_{args.fpr:g}"].to_dict()

    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for fname, df in (("probe_neuralhash_affine.csv", neuralhash_affine(images)),
                          ("probe_certphash_edges.csv", certphash_edges(images, thresholds)),
                          ("probe_flips_dct.csv", flips_dct(images))):
            df.to_csv(out / fname, index=False)
            print(f"== {fname}\n{df.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
