"""Download the weights needed by the learned hashes into ``models/`` (checksums pinned).

DinoHash (Singhi et al., ICML 2025 CODEML workshop):
    * adversarially fine-tuned DinoHash ONNX models (Hugging Face: proteus-photos/DINOHash);
    * the PCA files from the official repo (used for the *untrained* DinoHash baseline);
    * the original DINOv2-with-registers weights from Meta (also for the untrained baseline).

CertPHash (Yang et al., USENIX Security 2025), with --certphash: the released COCO/PhotoDNA
checkpoints (certified eps 1/2/4/8, normal, adversarial; ~63 MB) from the authors' Google
Drive ``saved_models`` folder, via ``gdown``.

Usage:
    python scripts/download_models.py               # DinoHash, everything
    python scripts/download_models.py --small       # DinoHash, only the ViT-S / 96-bit files (~180 MB)
    python scripts/download_models.py --certphash   # CertPHash checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/proteus-photos/DINOHash/resolve/main/"
GH = ("https://raw.githubusercontent.com/proteus-photos/dinohash-perceptual-hash/"
      "16733e9a6a9e6b3c557c3eba5b8ac19cb12c7847/hashes/")
FB = "https://dl.fbaipublicfiles.com/dinov2/"

# (url, file name, sha256, part of --small)
FILES = [
    (HF + "dinov2_vits14_reg_96bit_dynamic.onnx", "dinov2_vits14_reg_96bit_dynamic.onnx",
     "a36e158a27b655cd1f8888fa2d3c736f2ba0386dc6000f6ca762c1f3c72f332e", True),
    (HF + "dinov2_vitb14_reg_512bit_dynamic.onnx", "dinov2_vitb14_reg_512bit_dynamic.onnx",
     "cc399d9583877379ea89675f322bc220dc997b66c1d66766a7c12fc02fc562bd", False),
    (GH + "dinov2_vits14_reg_PCA.npy", "dinov2_vits14_reg_PCA.npy", None, True),
    (GH + "dinov2_vits14_reg_means.npy", "dinov2_vits14_reg_means.npy", None, True),
    (GH + "dinov2_vitb14_reg_PCA.npy", "dinov2_vitb14_reg_PCA.npy", None, False),
    (GH + "dinov2_vitb14_reg_means.npy", "dinov2_vitb14_reg_means.npy", None, False),
    (FB + "dinov2_vits14/dinov2_vits14_reg4_pretrain.pth", "dinov2_vits14_reg4_pretrain.pth", None, True),
    (FB + "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth", None, False),
]


# (Google Drive file id, path under models/certphash, sha256)
CERTPHASH_FILES = [
    ("1uMIH8XHn6Fg41t0NUI_6ldtH9gmqCy50", "coco_photodna_ep1/ckpt_best.pth",
     "ab7dc2ed7ec677780176d0b26d93ff57aa8503be843e7006f4a9b9483ece0b08"),
    ("1tn6Ym_KvH7JgeVwsjWpyIHftEDJRVCC5", "coco_photodna_ep2/ckpt_best.pth",
     "466a8f3f2a1c938ab5fa37501cb07e45d109e866b6c04d113b7afd8277c5240f"),
    ("1UfXogSDNRENPl-GEGQHesX3-DEuHm60L", "coco_photodna_ep4/ckpt_best.pth",
     "a3615a5e19cf0d4036c299b26d7458481f2bfcc18848e34b6d09e051190bdbcf"),
    ("1S2GPKyDMt9jsYif9JQtp7Q3ZePe-PDyo", "coco_photodna_ep8/ckpt_best.pth",
     "8877876dc8fc4d2d114fa5fa270de1b8cb32af76218f224635a9f3b34628fb0b"),
    ("1Y0z6M-bXl3_7DWuQVNnIiuyknbmOs0bz", "coco_photodna_ep0/ckpt_best.pth",
     "50eb2729d48450d6344500923d11f1720655dd0fc9e37b19d39f65a3865c9f4d"),
    ("1o191hi-PS1tH7LMUGfoErbWfslMCc6Vk", "base_adv/coco-pdq-ep8-photodna.pt",
     "76329e070765c28afc21eef0348872d1a8873131c794e83e7f00f5cd839e75c7"),
]


def download_certphash(out: Path) -> bool:
    import gdown

    ok = True
    for file_id, name, digest in CERTPHASH_FILES:
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"downloading {name} ...")
            gdown.download(id=file_id, output=str(dest), quiet=True)
        if sha256(dest) != digest:
            print(f"  CHECKSUM MISMATCH for {name}; delete it and retry")
            ok = False
        else:
            print(f"  {name}: ok ({dest.stat().st_size / 1e6:.1f} MB)")
    return ok


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "dinohash")
    ap.add_argument("--small", action="store_true", help="only the ViT-S / 96-bit files")
    ap.add_argument("--certphash", action="store_true", help="the CertPHash checkpoints instead (needs gdown)")
    args = ap.parse_args(argv)
    if args.certphash:
        return 0 if download_certphash(args.out.parent / "certphash") else 1
    args.out.mkdir(parents=True, exist_ok=True)

    ok = True
    for url, name, digest, small in FILES:
        if args.small and not small:
            continue
        dest = args.out / name
        if not dest.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        if digest is not None and sha256(dest) != digest:
            print(f"  CHECKSUM MISMATCH for {name}; delete it and retry")
            ok = False
        else:
            print(f"  {name}: ok ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
