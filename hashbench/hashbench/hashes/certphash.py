"""CertPHash: perceptual hash with certified robustness against evasion and collision.

Yang, Wang, Wang, Jha, Zhang. "CertPHash: Towards Certified Perceptual Hashing via Robust
Training", USENIX Security 2025. Official code: github.com/Yuchen413/CertPhash
(our copy: ../HASH_IMPLEMENTATIONS/CertPhash). Released models: Google Drive ``saved_models``,
fetched by ``scripts/download_models.py --certphash``.

A small ResNet (2.6M parameters) is trained to imitate PhotoDNA ("C-PDNA") on COCO at
64x64, with certified training (CROWN-IBP via auto_LiRPA) against l_inf perturbations of
size ``eps_target``. Pipeline (``train_verify/datasets.py`` ``ImageToHashAugmented``):
    1. RGB, scale to [0, 1].
    2. Resize to 64x64 (torchvision tensor resize: bilinear, antialiased), aspect ratio
       NOT kept.
    3. ImageNet normalisation.
    4. ResNet -> 144 values, regressed onto the 144 bytes of the PhotoDNA hash.
    5. hash = round(output) clipped to [0, 255] (the authors use relu(round(.))).
Two hashes match when their L1 distance is <= 1800 (the paper's PhotoDNA threshold).

Variants (all COCO / PhotoDNA, same architecture), the paper's three training regimes:
    certphash, certphash_ep{2,4,8}   certified training, eps_target = k/255 (ep1 is the
                                     paper's default, Sec. 6.1)
    certphash_base                   normal training (BASEPHASH, eps = 0)
    certphash_adv                    adversarial training (ADVPHASH, eps = 8/255)
The released MNIST / PDQ models use a different network on 28x28 digits and are not
wrapped here.

``load_certphash_model`` returns a differentiable ``nn.Module`` on [0, 1] images, like the
other neural hashes. Note: the authors apply perturbation budgets eps *after* ImageNet
normalisation, i.e. eps / std per channel in pixel space (about 4.4x larger than in our
[0, 1] convention).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hashbench.hashes.base import PerceptualHash

INPUT_SIZE = 64
N_VALUES = 144              # PhotoDNA hash: 144 bytes
MATCH_THRESHOLD = 1800      # L1, paper Sec. 5.2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_DIR = Path(os.environ.get(
    "HASHBENCH_MODELS", Path(__file__).resolve().parents[2] / "models")) / "certphash"

# variant -> checkpoint (relative to MODEL_DIR)
VARIANTS = {
    "ep1": "coco_photodna_ep1/ckpt_best.pth",
    "ep2": "coco_photodna_ep2/ckpt_best.pth",
    "ep4": "coco_photodna_ep4/ckpt_best.pth",
    "ep8": "coco_photodna_ep8/ckpt_best.pth",
    "base": "coco_photodna_ep0/ckpt_best.pth",
    "adv": "base_adv/coco-pdq-ep8-photodna.pt",
}


# --- network: resnet_v5 (ResNet9_v1_64 + BasicBlock2) of train_verify/models/resnet_v5.py --

class _Block(nn.Module):
    """conv-bn-relu-conv-bn + shortcut, *without* a ReLU after the addition (BasicBlock2)."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x)


class CertPHashNet(nn.Module):
    """[0, 1] RGB images (B, 3, 64, 64) -> 144 real-valued PhotoDNA-byte predictions.

    Parameter names match the authors' ``resnet_v5`` so their checkpoints load as-is."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)            # 64 -> 31
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = nn.Sequential(_Block(32, 64, 2), _Block(64, 64, 1), _Block(64, 64, 1))  # -> 16
        self.layer2 = nn.Sequential(_Block(64, 64, 2), _Block(64, 64, 1), _Block(64, 64, 1))  # -> 8
        self.linear1 = nn.Linear(64 * 8 * 8, 512)
        self.linear2 = nn.Linear(512, N_VALUES)  # no ReLU between the two linear layers
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1((x01 - self.mean) / self.std)))
        x = self.layer2(self.layer1(x))
        return self.linear2(self.linear1(torch.flatten(x, 1)))


def load_certphash_model(variant: str = "ep1", model_dir: Path = MODEL_DIR) -> CertPHashNet:
    path = Path(model_dir) / VARIANTS[variant]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/download_models.py --certphash")
    sd = torch.load(path, map_location="cpu", weights_only=True)  # plain state dict
    model = CertPHashNet()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if [k for k in missing if k not in ("mean", "std")] or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()


# --- PerceptualHash interface ---------------------------------------------------------

def preprocess(image: Image.Image) -> torch.Tensor:
    """Authors' pipeline up to normalisation (done inside the model): RGB uint8 -> [0, 1]
    float -> torchvision resize to 64x64 (bilinear, antialias=True), as in their datasets.py."""
    from torchvision.transforms import functional as TF

    x = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
    x = x.float() / 255.0
    return TF.resize(x, [INPUT_SIZE, INPUT_SIZE], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)


def outputs_to_hash(y: np.ndarray) -> np.ndarray:
    """Real-valued outputs -> uint8 hash values (authors: relu(round(y)); we also clip at 255)."""
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


class CertPHash(PerceptualHash):
    n_bits = N_VALUES  # 144 byte values, not bits

    def __init__(self, variant: str = "ep1", batch_size: int = 128):
        self.variant, self.batch_size = variant, batch_size
        self.name = "certphash" + ("" if variant == "ep1" else f"_{variant}")
        self.model = load_certphash_model(variant)

    @property
    def max_distance(self) -> int:
        return 255 * N_VALUES

    def distance(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(self.distances(np.asarray(a), np.asarray(b)))

    def distances(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """L1 distance between hashes along the last axis."""
        return np.abs(a.astype(np.int64) - b.astype(np.int64)).sum(axis=-1)

    def to_string(self, h: np.ndarray) -> str:
        return np.asarray(h, dtype=np.uint8).tobytes().hex()

    def from_string(self, s: str) -> np.ndarray:
        return np.frombuffer(bytes.fromhex(s), dtype=np.uint8).copy()

    @torch.no_grad()
    def outputs(self, images: list[Image.Image]) -> np.ndarray:
        """Real-valued network outputs, shape (len(images), 144)."""
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([preprocess(im) for im in images[i:i + self.batch_size]])
            out.append(self.model(batch).numpy())
        return np.concatenate(out)

    def hash_batch(self, images: list[Image.Image]) -> np.ndarray:
        return outputs_to_hash(self.outputs(images))

    def hash(self, image: Image.Image) -> np.ndarray:
        return self.hash_batch([image])[0]
