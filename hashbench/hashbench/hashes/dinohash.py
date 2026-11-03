"""DinoHash: adversarially robust perceptual hash on self-supervised DINOv2 features.

Singhi, Gupta, Struppek. "DINOHash: Learning Adversarially Robust Perceptual Hashes
from Self-Supervised Features", ICML 2025 CODEML workshop (arXiv 2503.11195).
Official code: github.com/proteus-photos/dinohash-perceptual-hash
Released models: huggingface.co/proteus-photos/DINOHash

Pipeline (``hashes/dinohash.py`` in the official repo):
    1. Resize to 224x224 (bilinear, aspect ratio NOT kept), scale to [0, 1].
    2. ImageNet normalisation.
    3. DINOv2 ViT with 4 register tokens -> layer-normed CLS token (384-d for ViT-S/14,
       768-d for ViT-B/14).
    4. Linear head = PCA: subtract the feature mean, project on the first ``bits``
       principal components ("logits").
    5. bit = logit >= 0.

Variants:
    trained   the released, adversarially fine-tuned model. Fine-tuning (APGD, eps=4/255
              in the paper) changes every backbone weight, and the released ViT-S/96-bit
              model also ships its own PCA head. Weights are read from the official ONNX
              file, so this is exactly the released model.
    untrained the original DINOv2 weights from Meta + the PCA files from the official
              repo, i.e. what the official code computes without a fine-tuned checkpoint
              ("DINOHash (untrained)" in the paper's Table 1). The ablation baseline.

The DINOv2 forward pass is re-implemented below (no torch.hub code) with parameter names
identical to Meta's, so official state dicts load as-is. It is differentiable, which the
white-box attacks need. One detail: DINOv2-with-registers resizes its 37x37 positional
embeddings to the 16x16 patch grid with antialiased bicubic interpolation, but the
official ONNX export patches this to ``antialias=False``. We follow the released ONNX
(``pos_antialias=False``) by default; see ``tests/test_dinohash.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hashbench.hashes.base import PerceptualHash

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_DIR = Path(os.environ.get(
    "HASHBENCH_MODELS", Path(__file__).resolve().parents[2] / "models")) / "dinohash"


@dataclass(frozen=True)
class DinoConfig:
    arch: str            # "vits14" / "vitb14"
    embed_dim: int
    num_heads: int
    depth: int = 12
    patch_size: int = 14
    num_register_tokens: int = 4
    pos_grid: int = 37   # DINOv2 was trained at 518 px = 37 x 14


VITS14 = DinoConfig("vits14", embed_dim=384, num_heads=6)
VITB14 = DinoConfig("vitb14", embed_dim=768, num_heads=12)


# --- DINOv2 vision transformer (forward pass of facebookresearch/dinov2) -----------

class _PatchEmbed(nn.Module):
    def __init__(self, cfg: DinoConfig):
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).unbind(2)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # exact (erf) GELU, as nn.GELU()


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class _Block(nn.Module):
    def __init__(self, cfg: DinoConfig):
        super().__init__()
        d = cfg.embed_dim
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.attn = _Attention(d, cfg.num_heads)
        self.ls1 = _LayerScale(d)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = _Mlp(d)
        self.ls2 = _LayerScale(d)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoV2(nn.Module):
    """DINOv2 ViT with registers; returns the layer-normed CLS token."""

    def __init__(self, cfg: DinoConfig, pos_antialias: bool = False):
        super().__init__()
        self.cfg = cfg
        self.pos_antialias = pos_antialias
        d = cfg.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cfg.pos_grid ** 2, d))
        self.register_tokens = nn.Parameter(torch.zeros(1, cfg.num_register_tokens, d))
        self.mask_token = nn.Parameter(torch.zeros(1, d))  # unused at inference; kept so state dicts load
        self.patch_embed = _PatchEmbed(cfg)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d, eps=1e-6)

    def interpolate_pos_encoding(self, rows: int, cols: int) -> torch.Tensor:
        """Resize the 37x37 positional embeddings to the (rows/14, cols/14) patch grid."""
        m, p = self.cfg.pos_grid, self.cfg.patch_size
        grid = (rows // p, cols // p)
        if grid == (m, m):
            return self.pos_embed
        pos = self.pos_embed.float()
        patch_pos = pos[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=grid, mode="bicubic", antialias=self.pos_antialias)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
        return torch.cat((pos[:, :1], patch_pos), dim=1).to(self.pos_embed.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, rows, cols = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(rows, cols)
        x = torch.cat((x[:, :1], self.register_tokens.expand(b, -1, -1), x[:, 1:]), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class DinoHashModel(nn.Module):
    """Differentiable DinoHash: [0, 1] RGB images (B, 3, 224, 224) -> per-bit logits."""

    def __init__(self, cfg: DinoConfig, bits: int, pos_antialias: bool = False):
        super().__init__()
        self.backbone = DinoV2(cfg, pos_antialias=pos_antialias)
        self.head = nn.Linear(cfg.embed_dim, bits)  # PCA: W = components, b = -mean @ W.T
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x01 - self.mean) / self.std))


# --- loading ------------------------------------------------------------------------

def state_dict_from_onnx(path: Path) -> dict[str, torch.Tensor]:
    """Weights of an official DinoHash ONNX export (nn.Sequential(dinov2, linear))."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    sd = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).copy()
        if init.name.startswith("0."):
            sd["backbone." + init.name[2:]] = torch.from_numpy(arr)
        elif init.name.startswith("1."):
            sd["head." + init.name[2:]] = torch.from_numpy(arr.reshape(-1) if init.name == "1.bias" else arr)
    return sd


def state_dict_untrained(cfg: DinoConfig, bits: int, model_dir: Path) -> dict[str, torch.Tensor]:
    """Original DINOv2 weights + the official repo's PCA head (the paper's untrained baseline)."""
    base = torch.load(model_dir / f"dinov2_{cfg.arch}_reg4_pretrain.pth", map_location="cpu", weights_only=True)
    pca = np.load(model_dir / f"dinov2_{cfg.arch}_reg_PCA.npy")      # (n_components, dim), rows = components
    means = np.load(model_dir / f"dinov2_{cfg.arch}_reg_means.npy")  # (1, dim)
    w = pca[:bits].astype(np.float32)
    sd = {"backbone." + k: v for k, v in base.items()}
    sd["head.weight"] = torch.from_numpy(w)
    sd["head.bias"] = torch.from_numpy((-means @ w.T).reshape(-1).astype(np.float32))
    return sd


# (config, bits, onnx file of the released fine-tuned model)
VARIANTS = {
    "vits14_96": (VITS14, 96, "dinov2_vits14_reg_96bit_dynamic.onnx"),
    "vitb14_512": (VITB14, 512, "dinov2_vitb14_reg_512bit_dynamic.onnx"),
}


def load_dinohash_model(variant: str = "vits14_96", trained: bool = True, pos_antialias: bool = False,
                        model_dir: Path = MODEL_DIR) -> DinoHashModel:
    cfg, bits, onnx_file = VARIANTS[variant]
    path = model_dir / onnx_file
    if trained and not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/download_models.py")
    sd = state_dict_from_onnx(path) if trained else state_dict_untrained(cfg, bits, model_dir)
    model = DinoHashModel(cfg, bits, pos_antialias=pos_antialias)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The ONNX export drops the unused mask token; nothing else may be missing.
    if [k for k in missing if k not in ("backbone.mask_token", "mean", "std")] or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval()


# --- PerceptualHash interface -----------------------------------------------------------

def preprocess(image: Image.Image) -> torch.Tensor:
    """Official preprocessing: torchvision Resize((224, 224)) on a PIL image (= PIL bilinear) + ToTensor."""
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class DinoHash(PerceptualHash):
    """DinoHash. ``trained=False`` gives the untrained (plain DINOv2 + PCA) baseline."""

    def __init__(self, variant: str = "vits14_96", trained: bool = True, pos_antialias: bool = False,
                 batch_size: int = 32):
        cfg, bits, _ = VARIANTS[variant]
        self.variant, self.trained, self.batch_size = variant, trained, batch_size
        self.n_bits = bits
        self.name = "dinohash" + ("" if variant == "vits14_96" else f"_{variant}") + ("" if trained else "_untrained")
        self.model = load_dinohash_model(variant, trained, pos_antialias)

    @torch.no_grad()
    def logits(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([preprocess(im) for im in images[i:i + self.batch_size]])
            out.append(self.model(batch).numpy())
        return np.concatenate(out)

    def hash_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self.logits(images) >= 0

    def hash(self, image: Image.Image) -> np.ndarray:
        return self.hash_batch([image])[0]
