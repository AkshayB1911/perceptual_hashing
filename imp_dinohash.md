# DinoHash: implementation notes

*BTP: Perceptual Hashing. Written 2026-10-23.*

This document describes how DinoHash was implemented in `hashbench/`, where every
piece came from, and how the implementation was verified against the official release.

Code: `hashbench/hashbench/hashes/dinohash.py`
Tests: `hashbench/tests/test_dinohash.py`
Weights: `hashbench/scripts/download_models.py` → `hashbench/models/dinohash/`

---

## 1. Sources

### 1.1 Paper
- **DINOHash: Learning Adversarially Robust Perceptual Hashes from Self-Supervised
  Features.** Shree Singhi, Aayush Gupta, Lukas Struppek. ICML 2025 Workshop CODEML.
  arXiv 2503.11195. Local copy: `DinoHash.pdf`.
- What the paper specifies:
  - DINOv2 ViT-B/14 with register tokens as the feature extractor.
  - PCA "whitening" from 768 to 96 dimensions, then binarisation by sign.
  - Adversarial fine-tuning against an APGD attack. The fine-tuned "student" network
    is trained to reproduce the original network's hash logits on perturbed images,
    using cross-entropy on the logits plus a clean-image consistency term
    (Eq. 2–3; 20k steps, batch size 256, AdamW, 10-step APGD).
  - Matching: count the equal bits; FPR from a Binomial(k, 0.5) model.
  - Results: 72.2% bit accuracy under a 100-step APGD attack at ε = 4/255, vs 0.3%
    for the untrained variant and 0.1% for NeuralHash.

### 1.2 Official code (read only, not executed)
- Repository: <https://github.com/proteus-photos/dinohash-perceptual-hash>, commit
  `16733e9a6a9e6b3c557c3eba5b8ac19cb12c7847` (2025-10-31).
- Files studied:

| File | What we took from it |
|---|---|
| `hashes/dinohash.py` | The hashing pipeline: preprocessing, `normalize`, `outs -= means; outs @ components; [:, :pca_dims]`, bit = `outs >= 0` |
| `model_to_onnx.py` | How the released ONNX models were built: `nn.Sequential(dinov2, linear)`; the linear layer holds the PCA (`weight = components`, `bias = -means @ components`); a monkey patch that sets `antialias=False` in the positional-embedding interpolation |
| `node_package/src/inference.js` | Confirms the deployed preprocessing (224×224, ImageNet normalisation) and binarisation (`x >= 0`) |
| `main.py`, `transformer.py`, `apgd_attack.py`, `apgd_train.py` | The evaluation protocol, the transformations and the attack/training code (reference for the attack step) |

- Only the source files were downloaded, into a scratch folder. The repository is
  about 1 GB with checkpoints, and none of its code was run.

### 1.3 Released models (Hugging Face)
<https://huggingface.co/proteus-photos/DINOHash> (MIT licence). Files used:

| File | Model | Bits | SHA-256 |
|---|---|---|---|
| `dinov2_vits14_reg_96bit_dynamic.onnx` | DINOv2 ViT-S/14 + registers | 96 | `a36e158a27b655cd1f8888fa2d3c736f2ba0386dc6000f6ca762c1f3c72f332e` |
| `dinov2_vitb14_reg_512bit_dynamic.onnx` | DINOv2 ViT-B/14 + registers | 512 | `cc399d9583877379ea89675f322bc220dc997b66c1d66766a7c12fc02fc562bd` |

Checksums were verified against the hashes Hugging Face publishes for these files.
The download script pins them.

### 1.4 Other files
- **PCA files** from the official repo (same commit), used for the *untrained*
  baseline:
  - `hashes/dinov2_vits14_reg_PCA.npy` (384×384) and `..._means.npy` (1×384)
  - `hashes/dinov2_vitb14_reg_PCA.npy` (768×768) and `..._means.npy` (1×768)
- **Original DINOv2 weights** from Meta (also for the untrained baseline):
  - `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth`
  - `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`
- **DINOv2 architecture:** <https://github.com/facebookresearch/dinov2>, commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Read only, to reproduce the forward
  pass. Files: `hub/backbones.py`, `models/vision_transformer.py`,
  `layers/attention.py`, `layers/block.py`, `layers/layer_scale.py`, `layers/mlp.py`,
  `layers/patch_embed.py`.

---

## 2. The DinoHash pipeline

For an input image *x*:

```
x (any size, PIL)
 └─ RGB, resize to 224×224 (bilinear, aspect ratio NOT kept), scale to [0, 1]
 └─ normalise: (x − mean) / std,  mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
 └─ DINOv2 ViT/14 with 4 register tokens  →  layer-normed CLS token  z ∈ R^D
       D = 384 (ViT-S/14) or 768 (ViT-B/14)
 └─ PCA head:  logits = W z + b,   W = first k principal components (k × D),
                                   b = −μ Wᵀ  (μ = mean feature vector)
 └─ hash bit i = 1  if logits_i ≥ 0
```

Two hashes are compared by Hamming distance. The paper's similarity score is the
number of matching bits, *M = k − Hamming distance*.

---

## 3. Implementation

### 3.1 Why re-implement DINOv2 instead of calling the official code?
The official code builds the backbone with
`torch.hub.load('facebookresearch/dinov2', ...)`, which downloads and runs code from
GitHub at run time. We wrote the forward pass ourselves (class `DinoV2`, about 80
lines) because:

1. **No remote code at run time.** Results don't depend on whatever `main` of the
   DINOv2 repo is when the experiment runs.
2. **Full control for attacks.** The model is a plain `nn.Module` mapping [0, 1]
   pixels to logits, which the white-box attacks (APGD, Struppek et al.) need.
3. **Verifiability.** Each part can be checked against the official ONNX export
   (Section 4).

Parameter names match Meta's state dict exactly (`cls_token`, `pos_embed`,
`register_tokens`, `patch_embed.proj.*`, `blocks.{i}.norm1/attn.qkv/attn.proj/ls1.gamma/norm2/mlp.fc1/mlp.fc2/ls2.gamma`, `norm.*`),
so official weights load without any renaming.

### 3.2 Architecture details reproduced

| Component | Detail (from `dinov2` source) | ViT-S/14 | ViT-B/14 |
|---|---|---|---|
| Patch embedding | `Conv2d(3, D, kernel=14, stride=14)`, flatten → (B, N, D) | D=384 | D=768 |
| Tokens | `[CLS] + patches`, add positional embedding, then insert 4 register tokens after CLS | 1+4+256 tokens at 224 px | same |
| Positional embedding | Trained at 518 px (37×37 grid); bicubic-resized to 16×16 for 224 px input | | |
| Transformer blocks | 12 × pre-norm: `x + ls1(attn(norm1(x)))`, `x + ls2(mlp(norm2(x)))` | 6 heads | 12 heads |
| Attention | `qkv` Linear with bias, scaled dot-product attention, `proj` Linear | head dim 64 | head dim 64 |
| MLP | `fc1` (D→4D), exact (erf) GELU, `fc2` (4D→D) | | |
| LayerScale | `x * gamma` (`init_values=1.0`) | | |
| LayerNorm | `eps = 1e-6` | | |
| Output | `norm(x)[:, 0]` (the `x_norm_clstoken` of DINOv2) | | |
| `mask_token` | Unused at inference; kept so Meta's state dict loads | | |

`DinoHashModel` wraps the backbone with the ImageNet normalisation (as registered
buffers) and the PCA head `nn.Linear(D, k)`. Its input is a [0, 1] image tensor of
shape (B, 3, 224, 224), and its output is the per-bit logits. Attacks can therefore
work directly in pixel space with ε budgets such as 4/255.

### 3.3 Positional-embedding interpolation (a subtle detail)
DINOv2-with-registers (`dinov2_vits14_reg`, `dinov2_vitb14_reg` in `hub/backbones.py`)
is built with `interpolate_antialias=True, interpolate_offset=0.0`. The official
DinoHash ONNX export (`model_to_onnx.py`) monkey-patches this to **`antialias=False`**
so that the model can be exported to ONNX.

- Our `DinoV2` has a flag `pos_antialias`. The default is `False`, which reproduces
  the **released** ONNX models exactly.
- `pos_antialias=True` gives the behaviour of the official repo's PyTorch path.
- Measured effect on 29 test images:

| Variant | max logit difference | bits changed per image (max / mean) |
|---|---|---|
| ViT-S/96 | 0.057 | 1 / 0.31 |
| ViT-B/512 | 0.040 | 10 / 4.41 |

So the released ONNX and the repo's PyTorch code do not give identical hashes. This
should be mentioned whenever DinoHash numbers from different sources are compared.

### 3.4 Preprocessing
`preprocess(image)` = `image.convert("RGB").resize((224, 224), BILINEAR)` / 255.

The official code uses torchvision's `Resize((224, 224))` + `ToTensor()` on a PIL
image, which calls PIL's bilinear resize internally. A test asserts the two produce
**identical tensors**. `convert("RGB")` was added so that grayscale and RGBA images
work; the official code would fail on 4-channel input.

### 3.5 Loading weights: two variants

**Trained (the released, adversarially fine-tuned DinoHash)**, `state_dict_from_onnx()`.
- The ONNX file stores the weights of `nn.Sequential(dinov2, linear)` as initializers
  named `0.<dinov2 param>` and `1.weight` / `1.bias`.
- We read them with the `onnx` package, map `0.*` → `backbone.*` and `1.*` → `head.*`,
  and reshape the bias from (1, k) to (k,).
- The ONNX export has no `mask_token`. That is the only missing parameter, and it is
  unused.

**Untrained (the paper's ablation)**, `state_dict_untrained()`.
- Meta's original DINOv2 weights, plus a head built from the official PCA files:
  `W = PCA[:k]` (rows are components) and `b = −means · Wᵀ`.
- This is what the official `DINOHash` class computes when no fine-tuned checkpoint is
  loaded.

### 3.6 Registered hashers
In `hashbench/hashes/__init__.py`. These are lazily imported, so pHash and PDQ still
work without torch.

| Name | Backbone | Bits | Weights |
|---|---|---|---|
| `dinohash` | ViT-S/14 + reg | 96 | released, adversarially fine-tuned |
| `dinohash_vitb14_512` | ViT-B/14 + reg | 512 | released, adversarially fine-tuned |
| `dinohash_untrained` | ViT-S/14 + reg | 96 | original DINOv2 + official PCA |
| `dinohash_vitb14_512_untrained` | ViT-B/14 + reg | 512 | original DINOv2 + official PCA |

The 96-bit ViT-S variant is the default because 96 bits makes it directly comparable
to NeuralHash.

The `DinoHash` class implements the common `PerceptualHash` interface:
- `hash(image)` → bool array of shape (k,);
- `hash_batch(images)` → batched inference (32 images per forward pass);
- `logits(images)` → raw logits, useful to see how close each bit is to flipping;
- `.model` → the differentiable `nn.Module`.

### 3.7 Benchmark integration
Changes made to `hashbench/benchmark.py` for neural hashes:
- **One model load per process.** Matchers are cached in `_MATCHER_CACHE` instead of
  being rebuilt for every image.
- **Batching.** All attacked copies of an image (about 80) are hashed in one batched
  call via `hash_batch`.
- **CPU split.** Each worker process gets `cpu_count / workers` torch threads, so
  several workers don't oversubscribe the CPU.
- `PerceptualHash.hash_batch` was added to the base class (default: loop over
  `hash`), so classical and neural hashes share the same code path.

---

## 4. Verification

### 4.1 Are the released models really fine-tuned?
The ONNX initializers were compared tensor-by-tensor with Meta's original DINOv2
weights:

| Model | Backbone tensors | Identical to DINOv2 | Changed | Largest change |
|---|---|---|---|---|
| ViT-S/96 | 175 | 0 | 175 | `blocks.2.attn.qkv.bias` (0.28) |
| ViT-B/512 | 175 | 0 | 175 | `blocks.1.attn.qkv.bias` (0.66) |

Both released models are fine-tuned. The PCA heads were checked as well:

- **ViT-B/512:** the head equals the repo's PCA file (max difference 7e-9 for the
  weights, 3e-7 for the bias).
- **ViT-S/96:** the head is orthonormal (WWᵀ = I to 4e-7) but **does not match** the
  repo's `dinov2_vits14_reg_PCA.npy`. The best |cosine| with any `.npy` component has
  a median of 0.20, so it is a different projection, probably PCA refit after
  fine-tuning. We therefore always take the trained head from the ONNX file, never
  from the `.npy`.

### 4.2 Our PyTorch model vs. the official ONNX model
29 images (12 samples from `data/samples` and 17 from `tests/data/meta_pdq`), official
ONNX run with `onnxruntime` vs our model with the same weights:

| Variant | max \|logit difference\| | bits differing (all images) |
|---|---|---|
| ViT-S/96 | 4.2e-6 | **0** |
| ViT-B/512 | 5.7e-6 | **0** |

The remaining 1e-6 differences are float32 rounding between the PyTorch and
onnxruntime kernels. For context, the smallest |logit| per image has a median of
0.0045 (ViT-S) and 0.0003 (ViT-B). Some bits are very close to the decision boundary,
which is exactly why exact numerical agreement matters and why tiny perturbations can
flip bits.

### 4.3 Automated tests (`tests/test_dinohash.py`)
1. Preprocessing equals the official torchvision `Resize((224,224)) + ToTensor()`
   exactly.
2. Both variants match the official ONNX: bit-identical, logits within 1e-4.
3. The untrained model differs from the trained one (> 20% of bits).
4. The hasher interface: 96 bool bits, `hash_batch` consistent with `hash`, a JPEG
   re-encode at quality 70 changes ≤ 10 bits.
5. Gradients flow from the logits back to the input pixels, as white-box attacks need.

The tests skip automatically if the model files have not been downloaded.

---

## 5. First results (smoke test, 12 images; not thesis numbers)

Match rate under the threshold calibrated at FPR = 5%:

| Attack | DinoHash | DinoHash (untrained) | PDQ | PDQ + dihedral | pHash |
|---|---|---|---|---|---|
| center crop | 1.00 | 1.00 | 0.45 | 0.45 | 0.63 |
| corner crop | 1.00 | 1.00 | 0.31 | 0.29 | 0.46 |
| translate | 1.00 | 1.00 | 0.31 | 0.29 | 0.44 |
| rotate | 0.97 | 0.99 | 0.56 | 0.58 | 0.63 |
| border | 1.00 | 1.00 | 0.56 | 0.56 | 0.77 |
| horizontal flip | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| vertical flip | 0.83 | 0.83 | 0.00 | 1.00 | 0.00 |
| rescale (to 10%) | 0.94 | 0.96 | 1.00 | 1.00 | 1.00 |
| gaussian blur | 0.97 | 0.95 | 1.00 | 1.00 | 1.00 |

Overall AUC: DinoHash 0.999, untrained 0.998, pHash 0.938, PDQ 0.913.

Observations:
- DinoHash is far more robust than the classical hashes to geometric changes.
- It is slightly weaker under extreme downscaling and heavy blur.
- *Correction (2026-11-03): the next point holds only at loose thresholds such as this FPR of 5%. On
  the full COCO benchmark at FPR 1e-3, untrained matches 94.3% and trained 89.7%; see F16 and F17 in
  `benchmark_findings.md`.*
- **Trained and untrained DinoHash behave the same under benign transformations.** The
  adversarial training targets *adversarial* perturbations. Its benefit has to be shown
  with gradient-based attacks, the next step in `learning_models_plans.md`.
- 12 images give only 66 impostor pairs, so thresholds are unreliable. Real results
  need 1,000+ images.

Performance on the local CPU (6 threads, no GPU): about 63 ms/image for ViT-S/96 and
about 210 ms/image for ViT-B/512.

---

## 6. Limitations and differences from the official setup

- **Benchmark data.** The paper evaluates on 2M DiffusionDB images with its own
  transformation pipeline ("screenshot" crop + JPEG, then erase/text). Our benchmark
  uses a parameterised attack suite on a dataset still to be chosen, so numbers are not
  directly comparable with the paper's figures.
- **Which model the paper's numbers belong to.** The paper describes a 96-bit ViT-B/14
  model. The released models are ViT-S/14 at 96 bits and ViT-B/14 at 512 bits. Which
  release corresponds exactly to the paper's tables is not stated.
- **Antialias difference.** The released ONNX and the repo's PyTorch code differ
  slightly (Section 3.3); we follow the released ONNX.
- **No training reproduced.** Adversarial fine-tuning needs a GPU (20k steps × batch
  256 on a ViT). We use the released weights.

---

## 7. How to use

```powershell
cd D:\AKSHAY\sem7\btp\perceptual_hashing\hashbench
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[neural]"
python scripts/download_models.py          # --small for ViT-S only
pytest tests/test_dinohash.py
```

```python
from PIL import Image
from hashbench import get_hasher

dh = get_hasher("dinohash")                  # or dinohash_untrained, dinohash_vitb14_512, ...
a = dh.hash(Image.open("a.jpg"))             # 96 bools
b = dh.hash(Image.open("b.jpg"))
print(dh.distance(a, b))                     # Hamming distance, 0..96

# differentiable model for attacks: [0,1] images (B,3,224,224) -> logits (B,96)
from hashbench.hashes.dinohash import load_dinohash_model, preprocess
model = load_dinohash_model("vits14_96", trained=True)
logits = model(preprocess(Image.open("a.jpg"))[None])
```

Benchmark:
```powershell
python -m hashbench.benchmark --images <IMAGE_DIR> --out results/run1 --algorithms phash pdq dinohash dinohash_untrained --workers 3
python -m hashbench.report --run results/run1
```
