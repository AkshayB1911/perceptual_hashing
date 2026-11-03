# hashbench

Implementations of perceptual image hashes and a benchmark of their robustness to
image transformations. Part of the BTP on perceptual hashing.

| Algorithm | Bits | Module | Validated against |
|---|---|---|---|
| pHash | 64 | `hashbench/hashes/phash.py` | `imagehash.phash`: bit-identical |
| PDQ | 256 | `hashbench/hashes/pdq.py` | Meta's C++ core (via `pdqhash`) and Meta's published test hashes; see below |
| PDQ + dihedral matching | 256 | `PDQHash.hash_dihedral` | as above, all 8 variants |
| DinoHash (`dinohash`) | 96 | `hashbench/hashes/dinohash.py` | Released ONNX model: bit-identical, logits within 6e-6 |
| DinoHash ViT-B (`dinohash_vitb14_512`) | 512 | same | as above |
| untrained DinoHash (`dinohash_untrained`, `dinohash_vitb14_512_untrained`) | 96 / 512 | same | original DINOv2 + official PCA (the paper's ablation) |
| NeuralHash (`neuralhash`) | 96 | `hashbench/hashes/neuralhash.py` | Reads Apple's Espresso files directly. Weights identical to the AppleNeuralHash2ONNX conversion; hashes match `nnhash.py` + onnxruntime (see below) |
| CertPHash (`certphash` = certified, ε=1/255; `certphash_ep2/ep4/ep8`) | 144 bytes (L1) | `hashbench/hashes/certphash.py` | Authors' `resnet_v5` + preprocessing: outputs bit-identical (difference exactly 0) for all 6 released models |
| CertPHash baselines (`certphash_base` = normal training, `certphash_adv` = adversarial training) | 144 bytes (L1) | same | as above |

### How faithful is the PDQ implementation?

`pdq.py` is a port of Meta's C++ reference (`ThreatExchange/pdq/cpp`), operation by
operation, including its float32 arithmetic and summation order.

* **Hashing core vs Meta's C++** (`tests/test_pdq_reference.py`): bit-identical hash and
  quality on natural images of random sizes. It is also bit-identical on degenerate
  images (flat, gradients, noise, checkerboards, sizes 5-65 px), where the bits are
  decided purely by float rounding. Images smaller than 5 px give the all-zero hash,
  as in the reference.
* **Meta's Python reference test vectors** (`tests/data/meta_pdq`): 17 of 18 hashes
  identical. The exception, `q0003.jpg`, has PDQ quality 3 (nearly flat). Its
  coefficients are within 0.005 of the median, so JPEG-decoder version differences
  flip 10 bits. Meta recommends discarding hashes with quality < 50.
* **Meta's C++ tool test vectors**: 2-8 bits away. That tool decodes JPEGs with CImg
  and squashes images to 512x512. Meta's own Python and C++ references disagree with
  each other by the same amount (8 bits on `bridge-1-original.jpg`).

Image loading is not part of PDQ, and Meta's tools differ here. The C++ tool squashes
images larger than 512 px to 512x512; the Python reference shrinks them to fit
512x512 keeping the aspect ratio. `PDQHash()` hashes pixels as given, like the C++
core. `PDQHash(max_side=512)` reproduces the Python reference. One documented
difference from the `pdqhash` wrapper: it computes luma in float64, while we follow
Meta's C++ (float32). This only changes bits on degenerate images.

Planned: SSCD (add a `PerceptualHash` subclass in
`hashbench/hashes/` and register it in `hashbench/hashes/__init__.py`). See
`../learning_models_plans.md`.

### DinoHash notes

* DINOv2 is re-implemented in plain PyTorch (`DinoV2`, about 80 lines, no `torch.hub`) with
  Meta's parameter names. Weights load directly from the official files, and the model is
  differentiable for white-box attacks (`load_dinohash_model(...)` returns an `nn.Module`
  mapping [0, 1] images to per-bit logits).
* **Both released models are adversarially fine-tuned.** All 175 backbone tensors differ
  from the original DINOv2. The ViT-B/512 head equals the PCA file in the official repo.
  The ViT-S/96 model ships a different (orthonormal) PCA head, so we read the head from
  the ONNX file rather than the repo's `.npy`.
* The released ONNX resizes DINOv2's positional embeddings with `antialias=False`, while
  DINOv2-with-registers in PyTorch uses `antialias=True`. That changes up to 1/96 (ViT-S)
  or 10/512 (ViT-B) bits. We follow the released ONNX; `pos_antialias=True` gives the
  PyTorch behaviour.
* Preprocessing: resize to 224x224 (bilinear, aspect ratio not kept) + ImageNet
  normalisation. Identical to the official torchvision transform (tested).
* CPU speed: about 63 ms/image (ViT-S) and 210 ms/image (ViT-B).

### NeuralHash notes

* Apple's model is **not distributed** (Apple's files; Struppek et al. do not share them
  either). Copy `NeuralHashv3b[_fp16]-current.espresso.{net,shape,weights}` and
  `neuralhash_128x96_seed1.dat` from the Vision framework of macOS 11.4+ or an iOS 14.7+
  firmware, **unchanged**, into `models/neuralhash/` (git-ignored). Do not redistribute them.
  We use the iOS build (`_fp16`: fp16 weights, upcast exactly to float32).
* **No third-party converter.** `neuralhash.py` reads Apple's Espresso format itself: the
  LZFSE-compressed `.net` (225 layers, decompressed with `pyliblzfse`) and the `.weights`
  blob table. Each of the 91 weighted layers is checked against our block table (channels,
  kernel, stride, groups, SAME/valid padding, fused ReLU, instance-norm flag) before its
  weights are copied. The network is MobileNetV3-like (1.8M parameters, instance norm,
  hard-swish, squeeze-excitation), giving 128-d features, then 96 hyperplanes.
  `model.onnx` from AppleNeuralHash2ONNX is accepted too (`source="onnx"`).
* **Validation** (`tests/test_neuralhash.py`, against the AppleNeuralHash2ONNX/TNN
  conversion of the same files run by `nnhash.py` in onnxruntime; that converter only reads
  fp32 weights, so `scripts/neuralhash_onnx_reference.py` first writes an fp32 copy of the iOS
  files):
  * all 148 weight tensors are bitwise identical to the converter's;
  * hashes are identical on 28 of 29 test images. The exception is one bit of
    `bridge-6-flipy.jpg`, whose logit is 8e-5 (logits are around ±500);
  * logits agree to about 5e-6 relative, except on low-contrast images (`small.jpg`: 2e-3).
    There, onnxruntime's instance norm loses digits when mean² ≫ variance (mean²/var ≈
    34,000 in the first layer). Compared with a float64 run, onnxruntime is off by 7e-3
    and we are off by 3e-5; ours is the more accurate.
* Preprocessing as `nnhash.py`: resize to 360x360 (PIL bicubic, aspect ratio not kept),
  scale to [-1, 1]. `load_neuralhash_model()` returns a differentiable `nn.Module` taking
  [0, 1] images, like DinoHash.
* Hashes computed on an Apple device can differ by a few bits (the device computes in fp16).

### CertPHash notes

* Yang et al., USENIX Security 2025. A 2.6M-parameter ResNet trained on COCO (64x64) to
  imitate **PhotoDNA**, i.e. the paper's C-PDNA. The output is 144 values in 0..255,
  compared by **L1**; the paper's match threshold is 1800 (normalized 1800/36720 = 0.049).
  This is the only hash in the project that isn't binary: it overrides `distances`,
  `max_distance` and the string encoding of `PerceptualHash`.
* Released models (`python scripts/download_models.py --certphash`, ~63 MB, checksums
  pinned): certified training at ε_target = 1/255 (`certphash`, the paper's default),
  2/255, 4/255 and 8/255, plus the paper's two baselines, normal training
  (`certphash_base`) and adversarial training (`certphash_adv`), all with the same
  architecture. The MNIST/PDQ models use a different network and are not wrapped.
  There is **no** certified COCO + PDQ model.
* **Fidelity to real PhotoDNA** (not reported in the paper), measured on the authors' COCO
  validation set (10,000 images at 100x100 with their real PhotoDNA hashes,
  `data/certphash/data/coco-val.csv`; test `test_imitates_real_photodna_on_coco_val`):

  | model | mean L1 to PhotoDNA | within 1800 | correct nearest neighbour among 10k |
  |---|---|---|---|
  | `certphash` (ε=1/255) | 933 | 96.9% | 99.9% |
  | `certphash_ep2` | 1266 | 89.8% | 98.2% |
  | `certphash_ep4` | 1681 | 73.6% | 97.9% |
  | `certphash_ep8` | 2551 | 15.9% | 97.7% |
  | `certphash_base` (normal) | 1367 | 93.1% | 99.4% |
  | `certphash_adv` (adversarial) | 439 | 99.6% | 100.0% |

  The larger the certified ε, the further the hash drifts from PhotoDNA: a
  robustness/fidelity trade-off. At ε=8/255 only 16% of images are within the PhotoDNA
  threshold of their own real hash. CPU speed: about 2.5 ms/image.
* Preprocessing as in the authors' `train_verify/datasets.py`: RGB, scale to [0, 1],
  torchvision resize to 64x64 (bilinear, antialiased, aspect ratio not kept), ImageNet
  normalization. hash = round(output) clipped to [0, 255]. Outputs can exceed 255; the
  authors use `relu(round(.))`, and their functionality CSVs then wrap values above 255
  through `astype(uint8)`.
* Things to be aware of in the authors' code (relevant when comparing with their numbers):
  * `attack/benign0_func_check.py` calls `get_dataset(args.dataset, args.target)`, which
    passes `target` into the `additional_transforms` slot. The functionality evaluation
    therefore runs **without** the ImageNet normalization the models were trained with.
  * `train_verify/verify_evasion.py` (CNER) certifies with `PerturbationL0Norm`, and on
    COCO compares PDQ-style bits (`post_pdq`, threshold 90) rather than PhotoDNA bytes
    (`post_photodna`, threshold 1800). It counts an evasion when
    `max(L1(lb), L1(ub)) >= threshold`, which is smaller than the worst-case L1 over the
    output box.
  * Perturbation budgets ε are applied after ImageNet normalization, i.e. ε/std per channel
    in pixel space (about 4.4x larger than ε on [0, 1] images).

## Setup

```powershell
cd hashbench
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[reference,dev]"
# learned hashes: CPU build of torch, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[neural]"
python scripts/download_models.py          # DinoHash weights (~850 MB; --small for ~180 MB)
python scripts/download_models.py --certphash   # CertPHash checkpoints (~63 MB)
# NeuralHash: copy Apple's 4 files into models/neuralhash/ (see NeuralHash notes)
```

## Usage

```powershell
# 1. Check our implementations against the reference libraries (all distances must be 0)
python scripts/validate_against_reference.py data/samples

# 2. Unit tests
pytest

# 3. Run the benchmark on a folder of images
python -m hashbench.benchmark --images <IMAGE_DIR> --out results/<run_name> [--limit 1000] \
    [--algorithms phash pdq pdq_dihedral dinohash dinohash_untrained neuralhash]

# 4. Summary tables and plots
python -m hashbench.report --run results/<run_name> --fpr 0.01
```

```python
from PIL import Image
from hashbench import get_hasher

pdq = get_hasher("pdq")
a = pdq.hash(Image.open("a.jpg"))
b = pdq.hash(Image.open("b.jpg"))
print(pdq.distance(a, b))          # Hamming distance, 0..256
```

## How the evaluation works

* **Genuine pairs**: each image vs each transformed copy (23 attack families,
  2-6 severities each; see `ATTACK_SUITE` in `hashbench/attacks.py`). These should match.
* **Impostor pairs**: originals of different images. These should not match.
* Distances are normalized Hamming distances (bits differing / hash length), so 64-bit and
  256-bit hashes are on the same scale.
* **Calibrated threshold**: for each algorithm, the largest threshold whose false match
  rate on impostor pairs is <= `--fpr`. Every algorithm gets the same false-positive
  budget, so the match rates under attack are directly comparable. The report also
  shows the commonly used thresholds (pHash 10/64, PDQ 31/256).

Outputs in the run folder: `genuine.csv`, `impostor.csv`, `hashes.csv`, `config.json`,
`overall.csv` (AUC, EER, thresholds), `summary.csv` (per attack and severity), and
`attack_curves.png`, `mean_distance_curves.png`, `distance_hist.png`, `roc.png`.

## Layout

```
hashbench/
  hashes/base.py      PerceptualHash interface, Hamming distance, hex conversion
  hashes/dct.py       DCT-II basis matrices (shared by pHash and PDQ)
  hashes/phash.py     pHash
  hashes/pdq.py       PDQ (Jarosz filter, decimation, quality metric, dihedral hashes)
  hashes/dinohash.py  DinoHash (DINOv2 in PyTorch)
  hashes/neuralhash.py NeuralHash (Espresso reader + MobileNetV3 in PyTorch)
  hashes/certphash.py CertPHash (ResNet, PhotoDNA-style 144-byte hash, L1)
  attacks.py          image transformations and the severity suite
  metrics.py          match rate, threshold at FPR, ROC/AUC, EER
  benchmark.py        runs hashes x attacks over an image folder
  report.py           tables and plots
scripts/validate_against_reference.py
tests/
data/samples/         12 sample images from the reference repos' test suites
```

## Notes

* `data/samples` is only a smoke test. 12 images give 66 impostor pairs, far too
  few to calibrate a threshold at a low FPR. Use at least ~1000 images for real results.
* Originals are downscaled to a longest side of 512 px before attacking (`--max-side`),
  for speed. Set `--max-side 0` to keep full resolution.
