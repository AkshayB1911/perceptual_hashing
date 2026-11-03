# Plan: learning-based perceptual hashes

*BTP: Perceptual Hashing. Written 2026-10-23.*

The thesis focuses on learning-based perceptual hashes: NeuralHash, DinoHash and
CertPHash. The classical hashes (pHash, PDQ), already implemented bit-exactly in
`hashbench/`, are baselines.

## 1. Feasibility of the three main models

| | NeuralHash (Apple) | DinoHash (2025) | CertPHash (USENIX Sec 2025) |
|---|---|---|---|
| **Code** | Yes. Attack code in `neuralHash/` (Struppek et al.) | Yes, `proteus-photos/dinohash-perceptual-hash`, MIT licence | Yes, `Yuchen413/CertPhash` |
| **Weights** | **Not distributed.** Must be extracted (see below) | Yes, on Hugging Face `proteus-photos/DINOHash`, ONNX: 96-bit ViT-S/14, 512-bit ViT-B/14 | Yes, on Google Drive (`saved_models`): certified, adversarially trained and normal models |
| **Model** | MobileNetV3 (about 1.8M parameters), 128-dim features, 96 random hyperplanes → 96 bits | DINOv2 vision transformer → PCA → sign → 96 or 512 bits; input 224×224 with ImageNet normalisation | ResNet (about 3M parameters) on 64×64 images, trained to imitate an existing hash |
| **CPU only?** | Yes, including attacks (small model) | Hashing yes. Attacks on ViT-B are slow; use ViT-S locally and a GPU for scale | Hashing yes. Training no: authors used an A100 40 GB, about 6 h on COCO |
| **Built-in defence** | None. Collisions about 90% success; a gradient attack flips a bit in about 5 steps | Adversarial training (APGD): 72.2% bit accuracy at ε=4/255 vs 0.1% for NeuralHash | Certified training: guaranteed no-evasion within ε (IBP/CROWN via auto_LiRPA) |
| **Main risk** | Legal/ethical access to Apple's model | Low | Compute; the 64×64 input makes comparisons with other hashes harder |

### NeuralHash
- AppleNeuralHash2ONNX converts four files: `neuralhash_128x96_seed1.dat`,
  `NeuralHashv3b-current.espresso.{net,shape,weights}`. They can be taken from a
  macOS 11.4+ install, or from an iOS 14.7+ firmware file (`.ipsw`) from ipsw.me,
  mounted on Linux with apfs-fuse. No Apple device is needed.
- Hashes from ONNX Runtime can be a few bits off from those computed on an iPhone.
- Most academic work extracted it this way, but those are Apple's files and must not
  be redistributed. **Check with the supervisor before using it.**
- **Fallback:** DinoHash's own "untrained" variant (DINOv2 with no adversarial
  training) serves as the undefended deep hash.

### DinoHash
- Released in PyTorch and ONNX formats and as an npm package. Trained on DiffusionDB
  (CC-BY-NC); DINOv2 weights are CC-BY-SA-4.0.
- Adversarial fine-tuning: a student model is trained to reproduce the original
  model's hash under APGD-perturbed inputs, using cross-entropy on the logits plus a
  clean-image consistency term. 20k steps, batch size 256, AdamW.
- The paper's baselines are NeuralHash, DCT-DWT and Stable Signature (a watermark).

### CertPHash
- It trains a small network to **imitate an existing hash**, using PhotoDNA or PDQ
  hashes as training labels, and adds three objectives: anti-evasion, anti-collision
  and functionality. The released variants are referred to as C-PDNA and C-PDQ.
- Datasets: COCO (64×64), MNIST, CelebA and NSFW-56K. NSFW-56K is not distributed.
- **Checking that a hash can't be evaded (certified no-evasion rate, CNER):** about
  2.5 GB of GPU memory and 0.02 s per image. A free Colab or Kaggle GPU is enough.
- **Checking that it can't be forced into a collision (certified no-collision rate,
  CNCR):** about 34 GB and 395 s per image. Skip it or cite it.
- Reported results: average ROC-AUC under benign transformations is 0.98 for
  CertPHash, vs 0.92 for PDQ, 0.93 for PhotoDNA and 0.84 for NeuralHash.

## 2. Other hashes worth adding

1. **SSCD** (Meta, 2022). A copy-detection model with official pretrained weights.
   It outputs a 512–1024-dimensional vector; binarise with PCA and sign, as DinoHash
   does. A strong learned baseline from a different family.
2. **DinoHash untrained vs. adversarially trained.** Same architecture with and
   without the defence: a clean ablation.
3. **C-PDQ vs. our exact PDQ.** Measures how far CertPHash's learned imitation drifts
   from the real hash.
4. Optional: HybridHash, ViT2Hash, SimDINO (2026). These were evaluated in
   DualShield; weight availability is not yet checked.
5. Not usable: **PhotoDNA** is closed source. Use C-PDNA as its stand-in.
6. Classical baselines (done): pHash, PDQ, and PDQ with rotation/flip-aware matching.
   aHash, dHash and wHash are trivial to add via `imagehash`.

## 3. Attacks and defences

The papers in the repository fall into two families:
- **Perceptual hashing (copy detection):** NeuralHash, DinoHash, CertPHash, the CSAM
  law and ethics paper, and the survey. This is the core of the thesis.
- **Deep hashing for retrieval (same-class search):** DHTA, UAH and DarkHash attack
  *retrieval* models, not copy detectors. Their ideas can be borrowed, but the thesis
  should state this distinction.

### Attacks, in order of implementation

| Attack | Type | Source | Effort |
|---|---|---|---|
| Benign transformations | none | `hashbench/attacks.py` (23 kinds, several strengths) | Done |
| Gradient evasion (change the hash while barely changing the image) | white-box | Struppek et al. (`neuralHash/adv2_*`); APGD (DinoHash) | Low: code exists |
| Gradient collision (make an image match a target hash) | white-box | Struppek et al. (`neuralHash/adv1_*`); extend with DHTA's "target a set of images" anchor code | Low to medium |
| Query-only evasion | black-box | Jain et al. (2021); NES as in DualShield | Medium |
| Attacks on PDQ | black-box | Prokos et al., "Squint Hard Enough" (USENIX 2023) | Medium |
| Recovering image content from the hash | privacy | Struppek et al. (`adv4_*`); Madden et al. (NeurIPS-W 2024) | Medium |
| Backdoor implanted during fine-tuning | training-time | DarkHash, adapted from retrieval models to perceptual hashes | High, but potentially novel |

Threat-model settings to report: white-box vs black-box, ℓ∞ budgets
(1/255, 2/255, 4/255, 8/255) and ℓ2 budgets, image quality (SSIM/LPIPS), success rate
and number of steps or queries.

### Defences to compare
- **Adversarial training:** DinoHash.
- **Certified training:** CertPHash.
- **Randomized smoothing at matching time,** plus DualShield, a plug-in that needs no
  retraining (arXiv 2608.03101, Aug 2026, not yet peer-reviewed). DualShield adds
  Gaussian noise (σ=0.10) at matching time and an imperceptible ℓ∞ ≤ 16/255
  hardening perturbation to reference images before publication.
- **Deliberately adding randomness to the hashes:** Madden et al.

## 4. Proposed thesis shape

> A unified evaluation of classical vs. learned perceptual hashes under benign,
> adversarial and privacy attacks, and how far today's defences actually help.

- **Baselines:** pHash and PDQ (bit-exact implementations in `hashbench/`).
- **Subjects:** NeuralHash (or untrained DinoHash), DinoHash and CertPHash.
- **Extra learned baseline:** SSCD.
- **Fair comparison:** every algorithm gets the same false-match rate (threshold
  calibrated on pairs of different images), with ROC/AUC and match rates per attack
  and strength. This is already implemented in `hashbench`.

Candidate contributions of our own, all feasible with limited compute:
- **(a) Transferability.** Do attacks crafted on one hash still work on the others,
  e.g. from undefended models to the defended DinoHash or C-PDQ?
- **(b) Randomized smoothing for DinoHash or NeuralHash,** without retraining.
- **(c) DarkHash-style backdoor in perceptual-hash fine-tuning.** The most novel
  option and the riskiest.

## 5. Implementation roadmap

1. **DinoHash** (no extraction, no GPU needed). Add a `DinoHash` class in
   `hashbench/hashes/` using the 96-bit ONNX model via `onnxruntime`, register it, and
   run it through the existing benchmark.
   **Done (2026-10-23).** `hashbench/hashes/dinohash.py`:
   - DINOv2 re-implemented in PyTorch (differentiable, ready for step 6).
   - Bit-identical to the released ONNX models (ViT-S/96 and ViT-B/512).
   - Variants: `dinohash`, `dinohash_vitb14_512`, `dinohash_untrained`,
     `dinohash_vitb14_512_untrained`.

   Findings:
   - Both released models are adversarially fine-tuned.
   - The ViT-S/96 model ships its own PCA head, not the repo's `.npy`.
   - The ONNX export uses `antialias=False` for the positional embeddings, unlike
     PyTorch; this changes up to 10/512 bits.
   - Smoke test (12 images): DinoHash survives crops, translation, rotation and
     borders, where pHash and PDQ fail. Trained ≈ untrained on benign
     transformations, so the difference must be shown with adversarial attacks
     (step 6).
2. **Dataset decision.** Use 1,000–2,000 images, e.g. ImageNette or COCO val2017.
   CertPHash's COCO 100×100 archive is another option for comparability with it.
3. **CertPHash.** Load the released C-PDQ and C-PDNA models. Evaluate them on the
   benchmark at their 64×64 input size; run CNER verification on a cloud GPU.
4. **NeuralHash** (after supervisor approval). Extract and convert to ONNX, then wrap
   it as a `PerceptualHash`.
5. **SSCD.** Official TorchScript weights; binarise with PCA and sign.
6. **White-box attacks.** Port the Struppek et al. evasion and collision attacks and
   APGD into a model-agnostic `hashbench/adversarial.py` that works for any
   differentiable hash.
7. **Black-box and privacy attacks.** NES-based evasion; hash-to-content leakage.
8. **Defences.** Randomized smoothing, then the comparison with adversarial training
   (DinoHash) and certified training (CertPHash).
9. **Contribution experiment:** (a), (b) or (c) from Section 4.

## 6. Open decisions

- [ ] NeuralHash: may we extract and use Apple's model? (Ask the supervisor.)
- [ ] GPU: Kaggle (about 30 free GPU hours a week), Colab, or an institute server.
  The local machine has no NVIDIA GPU.
- [ ] Benchmark dataset choice.
- [ ] Which contribution direction to pursue: (a), (b) or (c).

## Sources

- DinoHash: [Hugging Face weights](https://huggingface.co/proteus-photos/DINOHash) ·
  [code](https://github.com/proteus-photos/dinohash-perceptual-hash) ·
  [ICML 2025 CODEML](https://icml.cc/virtual/2025/48181) ·
  [OpenReview](https://openreview.net/forum?id=HrGa8Mq2NE) · `DinoHash.pdf` in this repo
- CertPHash: [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/yang-yuchen) ·
  [artifact appendix](https://www.usenix.org/system/files/usenixsecurity25-appendix-yang-yuchen.pdf) ·
  [code](https://github.com/Yuchen413/CertPhash) · `CertPHash.pdf` in this repo
- NeuralHash: [AppleNeuralHash2ONNX](https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX) ·
  [Learning to Break Deep Perceptual Hashing](https://arxiv.org/pdf/2111.06628) ·
  `Learning to break nerual hash.pdf` and `neuralHash/` in this repo
- [DualShield, arXiv 2608.03101](https://arxiv.org/html/2608.03101v1)
- [Madden et al., Robustness of Practical Perceptual Hashing to Evasion and Inversion](https://arxiv.org/pdf/2406.00918)
- [Squint Hard Enough, USENIX Security 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/prokos)
- [Adversarial Detection Avoidance Attacks (Jain et al.)](https://arxiv.org/pdf/2106.09820)
- SSCD: [paper](https://arxiv.org/pdf/2202.10261) ·
  [Hugging Face mirror](https://huggingface.co/m3/sscd-copy-detection/tree/main)
- [Hamming Distributions of Popular Perceptual Hashing Techniques](https://arxiv.org/pdf/2212.08035)
- Deep hashing for retrieval (this repo): `DHTA.pdf`, `UAH.pdf`, `DarkHash.pdf`
