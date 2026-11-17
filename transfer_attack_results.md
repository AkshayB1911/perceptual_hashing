# Transfer attack results (attack #1 of `future_attacks.md`)

*BTP: Perceptual Hashing. Run 2026-11-17 (Kaggle T4 + laptop CPU). Data:
`hashbench/results/transfer_v200/` (DinoHash and CertPHash sources), `hashbench/results/transfer_v200_neuralhash/`
(NeuralHash source), `hashbench/results/transfer_v200_matrix/` (matrices, heatmaps, raw.csv).*

## 1. Question

If an attacker has the weights of hash **A** (white-box) and attacks an image, does the same attacked
image also fool hash **B**, whose weights they never used? This is the threat model of an attacker who
uses a public substitute model instead of the deployed one. It is a black-box attack with zero queries.

## 2. Setup

| Item | Value |
|---|---|
| Images | `V200` (200 COCO val2017 images) |
| Sources (attacked) | DinoHash trained / untrained, NeuralHash, CertPHash base / adv / ε1 / ε2 / ε4 / ε8 (9) |
| Targets (re-hashed) | the same 9 + **pHash** and **PDQ** (11) |
| Attack on the source | Same as the V200 sweep: APGD + small-step PGD ensemble, 100 steps, ℓ∞, warm start across ε, best iterate by hash distance on the 8-bit image |
| Budgets | Evasion ε = 2, 4, 8 /255; random-target collision ε = 8, 16 /255 (target: image i + n/2) |
| Success | At each **target's** calibrated FPR 1e-3 threshold (DinoHash-T 17 bits, Dino-U 21, NeuralHash 26, pHash 16/64, PDQ 100/256, CertPHash L1 2581–5435) |
| Fair comparison | Images are saved at the source's input size (224 / 360 / 64 px). Target B compares B(clean at that size) with B(attacked at that size), so only the perturbation differs, not the resizing |
| Control | The clean image plus **random ±ε noise** of the same size. A transfer only counts if it beats this |
| Collisions | Counted only over pairs that did not already match before the attack |
| Code | `scripts/generate_transfer_images.py`, `scripts/transfer_matrix.py`, Kaggle package `kaggle/transfer/` (the older scripts are unchanged) |
| Run time | Kaggle T4, 8 sources: 2.1 h. NeuralHash on laptop CPU: 3.6 h. Matrix (CPU re-hashing): about 1.1 h |
| Check | Re-hashing each saved image with its own source reproduces the white-box result for **8,998 of 9,000** images. The 2 exceptions are CertPHash cases exactly at the threshold (1 L1 unit), caused by CertPHash's float resize vs the 8-bit clean PNG |

Short names: Dino-T = DinoHash trained, Dino-U = DinoHash untrained, NH = NeuralHash, C-x = CertPHash
variant. **Bold** = diagonal = white-box success on the source itself.

## 3. Results

### 3.1 Evasion transfer, ε = 8/255 (success %)

| source \ target | Dino-T | Dino-U | NH | C-base | C-adv | C-ε1 | C-ε2 | C-ε4 | C-ε8 | pHash | PDQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dino-T | **100** | 61.5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dino-U | 10.0 | **100** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NH | 2.5 | 0.5 | **93.5** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| C-base | 17.0 | 7.0 | 1.0 | **15.0** | 0.5 | 0 | 1.0 | 0.5 | 0 | 0 | 0 |
| C-adv | 15.5 | 10.5 | 3.0 | 0 | **1.5** | 0 | 0 | 0 | 0 | 0 | 0 |
| C-ε1 | 21.5 | 6.5 | 3.5 | 0 | 0.5 | **1.0** | 0.5 | 0 | 0 | 0 | 0 |
| C-ε2 | 20.0 | 6.0 | 2.0 | 0 | 0 | 0 | **4.0** | 1.5 | 0 | 0 | 0 |
| C-ε4 | 21.0 | 6.5 | 2.0 | 0 | 0 | 0 | 2.0 | **2.5** | 0 | 0 | 0 |
| C-ε8 | 20.5 | 6.0 | 0.5 | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 |

**Random-noise control at ε = 8/255** (same rows; all other targets ≤ 1%):

| source (image size) | Dino-T | Dino-U | NH |
|---|---|---|---|
| Dino-T, Dino-U (224 px) | 0–0.5 | 0–0.5 | 0 |
| NH (360 px) | 0 | 0 | 0 |
| any CertPHash (64 px) | 26–30.5 | 8–9.5 | 0.5–1 |

### 3.2 The DinoHash pair across budgets (evasion success %, mean bits flipped)

| Direction | ε=2 | ε=4 | ε=8 | Noise control |
|---|---|---|---|---|
| Dino-T → Dino-U (threshold 21 bits) | 30.5% (15.3 bits) | 46.5% (21.3) | 61.5% (26.0) | 0–0.5% |
| Dino-U → Dino-T (threshold 17 bits) | 7.5% (9.6 bits) | 9.5% (10.4) | 10.0% (10.5) | 0% |

All other off-diagonal evasion cells at ε = 2 and 4 are ≤ 2.5%.

### 3.3 Collision transfer, ε = 16/255 (success %, pairs not matching before)

| source \ target | Dino-T | Dino-U | NH | C-base | C-adv | C-ε1 | C-ε2 | C-ε4 | C-ε8 | pHash | PDQ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dino-T | **97.0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Dino-U | 0 | **100** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NH | 0 | 0 | **56.0** | 1.0 | 1.5 | 1.0 | 0.5 | 0.5 | 0 | 0 | 0 |
| C-base | 0 | 0 | 0.5 | **87.0** | 7.5 | 5.5 | 1.0 | 1.0 | 0 | 0.5 | 0.5 |
| C-adv | 0 | 0 | 0.5 | 18.0 | **50.0** | 24.5 | 3.5 | 1.5 | 0 | 0.5 | 1.0 |
| C-ε1 | 0 | 0 | 0.5 | 8.0 | 21.0 | **53.0** | 3.0 | 1.5 | 0 | 1.0 | 0 |
| C-ε2 | 0 | 0 | 0.5 | 7.0 | 15.0 | 12.0 | **20.5** | 2.5 | 0 | 0.5 | 0.5 |
| C-ε4 | 0 | 0 | 0 | 4.0 | 6.5 | 6.0 | 3.0 | **18.5** | 0 | 0 | 0 |
| C-ε8 | 0 | 0 | 0 | 2.5 | 4.5 | 4.0 | 1.0 | 0.5 | **15.2** | 0.5 | 0 |

At ε = 8/255 every off-diagonal collision cell is ≤ 5%. All tables, including ε = 2, 4 and the noise
controls, are in `results/transfer_v200_matrix/summary.md`; heatmaps are `transfer_*.png` there.

## 4. Findings

1. **Transfer is weak almost everywhere.** Apart from the DinoHash pair and the CertPHash family, an
   image attacked on one hash almost never fools another (≤ 3.5% evasion at ε = 8/255). Without the
   target's weights, our white-box results do not carry over. A real attacker would need queries
   (black-box, `future_attacks.md` #2/#3) or the actual model.

2. **pHash and PDQ are never fooled by transferred noise** (0% evasion; ≤ 1% collision). Small
   pixel noise tuned for a neural network does not move their low-frequency DCT bits.

3. **NeuralHash is not reached by transfer either** (≤ 3.5%), and its own attacks don't transfer out
   (≤ 2.5%).

4. **The DinoHash pair transfers, but asymmetrically.**
   - Attacks on **trained** DinoHash fool **untrained** DinoHash: 61.5% at ε = 8 (26 bits flipped).
   - Attacks on **untrained** fool **trained** only 10% of the time (about 10 bits flipped, under its
     17-bit threshold), and this does not grow with ε.
   - The noise control is about 0% for both, so this is real transfer.
   - **What it means for the "gradient masking" question:** if trained DinoHash's robustness were only
     hidden gradients, attacks made on the untrained twin (same backbone, clean gradients) should
     break it easily. They don't (10%), while the direct white-box attack does (100%). So the trained
     model really changed its features; the white-box weakness is not a masking artefact. This
     agrees with Sun et al., who found DinoHash *more* robust than NeuralHash under black-box attacks.
   - The direction "robust → standard transfers better" matches what is known for classifiers:
     perturbations against a more robust model are more general.

5. **CertPHash → DinoHash trained looks like 17–21% at ε = 8, but it is not transfer.** Random noise
   on the same 64-px images does *better* (26–30.5%). The cause is the resizing: a 64-px image
   upscaled to 224 px with ±8/255 noise per pixel becomes blocky noise that trained DinoHash is
   sensitive to. This is a side finding about **trained DinoHash's sensitivity to upscaled noise**,
   not an attack effect. Without the control we would have reported it wrongly.

6. **Collisions transfer only inside the CertPHash family**, between variants with the same
   architecture and training data (e.g. C-adv → C-ε1 24.5% at ε = 16). **CertPHash ε8 receives
   0%** from every source, the strongest result for certified training here.

7. **Collisions never transfer across families** (≤ 1.5%). Forging a match on a hash needs that hash.

## 5. Caveats

- **Single-source attacks.** Attacking an ensemble of sources (e.g. Dino-U + NH together) usually
  transfers better. That will be tested with joint evasion (`future_attacks.md` #7).
- **Perturbation at the model's input size.** Images are 224 / 360 / 64 px, so cross-size transfer
  includes a resize. The noise control measures this effect; finding 5 shows it matters for 64 px
  sources.
- **n = 200 per cell.** A 95% interval around 10% is roughly ±4 points; cells ≤ 2% are effectively
  zero.
- **Collision rows** exclude pairs that already matched before the attack (only CertPHash ε8: 1% of
  pairs; 0% for every other target).

## 6. Rerun

```bash
# DinoHash + CertPHash sources on Kaggle (never uploads NeuralHash files)
bash kaggle/transfer/push_transfer.sh <kaggle_username>
# NeuralHash source locally (from hashbench/)
python scripts/generate_transfer_images.py --list data/splits/V200.txt \
    --out results/transfer_v200_neuralhash --device cpu --hashes neuralhash
# matrix (from hashbench/; resumable)
python scripts/transfer_matrix.py --gen results/transfer_v200 results/transfer_v200_neuralhash \
    --out results/transfer_v200_matrix
```
