# Adversarial results: all tables

*BTP: Perceptual Hashing. Collected 2026-11-11. This file only gathers the tables. Method details
and discussion are in:*
- *`adversarial_testing_report.md`: the V200 sweep (§1–4 below);*
- *`dinohash_table1_reproduction.md`: the DinoHash paper's Table 1 (§5).*

## Setup (§1–4: our V200 sweep)

| Item | Value |
|---|---|
| Images | `V200`: 200 COCO val2017 images, n = 200 per cell |
| Threat model | White-box, ℓ∞ budget ε (in /255, [0, 1] pixel space) |
| Attack | APGD + small-step PGD from the same start (better result kept per image), 100 steps per ε, warm start from the previous ε |
| Check | Success is recomputed on the image rounded to 8 bits |
| Thresholds | Calibrated at FPR 1e-3 unless stated: DinoHash trained 17 bits, untrained 21, NeuralHash 26, CertPHash L1 base 4936 / adv 5435 / ε1 4870 / ε2 3912 / ε4 3306 / ε8 2581 |
| Data | `hashbench/results/adv_v200_kaggle/` (DinoHash, CertPHash), `hashbench/results/adv_v200_neuralhash/` |

---

## 1. Evasion: the attacked image no longer matches its original

### 1.1 Success rate (%)

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| DinoHash untrained | 100 | 100 | 100 | 100 |
| DinoHash trained | 97.5 | 100 | 100 | 100 |
| NeuralHash | 0.5 | 0.5 | 18.0 | 94.5 |
| CertPHash base | 0 | 0 | 0 | 11.5 |
| CertPHash ε=2 | 0 | 0 | 1.0 | 3.5 |
| CertPHash ε=4 | 0 | 0 | 0 | 2.5 |
| CertPHash adv | 0 | 0 | 0.5 | 1.5 |
| CertPHash ε=1 | 0 | 0 | 0 | 1.0 |
| CertPHash ε=8 | 0 | 0 | 0 | 0 |

### 1.2 Bit accuracy (% of bits unchanged; lower = more broken)

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| DinoHash untrained | 20.6 | 2.3 | 0.1 | 0.0 |
| DinoHash trained | 45.6 | 22.3 | 10.3 | 3.7 |
| NeuralHash | 94.4 | 88.9 | 78.4 | 59.2 |

### 1.3 CertPHash evasion success (%) at the paper's L1 1800 threshold

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| base | 1.0 | 6.0 | 33.0 | 99.0 |
| adv | 0 | 0.5 | 5.0 | 60.5 |
| ε=1 | 0 | 0 | 2.0 | 48.0 |
| ε=2 | 0.5 | 1.5 | 3.0 | 43.0 |
| ε=4 | 0 | 0 | 2.5 | 22.5 |
| ε=8 | 0 | 0 | 0 | 0 |

### 1.4 CertPHash mean L1 distance moved by evasion

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| base | 453 | 902 | 1,767 | 3,404 |
| adv | 260 | 519 | 1,063 | 2,177 |
| ε=1 | 225 | 445 | 913 | 1,914 |
| ε=2 | 235 | 451 | 905 | 1,847 |
| ε=4 | 202 | 389 | 771 | 1,576 |
| ε=8 | 133 | 247 | 472 | 937 |

---

## 2. Collision, random target: make an image match an unrelated one (success %)

| Hash | ε=1 | ε=2 | ε=4 | ε=8 | ε=16 | ε=32 |
|---|---|---|---|---|---|---|
| DinoHash untrained | 92.5 | 100 | 100 | 100 | 100 | 100 |
| DinoHash trained | 51.0 | 92.0 | 97.5 | 99.5 | 100 | 100 |
| NeuralHash | 0 | 0 | 3.0 | 16.5 | 56.5 | 89.5 |
| CertPHash base | 1.0 | 1.0 | 6.5 | 25.5 | 88.0 | 100 |
| CertPHash adv | 0.5 | 2.0 | 3.0 | 11.0 | 51.0 | 94.5 |
| CertPHash ε=1 | 1.0 | 1.0 | 3.5 | 9.0 | 53.0 | 93.5 |
| CertPHash ε=2 | 0.5 | 1.0 | 1.0 | 3.5 | 20.5 | 83.5 |
| CertPHash ε=4 | 0 | 0.5 | 1.0 | 2.0 | 19.0 | 78.5 |
| CertPHash ε=8 | 1.0 | 1.0 | 1.0 | 1.5 | 16.0 | 66.5 |

Before any attack, no random pair matched (except 1% for CertPHash ε=8).

---

## 3. Collision, closest target: match the most similar other image in V200

### 3.1 Success rate (%), calibrated thresholds

| Hash | Already matched | ε=1 | ε=2 | ε=4 | ε=8 | ε=16 | ε=32 |
|---|---|---|---|---|---|---|---|
| DinoHash untrained | 10.5 | 99.5 | 100 | 100 | 100 | 100 | 100 |
| DinoHash trained | 6.0 | 92.5 | 97.5 | 100 | 100 | 100 | 100 |
| NeuralHash | 14.0 | 38.0 | 63.5 | 80.5 | 93.5 | 97.0 | 98.0 |
| CertPHash base | 8.0 | 19.5 | 33.0 | 65.5 | 100 | 100 | 100 |
| CertPHash adv | 11.0 | 18.0 | 34.0 | 69.5 | 96.0 | 100 | 100 |
| CertPHash ε=1 | 15.5 | 22.0 | 32.0 | 60.5 | 91.5 | 100 | 100 |
| CertPHash ε=2 | 10.5 | 12.5 | 15.5 | 28.0 | 59.5 | 94.5 | 100 |
| CertPHash ε=4 | 9.5 | 12.0 | 16.5 | 25.0 | 48.5 | 89.0 | 100 |
| CertPHash ε=8 | 12.5 | 15.5 | 20.5 | 27.0 | 50.5 | 78.5 | 98.5 |

### 3.2 CertPHash closest-target collision success (%) at the paper's L1 1800

| Hash | ε=8 | ε=16 | ε=32 |
|---|---|---|---|
| base | 4.0 | 51.5 | 98.5 |
| ε=8 | 13.0 | 44.5 | 91.5 |
| ε=4 | 7.5 | 33.5 | 91.0 |
| ε=2 | 6.5 | 24.5 | 80.0 |
| ε=1 | 1.0 | 7.5 | 66.5 |
| adv | 1.0 | 2.5 | 44.0 |

---

## 4. Image quality: mean SSIM of evasion images (1 = identical)

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| CertPHash (range across versions) | 0.997–0.998 | 0.990–0.992 | 0.966–0.973 | 0.895–0.919 |
| NeuralHash | 0.996 | 0.985 | 0.950 | 0.859 |
| DinoHash trained | 0.995 | 0.988 | 0.974 | 0.952 |
| DinoHash untrained | 0.995 | 0.988 | 0.984 | 0.984 |

At ε=32, SSIM is about 0.57–0.64 (CertPHash) and 0.58–0.68 (NeuralHash), so those attacks are clearly
visible. DinoHash attacks don't use their full budget (mean max change at ε=8: 6.4/255 trained,
3.2/255 untrained).

---

## 5. DinoHash paper Table 1 rerun: the authors' attack

Setup:
- the authors' APGD (`c7008888b7`), unmodified: 100 steps, random start, BCE loss;
- the first 50 V200 images.

Data:
- `hashbench/results/dinohash_table1_repro_combined/`;
- details and the GitHub script problem: `dinohash_table1_reproduction.md`.

### 5.1 Bit accuracy (%), as in the paper

| Model | Paper ε=4 | Ours ε=4 | Paper ε=8 | Ours ε=8 |
|---|---|---|---|---|
| DinoHash trained | 72.2 | 68.9 | 64.8 | 56.1 |
| DinoHash untrained (Meta DINOv2 + `PCA_96`) | 0.3 | 0.0 | 0.0 | 0.0 |
| NeuralHash (attacked correctly) | 0.1 | 75.6 | 0.0 | 58.4 |
| NeuralHash, the authors' script logic | 0.1 | 0.0 | 0.0 | 0.0 |

### 5.2 The same run as evasion success at our calibrated thresholds

| Model | Mean bits flipped ε=4 | Success ε=4 | Mean bits flipped ε=8 | Success ε=8 |
|---|---|---|---|---|
| DinoHash trained (> 17 bits) | 29.6 | 88% | 42.1 | 100% |
| DinoHash untrained (> 21 bits) | 96.0 | 100% | 96.0 | 100% |
| NeuralHash (> 26 bits) | 23.5 | 36% | 40.0 | 100% |

### 5.3 Authors' attack vs our sweep, trained DinoHash at ε=4

| Attack | Bits unchanged | Evasion success |
|---|---|---|
| Authors' APGD (50 images) | 68.9% | 88% |
| Our sweep (200 images) | 10.3% | 100% |

On NeuralHash at ε=4 the authors' attack did better than our sweep: 36% vs 18% success. Their BCE
loss seems to suit NeuralHash better than our hinge loss, so our NeuralHash success rates are
probably somewhat too low.

---

## 6. Comparison with the CertPHash paper (Table 4, COCO, empirical evasion)

The paper's attack is ℓ2 PGD (ℓ2 ≈ 3.7, at most 100 steps, final ℓ∞ ≈ 0.035 ≈ 9/255), following
Struppek et al. Our ℓ∞ 8/255 on the 64×64×3 input has ℓ2 ≤ 3.5, so our ε=8 column at L1 1800 is the
matching one.

| Model | Paper ER (%) | Ours, ε=8 at L1 1800 (%) |
|---|---|---|
| base | 89.2 | 99.0 |
| adv | 68.8 | 60.5 |
| ε=1 | 54.6 | 48.0 |
| ε=2 | – | 43.0 |
| ε=4 | 18.7 | 22.5 |
| ε=8 | 1.0 | 0.0 |

**Agreement:**
- Same ranking (base > adv > ε1 > ε4 > ε8).
- All values within about 10 points of the paper's.
- The paper's main claim reproduces: the ε=8 model is almost never evaded.

**Caveat:** this holds at L1 1800 only. At the calibrated thresholds (§1.1) almost nothing evades
CertPHash.

---

## 7. Paper-reproduction status

| Paper result | Reproduces? |
|---|---|
| CertPHash Table 4 (empirical evasion) | ✅ Yes, within about 10 points, same ranking |
| DinoHash Table 1, trained / untrained | ✅ Yes, with the authors' attack (68.9 vs 72.2; 0 vs 0.3) |
| DinoHash Table 1, NeuralHash | ❌ No: 75.6 vs 0.1. The paper's value matches the logic of the authors' buggy script |
| DinoHash's robustness against a stronger attack | Not claimed by the paper. Our sweep leaves 10.3% of bits at ε=4 |
