# Benign benchmark, Run 1: observations, checks and explanations

*BTP: Perceptual Hashing. Written 2026-11-01. Run: `hashbench/results/coco_run1`.*

> **Superseded by `benchmark_findings.md`,** which covers all 12 hashes and updates F1, F2, F3, F5, F9, F10,
> F11 and F12. Kept as the record of Run 1 on its own.

## Setup

- **Hashes:** pHash, PDQ, PDQ-dihedral, NeuralHash, and CertPHash in six variants:
  certified ε = 1/255 (`certphash`), 2, 4, 8 /255, plus normal training (`base`) and
  adversarial training (`adv`).
- **Data:** COCO val2017.
  - Impostor pairs (different images): 200,000 pairs drawn from all 5,000 images.
  - Attack suite: 23 attack types with 88 parameter settings, on the 1,000 images in
    `data/splits/V1k.txt`. That gives 880,000 genuine distances.
- **Thresholds:** for each hash, calibrated to a false-match rate (FPR) of 1e-3; 1e-2 and
  1e-4 are also reported. FPR 1e-4 rests on only 20 impostor pairs per hash.
- **Pending:** DinoHash runs in Run 2.

### Where the data is

| What | Path |
|---|---|
| Raw distances, configuration, log | `hashbench/results/coco_run1/{genuine,impostor,hashes}.csv`, `config.json`, `run.log` |
| Report tables and plots | `overall.csv`, `summary.csv` (with 95% intervals), `attack_curves.png`, `mean_distance_curves.png`, `distance_hist.png`, `roc.png` |
| Analysis tables | `hashbench/results/coco_run1/analysis/*.csv` |
| Thesis heatmap | `hashbench/results/coco_run1/analysis/family_heatmap.png` |

The analysis tables are:
- `family_match_rates.csv`: match rate per attack family, with a bootstrap 95% interval
  over images;
- `per_param.csv`: one row per attack and strength;
- `zero_fpr.csv`: match rates at a threshold with no false matches at all;
- `exact_invariances.csv`: share of copies whose hash is exactly unchanged;
- `closest_impostors.csv`: the closest pairs of different images;
- `bit_balance.csv`: bit statistics of each hash;
- `image_difficulty_corr.csv`: whether the same images are hard for every hash;
- `black_frame_*.csv`, `dark_image_impostors.csv`, `image_mean_luma.csv`: the
  black-frame and dark-image checks;
- `probe_*.csv`: the targeted probes.

**To reproduce** (from `hashbench/`):

```
python -m hashbench.report --run results/coco_run1 --fpr 0.001 0.01 0.0001
python scripts/analyze_run.py --run results/coco_run1 --fpr 0.001
python scripts/analyze_closest_pairs.py --run results/coco_run1 --images data/coco/val2017
python scripts/probe_invariances.py --run results/coco_run1 --images data/coco/val2017 --list data/splits/V200.txt --n 50
```

---

## 1. Headline numbers

**Mean match rate over all 88 settings (%)**

| Hash | FPR 1e-2 | FPR 1e-3 | FPR 1e-4 | Published threshold | AUC |
|---|---|---|---|---|---|
| CertPHash adv | 94.0 | **91.8** | 90.2 | 76.9 | 0.988 |
| CertPHash (ε=1) | 93.9 | **91.3** | 88.4 | 74.3 | 0.988 |
| NeuralHash | 94.2 | **89.6** | 83.0 | – | 0.990 |
| CertPHash base | 92.5 | **86.3** | 75.6 | 67.0 | 0.987 |
| CertPHash ε=8 | 90.4 | **85.0** | 78.7 | 76.2 | 0.978 |
| PDQ-dihedral | 82.2 | **80.9** | 79.8 | 66.2 | 0.923 |
| pHash | 83.8 | **80.2** | 78.3 | 73.6 | 0.946 |
| CertPHash ε=2 | 91.8 | **79.6** | 71.5 | 66.7 | 0.987 |
| PDQ | 80.3 | **79.3** | 78.6 | 63.9 | 0.916 |
| CertPHash ε=4 | 89.6 | **78.2** | 68.9 | 66.1 | 0.985 |

Averages weight every setting equally, so attacks with more strength levels count for more.
The family view below is fairer.

**Match rate by attack family at FPR 1e-3** (95% bootstrap intervals are ±1 point or less,
except flips, which are about ±2)

| Hash | Compression | Filtering/noise | Photometric | Geometric | Flips | Overlays |
|---|---|---|---|---|---|---|
| NeuralHash | 0.98 | 0.98 | 0.93 | **0.86** | 0.31 | **0.72** |
| CertPHash adv | 1.00 | 1.00 | 1.00 | 0.80 | 0.21 | 0.97 |
| CertPHash ε=1 | 1.00 | 1.00 | 1.00 | 0.79 | 0.21 | 0.97 |
| CertPHash base | 1.00 | 1.00 | 1.00 | 0.65 | 0.15 | 0.93 |
| CertPHash ε=8 | 1.00 | 1.00 | 0.96 | 0.64 | 0.12 | 0.93 |
| pHash | 1.00 | 1.00 | 1.00 | 0.46 | **0.00** | 0.96 |
| CertPHash ε=2 | 1.00 | 1.00 | 1.00 | 0.45 | 0.10 | 0.92 |
| PDQ | 1.00 | 1.00 | 1.00 | 0.42 | **0.00** | 1.00 |
| CertPHash ε=4 | 1.00 | 0.998 | 0.99 | 0.41 | 0.08 | 0.90 |
| PDQ-dihedral | 1.00 | 1.00 | 1.00 | 0.40 | **1.00** | 1.00 |

---

## 2. Correctness checks (does the run behave as it should?)

| # | Check | Expected | Observed | Verdict |
|---|---|---|---|---|
| 1 | Row counts | 1000 × 88 × 10 genuine; 200k × 10 impostor | 880,000 / 2,000,000, no failed attacks | ✅ |
| 2 | Distance between unrelated images, binary hashes | ≈ 0.5 (bits uncorrelated) | NeuralHash 0.496, PDQ 0.500, pHash 0.491 | ✅ (pHash below 0.5 because of check 3) |
| 3 | pHash bit balance | – | One bit is **always 1** | ✅ Expected: it is the DC coefficient (mean brightness), always above the median. imagehash has the same property, so pHash carries **63** useful bits |
| 4 | Grayscale, pHash | Distance exactly 0 (pHash only uses luma) | 100% exactly 0 | ✅ |
| 5 | Grayscale, PDQ | ≈ 0 | 95% exactly 0, all matched | ✅ PIL rounds luma to integers; PDQ uses float luma, so a few bits flip |
| 6 | Aspect-ratio stretch | ≈ no effect: every hash resizes to a fixed square and ignores aspect ratio | 100% matched for all hashes; pHash 95% exactly identical | ✅ |
| 7 | Grayscale, hue and saturation on CertPHash | No change for photos that are already grayscale | Exactly 0 for 2.8% of images, in every variant | ✅ Matches COCO's share of grayscale photos |
| 8 | PDQ-dihedral on flips | 100% matched (8 orientations derived from the DCT) | 100%; exactly 0 in 16% (horizontal) and 45% (vertical) | ✅ Small differences come from the filter and decimation not being exactly symmetric |
| 9 | pHash and PDQ on flips | A flip negates every DCT coefficient with an odd frequency, so about half the bits change | Probe: distance 0.49–0.50, the same as unrelated images | ✅ Theory confirmed |
| 10 | NeuralHash against the published curves (Struppek et al., `neuralHash/images/transformations.png`, ImageNet) | Brightness 1.25 → ≈0.105, 1.5 → ≈0.23; contrast 1.25 → ≈0.057, 1.5 → ≈0.127; rotation 4° → ≈0.08, 16° → ≈0.21, 64° → ≈0.39; ≈0 when darkening | Brightness 1.3 → 0.132, 1.6 → 0.267; contrast 1.3 → 0.075, 1.6 → 0.153; rotation 5° → 0.109, 20° → 0.27, 45° → 0.393; darkening ≤ 0.003 | ✅ Independent confirmation of our NeuralHash and attack code |
| 11 | CertPHash against its paper | Reported ROC-AUC for C-PDNA under benign transformations ≈ 0.98 | 0.988 (ε=1) | ✅ |
| 12 | Statistical precision | – | Per setting: 95% interval ≤ ±3.1 points (n = 1,000). Per family: ≤ ±1 point | ✅ |

The CertPHash paper reports AUC 0.84 for NeuralHash; we measure 0.990. Their
transformation set, strengths and dataset differ from ours, so the two numbers measure
different things. Point this out if both are cited.

---

## 3. Findings and why they happen

### F1. Compression, noise, filtering and colour don't separate the hashes
Every hash scores about 100% on these.
- **Why:** every hash shrinks the image heavily before hashing (pHash to 32×32, PDQ and
  CertPHash to 64×64, NeuralHash to 360×360 followed by a CNN). JPEG artefacts, noise and
  blur are fine detail that disappears in that step, and the low-frequency structure the
  hashes rely on survives.
- pHash and PDQ threshold at the median, which doesn't change when all pixels are scaled
  by the same factor. That's why pHash is *exactly* unchanged for 50% of brightness edits:
  the darkening ones.
- **Consequence:** benign tests can't distinguish learned from classical hashes on these
  attacks. Geometry, flips and overlays carry all the differences.

### F2. Geometric edits split classical and learned hashes
- Translating by 5%: PDQ 2.4%, pHash 17%, NeuralHash 100%.
- Corner crop keeping 80%: PDQ 0.2%, pHash 0.7%, NeuralHash 94%.
- **Why the classical hashes fail:** DCT basis functions are tied to fixed positions. A
  shift changes the phase of the low-frequency coefficients, so their signs relative to the
  median change.
- **Why PDQ fails more than pHash:** PDQ uses a 16×16 block of frequencies (up to 16
  cycles across the image), while pHash uses the lowest 8×8. Higher frequencies make PDQ
  more sensitive to shifts.
- **Why NeuralHash survives:** a CNN with global pooling is roughly shift-tolerant, and it
  was trained to be robust to such edits.

### F3. Flips: classical hashes see a completely new image; learned hashes see horizontal flips more often
- pHash and PDQ: 0% (explained by check 9). PDQ-dihedral: 100%, by design.
- Every learned model matches horizontal flips much more often than vertical ones:
  - NeuralHash 48% vs 14%;
  - CertPHash ε=1 32% vs 10%.
- **Why (hypothesis):** horizontal flips are a standard training augmentation, and
  left-right mirrored photos still look natural. Upside-down photos are rare, so nothing in
  training encourages robustness to vertical flips. The asymmetry is consistent across 7
  independently trained models.

### F4. NeuralHash ignores darkening and contrast reduction, but not brightening or gamma
- Brightness 0.5 and 0.7: 100%. Brightness 1.6: 55%. Gamma 2.0: 49%. Gamma 0.5: 80%.
- **Why:** the network starts with convolution → **instance normalisation**. That pair
  removes any change of the form a·x + b to the input.
- **Probe** (`probe_neuralhash_affine.csv`, 50 images):

  | Edit | Bits changed (of 96) |
  |---|---|
  | Scaling around mid-grey | **0.00** |
  | Plain darkening | 0.08–0.22 (the convolution pads with zeros, which isn't affine-invariant) |
  | Gamma 2.0 | 27.8 (non-linear) |
  | Brightness 1.6 with clipping | 27.4 (non-linear) |

  Brightening fails because PIL clips values at 255, which isn't affine.
- This matches Struppek et al.'s published curves (check 10).

### F5. NeuralHash is fragile to a grey square in the centre; pHash and PDQ are not
- Occlusion covering 10% of the image: NeuralHash 39%, PDQ 100%. Covering 25%: NeuralHash
  6.5%, PDQ 99%, pHash 83%.
- **Why (hypothesis):**
  - The square is grey (128), close to the average image brightness, so the coarse
    brightness layout that pHash and PDQ measure barely changes.
  - NeuralHash encodes image content. COCO objects tend to be in the middle, so a central
    square hides the main object.

  Not probed further.

### F6. The CertPHash ε sweep isn't monotonic, and the cause is a black-pixel shortcut in the ε=2 and ε=4 models
- **Observation:**
  - ε=2 and ε=4 match only 24–26% with a 2% black border or a 2% shift.
  - ε=1, ε=8 and adv stay at 100% on the same edits.
  - The same pattern explains their poor rotation results.
- **Ruling out a bug:** all six variants run through the same code, and their outputs were
  already verified bit-identical to the authors' implementation.
- **Probe** (`probe_certphash_edges.csv`): overwrite only the outer 1 px of the 64×64
  input, with no geometric change.

  | Edge fill | ε=2 / ε=4 match rate |
  |---|---|
  | Black | **30%** |
  | Grey, white or replicated edge | 98% |

  The weakness is to **black** pixels specifically. Border, translation and rotation all
  add black pixels, so this single cause explains all three failures. The base model shows
  it to a lesser degree (82% at 1 px, 36% at 4 px). ε=1, ε=8 and adv don't show it.
- **The same flaw causes false matches between different images**
  (`black_frame_impostors.csv`, `dark_image_impostors.csv`):

  | Pair type | Share of all pairs | Share of the 100 closest pairs (base / ε=2 / ε=4) | Other hashes |
  |---|---|---|---|
  | Both photos black-framed (1% of COCO photos) | 0.01% | 11–12% | 0% |
  | Both among the darkest 10% | 1% | 14% / 24% / 34% | 1–8% |

  Example: the photos of a child with a cake (000000102820) and of a motorbike
  (000000226662) are among the closest pairs for three models. Their only shared feature is
  a black frame.
- **Interpretation:**
  - The ordering of the certified models on benign edits is driven by **habits specific to
    each training run** ("shortcuts"), not only by ε.
  - Only one checkpoint was released per ε, so the effect of ε can't be separated from
    run-to-run variation. This must be stated in the thesis.
  - Why black specifically is unknown. Two possibilities to test:
    - real PhotoDNA itself reacts to black borders, and the models copied it (testable
      with the real PhotoDNA labels in `coco-val.csv`);
    - an artefact of training.

### F7. CertPHash ε=8 has a narrow output range and is sensitive to large brightness shifts
- **Observations:**
  - Lowest AUC (0.978).
  - Different images are closer together than for the other variants: mean distance 0.155
    against about 0.21.
  - Darkening by half: 52% matched.
- **First hypothesis (ε=8 confuses dark images): refuted.**
  - Its closest pairs aren't dark (average brightness 106; all images: 114).
  - Its darkening failures are worst on the *brightest* photos (31% matched) and mildest on
    the darkest (78%).
- **Explanation consistent with the data:** halving a bright image moves pixel values by a
  large absolute amount. Different images map to close outputs under this model, so such a
  shift looks like "a different image".

### F8. A larger certified ε costs both fidelity and discrimination
- AUC falls steadily: 0.988 (ε=1) → 0.987 (ε=2) → 0.985 (ε=4) → 0.978 (ε=8). This holds
  even though the per-attack ranking isn't monotonic (F6).
- Combined with the earlier fidelity measurement (share of hashes within the PhotoDNA
  threshold of the real PhotoDNA hash: 96.9% at ε=1, 15.9% at ε=8), certified training at
  larger ε costs both closeness to PhotoDNA and the ability to tell images apart.
- This is direct evidence for the "price of robustness" storyline.

### F9. Published thresholds are very conservative, except for the ε=2/4/8 CertPHash models
- At pHash 10/64, PDQ 31/256 and CertPHash L1 1800, there are zero false matches in 200,000
  pairs for pHash, PDQ, PDQ-dihedral and CertPHash ε=1, adv and base. But these thresholds
  miss 25–35% of copies.
- At the paper's 1800 threshold, CertPHash ε=2, ε=4 and ε=8 **do** produce false matches:
  7, 4 and 6 of 200,000. That's about 2–4 × 10⁻⁵, which is significant at deployment scale
  (billions of comparisons). Two each for ε=2 and ε=4 are black-framed pairs (F6).
- **Consequence:** calibrated thresholds are needed for a fair comparison. The published
  CertPHash threshold doesn't carry over to its larger-ε models.

### F10. With no false matches allowed, the hashes still catch 63–88% of copies
- Threshold just below the closest impostor pair (`zero_fpr.csv`):

  | Hash | Match rate |
  |---|---|
  | adv | 88% |
  | ε=1 | 82% |
  | PDQ-dihedral | 79% |
  | PDQ | 77% |
  | pHash | 76% |
  | NeuralHash | 75% |
  | ε=8 | 73% |
  | base | 69% |
  | ε=4 | 64% |
  | ε=2 | 63% |

- PDQ has the narrowest spread of distances between unrelated images (standard deviation
  0.032, closest pair 0.352). That's why it holds up best as the FPR gets stricter.

### F11. Different hashes fail on different images, which suggests combining them
- How strongly the per-image match rates of two hashes are correlated
  (`image_difficulty_corr.csv`):

  | Pair | Correlation |
  |---|---|
  | Within families: PDQ / PDQ-dihedral | 0.81 |
  | Within families: CertPHash variants | 0.3–0.76 |
  | Within families: pHash / PDQ | 0.39 |
  | NeuralHash vs anything else | ≤ 0.12 |
  | Classical vs CertPHash | ≈ −0.1 to 0.06 |

- Failures come from the hash, not from "hard images".
- **Consequence:** combining hashes is promising (Idea 4 in `thesis_research_plan.md`):
  - NeuralHash covers geometric edits;
  - PDQ covers occlusion, overlays and brightness;
  - dihedral matching covers flips.

### F12. Bit statistics (Idea 6)

| Hash | Share of 1s per bit | Mean bit correlation | Note |
|---|---|---|---|
| PDQ | 0.46–0.53 | 0.013 | Close to ideal |
| NeuralHash | 0.40–0.60 | 0.053 (max 0.27) | Per-bit entropy sums to 95.4 of 96, an upper bound; correlations reduce the true effective length further |
| pHash | – | – | 63 useful bits (check 3) |

PDQ quality can't be studied on COCO: only 4 of the 1,000 attacked images have quality
below 50.

---

## 4. Corrections to earlier statements

- The earlier statement "ε=8 does better than ε=2/4 on borders, unexplained" is now
  explained (F6). ε=2 and ε=4 have the black-pixel shortcut and ε=8 doesn't; it isn't an
  effect of ε itself.
- My first explanation for ε=8 (dark images) was tested and refuted (F7).

## 5. Limitations

- One released checkpoint per CertPHash ε, so effects of ε and of the individual training
  run can't be separated.
- COCO-specific rates: black frames are 1% of images, there are few low-quality images, and
  objects tend to be in the centre.
- Originals are downscaled to 512 px before attacking. Averages weight all settings
  equally.
- FPR 1e-4 thresholds rest on 20 impostor pairs per hash.
- F3, F5 and the cause of the black-pixel shortcut are hypotheses that haven't been probed.

## 6. Next

1. **Run 2:** DinoHash ViT-S, trained and untrained (about 4 h, checkpointed).
2. Optional **Run 3:** ViT-B on V200 only.
3. Merge all runs into one table and heatmap.
4. Follow-ups:
   - Does real PhotoDNA react to black frames? (Use `coco-val.csv`.)
   - Probe NeuralHash's sensitivity to the position of the occluding square.
