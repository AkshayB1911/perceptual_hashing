# Benign benchmark of all hashes: results, checks and explanations

*BTP: Perceptual Hashing. Written 2026-11-03. Combines Run 1 (`hashbench/results/coco_run1`) and
Run 2 (`hashbench/results/coco_run2`). Supersedes `benchmark_findings_run1.md`.*

## Main conclusions

1. **No hash wins everywhere.** DinoHash leads on geometric edits, CertPHash and the classical
   hashes on compression and noise, and PDQ-dihedral on flips (F13, F14, F21).
2. **The ranking depends on the false-match rate.** DinoHash leads at FPR 1e-2. CertPHash adv
   leads at 1e-4, where trained DinoHash drops to 75% (F21).
3. **Untrained DinoHash beats trained DinoHash on benign edits**: 94.3% against 89.7% at FPR 1e-3.
   Both match copies equally well at the same number of bits. Trained DinoHash loses because it
   produces more false matches, so it needs a stricter threshold (F16). This doesn't show that
   fine-tuning is harmful: the two models also have different final layers (F17), and this
   benchmark doesn't test adversarial robustness, which is what fine-tuning targets.
4. **DinoHash's false matches look semantic.** All 12 of its closest pairs of different images
   show the same kind of subject, such as two zebra photos (F18). This helps explain why the
   false-match formula in the DinoHash paper is badly wrong on real images: at its 1e-3 threshold,
   we measure 2.3% (F19).
5. **DinoHash has a 14-pixel patch-grid weakness.** A 1 px shift already changes about 5 bits, and
   shifts by a whole patch change far fewer bits than shifts by part of one (F15).
6. **Hashes from different families fail independently.** Matching if any of three hashes matches
   gives about 98% at a measured FPR of about 1e-3, higher than any single hash (F22).

---

## Setup

- **Hashes (12):**
  - classical: pHash, PDQ, PDQ-dihedral;
  - learned: NeuralHash; CertPHash in six variants (certified ε = 1, 2, 4, 8 /255, plus normal
    training `base` and adversarial training `adv`);
  - DinoHash ViT-S/14, 96 bits: the released fine-tuned model (`dinohash`, "trained") and original
    DINOv2 with the repo's PCA head (`dinohash_untrained`, "untrained").
- **Data:** COCO val2017.
  - Impostor pairs (different images): 200,000 pairs from all 5,000 images. **The pairs are the
    same for every hash.**
  - Attack suite: 23 attack types with 88 settings, on the 1,000 images in `data/splits/V1k.txt`.
    That gives 88,000 genuine distances per hash.
- **Thresholds:** for each hash, the largest threshold whose false-match rate (FPR) on its own
  impostor pairs doesn't exceed the target. The main target is 1e-3; 1e-2 and 1e-4 are also
  reported. FPR 1e-4 rests on only about 20 impostor pairs per hash.
- **Machines:** Run 1 on the laptop (Ryzen 5 5600H, Windows). Run 2 on a desktop (Intel i7-12700,
  Ubuntu), taking 2 h 30 min. Check 14 shows the machines give identical DinoHash results.

### Where the data is

| What | Path |
|---|---|
| **Combined comparison** | `hashbench/results/comparison/` |
| Raw distances, configuration, log | `hashbench/results/coco_run{1,2}/{genuine,impostor,hashes}.csv`, `config.json`, `run.log` |
| Report tables and plots per run | `overall.csv`, `summary.csv`, `attack_curves.png`, `mean_distance_curves.png`, `distance_hist.png`, `roc.png` |
| Analysis tables per run | `hashbench/results/coco_run{1,2}/analysis/` |
| DinoHash probes | `coco_run2/analysis/probe_patch_grid.csv`, `closest_pairs_dinohash.png`, `closest_pairs_other_hashes.csv` |
| Cross-machine check | `hashbench/results/crosscheck_{laptop,linux}_dinohash/` |

The comparison folder contains:
- `headline.csv`, `family.csv`, `per_setting.csv`: the match rates;
- `achieved_fpr.csv`: the false-match rate each threshold really gives;
- `bootstrap.csv`, `bootstrap_differences.csv`: 95% intervals;
- `binomial_vs_observed.csv`: measured false matches against the paper's formula;
- `bits_trained_vs_untrained.csv`: bits changed per setting;
- `impostor_corr.csv`, `image_difficulty_corr.csv`: whether hashes make the same mistakes;
- `ensemble.csv`: combined hashes;
- `family_heatmap.png`, `roc_low_fpr.png`: thesis figures.

**To reproduce** (from `hashbench/`):

```
python -m hashbench.report --run results/coco_run2 --fpr 0.001 0.01 0.0001
python scripts/analyze_run.py --run results/coco_run2 --fpr 0.001
python scripts/analyze_closest_pairs.py --run results/coco_run2 --images data/coco/val2017
python scripts/probe_dinohash.py --run results/coco_run2 --other-run results/coco_run1 --images data/coco/val2017 --list data/splits/V200.txt --n 50
python scripts/compare_all_runs.py --runs results/coco_run1 results/coco_run2 --out results/comparison
```

The Run 1 equivalents are the first three commands with `coco_run1`, plus
`python scripts/probe_invariances.py --run results/coco_run1 --images data/coco/val2017 --list data/splits/V200.txt --n 50`.

---

## 1. Results

### Overall

**Mean match rate over all 88 settings (%)**, sorted by FPR 1e-3.

| Hash | FPR 1e-2 | FPR 1e-3 | 95% interval | Exactly 1e-3 | FPR 1e-4 | Equal-family mean | AUC |
|---|---|---|---|---|---|---|---|
| DinoHash untrained | 98.7 | **94.3** | 93.0–94.5 | 94.3 | 83.0 | 89.1 | 0.999 |
| CertPHash adv | 94.0 | **91.8** | 91.6–92.0 | 91.8 | 90.2 | 83.2 | 0.988 |
| CertPHash ε=1 | 93.9 | **91.3** | 90.9–91.6 | 91.3 | 88.4 | 82.8 | 0.988 |
| DinoHash trained | 98.0 | **89.7** | 88.2–91.5 | 91.1 | 75.4 | 84.4 | 0.999 |
| NeuralHash | 94.2 | **89.6** | 88.5–89.9 | 89.8 | 83.0 | 79.7 | 0.990 |
| CertPHash base | 92.5 | **86.3** | 84.9–87.3 | 86.3 | 75.6 | 78.8 | 0.987 |
| CertPHash ε=8 | 90.4 | **85.0** | 84.1–85.7 | 85.0 | 78.7 | 77.5 | 0.978 |
| PDQ-dihedral | 82.2 | **80.9** | 80.5–81.0 | 81.0 | 79.8 | 90.0 | 0.923 |
| pHash | 83.8 | **80.2** | 80.1–82.0 | 81.6 | 78.3 | 73.6 | 0.946 |
| CertPHash ε=2 | 91.8 | **79.6** | 77.1–81.5 | 79.6 | 71.5 | 74.4 | 0.987 |
| PDQ | 80.3 | **79.3** | 79.2–79.4 | 79.5 | 78.6 | 73.7 | 0.916 |
| CertPHash ε=4 | 89.6 | **78.2** | 75.8–79.8 | 78.2 | 68.9 | 73.1 | 0.985 |

- **95% interval:** resamples images and recalibrates the thresholds each time, so it includes
  threshold uncertainty.
- **Exactly 1e-3:** interpolates between the two nearest thresholds, so all hashes sit at the same
  FPR (see "Reading these numbers").
- **Equal-family mean:** the six attack families count equally, so attacks with more strength
  levels don't dominate.

**Differences that the intervals support, at FPR 1e-3** (paired, in percentage points)

| Comparison | Difference | 95% interval | Reading |
|---|---|---|---|
| DinoHash untrained − trained | +4.6 | +1.9 to +5.2 | Real |
| DinoHash untrained − CertPHash adv | +2.5 | +1.2 to +2.8 | Real |
| DinoHash trained − CertPHash adv | −2.0 | −3.7 to −0.2 | Barely real |
| DinoHash trained − NeuralHash | +0.2 | −1.3 to +2.8 | **Tie** |
| CertPHash adv − ε=1 | +0.5 | +0.3 to +0.8 | Real but small |
| DinoHash trained − pHash | +9.5 | +7.7 to +11.3 | Real |

### By attack family

**Match rate at FPR 1e-3** (`family_heatmap.png`)

| Hash | Compression | Filtering/noise | Photometric | Geometric | Flips | Overlays |
|---|---|---|---|---|---|---|
| DinoHash untrained | 0.94 | 0.93 | 0.99 | **0.95** | 0.64 | 0.89 |
| CertPHash adv | 1.00 | 1.00 | 1.00 | 0.80 | 0.21 | 0.97 |
| CertPHash ε=1 | 1.00 | 1.00 | 1.00 | 0.79 | 0.21 | 0.97 |
| DinoHash trained | 0.91 | **0.83** | 0.98 | **0.90** | 0.58 | 0.86 |
| NeuralHash | 0.98 | 0.98 | 0.93 | 0.86 | 0.31 | **0.72** |
| CertPHash base | 1.00 | 1.00 | 1.00 | 0.65 | 0.15 | 0.93 |
| CertPHash ε=8 | 1.00 | 1.00 | 0.96 | 0.64 | 0.12 | 0.93 |
| PDQ-dihedral | 1.00 | 1.00 | 1.00 | 0.40 | **1.00** | 1.00 |
| pHash | 1.00 | 1.00 | 1.00 | 0.46 | **0.00** | 0.96 |
| CertPHash ε=2 | 1.00 | 1.00 | 1.00 | 0.45 | 0.10 | 0.92 |
| PDQ | 1.00 | 1.00 | 1.00 | 0.42 | **0.00** | 1.00 |
| CertPHash ε=4 | 1.00 | 0.998 | 0.99 | 0.41 | 0.08 | 0.90 |

### Reading these numbers

- **"Same FPR" is only approximately the same.** Hashes measured in whole bits can only move their
  threshold one bit at a time, and the method never goes over the target. So their real FPR is
  below 1e-3:

  | Hash | Real FPR at the 1e-3 threshold |
  |---|---|
  | All CertPHash variants | 0.00100 |
  | DinoHash untrained | 0.00099 |
  | NeuralHash | 0.00092 |
  | PDQ-dihedral | 0.00089 |
  | DinoHash trained | 0.00080 |
  | PDQ | 0.00061 |
  | pHash | **0.00018** |

  This understates pHash and trained DinoHash by about 1.4 points each. The "Exactly 1e-3" column
  corrects it. This caveat applies to all earlier Run 1 numbers too.
- **Treat small gaps as ties.** Trained DinoHash vs NeuralHash is the clearest example.
- **Averages hide weak spots.** The lowest match rate on any single setting is 28% for untrained
  DinoHash (vertical flip), 10% for trained DinoHash (rescale to 10%), 6.5% for NeuralHash (25%
  occlusion) and 3.2% for CertPHash adv (corner crop keeping 70%).

---

## 2. Correctness checks (does the data behave as it should?)

| # | Check | Expected | Observed | Verdict |
|---|---|---|---|---|
| 1 | Row counts | 1000 × 88 per hash; 200k impostor pairs per hash | 88,000 / 200,000 for all 12, no failed attacks | ✅ |
| 2 | Distance between unrelated images, binary hashes | ≈ 0.5 if bits are balanced | NeuralHash 0.496, PDQ 0.500, pHash 0.491, DinoHash untrained 0.481, trained 0.460 | ✅ See checks 3 and 13 |
| 3 | pHash bit balance | – | One bit is **always 1** | ✅ Expected: it is the DC coefficient (mean brightness), always above the median. imagehash has the same property, so pHash carries **63** useful bits |
| 4 | Grayscale, pHash | Distance exactly 0 (pHash only uses luma) | 100% exactly 0 | ✅ |
| 5 | Grayscale, PDQ | ≈ 0 | 95% exactly 0, all matched | ✅ PIL rounds luma to integers; PDQ uses float luma, so a few bits flip |
| 6 | Aspect-ratio stretch | ≈ no effect: every hash resizes to a fixed square | 100% matched for all 12; exactly unchanged for 95% (pHash), 41% (DinoHash trained), 31% (untrained) | ✅ |
| 7 | Grayscale, hue and saturation on photos that are already grayscale | No change | CertPHash exactly 0 for 2.8% of images; both DinoHash models 2.9% on grayscale | ✅ Matches COCO's share of grayscale photos |
| 8 | PDQ-dihedral on flips | 100% matched (8 orientations derived from the DCT) | 100%; exactly 0 in 16% (horizontal) and 45% (vertical) | ✅ Small differences come from the filter and decimation not being exactly symmetric |
| 9 | pHash and PDQ on flips | A flip negates every DCT coefficient with an odd frequency, so about half the bits change | Probe: distance 0.49–0.50, the same as unrelated images | ✅ Theory confirmed |
| 10 | NeuralHash against the published curves (Struppek et al., `neuralHash/images/transformations.png`, ImageNet) | Brightness 1.25 → ≈0.105, 1.5 → ≈0.23; contrast 1.25 → ≈0.057, 1.5 → ≈0.127; rotation 4° → ≈0.08, 16° → ≈0.21, 64° → ≈0.39; ≈0 when darkening | Brightness 1.3 → 0.132, 1.6 → 0.267; contrast 1.3 → 0.075, 1.6 → 0.153; rotation 5° → 0.109, 20° → 0.27, 45° → 0.393; darkening ≤ 0.003 | ✅ Independent confirmation of our NeuralHash and attack code |
| 11 | CertPHash against its paper | Reported ROC-AUC for C-PDNA under benign transformations ≈ 0.98 | 0.988 (ε=1) | ✅ |
| 12 | Statistical precision | – | Per setting: 95% interval ≤ ±3.1 points (n = 1,000). Per family: mostly ≤ ±1 point | ✅ |
| 13 | DinoHash mean impostor distance, explained by bit skew | Mean distance = Σ 2p(1−p) for bit frequencies p | Predicted 44.17 and 46.16 bits; measured 44.19 and 46.19 | ✅ The values below 0.5 are entirely explained by unbalanced bits |
| 14 | Run 2 machine vs laptop | Identical hashes | 50 images compared, 24 of them with all attacks (2,112 genuine distances per hash): 100% identical, 0 bits different | ✅ |
| 15 | Same inputs in both runs | Same images, attacks and impostor pairs | Identical keys; 200,000 of 200,000 impostor pairs shared | ✅ |
| 16 | Run 2 internal consistency | Impostor distances follow from `hashes.csv`; returned tables follow from raw data | 0 differences; all report and analysis tables recompute exactly on the laptop | ✅ |
| 17 | Grayscale vs saturation 0 on DinoHash | The same image, so the same result | 8.9 / 9.4 bits and 95.7% / 98.7% on both attacks | ✅ |
| 18 | Hash names not swapped | Laptop models reproduce each name's hashes | Identical under each name (check 14) | ✅ Untrained > trained is not a labelling mistake |
| 19 | Severity trend, DinoHash | Bits changed grow with strength | True everywhere except small shifts: 2% changes 7.9 bits and 5% changes 7.5 | ✅ Explained by the patch grid (F15) |
| 20 | DinoHash AUC against the early smoke test | Similar | 0.999 both | ✅ |

**Comparison with the DinoHash paper** (`DinoHash.pdf`). We can't reproduce its numbers directly,
because it evaluates a ViT-B/14 96-bit model on a combined pipeline (60% centre crop, then JPEG 30,
then one edit, then random erasing and text) on other data. Its statements compare as follows:

| Paper | Ours | Verdict |
|---|---|---|
| DINOHash outperforms NeuralHash across its transformation pipelines | Better on hard crops (50% centre crop: 58% vs 33%), occlusion, large rotation and shift, flips. Worse on blur, impulse noise, downscaling | Compatible: its pipelines all start with a heavy crop, which favours DinoHash. A tie on our mean |
| False-positive rate follows Binomial(96, 0.5) | Far heavier tail (F19) | **Not supported on COCO** |
| Trained much better than untrained under attack | Not tested yet; the paper gives no benign comparison of the two | Open. Our result on benign edits is new (F16) |

The CertPHash paper reports AUC 0.84 for NeuralHash; we measure 0.990. Their transformation set,
strengths and dataset differ from ours, so the two numbers measure different things. Point this out
if both are cited.

---

## 3. Findings: DinoHash and the combined comparison

### F13. DinoHash is the most robust hash to geometric change

| Setting (match rate, FPR 1e-3) | DinoHash untrained | DinoHash trained | NeuralHash | CertPHash adv | PDQ |
|---|---|---|---|---|---|
| Centre crop keeping 50% | **78%** | 58% | 33% | 6% | 0.4% |
| Corner crop keeping 70% | **89%** | 77% | 57% | 3% | 0.1% |
| Translate 20% | **96%** | 91% | 55% | 8% | 0% |
| Rotate 45° | **43%** | 24% | 9% | 13% | 0.1% |
| Rotate and crop 20° | **82%** | 62% | 51% | 32% | 0.3% |
| Horizontal flip | **100%** | 99% | 48% | 33% | 0% |

- **Why (hypothesis):** DINOv2 was trained to give similar features to different crops of one image
  (the paper cites this invariance), and its class token summarises the whole image instead of
  fixed positions. Not probed separately.

### F14. DinoHash is the only family that loses matches on compression, blur and noise

This changes F1, which held for the Run 1 hashes only.

| Setting (match rate, FPR 1e-3) | DinoHash untrained | DinoHash trained | NeuralHash | CertPHash adv, PDQ |
|---|---|---|---|---|
| Rescale to 10% | 35% | **10%** | 80% | 100% |
| Gaussian blur σ=5 | 55% | **17%** | 77% | 100% |
| Gaussian blur σ=3 | 93% | 63% | 99% | 100% |
| Salt and pepper 10% | 63% | **37%** | 97% | 100% |
| Median filter 7 | 95% | 79% | 99.9% | 100% |
| JPEG quality 10 | 94% | 88% | 100% | 100% |
| Gaussian noise 40 | 93% | 88% | 99.7% | 100% |

- **Why (partly explained):** PDQ and CertPHash work from a 64×64 input. Rescaling a 512 px image to
  10% leaves about 51 px, so they lose almost nothing. DinoHash uses a 224×224 input, so these edits
  remove detail that it sees.
- **Input size isn't the whole story:** NeuralHash uses a larger input (360×360) and holds up much
  better. So DinoHash's sensitivity comes from the model as well. We have not isolated why.
- **Note:** the paper lists blur among the edits DINOv2 is invariant to. Bits do change steadily
  with σ here: 4.2 at σ=1 and 23.9 at σ=5 for the trained model.

### F15. DinoHash's patch grid: small shifts cost bits, whole-patch shifts cost far fewer

- **Observation:** any small geometric change immediately costs about 5–8 bits, then grows slowly.
  Rotating 1° changes 5.5 bits, a 2% border 6.8, and a 2% shift 7.9. A 5% shift changes *fewer* bits
  (7.5) than a 2% shift. Blur at σ=0.5 changes only about 1 bit.
- **Hypothesis:** DinoHash cuts its input into 14×14 patches, and a shift by part of a patch changes
  every patch's content. In the benchmark, 2% is about 4.5 px and 5% is about 11 px, which is closer
  to 14. So shifts by exactly 14 or 28 px should change fewer bits.
- **Probe** (`probe_patch_grid.csv`, 50 images, shifting the 224×224 input; mean bits changed,
  standard error at most 0.83):

  | Shift (px) | Trained, black fill | Trained, wrap-around | Untrained, black fill | Untrained, wrap-around |
  |---|---|---|---|---|
  | 1 | 4.7 | 5.3 | 5.6 | 6.6 |
  | 4 | 7.3 | 8.6 | 9.2 | 11.1 |
  | 7 | 7.9 | 9.4 | 9.3 | 12.4 |
  | 10 | 7.1 | 9.8 | 8.7 | 12.0 |
  | **14** | **4.9** | **7.3** | **5.2** | **6.5** |
  | 18 | 8.3 | 11.2 | 10.2 | 14.5 |
  | 21 | 8.9 | 11.9 | 10.0 | 15.2 |
  | 25 | 8.0 | 11.3 | 9.6 | 14.8 |
  | **28** | **6.7** | **9.6** | **7.4** | **8.8** |

- **Prediction confirmed:** bits drop sharply at 14 and 28 px in all four columns. In three of the
  four columns, shifting 14 px costs about as much as shifting 1 px.
- **Why it isn't zero at 14 px:** a shifted patch gets a different position embedding, and content
  enters and leaves at the edges.
- **Consequence:** at a 17-bit threshold, a tiny shift already uses about a third of the budget.
- **Research idea:** averaging the hash over sub-patch shifts might remove this cost, and it fits
  the randomized-smoothing direction (Idea 2 in `thesis_research_plan.md`). Not tested.

### F16. Trained DinoHash loses to untrained because of false matches, not because it misses copies

Match rate and FPR at the same number of bits:

| Threshold (bits) | Trained: match | Trained: FPR | Untrained: match | Untrained: FPR |
|---|---|---|---|---|
| 17 | **89.7%** | 0.00080 | 89.3% | 0.00022 |
| 18 | 91.2% | 0.00102 | 90.8% | 0.00036 |
| 21 | **94.3%** | **0.00200** | **94.3%** | **0.00099** |

- **At 21 bits, both match exactly 94.3% of copies, but trained DinoHash has twice the false matches.**
  To stay within FPR 1e-3, it must use 17 bits while untrained can use 21. That one difference
  produces the gap.
- Trained DinoHash actually changes **fewer** bits on average (8.14 vs 8.51), on 65 of the 88
  settings.
- **Where the extra false matches come from:**
  - **Unbalanced bits.** The trained model has 7 bits outside 0.2–0.8 frequency, with extremes 0.12
    and 0.98. The untrained model has 1 such bit, which is nearly constant (0.001). Check 13 shows
    this fully explains the mean: different images are 2 bits closer on average (44.2 vs 46.2).
  - **A heavier tail beyond that.** The threshold gap is 4 bits, so about 2 bits come from the tail.
- **How much is real:** at exactly FPR 1e-3 the gap is about 3.3 points (94.3 vs 91.1) instead of 4.6,
  because the 18-bit threshold just misses the target. At FPR 1e-2 it shrinks to 0.7 points. So the
  earlier smoke-test remark that the two "behave the same" only holds at loose thresholds.

### F17. "Trained vs untrained" isn't only the effect of fine-tuning

- **Same image, different hash:** the two models' hashes differ on **48.6 of 96 bits**, about what
  unrelated hashes would give, although the paper's training aims to keep the original outputs.
- **Cause found: the final layers differ.**
  - Head: the released ViT-S head is a different projection from the repo's PCA file. Its best
    |cosine| with any PCA row averages 0.22, as already noted in `imp_dinohash.md`.
  - Backbone: weights moved by 8.75% in relative size.
- **But the two still see images similarly:** they agree on which different images look close
  (Spearman 0.30, against ≤ 0.04 with any other family, F22). 97% of trained DinoHash's 100 closest
  pairs are in the untrained model's closest 1%.
- **Consequence for the thesis:** F16 can't be attributed to fine-tuning alone. The head alone might
  cause the bit skew. Any robustness-vs-privacy comparison using these two models (Idea 1) has the
  same problem and should separate the two changes.
- **How to separate them:** build the two mixed models (trained backbone with PCA head, original
  backbone with trained head) and rerun. All the files are already on the laptop.

### F18. DinoHash's closest different-image pairs share a subject

- **Observation** (`closest_pairs_dinohash.png`): all 12 closest pairs (6–11 bits apart) show the
  same kind of subject: zebras (5 pairs), elephants (4), motorbikes, skiers and pizza. They aren't
  copies of one photo.
- **Other hashes don't consider these pairs close** (`closest_pairs_other_hashes.csv`). For the 100
  closest pairs under trained DinoHash:

  | Other hash | Median rank among all 200,000 pairs | In that hash's closest 1% |
  |---|---|---|
  | DinoHash untrained | Top 0.17% | 97% |
  | NeuralHash | Top 26% | 5% |
  | CertPHash ε=1 | Top 25% | 6% |
  | CertPHash adv | Top 35% | 9% |
  | PDQ | Top 40% | 0% |
  | pHash | Top 55% | 1% |

- **Not the Run 1 black-frame shortcut:** both-framed pairs are 0% of DinoHash's 100 closest, and
  both-darkest-10% pairs are 1%.
- **Why (hypothesis):** DINOv2 features are built to capture what an image shows, so two photos of
  zebras land close together. That helps matching copies under heavy crops (F13), and it also causes
  these false matches.
- **Consequence:**
  - False matches on a real service depend on its content. A collection with many similar photos
    could have a much higher FPR than COCO.
  - For the thesis, a DinoHash false match may reveal the subject of an image. This supports the
    robustness-vs-privacy storyline (Idea 1).
- **Limit:** this comes from looking at 12 pairs, not from labels. COCO's category annotations would
  test it properly. They aren't on disk yet and would need a download.

### F19. The paper's false-match formula is far too optimistic on real images

The DinoHash paper assumes independent 50/50 bits, so that Binomial(96, 0.5) gives the FPR.

| Hash | Threshold for FPR 1e-3, measured | Binomial's prediction there | Measured ÷ predicted | Binomial's own 1e-3 threshold | Measured FPR at it |
|---|---|---|---|---|---|
| DinoHash trained | 17 bits | 5.0 × 10⁻¹¹ | **16 million ×** | 32 bits | **2.3%** |
| DinoHash untrained | 21 bits | 1.4 × 10⁻⁸ | 73,000 × | 32 bits | **1.3%** |
| NeuralHash | 26 bits | 4.0 × 10⁻⁶ | 229 × | 32 bits | 0.83% |
| pHash | 16 bits of 64 | 3.9 × 10⁻⁵ | 4.8 × | 19 bits | 0.12% |
| PDQ | 100 bits of 256 | 2.8 × 10⁻⁴ | 2.2 × | 102 bits | 0.13% |

- **Consequence:** picking a threshold from the formula gives 13–23 times more false matches on
  DinoHash than intended. Thresholds must be measured on images.
- **Why:** the bits aren't balanced (F16), and semantic similarity moves many bits together (F18).
  Average bit correlation is low (0.045), so the dependence shows up in the tail and not as a simple
  pairwise correlation. PDQ, which targets uniform bits, is close to the formula.
- **Limit:** this is measured on COCO at FPR ≥ about 1e-4, with the released ViT-S. The paper's own
  numbers use ViT-B.
- PDQ-dihedral is left out of this table: it takes the smallest distance over 8 orientations, so the
  formula doesn't apply to it. Its row in `binomial_vs_observed.csv` should be ignored.

### F20. Fine-tuning changes *which* edits cost bits

Mean bits changed, trained vs untrained (`bits_trained_vs_untrained.csv`):

| Trained changes fewer bits | Trained | Untrained | | Trained changes more bits | Trained | Untrained |
|---|---|---|---|---|---|---|
| Gaussian noise 20 | 5.7 | 8.5 | | Gaussian blur σ=5 | 23.9 | 21.0 |
| Gaussian noise 10 | 3.2 | 5.0 | | Gaussian blur σ=3 | 15.9 | 13.0 |
| Occlusion 10% | 13.4 | 15.4 | | Gaussian blur σ=2 | 11.5 | 8.8 |
| JPEG 30 | 4.5 | 6.2 | | Brightness 0.5 | 4.6 | 2.7 |

- **Why (hypothesis):** the paper fine-tunes against small additive pixel perturbations, and noise
  and JPEG are the nearest benign relatives. Nothing in that training rewards blur or brightness
  robustness. The head difference (F17) could also contribute.

### F21. The ranking changes with the false-match rate and with how attacks are weighted

(`roc_low_fpr.png`)

| View | Best | Then |
|---|---|---|
| FPR 1e-2, mean of settings | DinoHash untrained 98.7 | trained 98.0, NeuralHash 94.2 |
| FPR 1e-3, mean of settings | DinoHash untrained 94.3 | CertPHash adv 91.8, ε=1 91.3 |
| FPR 1e-4, mean of settings | CertPHash adv 90.2 | ε=1 88.4, DinoHash untrained 83.0 |
| FPR 1e-3, families equal | PDQ-dihedral 90.0 | DinoHash untrained 89.1, trained 84.4 |
| Settings ≥ 95% at FPR 1e-3 | CertPHash adv 78 of 88 | ε=1 77, DinoHash untrained 70 |

- **Why:**
  - DinoHash has the best AUC, but some different images are very close under it (6–9 bits), so it
    loses most as the FPR tightens. Its curve crosses CertPHash adv near FPR 4e-4 (untrained) and 1.1e-3 (trained).
  - PDQ-dihedral scores highest with equal families only because flips count as a full family and
    it gets exactly 100% there, though its geometric score is the lowest (0.40).
- **Consequence:** the thesis shouldn't name one overall winner. Report results at a stated FPR, by
  family. AUC alone misleads: untrained and trained DinoHash both show 0.999 and differ by about 8
  points at FPR 1e-4.

### F22. Different hash families fail independently, and combining them works

This measures what F11 suggested.

- **Independent mistakes.** Spearman correlations:

  | Pair | Which copies fail (per image) | Which different images look close |
  |---|---|---|
  | DinoHash trained / untrained | 0.51 | 0.30 |
  | CertPHash ε=1 / adv | 0.66 | 0.90 |
  | DinoHash vs any other family | ≤ 0.08 | ≤ 0.04 |
  | NeuralHash vs any other family | ≤ 0.12 | ≤ 0.07 |

- **Combined rule:** match if any hash matches, with the FPR budget split evenly between the hashes.
  False matches are measured directly, because all hashes share the same 200,000 pairs.

  | Combination | Measured FPR | Mean match | Geometric | Flips | Equal-family mean |
  |---|---|---|---|---|---|
  | *Best single: DinoHash untrained* | *0.00099* | *94.3* | *0.95* | *0.64* | *89.1* |
  | DinoHash trained + PDQ-dihedral | 0.0008 | 95.7 | 0.87 | 1.00 | 97.7 |
  | DinoHash trained + CertPHash adv | 0.0009 | 96.2 | 0.92 | 0.58 | 91.3 |
  | DinoHash untrained + PDQ-dihedral | 0.0007 | 97.1 | 0.91 | 1.00 | 98.5 |
  | DinoHash untrained + CertPHash adv | 0.0009 | 97.1 | 0.94 | 0.62 | 92.4 |
  | DinoHash trained + CertPHash adv + PDQ-dihedral | 0.0010 | 96.9 | 0.90 | 1.00 | 98.4 |
  | DinoHash untrained + CertPHash adv + PDQ-dihedral | 0.0009 | **97.7** | 0.93 | 1.00 | **98.8** |

- All combinations score 1.00 on compression, filtering and photometric, and at least 0.98 on
  overlays.
- **Limits:**
  - Thresholds are set and measured on the same pairs, so these are slightly optimistic. They need
    a held-out check.
  - An "any hash" rule makes it **easier** to force a false match, since an attacker only has to
    fool the weakest hash. Evasion gets harder and collisions get easier, which matters for Idea 4.

---

## 4. Findings: Run 1 hashes (updated with DinoHash where relevant)

### F1. Compression, noise, filtering and colour don't separate the Run 1 hashes
pHash, PDQ, CertPHash and (mostly) NeuralHash score about 100% on these. **DinoHash does not** (F14).
- **Why:** those hashes shrink the image heavily before hashing (pHash to 32×32, PDQ and CertPHash
  to 64×64, NeuralHash to 360×360 followed by a CNN). JPEG artefacts, noise and blur are fine detail
  that disappears in that step, and the low-frequency structure the hashes rely on survives.
- pHash and PDQ threshold at the median, which doesn't change when all pixels are scaled by the same
  factor. That's why pHash is *exactly* unchanged for 50% of brightness edits: the darkening ones.
- **Consequence:** among the Run 1 hashes, geometry, flips and overlays carry all the differences.

### F2. Geometric edits split classical and learned hashes
- Translating by 5%: PDQ 2.4%, pHash 17%, NeuralHash 100%, DinoHash 99–100%.
- Corner crop keeping 80%: PDQ 0.2%, pHash 0.7%, NeuralHash 94%, DinoHash 93–98%.
- **Why the classical hashes fail:** DCT basis functions are tied to fixed positions. A shift changes
  the phase of the low-frequency coefficients, so their signs relative to the median change.
- **Why PDQ fails more than pHash:** PDQ uses a 16×16 block of frequencies (up to 16 cycles across
  the image), while pHash uses the lowest 8×8. Higher frequencies make PDQ more sensitive to shifts.
- **Why NeuralHash survives:** a CNN with global pooling is roughly shift-tolerant, and it was
  trained to be robust to such edits.

### F3. Flips: classical hashes see a completely new image; learned hashes see horizontal flips more often
- pHash and PDQ: 0% (explained by check 9). PDQ-dihedral: 100%, by design.
- Every learned model matches horizontal flips much more often than vertical ones:
  - DinoHash untrained 100% vs 28%, trained 99% vs 17% (7.7 vs 25.6 bits for trained);
  - NeuralHash 48% vs 14%;
  - CertPHash ε=1 32% vs 10%.
- **Why (hypothesis):** horizontal flips are a standard training augmentation, and left-right
  mirrored photos still look natural. Upside-down photos are rare, so nothing in training encourages
  robustness to vertical flips. The asymmetry is consistent across 9 models, and it is strongest in
  DinoHash, whose self-supervised training is built around such invariances.

### F4. NeuralHash ignores darkening and contrast reduction, but not brightening or gamma
- Brightness 0.5 and 0.7: 100%. Brightness 1.6: 55%. Gamma 2.0: 49%. Gamma 0.5: 80%.
- **Why:** the network starts with convolution → **instance normalisation**. That pair removes any
  change of the form a·x + b to the input.
- **Probe** (`probe_neuralhash_affine.csv`, 50 images):

  | Edit | Bits changed (of 96) |
  |---|---|
  | Scaling around mid-grey | **0.00** |
  | Plain darkening | 0.08–0.22 (the convolution pads with zeros, which isn't affine-invariant) |
  | Gamma 2.0 | 27.8 (non-linear) |
  | Brightness 1.6 with clipping | 27.4 (non-linear) |

  Brightening fails because PIL clips values at 255, which isn't affine.
- This matches Struppek et al.'s published curves (check 10).
- DinoHash holds these edits at 98–100% on the same settings.

### F5. NeuralHash is fragile to a grey square in the centre; pHash and PDQ are not
- Occlusion covering 10% of the image: NeuralHash 39%, DinoHash 77–81%, PDQ 100%. Covering 25%:
  NeuralHash 6.5%, DinoHash 42–56%, pHash 83%, PDQ 99%.
- **Why (hypothesis):**
  - The square is grey (128), close to the average image brightness, so the coarse brightness layout
    that pHash and PDQ measure barely changes.
  - The learned hashes encode image content. COCO objects tend to be in the middle, so a central
    square hides the main object. DinoHash's better score fits its whole-image features (F13).

  Not probed further.

### F6. The CertPHash ε sweep isn't monotonic, and the cause is a black-pixel shortcut in the ε=2 and ε=4 models
- **Observation:**
  - ε=2 and ε=4 match only 24–26% with a 2% black border or a 2% shift.
  - ε=1, ε=8 and adv stay at 100% on the same edits.
  - The same pattern explains their poor rotation results.
- **Ruling out a bug:** all six variants run through the same code, and their outputs were already
  verified bit-identical to the authors' implementation.
- **Probe** (`probe_certphash_edges.csv`): overwrite only the outer 1 px of the 64×64 input, with no
  geometric change.

  | Edge fill | ε=2 / ε=4 match rate |
  |---|---|
  | Black | **30%** |
  | Grey, white or replicated edge | 98% |

  The weakness is to **black** pixels specifically. Border, translation and rotation all add black
  pixels, so this single cause explains all three failures. The base model shows it to a lesser
  degree (82% at 1 px, 36% at 4 px). ε=1, ε=8 and adv don't show it.
- **The same flaw causes false matches between different images**
  (`black_frame_impostors.csv`, `dark_image_impostors.csv`):

  | Pair type | Share of all pairs | Share of the 100 closest pairs (base / ε=2 / ε=4) | Other hashes |
  |---|---|---|---|
  | Both photos black-framed (1% of COCO photos) | 0.01% | 11–12% | 0% |
  | Both among the darkest 10% | 1% | 14% / 24% / 34% | 1–8% |

  Example: the photos of a child with a cake (000000102820) and of a motorbike (000000226662) are
  among the closest pairs for three models. Their only shared feature is a black frame.
- **Interpretation:**
  - The ordering of the certified models on benign edits is driven by **habits specific to each
    training run** ("shortcuts"), not only by ε.
  - Only one checkpoint was released per ε, so the effect of ε can't be separated from run-to-run
    variation. This must be stated in the thesis.
  - Why black specifically is unknown. Two possibilities to test:
    - real PhotoDNA itself reacts to black borders, and the models copied it (testable with the real
      PhotoDNA labels in `coco-val.csv`);
    - an artefact of training.

### F7. CertPHash ε=8 has a narrow output range and is sensitive to large brightness shifts
- **Observations:**
  - Lowest CertPHash AUC (0.978).
  - Different images are closer together than for the other variants: mean distance 0.155 against
    about 0.21.
  - Darkening by half: 52% matched.
- **First hypothesis (ε=8 confuses dark images): refuted.**
  - Its closest pairs aren't dark (average brightness 106; all images: 114).
  - Its darkening failures are worst on the *brightest* photos (31% matched) and mildest on the
    darkest (78%).
- **Explanation consistent with the data:** halving a bright image moves pixel values by a large
  absolute amount. Different images map to close outputs under this model, so such a shift looks
  like "a different image".

### F8. A larger certified ε costs both fidelity and discrimination
- AUC falls steadily: 0.988 (ε=1) → 0.987 (ε=2) → 0.985 (ε=4) → 0.978 (ε=8). This holds even though
  the per-attack ranking isn't monotonic (F6).
- Combined with the earlier fidelity measurement (share of hashes within the PhotoDNA threshold of
  the real PhotoDNA hash: 96.9% at ε=1, 15.9% at ε=8), certified training at larger ε costs both
  closeness to PhotoDNA and the ability to tell images apart.
- This is direct evidence for the "price of robustness" storyline. F16 is a possible parallel on the
  DinoHash side, subject to F17.

### F9. Published thresholds are very conservative, except for the ε=2/4/8 CertPHash models
- At pHash 10/64, PDQ 31/256 and CertPHash L1 1800, there are zero false matches in 200,000 pairs
  for pHash, PDQ, PDQ-dihedral and CertPHash ε=1, adv and base. But these thresholds miss 25–35% of
  copies.
- At the paper's 1800 threshold, CertPHash ε=2, ε=4 and ε=8 **do** produce false matches: 7, 4 and 6
  of 200,000. That's about 2–4 × 10⁻⁵, which is significant at deployment scale (billions of
  comparisons). Two each for ε=2 and ε=4 are black-framed pairs (F6).
- DinoHash and NeuralHash have no comparable published threshold in our sources. DinoHash's formula
  gives thresholds that are too loose (F19).
- **Consequence:** calibrated thresholds are needed for a fair comparison. The published CertPHash
  threshold doesn't carry over to its larger-ε models.

### F10. With no false matches allowed, the hashes still catch 44–88% of copies
- Threshold just below the closest impostor pair (`zero_fpr.csv`):

  | Hash | Match rate |
  |---|---|
  | CertPHash adv | 88% |
  | CertPHash ε=1 | 82% |
  | PDQ-dihedral | 79% |
  | PDQ | 77% |
  | pHash | 76% |
  | NeuralHash | 75% |
  | CertPHash ε=8 | 73% |
  | CertPHash base | 69% |
  | CertPHash ε=4 | 64% |
  | CertPHash ε=2 | 63% |
  | DinoHash untrained | **59%** |
  | DinoHash trained | **44%** |

- DinoHash is last here because its closest different-image pair is only 6 bits apart (trained) or 9
  (untrained), so its threshold has to be extremely tight (F18).
- PDQ has the narrowest spread of distances between unrelated images (standard deviation 0.032,
  closest pair 0.352). That's why it holds up best as the FPR gets stricter.

### F11. Different hashes fail on different images
- How strongly the per-image match rates of two hashes are correlated (`image_difficulty_corr.csv`):

  | Pair | Correlation |
  |---|---|
  | Within families: PDQ / PDQ-dihedral | 0.81 |
  | Within families: CertPHash variants | 0.26–0.76 |
  | Within families: DinoHash trained / untrained | 0.51 |
  | Within families: pHash / PDQ | 0.39 |
  | NeuralHash vs anything else | ≤ 0.12 |
  | DinoHash vs any other family | ≤ 0.08 |
  | Classical vs CertPHash | ≈ −0.13 to 0.06 |

- Failures come from the hash, not from "hard images". F22 measures what combining them gains.

### F12. Bit statistics (Idea 6)

| Hash | Share of 1s per bit | Mean bit correlation | Note |
|---|---|---|---|
| PDQ | 0.46–0.53 | 0.013 | Close to ideal |
| NeuralHash | 0.40–0.60 | 0.053 (max 0.27) | Per-bit entropy sums to 95.4 of 96 |
| DinoHash untrained | 0.001–0.73 | 0.041 (max 0.29) | Entropy sum 93.0 of 96; one almost constant bit |
| DinoHash trained | 0.12–0.98 | 0.045 (max 0.23) | Entropy sum 89.8 of 96; two bits are 1 in about 98% of images |
| pHash | – | – | 63 useful bits (check 3) |

The entropy sums are upper bounds, since correlations reduce the true effective length further.

PDQ quality can't be studied on COCO: only 4 of the 1,000 attacked images have quality below 50.

---

## 5. Corrections to earlier statements

- **F1** said benign tests can't separate hashes on compression and noise. That holds only for the
  Run 1 hashes; DinoHash loses matches there (F14).
- **`imp_dinohash.md`** said trained and untrained DinoHash behave the same under benign
  transformations, based on 12 images at FPR 5%. That holds only at loose thresholds (F16). The file
  now carries a note.
- **Earlier CertPHash note:** "ε=8 does better than ε=2/4 on borders, unexplained" is explained (F6).
  ε=2 and ε=4 have the black-pixel shortcut and ε=8 doesn't; it isn't an effect of ε itself.
- **My first explanation for ε=8** (dark images) was tested and refuted (F7).
- **All Run 1 "FPR 1e-3" scores** carry the whole-bit caveat in Section 1. The numbers themselves are
  unchanged.

## 6. Limitations

- **DinoHash model:** the paper describes ViT-B/14 at 96 bits. The released 96-bit model is ViT-S/14,
  and that is what we tested. The released ViT-B file we have is 512 bits.
- **Trained vs untrained differ in head as well as backbone** (F17).
- **Benign edits only, one at a time.** No adversarial results yet, and no combined pipeline like the
  paper's.
- One released checkpoint per CertPHash ε, so effects of ε and of the individual training run can't
  be separated.
- COCO-specific rates: black frames are 1% of images, there are few low-quality images, objects tend
  to be in the centre, and some subjects (such as zebras) repeat often, which affects F18 and F19.
- Originals are downscaled to 512 px before attacking. Mean scores weight all settings equally.
- FPR 1e-4 thresholds rest on about 20 impostor pairs per hash.
- Combination results (F22) are set and measured on the same pairs.
- Still hypotheses: F3, F5, F13, F14 (the model-side part), F18's cause, F20 and the cause of the
  black-pixel shortcut. F15 and F4 are probed.

## 7. Next

1. **Phase C, adversarial evaluation:** the test of what DinoHash's fine-tuning is for. Trained vs
   untrained on the benign results here, set against attacks, is the core trade-off. Build
   `hashbench/adversarial.py` (model-agnostic PGD/APGD evasion and collision); run the full sweep on
   a free Kaggle GPU, since the laptop has no CUDA.
2. **Test F18 with labels:** do close pairs share COCO categories? Needs the annotations download
   (ask first).
3. **Paper's combined pipeline:** run crop + JPEG + one edit + erase + text, to compare with the paper
   directly.
4. **Optional Run 3:** ViT-B at 512 bits on V200 only.
5. **Follow-ups from Run 1:**
   - Does real PhotoDNA react to black frames? (Use `coco-val.csv`.)
   - Probe NeuralHash's sensitivity to the position of the occluding square.
6. **Ideas raised here:** sub-patch shift averaging (F15), and a held-out test of combined hashes
   under attack (F22, Idea 4).
7. **Deferred — separate head from backbone (F17):** rerun with the two mixed DinoHash models. No
   download needed. It refines how F16 is stated but doesn't block the adversarial work, so it is
   parked until after Phase C.
