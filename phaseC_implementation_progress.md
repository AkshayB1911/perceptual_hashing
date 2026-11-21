# Phase C (adversarial robustness): implementation progress

*BTP: Perceptual Hashing. Status as of 2026-11-22 (summary table, §8, §9); 2026-11-17 (updates, §6.1); the rest of §1–§9 dates from 2026-11-04. Phase C is defined in
`thesis_research_plan.md` §2; the benign results it builds on are in `benchmark_findings.md`.*

## Summary

| Part | Status |
|---|---|
| Plan reordered (adversarial work first, mixed DinoHash head/backbone rerun deferred) | ✅ Done |
| Attack module `hashbench/hashbench/adversarial.py` (PGD, APGD, evasion, collision) | ✅ Done; §6.1 issue fixed 2026-11-05 |
| Unit and smoke tests `hashbench/tests/test_adversarial.py` | ✅ 13 pass |
| Sweep script `hashbench/scripts/run_adversarial.py` (models × ε × mode, resumable) | ✅ Done |
| Single-image sanity check | ✅ Done |
| CPU timing pilot and ε pilot (10 images; 3 for DinoHash) | ✅ Done, **pilot only** |
| Fix best-iterate selection (§6.1) | ✅ Done (2026-11-05: selection by hash distance, APGD + small-step PGD ensemble, warm start across ε) |
| Kaggle package (code, models, images, thresholds, notebook) | ✅ Built (`kaggle/`) |
| Full sweep on `V200` (200 images, ε = 1, 2, 4, 8 /255) | ✅ Done: `results/adv_v200_kaggle` (DinoHash, CertPHash), `results/adv_v200_neuralhash` |
| Write-up of adversarial findings | ✅ `adversarial_testing_report.md` (§4 partly superseded, see below) |
| Reproduction of the DinoHash paper's Table 1 with the authors' own attack (50 images) | ✅ Done 2026-11-11: `dinohash_table1_reproduction.md` |
| All adversarial tables in one file, incl. the CertPHash paper comparison | ✅ `adversarial_results_tables.md` |
| Catalogue of the 12 further attacks (details, feasibility + novelty ranking, order) | ✅ Written 2026-11-16: `future_attacks.md`; #1 transfer done |
| Transfer attack (#1): 9 sources × 11 targets incl. pHash / PDQ, V200, with random-noise control | ✅ Done 2026-11-17: `transfer_attack_results.md`, `results/transfer_v200*` |
| NES black-box attack (#2): DinoHash ×2, NeuralHash, 6 CertPHash on V200 (Kaggle) | ✅ Done 2026-11-18: `nes_attack_results.md`, `results/nes_v200_*` |
| NES like-for-like check: DinoHash trained at NeuralHash's 26-bit threshold | ✅ 2026-11-18: at 26 bits Dino-T 41.5% / 6.5% evasion ε8/ε4 vs NH 5.0 / 0.5 (`nes_attack_results.md` §5.3) |
| NES in the Sun et al. 2026 setting: DinoHash ×2, NeuralHash, 6 CertPHash on V200 (Kaggle) | ✅ Done 2026-11-19: Dino-T 41.2%, Dino-U 54.7%, NH 24.5%, CertPHash ≤ 2.5% (`nes_sun_attack_results.md`, `results/nes_sun_v200_*`) |
| Sun et al. NES, 500-step variant (100 images; NeuralHash all budgets, DinoHash ℓ2 180) | ✅ Done 2026-11-21: NH 63/69/63%, Dino-T 84% at ℓ2 180 (`nes_sun_attack_results.md` §6, `results/nes_sun500_v100_*`) |
| NES on pHash + PDQ (CPU, separate Linux PC; package `offline/phash_pdq/`) | ✅ Done 2026-11-18: `results/nes_v200_classical/`, `nes_attack_results.md` §6 |
| Conclusion of the black-box work | DinoHash is easier to evade than NeuralHash in every setting tried (white-box, our NES, Sun et al. literal, Sun et al. 500 steps). Sun et al.'s opposite ranking could not be reproduced; their perturbations are larger than their stated settings produce (`nes_sun_attack_results.md` §5–6) |

The ε tables in §5 come from a pilot on very few images. They show that the pipeline works
and give early signals, but **they are not thesis results**. Thesis numbers are in
`adversarial_testing_report.md` and `dinohash_table1_reproduction.md`.

### Update 2026-11-11: DinoHash Table 1 reproduction

The authors' APGD was vendored unmodified and run on the first 50 `V200` images (bit accuracy %,
ε = 4 / 8 /255):

| Model | Paper | Ours |
|---|---|---|
| DinoHash trained | 72.2 / 64.8 | 68.9 / 56.1 |
| DinoHash untrained | 0.3 / 0.0 | 0.0 / 0.0 |
| NeuralHash (attacked correctly) | 0.1 / 0.0 | 75.6 / 58.4 |
| NeuralHash, the authors' script logic | 0.1 / 0.0 | 0.0 / 0.0 |

The DinoHash numbers reproduce; the NeuralHash number does not. The authors' committed
`neuralhash_attack_results.py` attacks and scores DinoHash but compares the result with NeuralHash
bits. Details, GitHub links and line numbers are in `dinohash_table1_reproduction.md`.

Corrections to `adversarial_testing_report.md` §4:
- the released ViT-S model **is** the paper's model (its head equals `PCA_96`), so the
  "ViT-B vs ViT-S" explanation no longer applies;
- the 50-image numbers replace the 5-image probe.

### Update 2026-11-05: full `V200` white-box sweep (thesis numbers)

This replaces the pilot tables in §5.3. All tables (incl. CertPHash at L1 1800, L1 moved,
SSIM) are in `adversarial_results_tables.md`; discussion is in `adversarial_testing_report.md`.

**Setup**

| Item | Value |
|---|---|
| Images | `V200` (200 COCO val2017 images), n = 200 per cell |
| Hashes | DinoHash trained / untrained, NeuralHash, CertPHash base / adv / ε1 / ε2 / ε4 / ε8 (9 models) |
| Threat model | White-box, ℓ∞, ε in /255 on [0, 1] pixels |
| Attack | `--method ensemble`: APGD and small-step PGD (step min(ε/10, 0.25/255)) from the same random start, the better result kept per image; 100 steps per ε; warm start from the previous ε |
| Best iterate | Chosen by hash distance on the 8-bit rounded image (the §6.1 fix) |
| Modes and budgets | Evasion ε = 1, 2, 4, 8; collision (random target i → i + n/2) and closest-target collision ε = 1, 2, 4, 8, 16, 32 |
| Thresholds | Calibrated FPR 1e-3 (§4.1) |
| Hardware and time | DinoHash + CertPHash on a Kaggle T4: about 49 s/img per DinoHash model, 0.8 s/img per CertPHash model, 5.7 GPU-h in total. NeuralHash on the laptop CPU: about 111 s/img, 6.2 h |
| Monotonicity check | `monotonicity.csv`: **0 cases** where a larger ε did worse (the §6.1 problem is gone) |
| Data | `hashbench/results/adv_v200_kaggle/`, `hashbench/results/adv_v200_neuralhash/`; plots in `results/adv_v200_plots/`, `results/benign_vs_adversarial.pdf` |

**Evasion success (%)**

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| DinoHash untrained | 100 | 100 | 100 | 100 |
| DinoHash trained | 97.5 | 100 | 100 | 100 |
| NeuralHash | 0.5 | 0.5 | 18.0 | 94.5 |
| CertPHash base | 0 | 0 | 0 | 11.5 |
| CertPHash adv / ε1 / ε2 / ε4 | 0 | 0 | 0–1.0 | 1.0–3.5 |
| CertPHash ε8 | 0 | 0 | 0 | 0 |

**Bit accuracy under evasion (% bits unchanged)**

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| DinoHash untrained | 20.6 | 2.3 | 0.1 | 0.0 |
| DinoHash trained | 45.6 | 22.3 | 10.3 | 3.7 |
| NeuralHash | 94.4 | 88.9 | 78.4 | 59.2 |

**Collision success (%), random target / closest target**

| Hash | ε=2 | ε=8 | ε=32 |
|---|---|---|---|
| DinoHash untrained | 100 / 100 | 100 / 100 | 100 / 100 |
| DinoHash trained | 92.0 / 97.5 | 99.5 / 100 | 100 / 100 |
| NeuralHash | 0 / 63.5 | 16.5 / 93.5 | 89.5 / 98.0 |
| CertPHash base | 1.0 / 33.0 | 25.5 / 100 | 100 / 100 |
| CertPHash ε8 | 1.0 / 20.5 | 1.5 / 50.5 | 66.5 / 98.5 |

**Main findings**
1. **DinoHash (both versions) is the easiest to break**: evaded from ε = 1–2/255 and collided
   almost always. The DinoHash paper's robustness holds only against its weaker attack
   (`dinohash_table1_reproduction.md`).
2. **NeuralHash** needs ε = 4–8/255 for evasion. The authors' BCE APGD did better on it at ε=4
   (36% vs our 18%), so our NeuralHash numbers are probably a little too low; add APGD-BCE to
   the ensemble.
3. **CertPHash ε8 resists evasion best** (0% at ε=8 at the calibrated threshold), and the
   CertPHash paper's Table 4 reproduces at L1 1800 (within about 10 points, same ranking).
4. **Evasion robustness ≠ collision robustness:** closest-target collisions succeed on every
   hash at large ε (CertPHash ε8: 50.5% at ε=8, 98.5% at ε=32).
5. **The threshold matters a lot** for CertPHash: at L1 1800 evasion succeeds much more often
   (e.g. base 99% at ε=8) than at the calibrated thresholds (L1 2581–5435).

**Further attacks** (black-box, transfer, EOT, joint evasion, …) are catalogued and ranked in
`future_attacks.md`.

---

## 1. Plan change

In `benchmark_findings.md` §7 ("Next"):
- Phase C (adversarial evaluation) is now step 1, with the full sweep planned on a free
  Kaggle GPU because the laptop has no CUDA.
- The mixed DinoHash head/backbone rerun (F17) moved to the end as **"Deferred"**. It refines
  how F16 is stated but doesn't block the adversarial work.

The F17 caveat still applies to everything below: trained and untrained DinoHash differ in
both backbone *and* head, so a robustness gap between them can't be attributed to
fine-tuning alone until the mixed models are tested.

---

## 2. Attack module: `hashbench/hashbench/adversarial.py`

### 2.1 What it does
Model-agnostic white-box attacks on any hash whose network is a differentiable `nn.Module`
mapping [0, 1] RGB images to outputs:

| Hash family | Network output | Hash | Distance | Supported |
|---|---|---|---|---|
| DinoHash (trained, untrained) | 96 logits | bit = logit ≥ 0 | Hamming | ✅ |
| NeuralHash | 96 logits | bit = logit ≥ 0 | Hamming | ✅ |
| CertPHash (ε=1, 2, 4, 8, base, adv) | 144 real values | round, clip to [0, 255] | L1 | ✅ |
| pHash, PDQ | not differentiable | – | – | ❌ needs soft surrogates (later in Phase C) |

Two attack goals:
- **Evasion:** perturb an image so its hash moves beyond the match threshold from its own
  original hash. Success: `d(adv, original) > threshold`.
- **Collision:** perturb a source image so its hash comes within the threshold of a target
  image's hash. Success: `d(adv, target) <= threshold`.

Both match the benchmark's rule that two hashes match when `distance <= threshold`.

### 2.2 Optimisers
- **PGD** (fixed step): ℓ∞ step `α · sign(grad)`, ℓ2 step along the normalised gradient.
  Default step size ε/4 (ℓ∞) or ε/2 (ℓ2).
- **APGD** (Croce & Hein, ICML 2020), minimisation form:
  - initial step 2ε, per-sample step sizes, momentum 0.75;
  - checkpoints at the paper's schedule (0.22 n, then growing gaps);
  - at a checkpoint, the step halves if fewer than 75% of the steps since the last
    checkpoint improved the best loss, and the iterate *and its gradient* restart from the
    best point.
  - **Simplified:** the paper's second halving condition (step size and best loss both
    unchanged since the last checkpoint) is not checked. This is stated in the docstring.
- Optional random start inside the budget ball.
- Optional **EOT** (Expectation over Transformation): averages the loss over
  `eot_samples` random transforms. Needed later to attack randomised defences (research
  ideas 2 and 5).

### 2.3 Losses (lower is better for the attacker)
- **Bit hashes, evasion:** `mean_k relu(margin + logit_k · s_k)`, where `s_k ∈ {−1, +1}` is
  the sign of bit k on the clean image. Minimising it flips bits. Exactly-zero logits count
  as +1 so they still get a gradient.
- **Bit hashes, collision:** `mean_k relu(margin − logit_k · t_k)`, with `t_k` the target's
  signs.
- **CertPHash, evasion:** `−Σ |output − round(clean output)|`, i.e. maximise the L1 distance
  to the clean hash.
- **CertPHash, collision:** `Σ |output − round(target output)|`.
- `margin` defaults to 0 (flip the sign only). Logit scales differ by model (DinoHash about
  ±500, NeuralHash much smaller), so a non-zero margin would be model-specific.

### 2.4 Budget conventions
- ε is measured in **[0, 1] pixel space** of the model input, ℓ∞ or ℓ2.
- Every network applies its own normalisation (ImageNet, or [−1, 1] for NeuralHash) inside
  `forward`, so attacking its [0, 1] input is attacking pixel space directly. This is the
  same convention as CertPHash's certified training (ε/std added to the normalised input)
  and its PGD attack, so their ε = k/255 and ours match. (Corrected 2026-11-04: an earlier
  note claimed the authors' ε was 4.4× larger in pixel space. The real problem is in their
  CNER script `verify_evasion.py`, which certifies a one-value L0 change with PDQ bits and
  threshold 90, not an ℓ∞ ball with PhotoDNA's 1800; see hashbench/README.md.)
- Attacks run at each model's **input resolution** (DinoHash 224², NeuralHash 360²,
  CertPHash 64²). Originals are first downscaled to 512 px, as in the benign benchmark.

### 2.5 Honest reporting
- The final image is **quantised to 8 bits** (what saving a PNG does), and success is
  re-checked on the quantised image. A reported success therefore survives being saved as
  a real image.
- Reported metrics are of that 8-bit image versus the clean input: ℓ∞, ℓ2, PSNR and SSIM
  (Gaussian window 11, σ 1.5).
- Consequence: the reported ℓ∞ can exceed ε by up to half a quantisation level (0.5/255 ≈
  0.002). The continuous perturbation the optimiser produces is strictly within budget.
  CertPHash's 64² input is not on the 8-bit grid (antialiased resize), so its ℓ∞ shows this
  most clearly (0.0333 at ε = 8/255 = 0.0314).

### 2.6 API
```python
from hashbench.hashes import get_hasher
from hashbench.adversarial import AttackConfig, attack_batch, evade, collide, build_adapter

h = get_hasher("dinohash")
cfg = AttackConfig(norm="linf", eps=8/255, steps=100, method="apgd", device="cuda", seed=0)

r = evade(h, image, cfg, threshold=17)             # one image; threshold in bits (L1 for CertPHash)
r = collide(h, source, target, cfg, threshold=17)
rs = attack_batch(h, images, cfg, threshold=17)                    # batched evasion
rs = attack_batch(h, images, cfg, threshold=17, targets=targets)   # batched collision
```
- `AttackResult` fields: `mode, success, distance, orig_distance, steps_used, linf, l2,
  psnr, ssim, adv, adv_image`.
- `attack_batch` takes a prebuilt `adapter` (to avoid moving the model for every batch) and
  `keep_images=False` (to keep only the numbers).
- `build_adapter(hasher)` wraps DinoHash, NeuralHash or CertPHash and raises a clear error
  for pHash/PDQ.
- `default_threshold`: 1800 (the paper's L1 threshold) for CertPHash; 0 for bit hashes,
  where the caller should pass the calibrated threshold.

### 2.7 Changes made after the first version
1. Added the batched entry point `attack_batch`; `evade` and `collide` are now thin wrappers.
2. APGD fix: on a step-size halving, the gradient now also restarts from the best point
   (it previously reused the gradient of the current point).
3. Removed a dead helper function in `_apgd`.
4. Documented the half-quantisation-level ℓ∞ overshoot.

---

## 3. Tests: `hashbench/tests/test_adversarial.py` (13 pass)

Tiny dummy model (a fixed linear map to 16 logits), so these run anywhere without weights:

| Test | Checks |
|---|---|
| `test_respects_budget` (ℓ∞/ℓ2 × PGD/APGD, 4 cases) | Output stays in [0, 1] and inside the ε ball |
| `test_evasion_flips_bits` (PGD, APGD) | A large budget flips at least 4 of 16 bits on average |
| `test_collision_reduces_distance` (PGD, APGD) | Distance to the target goes down |
| `test_project_l2_shrinks_only_when_outside` | ℓ2 projection leaves points inside the ball unchanged |
| `test_zero_budget_is_a_no_op` | ε = 0 returns the clean image |

Real models (skipped when weights are missing):

| Test | Checks |
|---|---|
| `test_real_evasion_smoke` (dinohash, neuralhash, certphash_adv) | Pipeline runs; ℓ∞ ≤ ε + half a level; evasion never moves the hash closer; SSIM in [0, 1] |

Run from `hashbench/`: `python -m pytest tests/test_adversarial.py -q`

---

## 4. Sweep script: `hashbench/scripts/run_adversarial.py`

### 4.1 What it does
- For each hash, each ε (in /255 units) and each mode (evasion, collision), attacks every
  image in a fixed list and appends one row per image to `<out>/raw.csv`.
- **Success thresholds** are the benign benchmark's calibrated FPR 1e-3 thresholds, read
  from `results/coco_run1/overall.csv` and `results/coco_run2/overall.csv`, so benign and
  adversarial results use the same operating point. Converted to hash units with
  `floor(normalised threshold × max_distance)`:

  | Hash | Threshold |
  |---|---|
  | dinohash | 17 bits |
  | dinohash_untrained | 21 bits |
  | neuralhash | 26 bits |
  | certphash (ε=1) | L1 4870 |
  | certphash_ep2 | L1 3912 |
  | certphash_ep4 | L1 3306 |
  | certphash_ep8 | L1 2581 |
  | certphash_base | L1 4936 |
  | certphash_adv | L1 5435 |

  The CertPHash values are much looser than the paper's L1 1800 (see §5.3).
- **Collision targets are fixed:** image i of the list is pushed onto image (i + n/2) mod n.
  Every model and budget attacks the same pairs. With an even n the pairs are symmetric.
- **Resumable:** rows already in `raw.csv` are skipped, so a run interrupted by a Kaggle
  session timeout can be restarted with the same command. A rerun into the same folder with
  different attack settings is refused instead of mixing results. Per-batch seeds depend
  only on the batch content, so a resumed run reproduces the same random starts.
- **Outputs:** `config.json` (all settings, device, GPU name, torch version),
  `raw.csv`, `summary.csv`, and `summary.md`.
  - `summary.csv`: success rate with a 95% Wilson interval, mean distance, mean distance
    before the attack, share already successful before the attack, mean ℓ∞, PSNR, SSIM and
    seconds per image, per hash × mode × ε.
  - `summary.md`: the ε tables for evasion and collision, plus mean SSIM.
- `raw.csv` columns: `hash, mode, norm, eps_255, image, target, threshold, distance,
  orig_distance, success, linf, l2, psnr, ssim, seconds_per_image`.

### 4.2 Main options (defaults)
`--list` image list · `--limit N` · `--hashes` (all 9 learned variants) · `--eps 1 2 4 8` ·
`--modes evasion collision` · `--norm linf` · `--steps 100` · `--method apgd` ·
`--step-size` (PGD only) · `--margin 0` · `--fpr 0.001` · `--thresholds-from
results/coco_run1 results/coco_run2` · `--max-side 512` · `--batch-size 8` · `--device` ·
`--seed 0` · `--summary-only`

### 4.3 Commands
```
# from hashbench/
# full run (GPU)
python scripts/run_adversarial.py --list data/splits/V200.txt --out results/adv_v200 --device cuda --batch-size 16
# rebuild only the summary tables
python scripts/run_adversarial.py --out results/adv_v200 --summary-only
```

---

## 5. Results so far

### 5.1 Single-image sanity check
First sample image in `data/samples`, ℓ∞ ε = 8/255, APGD 40 steps, CPU. Thresholds here
were rough approximations of the calibrated ones.

| Model | Bits moved (evasion) | Evaded? |
|---|---|---|
| DinoHash untrained | 92 / 96 | ✅ |
| NeuralHash | 45 / 96 | ✅ |
| DinoHash trained | 11 / 96 | ❌ |

Trained DinoHash collision on this image: distance to the target 54 → 55 bits (no
progress). This only showed that the attack runs and breaks undefended models; the numbers
are one image and should not be cited.

### 5.2 Timing (laptop CPU, 4 images, ε = 8/255, 20 steps)

| Model | Seconds per image |
|---|---|
| DinoHash (each model) | about 3 |
| NeuralHash | about 0.8 |
| CertPHash (each variant) | about 0.08 |

At 100 steps this is about 15 s per image for each DinoHash model, so the full `V200` sweep
(200 images × 4 ε × 2 modes) would take about 13 h on the laptop for the two DinoHash models.
That is why the full run goes to a GPU.

### 5.3 ε pilot (ℓ∞, APGD 100 steps, CPU) — superseded by the V200 sweep (see the 2026-11-05 update at the top)
- **Images:** the first 10 images of `V200` for NeuralHash and CertPHash
  (`hashbench/results/adv_pilot_fast/`); the first 3 for each DinoHash model
  (`hashbench/results/adv_pilot_dino/`).
- **Thresholds:** benchmark FPR 1e-3 (§4.1).
- With n = 10 or n = 3 the 95% intervals are very wide (0% with n = 10 means 0–28%).

**Evasion success (%)**

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| DinoHash untrained | 3 | 100 | 100 | 100 | 100 |
| DinoHash trained ⚠️ | 3 | 100 | 100 | 100 | 100 |
| NeuralHash | 10 | 0 | 0 | 10 | 80 |
| CertPHash ε=1, ε=2, ε=4, ε=8, adv | 10 | 0 | 0 | 0 | 0 |
| CertPHash base | 10 | 0 | 0 | 0 | 40 |

**Collision success (%)**

| Hash | n | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|---|
| DinoHash untrained | 3 | 100 | 100 | 100 | 100 |
| DinoHash trained ⚠️ | 3 | 0 | 0 | 0 | 0 |
| NeuralHash | 10 | 0 | 0 | 0 | 30 |
| CertPHash base | 10 | 0 | 0 | 0 | 40 |
| CertPHash adv | 10 | 0 | 0 | 0 | 30 |
| CertPHash ε=1 | 10 | 0 | 0 | 0 | 20 |
| CertPHash ε=2, ε=4, ε=8 | 10 | 0 | 0 | 0 | 0 |

No collision pair was already a match before the attack.

**Mean distance after the attack**

| Hash | Mode | ε=1 | ε=2 | ε=4 | ε=8 | Before attack |
|---|---|---|---|---|---|---|
| NeuralHash (bits) | evasion | 4.1 | 8.0 | 18.6 | 36.9 | 0 |
| NeuralHash (bits) | collision | 45.5 | 44.5 | 40.3 | 33.1 | 47.0 |
| CertPHash ε=1 (L1) | evasion | 222 | 439 | 887 | 1909 | 0 |
| CertPHash ε=8 (L1) | evasion | 123 | 223 | 428 | 882 | 0 |
| CertPHash base (L1) | evasion | 496 | 991 | 1888 | 4203 | 0 |
| DinoHash untrained (bits) | evasion | 76.7 | 95.3 | 96.0 | 96.0 | 0 |
| DinoHash untrained (bits) | collision | 4.0 | 0.0 | 0.0 | 0.0 | 46.7 |

CertPHash evasion distance grows roughly linearly with ε, and the certified ε=8 model moves
about a fifth as far as the base model.

**Image quality (mean SSIM, evasion)**

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| CertPHash (range over variants) | 0.997 | 0.989–0.991 | 0.960–0.969 | 0.875–0.907 |
| NeuralHash | 0.995 | 0.982 | 0.940 | 0.835 |
| DinoHash trained | 0.995 | 0.983 | 0.948 | 0.829 |
| DinoHash untrained | 0.994 | 0.984 | 0.948 | 0.842 |

SSIM is measured at each model's input resolution, so values for different models aren't
strictly comparable.

**CertPHash at the paper's threshold (L1 1800), recomputed from the same pilot rows**

| Hash | Evasion ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| CertPHash base | 0 | 10 | 40 | 100 |
| CertPHash adv | 0 | 0 | 10 | 70 |
| CertPHash ε=1 | 0 | 0 | 0 | 40 |
| CertPHash ε=2 | 0 | 0 | 0 | 30 |
| CertPHash ε=4 | 0 | 0 | 0 | 20 |
| CertPHash ε=8 | 0 | 0 | 0 | 0 |

All CertPHash collisions are 0% at L1 1800 at every ε. At this threshold, evasion resistance
orders exactly by training strength. At the calibrated FPR 1e-3 threshold (L1 2581–5435),
almost nothing evades. **The thesis has to report both thresholds.**

### 5.4 Timing anomalies in the pilot
- One DinoHash trained collision batch at ε = 8/255 logged 4,180 s per image (about 3.5 h
  for 3 images); other DinoHash batches took about 14–25 s per image. The laptop most likely
  slept or was busy, so that batch's timing is meaningless.
- One NeuralHash evasion batch at ε = 2/255 logged 30.9 s per image against about 4 s for the
  others, probably for the same reason.
- The pilot therefore took much longer than the planned 20 minutes.

---

## 6. Known issues and limitations

### 6.1 Best-iterate selection ✅ fixed 2026-11-05

*Fixed: the best iterate is now chosen by hash distance on the 8-bit image, APGD runs together
with small-step PGD, and ε is warm-started. The `V200` sweep has 0 monotonicity violations. The
text below is kept as a record of the problem.*

Trained DinoHash, bits flipped per image (evasion):

| Image | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| 000000002149.jpg | 27 | 21 | 20 | 34 |
| 000000002923.jpg | 33 | 35 | 40 | 34 |
| 000000003553.jpg | 39 | 40 | 42 | 51 |

A larger budget should never do worse, because any perturbation allowed at ε=1 is also
allowed at ε=8. The non-monotonic values show the attack is not optimising reliably on
trained DinoHash.

- **Likely cause:** both optimisers keep the iterate with the lowest *loss*, not the one
  with the largest hash distance (smallest, for collisions). Trained DinoHash's logits are
  around ±500, so a few large-logit bits dominate the hinge loss and the lowest-loss iterate
  isn't the one that flips the most bits.
- **Planned fix:** track the best iterate by the attack's real objective (hash distance), and
  add a check that flags any model whose success or distance falls as ε grows.
- **Until then:** trained DinoHash's pilot numbers (both modes) can't be used. In particular,
  trained DinoHash may be *more* breakable than shown. Untrained DinoHash isn't affected
  in practice, since it saturates at 96/96 bits from ε=2.

### 6.2 Other limitations
1. **Simplified APGD:** the second step-halving condition isn't implemented (§2.2).
2. **Attacks at model input resolution:** the perturbation is added to the resized input
   (224 / 360 / 64 px), not to the full-size original before resizing. This is the standard
   setting in Struppek et al., but must be stated.
3. **Quantisation overshoot:** reported ℓ∞ can exceed ε by up to 0.5/255 (§2.5).
4. **Collision targets:** fixed (i → i + n/2) random pairs. CertPHash's paper instead pairs
   each image with the test image whose hash is already *closest*, so our collision rates will
   be lower and not directly comparable to theirs.
5. **Threshold choice:** results depend heavily on it, especially for CertPHash (§5.3).
6. **margin = 0:** bits are pushed only just past zero; large-margin variants weren't tried.
7. **pHash/PDQ not attackable yet:** they need differentiable soft surrogates.
8. **Small samples:** n = 3 or 10 per cell in the pilot.
9. **NeuralHash and Kaggle:** Apple's model files must not be redistributed, so they should
   not be uploaded to Kaggle, even as a private dataset. NeuralHash is cheap enough to run
   locally (about 2 h for the full `V200` sweep on the laptop CPU).
10. **F17 confound:** trained vs untrained DinoHash differ in head and backbone (§1).

---

## 7. Comparison with the attacks in the papers

From `research_papers/DinoHash.pdf` and `research_papers/CertPHash.pdf`.

| | DinoHash paper | CertPHash paper | Our sweep |
|---|---|---|---|
| Attack | APGD, 100 steps (training: APGD, 10 steps) | PGD (Struppek et al.), up to 100 steps; plus IBP/CROWN verification | APGD or PGD, 100 steps |
| Budget | ℓ∞ 4/255, 8/255 | Empirical: ℓ2 (about 3.7 evasion, 10.8 collision). Certified: ℓ∞ 1–8/255, ℓ2 0.05–0.2 | ℓ∞ 1, 2, 4, 8/255 (ℓ2 supported) |
| Evasion | ✅ | ✅ | ✅ |
| Collision | ❌ | ✅, closest target | ✅, fixed random pair |
| Success measure | Bit accuracy (share of the 96 bits still matching) | Threshold, L1 1800 | Benchmark FPR 1e-3 threshold |

**DinoHash paper, Table 1** (bit accuracy under 100-step APGD):

| Model | ε = 4/255 | ε = 8/255 |
|---|---|---|
| DinoHash trained | 72.2% | 64.8% |
| DinoHash untrained | 0.3% | 0.0% |
| NeuralHash | 0.1% | 0.0% |

**CertPHash paper, COCO, empirical ℓ2 PGD** (Tables 4 and 5):

| Model | Evasion success | Collision success |
|---|---|---|
| Base | 89.2% | 89.8% |
| Adv | 68.8% | 21.7% |
| Certified ε = 1/255 | 54.6% | 37.5% |
| Certified ε = 2/255 | – | 27.4% |
| Certified ε = 4/255 | 18.7% | – |
| Certified ε = 8/255 | 1.0% | 18.7% |

Certified results (verification, not attacks): 100% certified no-evasion and 95% certified
no-collision at ℓ∞ 8/255, against 0.01% and 0% for adversarial training.

**How the pilot compares (to be confirmed by the full run and the §6.1 fix):**
- **Bit accuracy is not the same as robustness at a matching threshold.** 72% bit accuracy
  at 4/255 means about 27 of 96 bits flipped, which is beyond our 17-bit threshold. So the
  paper's "robust" and our "evaded" can both be true; this is worth a paragraph in the thesis.
- **DinoHash:** our 3 trained images average about 40 bits flipped at 8/255, i.e. 59% bit
  accuracy, close to the paper's 64.8%.
- **CertPHash:** at L1 1800, our ℓ∞ 8/255 ordering (base 100, adv 70, ε=1 40, ε=4 20,
  ε=8 0%) is close to the paper's ℓ2 results (89 / 69 / 55 / 19 / 1%), although the attack
  norms differ.

---

## 8. Next steps

**Current list (2026-11-22).** Items 1–5 of the original list below are done (sweep additions:
bit accuracy, both CertPHash thresholds and closest-target collisions are done; ℓ2 was dropped).

1. ✅ Commit the new files and results (2026-11-17; transfer PNGs kept local, see §9). NES and
   Sun et al. code and results committed 2026-11-22.
2. Update `adversarial_testing_report.md` §4 (drop ViT-B explanation, add the 50-image numbers,
   soften the gradient-masking claim); fix the author line in `certphash.py`.
3. Add the authors' APGD-BCE as a third ensemble member (NeuralHash numbers are likely too low).
4. Build the next attacks in the order of `future_attacks.md`: transfer ✅ (2026-11-17), NES
   black-box ✅ (2026-11-18, all 11 hashes + the 26-bit DinoHash check), Sun et al. setting ✅
   (2026-11-19/21, literal + 500-step variant), then EOT,
   joint evasion, universal perturbation; test each on `V200`.
5. A sharding / merge helper for parallel runs on the institute cluster.
6. On the cluster: a 10-image pilot, then benign on 2,000 images and adversarial on 1,000 images.
7. Optional: the Sun et al. 500-step variant for DinoHash at ℓ2 40 and 90 (not run; expected to
   repeat the ℓ2 180 result, `nes_sun_attack_results.md` §6.2).

**Original list (2026-11-04), kept for reference:**

1. **Fix best-iterate selection** (§6.1) and add the ε-monotonicity check. Rerun the
   DinoHash pilot on the same 3 images to confirm (about 15 min if the laptop stays awake).
2. **Sweep additions:**
   - report bit accuracy for DinoHash and NeuralHash (computable from the saved distances),
     to compare with the DinoHash paper;
   - report CertPHash at both thresholds (L1 1800 and calibrated FPR 1e-3);
   - add a "closest target" collision option, to compare with CertPHash's Table 5;
   - optionally, an ℓ2 run like CertPHash's.
3. **Kaggle package:** code (`hashbench/hashbench`, `scripts/run_adversarial.py`), DinoHash
   ViT-S and CertPHash model files, the 200 `V200` images and the two `overall.csv` threshold
   files, uploaded as a private dataset. Point `HASHBENCH_MODELS` at the uploaded models
   folder and run with `--device cuda`. Exclude Apple's NeuralHash files.
4. **Full `V200` sweep:** DinoHash and CertPHash on Kaggle; NeuralHash locally.
5. **Write-up:** adversarial findings into `benchmark_findings.md`.
6. **Later in Phase C:** soft pHash/PDQ surrogates, black-box NES, transferability matrix (✅ 2026-11-17).

---

## 9. Files

| File | Status |
|---|---|
| `hashbench/hashbench/adversarial.py` | Committed (`0489873`) |
| `hashbench/scripts/run_adversarial.py` | Committed (`0489873`) |
| `hashbench/tests/test_adversarial.py` | Committed (`0489873`) |
| `hashbench/results/adv_pilot_fast/`, `adv_pilot_dino/`, `adv_pilot_dino_fixed/` | Pilot output |
| `hashbench/results/adv_v200_kaggle/`, `adv_v200_neuralhash/` | V200 sweep, committed (`0489873`) |
| `hashbench/results/adv_v200_plots/`, `benign_vs_adversarial.pdf` | Plots, committed (`f1b6e1a`) |
| `adversarial_testing_report.md`, `adversarial_results_tables.md` | Write-up and all tables, committed 2026-11-17 |
| `dinohash_table1_reproduction.md`, `scripts/reproduce_dinohash_table1.py`, `scripts/third_party/`, `kaggle/table1/`, `results/dinohash_table1_repro_*` | Table 1 reproduction, committed 2026-11-17 |
| `future_attacks.md` | Catalogue of further attacks, committed 2026-11-17 |
| `scripts/generate_transfer_images.py`, `scripts/transfer_matrix.py`, `kaggle/transfer/` | Transfer attack code, committed 2026-11-17 |
| `transfer_attack_results.md`, `results/transfer_v200_matrix/`, `results/transfer_v200*/generation.csv`, logs | Transfer results, committed 2026-11-17 |
| `hashbench/hashbench/blackbox.py`, `blackbox_classical.py`, `scripts/run_nes.py`, `scripts/run_nes_classical.py`, `tests/test_blackbox*.py`, `kaggle/nes/`, `offline/phash_pdq/` | NES attack code (Kaggle GPU runner and the offline pHash/PDQ CPU runner), committed 2026-11-22 |
| `nes_attack_results.md`, `results/nes_v200_*`, `results/nes_pilot_kaggle/` | NES results, committed 2026-11-22 |
| `hashbench/hashbench/blackbox_sun.py`, `scripts/run_nes_sun.py`, `tests/test_blackbox_sun.py`, `kaggle/nes_sun/`, `nes_sun_attack_results.md`, `results/nes_sun_v200_*`, `results/nes_sun500_v100_*` | Sun et al. 2026 NES replication (code + results), committed 2026-11-22 |
| `results/transfer_v200/images/`, `results/transfer_v200_neuralhash/images/` | 19,800 attacked/clean PNGs (about 600 MB), **kept local only** (git-ignored); regenerate with `transfer_attack_results.md` §6 |
| `phaseC_implementation_progress.md` | This file, last updated 2026-11-22 |
