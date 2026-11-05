# Phase C (adversarial robustness): implementation progress

*BTP: Perceptual Hashing. Status as of 2026-11-04. Phase C is defined in
`thesis_research_plan.md` §2; the benign results it builds on are in `benchmark_findings.md`.*

## Summary

| Part | Status |
|---|---|
| Plan reordered (adversarial work first, mixed DinoHash head/backbone rerun deferred) | ✅ Done |
| Attack module `hashbench/hashbench/adversarial.py` (PGD, APGD, evasion, collision) | ✅ Done, one known issue (§6.1) |
| Unit and smoke tests `hashbench/tests/test_adversarial.py` | ✅ 13 pass |
| Sweep script `hashbench/scripts/run_adversarial.py` (models × ε × mode, resumable) | ✅ Done |
| Single-image sanity check | ✅ Done |
| CPU timing pilot and ε pilot (10 images; 3 for DinoHash) | ✅ Done, **pilot only** |
| Fix best-iterate selection (§6.1) | ❌ Not done; blocks the full run |
| Kaggle package (code, models, images, thresholds, notebook) | ❌ Not built |
| Full sweep on `V200` (200 images, ε = 1, 2, 4, 8 /255) | ❌ Not run |
| Write-up of adversarial findings | ❌ Not done |

The ε tables in §5 come from a pilot on very few images. They show that the pipeline works
and give early signals, but **they are not thesis results**.

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

### 5.3 ε pilot (ℓ∞, APGD 100 steps, CPU)
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

### 6.1 Best-iterate selection (must fix before the full run)
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
6. **Later in Phase C:** soft pHash/PDQ surrogates, black-box NES, transferability matrix.

---

## 9. Files

| File | Status |
|---|---|
| `hashbench/hashbench/adversarial.py` | New, not committed |
| `hashbench/scripts/run_adversarial.py` | New, not committed |
| `hashbench/tests/test_adversarial.py` | New, not committed |
| `hashbench/results/adv_pilot_fast/` (config, raw, summary) | Pilot output, not committed |
| `hashbench/results/adv_pilot_dino/` (config, raw, summary) | Pilot output, not committed |
| `benchmark_findings.md` §7 | Edited (plan reorder), not committed |
| `phaseC_implementation_progress.md` | This file, not committed |
