# White-box adversarial testing: process, results and paper comparison

*BTP: Perceptual Hashing. Written 2026-11-05. Covers the full `V200` sweep in
`hashbench/results/adv_v200_kaggle` and `hashbench/results/adv_v200_neuralhash`, plus the investigation of
the disagreement with the DinoHash paper. The benign results it builds on are in `benchmark_findings.md`.*

## Main conclusions

1. **The best hash on benign edits is the easiest to break.** Both DinoHash versions are evaded on at
   least 97.5% of images at ε=1/255, and collisions on trained DinoHash reach 92% at ε=2/255.
2. **Certified CertPHash resists best at the paper's threshold, but not everywhere.** At L1 1800 and ε=8/255,
   evasion falls from 99% (base) to 0% (ε=8). Closest-target collisions at calibrated thresholds still
   reach 50.5% on ε=8.
3. **The threshold changes the CertPHash story.** At the calibrated FPR 1e-3 thresholds, which are much
   looser than L1 1800, almost nothing evades, including the undefended base model at small budgets.
4. **The disagreement with the DinoHash paper has likely explanations, and neither points to a fault in
   our attack:**
   - **Trained DinoHash (paper 72.2%, ours 10.3% at ε=4):** the paper's attack method gets stuck on this model. An
     exact copy of it gives 62.9% on our model, while small-step PGD from the same kind of setup reaches 30–35%.
     So the paper's figure probably overstates the model's robustness.
   - **NeuralHash (paper 0.1%, ours 78.4% at ε=4):** no attack we tried comes close to the paper. The authors'
     committed evaluation script doesn't attack NeuralHash at all, and copying what it computes gives 0.0%.
     So the paper's NeuralHash column may be an artefact. **This is strong evidence but not proof**, because we
     can't confirm that this script produced the table.

**Treat all DinoHash and NeuralHash claims about the paper as our findings to be checked with the authors.**
Those investigation runs used only 3–10 images.

---

## 1. What was tested

### 1.1 Threat model

A **white-box** attacker knows the network and its weights, and can add a small change to each pixel
of the image. The change is limited by a budget ε, the maximum change per pixel in [0, 1] pixel space
(ℓ∞ norm).

| Goal | What the attacker does | Success when |
|---|---|---|
| **Evasion** | Changes an image so its own copy is no longer found | `distance(attacked, original) > threshold` |
| **Collision, random target** | Changes an image to match a fixed unrelated image | `distance(attacked, target) <= threshold` |
| **Collision, closest target** | The same, but the target is the other image in the list with the closest clean hash | `distance(attacked, target) <= threshold` |

### 1.2 Hashes

The 9 learned versions were tested. pHash and PDQ aren't differentiable and are **out of scope** for
now.

| Hash | Input size | Output | Calibrated threshold (FPR 1e-3) |
|---|---|---|---|
| DinoHash trained | 224 px | 96 bits | 17 bits |
| DinoHash untrained | 224 px | 96 bits | 21 bits |
| NeuralHash | 360 px | 96 bits | 26 bits |
| CertPHash base | 64 px | 144 values | L1 4936 |
| CertPHash adv | 64 px | 144 values | L1 5435 |
| CertPHash ε=1 | 64 px | 144 values | L1 4870 |
| CertPHash ε=2 | 64 px | 144 values | L1 3912 |
| CertPHash ε=4 | 64 px | 144 values | L1 3306 |
| CertPHash ε=8 | 64 px | 144 values | L1 2581 |

The thresholds come from the benign benchmark, so benign and adversarial results share one operating
point. CertPHash is also reported at the paper's L1 1800.

### 1.3 Sweep settings

| Setting | Value |
|---|---|
| Images | `data/splits/V200.txt`, 200 COCO val2017 images, downscaled to at most 512 px first |
| Evasion budgets | ε = 1, 2, 4, 8 /255 |
| Collision budgets | ε = 1, 2, 4, 8, 16, 32 /255 |
| Attacks run | 28,800 (9 versions × 200 images × 16 mode and budget settings) |
| Machines | DinoHash and CertPHash on a Kaggle Tesla T4 GPU (5.7 h); NeuralHash on CPU |

Apple's NeuralHash files were kept off Kaggle.

---

## 2. The attack process

For one image at one budget:

1. **Prepare:** resize to the model's input size. Attacks act on this resized input, not on the full-size
   original.
2. **Set the reference:** the image's own hash (evasion) or the target's hash (collision).
3. **Choose a start:** at the smallest budget, a random point within the budget. At larger budgets, the best
   result from the next smaller budget (a **warm start**).
4. **Optimise for 100 steps,** twice from the same start:
   - **APGD** (Croce & Hein 2020), which starts with large steps of 2ε and shrinks them;
   - **small-step PGD**, with a step of `min(ε/10, 0.25/255)`.
5. **Keep the best result:** for each run, the step with the best real hash distance, measured on the 8-bit
   image. Between the two runs, the better one is kept for each image.
6. **Quantise and check:** round to 8 bits, as saving a PNG would, then compute the hash again. A reported
   success therefore survives saving as a real image.
7. **Record:** distance, success, ℓ∞, ℓ2, PSNR and SSIM.

### Losses

| Hash type | Evasion | Collision |
|---|---|---|
| Bit hashes (hinge, margin 0) | `mean relu(logit × original sign)`: push each bit past zero | `mean relu(−logit × target sign)` |
| CertPHash | Maximise L1 distance to the clean hash | Minimise L1 distance to the target hash |

### Fixes before the full run

The pilot kept the step with the lowest **loss**. This returned results that flipped fewer bits than
earlier steps, so a larger budget sometimes did worse, on trained DinoHash, NeuralHash and CertPHash base.
The current code selects by real hash distance, and adds the second optimiser and the warm start.

---

## 3. Results

All tables use n = 200 per cell. In this section, success rates use the calibrated FPR 1e-3 thresholds unless
stated otherwise. Parentheses give 95% Wilson intervals.

### 3.1 Evasion

**Success rate (%)**

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| DinoHash untrained | 100 (98–100) | 100 (98–100) | 100 (98–100) | 100 (98–100) |
| DinoHash trained | 97.5 (94–99) | 100 (98–100) | 100 (98–100) | 100 (98–100) |
| NeuralHash | 0.5 (0–3) | 0.5 (0–3) | 18.0 (13–24) | 94.5 (90–97) |
| CertPHash base | 0 (0–2) | 0 (0–2) | 0 (0–2) | 11.5 (8–17) |
| CertPHash ε=2 | 0 (0–2) | 0 (0–2) | 1.0 (0–4) | 3.5 (2–7) |
| CertPHash ε=4 | 0 (0–2) | 0 (0–2) | 0 (0–2) | 2.5 (1–6) |
| CertPHash adv | 0 (0–2) | 0 (0–2) | 0.5 (0–3) | 1.5 (1–4) |
| CertPHash ε=1 | 0 (0–2) | 0 (0–2) | 0 (0–2) | 1.0 (0–4) |
| CertPHash ε=8 | 0 (0–2) | 0 (0–2) | 0 (0–2) | 0 (0–2) |

**Bit accuracy (%, bits still equal to the original; lower means more broken)**

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| DinoHash untrained | 20.6 | 2.3 | 0.1 | 0.0 |
| DinoHash trained | 45.6 | 22.3 | 10.3 | 3.7 |
| NeuralHash | 94.4 | 88.9 | 78.4 | 59.2 |

**CertPHash evasion success (%) at the paper's L1 1800**

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| base | 1.0 | 6.0 | 33.0 | 99.0 (96–100) |
| adv | 0 | 0.5 | 5.0 | 60.5 (54–67) |
| ε=1 | 0 | 0 | 2.0 | 48.0 (41–55) |
| ε=2 | 0.5 | 1.5 | 3.0 | 43.0 (36–50) |
| ε=4 | 0 | 0 | 2.5 | 22.5 (17–29) |
| ε=8 | 0 | 0 | 0 | 0 (0–2) |

**CertPHash mean L1 distance moved by evasion**

| Hash | ε=1/255 | ε=2/255 | ε=4/255 | ε=8/255 |
|---|---|---|---|---|
| base | 453 | 902 | 1,767 | 3,404 |
| adv | 260 | 519 | 1,063 | 2,177 |
| ε=1 | 225 | 445 | 913 | 1,914 |
| ε=2 | 235 | 451 | 905 | 1,847 |
| ε=4 | 202 | 389 | 771 | 1,576 |
| ε=8 | 133 | 247 | 472 | 937 |

The distance grows roughly linearly with ε. The ε=8 model moves about 3.6 times less than base at
ε=8/255. Its output range is also narrower, so raw distances overstate this difference somewhat.

### 3.2 Collision, random target

**Success rate (%)**

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

Before any attack, no random pair matched, except 1% for CertPHash ε=8.

### 3.3 Collision, closest target

**Success rate (%)**

| Hash | Matched before attack | ε=1 | ε=2 | ε=4 | ε=8 | ε=16 | ε=32 |
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

**The "matched before attack" column matters.** Some closest pairs already match, so these success rates start
above zero. The closest image is chosen from only these 200, so a real attacker with a larger pool could do
better.

**CertPHash closest-target collision success (%) at the paper's L1 1800**

| Hash | ε=8 | ε=16 | ε=32 |
|---|---|---|---|
| base | 4.0 | 51.5 | 98.5 |
| ε=8 | 13.0 | 44.5 | 91.5 |
| ε=4 | 7.5 | 33.5 | 91.0 |
| ε=2 | 6.5 | 24.5 | 80.0 |
| ε=1 | 1.0 | 7.5 | 66.5 |
| adv | 1.0 | 2.5 | 44.0 |

Here ε=8 is easier to attack than ε=1 and adv at a fixed L1, likely because its narrow output range puts images
closer together to begin with. Its mean clean distance to the closest target is 3,450, compared with 5,367 for
ε=1. So **a defence that reduces evasion doesn't necessarily reduce collisions**, a point the thesis should make.

### 3.4 Image quality

**Mean SSIM of attacked images (evasion)**

| Hash | ε=1 | ε=2 | ε=4 | ε=8 |
|---|---|---|---|---|
| CertPHash (range across versions) | 0.997–0.998 | 0.990–0.992 | 0.966–0.973 | 0.895–0.919 |
| NeuralHash | 0.996 | 0.985 | 0.950 | 0.859 |
| DinoHash trained | 0.995 | 0.988 | 0.974 | 0.952 |
| DinoHash untrained | 0.995 | 0.988 | 0.984 | 0.984 |

Read this table with care:
- **DinoHash attacks don't use their full budget.** Its average maximum pixel change at ε=8/255 is 6.4/255 for
  trained and 3.2/255 for untrained. These SSIM values describe the attack found, not the worst an image at
  that budget could look.
- **SSIM is measured at each model's input size,** so it isn't strictly comparable between families.
- **CertPHash's reported ℓ∞ goes about 0.5/255 over budget** (8.5/255 at ε=8). Its 64×64 input isn't on the 8-bit
  grid, so rounding adds this. The optimiser's own result stays inside the budget.
- At ε=32/255, SSIM falls to about 0.57–0.64 for CertPHash and 0.58–0.68 for NeuralHash, so these attacks are
  clearly visible. They show how the models break, but aren't subtle attacks.

### 3.5 What the results show

- **Attack budget for practical evasion:** DinoHash falls at about 1/255, NeuralHash between 4 and 8/255. CertPHash
  mostly survives ε=8/255 at calibrated thresholds, but not at L1 1800 unless it is ε=8.
- **Trained vs untrained DinoHash:** training does help at small budgets (bit accuracy 45.6% vs 20.6% at ε=1;
  random-target collisions 51% vs 92.5%), but it isn't enough at the matching threshold. **This can't be credited
  to fine-tuning alone**, since the two also have different final layers (`benchmark_findings.md`, F17).
- **Evasion and collision differ:** NeuralHash is evaded at 94.5% at ε=8 but gets only 16.5% on random-target
  collisions.

---

## 4. Investigation: why our numbers differ from the DinoHash paper

### 4.1 The disagreement

DinoHash paper, Table 1 (100-step APGD, bit accuracy), against our full sweep:

| Model | Paper ε=4 | Ours ε=4 | Paper ε=8 | Ours ε=8 |
|---|---|---|---|---|
| DinoHash trained | 72.2% | **10.3%** | 64.8% | **3.7%** |
| DinoHash untrained | 0.3% | 0.1% | 0.0% | 0.0% |
| NeuralHash | 0.1% | **78.4%** | 0.0% | **59.2%** |

### 4.2 How it was investigated

1. Read the paper's method, and the authors' code at `github.com/proteus-photos/dinohash-perceptual-hash`, including
   its git history.
2. Wrote an exact copy of their APGD (the ℓ∞ path of `attack_single_run`) in a separate script.
3. Ran it beside our variants, on our verified models and the first images of `V200`, without quantising first. No
   project code or results were changed.

| Difference | Authors' attack | Our sweep |
|---|---|---|
| Loss | BCE to the flipped hash on `20 × unit(logits)`, which keeps pushing every bit | Hinge, which stops once a bit flips |
| Start | Clean image | Random, then warm start |
| Optimiser | APGD, starting at 2ε | APGD and small-step PGD |
| Result kept | Best loss | Best hash distance on the 8-bit image |
| Steps | 100 | 100 per optimiser, per budget, building on the previous budget |

### 4.3 Check that the copy is right

Untrained DinoHash, authors' attack (5 images): **0.0% at both ε=4 and ε=8**, against the paper's 0.3% and 0.0%. The copy
reproduces the paper where we'd expect it to.

### 4.4 Trained DinoHash: the paper's attack gets stuck

Bit accuracy on the first 5 images of `V200`, trained DinoHash, without quantising:

| Attack | ε=4 | ε=8 |
|---|---|---|
| Paper, Table 1 (different model and data) | 72.2% | 64.8% |
| **Authors' APGD, exact copy** | **62.9%** | **59.4%** |
| Our APGD, hinge, no warm start | 69.0% | 57.9% |
| Small-step PGD, hinge, 100 steps | 34.6% | 18.1% |
| Small-step PGD, BCE, 100 steps | 30.2% | 14.0% |
| Full sweep, same 5 images | 2.5% | 1.9% |
| Full sweep, all 200 images | 10.3% | 3.7% |

**Reading:**
- With the paper's method we land close to the paper: 62.9% vs 72.2%, and 59.4% vs 64.8%. The remaining gap is what
  we'd expect given a different model size (see §4.6), different images and just 5 samples.
- Changing only the **step size** cuts bit accuracy to about half at ε=4 and to between a third and a quarter at ε=8.
  Changing the loss makes little difference. So APGD's large steps are what get stuck.
- The sweep goes lower still because it builds on results from smaller budgets, so it has effectively had many more
  steps. It isn't a "100-step" number and **shouldn't be presented as one**.

**Conclusion:** the paper's 72.2% is what its attack finds, not how robust the model is against a stronger attack.
This is the pattern adversarial-robustness research warns about: a defence that looks strong because the evaluated
attack can't find the weakness.

This doesn't mean that training did nothing. Trained DinoHash remains harder than untrained at the same small budgets
(§3.5).

### 4.5 NeuralHash: we can't reproduce the paper, and its script doesn't attack NeuralHash

Bit accuracy, first 10 images unless stated:

| Attack | ε=4 | ε=8 |
|---|---|---|
| Paper, Table 1 | 0.1% | 0.0% |
| **Authors' APGD, exact copy** | **75.8%** | **58.1%** |
| Same, with the authors' [0, 1] input scaling | 75.9% | 58.0% |
| Our APGD, hinge | 79.5% | 61.3% |
| APGD, BCE, best by distance | 75.5% | 56.5% |
| Small-step PGD, hinge | 80.6% | 65.9% |
| Small-step PGD, BCE | 77.2% | 61.4% |
| Small-step PGD, hinge, **1,000 steps** (3 images) | 81.2% | – |
| Small-step PGD, BCE, **1,000 steps** (3 images) | 76.0% | – |
| Full sweep, all 200 images | 78.4% | 59.2% |

All variants agree. Neither input scaling nor 10× more steps gets anywhere near 0.1%. The same code takes untrained
DinoHash to 0.0%, so the attack itself is capable of it.

**Why the paper might show 0.1%.** The authors' only committed NeuralHash evaluation is
`neuralhash_attack_results.py`. In its earliest version (commit `f8c2fd5d65`, 2025-06-24):

1. It builds `APGDAttack(eps=...)` with no model. The `apgd_attack.py` in effect at that commit (`c7008888b7`, 2025-04-04)
   has `dinohash` **hard-coded in its loss**. So the attack optimises DinoHash, and NeuralHash receives no gradient.
2. `load_model` is never called, unlike in `attack_results.py`, so as far as we can read the DinoHash it attacks has the
   untrained weights.
3. NeuralHash supplies only the reference bits. The result is scored with `dinohash(adv_images)` against those bits.

So it measures how far an attack on untrained DinoHash can push DinoHash's bits away from NeuralHash's bits.

**Test.** We reproduced that calculation with our models, per image, on 5 images:

| | Result |
|---|---|
| Clean agreement, untrained DinoHash vs NeuralHash bits | 47.5% (about what unrelated hashes give) |
| After the attack, ε=4 | **0.0%** |
| After the attack, ε=8 | **0.0%** |

That matches the paper's near-zero column, while a real attack on NeuralHash gives 58–81%.

**Limits of this explanation:**
- **We can't prove this script produced the table.** It may have been run in another form, or the table may have other
  origins. The paper first appeared on arXiv in March 2025 (arXiv 2503.11195), before this commit, and we didn't compare
  arXiv versions, so the numbers may predate the script.
- **The committed script looks unable to run as written.** By our reading, it computes the reference from the first
  image of each batch only, which wouldn't fit the attack's per-batch loss at the default batch size, and passes 360 px
  images to a model with 14 px patches. We corrected this in our test, so it reproduces the logic, not the exact run.
- **Later versions changed.** `attack_results.py` now prints a variable named "accuracy" that measures the share of bits
  that *differ*. This code is from after the workshop paper (October 2025) and the June 2025 version measures matching
  bits, so it **doesn't** explain Table 1.

### 4.6 What remains unexplained

| Difference | Effect |
|---|---|
| **Model size:** the paper describes ViT-B/14 at 96 bits, and the released 96-bit model is ViT-S/14 | May explain part of 62.9 vs 72.2. A ViT-B/96 trained model isn't available |
| **Images:** DiffusionDB vs COCO | Unmeasured |
| **Sample size:** 3–10 images in these probes | Direction is clear; exact values aren't |

### 4.7 Can the attack results be trusted?

| Claim | Confidence | Why |
|---|---|---|
| Reported successes are real | **High** | Each was recomputed on the 8-bit image with verified models |
| Trained DinoHash is much weaker than 72% suggests | **High** | Even simple small-step PGD reaches 30–35% at ε=4 |
| NeuralHash is far from 0.1% at ε=4 | **High on our setup** | Six attack variants agree, including much longer runs |
| Paper's NeuralHash column is an artefact | **Moderate** | The script and test fit, but we can't confirm the origin |
| NeuralHash and DinoHash sweep values are the strongest possible | **No** | No attack can prove this; see below |

**Our sweep is slightly weak on NeuralHash.** BCE beats our hinge on NeuralHash: on the same 10 images at ε=8, the sweep flips
38.4 bits and APGD with BCE flips 42.2. This could raise NeuralHash's evasion rate near the 26-bit threshold (18% at ε=4), so
those numbers are lower bounds. It doesn't change the comparison with the paper. Results at the large-budget end of
§3.1–3.3 are least affected.

**How to word this in the thesis:** "We could not reproduce Table 1 of the DinoHash paper. With an exact port of the authors'
attack, we observe figures close to the paper's for DinoHash but not for NeuralHash. A smaller-step attack breaks trained DinoHash
much further. The NeuralHash column may reflect an evaluation error, which we raise as a possibility, not a settled fact."
Contact the authors before publishing a stronger claim.

---

## 5. Comparison with the CertPHash paper

Their results use an ℓ2 PGD attack, while ours use ℓ∞, so only the overall pattern is comparable.

| Model | Paper: evasion | Ours: evasion, L1 1800, ε=8/255 | Paper: collision | Ours: closest target, L1 1800, ε=16/255 |
|---|---|---|---|---|
| Base | 89.2% | 99.0% | 89.8% | 51.5% |
| Adv | 68.8% | 60.5% | 21.7% | 2.5% |
| Certified ε=1 | 54.6% | 48.0% | 37.5% | 7.5% |
| Certified ε=2 | – | 43.0% | 27.4% | 24.5% |
| Certified ε=4 | 18.7% | 22.5% | – | 33.5% |
| Certified ε=8 | 1.0% | 0% | 18.7% | 44.5% |

- **Evasion agrees well:** the same ordering, with similar values. Unlike DinoHash, CertPHash's claims hold up under our attack.
- **Collision doesn't compare directly.** The paper's pool is the whole test set and ours is 200 images. The ε=16 column is only
  chosen as the sweep budget nearest below the size noted in `run_adversarial.py` for their collisions (ℓ∞ about 0.10, or 25/255). Our ordering among certified versions
  runs the other way, consistent with §3.3.
- **Certificates weren't tested.** These are attack results. They can't confirm or refute a mathematical guarantee. `hashbench/README.md`
  also records that the released `verify_evasion.py` certifies a one-value L0 change with PDQ settings, so the paper's certified no-evasion
  rate can't be reproduced from it as described.

---

## 6. Limitations

1. **Attacks act on the resized model input,** not the full-size image. Attacking before the resize is a separate, harder setting.
2. **Warm start makes the ε-monotonicity check weak.** Each budget starts from the previous one's best result, so "no drops" holds by
   construction. It no longer independently shows that the attack converged.
3. **Sweep numbers aren't plain 100-step figures** (§4.4).
4. **No attack proves robustness.** Low success rates are upper bounds on how robust a model looks, not guarantees. Only strong attacks were
   tried here; there is no black-box attack or transfer test yet.
5. **DinoHash can't separate backbone from head** (F17), and each CertPHash ε has only one released model, so ε effects can't be
   separated from individual training runs.
6. **Calibrated thresholds rest on COCO,** and the hash thresholds move in whole bits, as noted in `benchmark_findings.md`.
7. **Closest targets come from 200 images,** and some already match before attack (§3.3).
8. **The investigation's sample is small:** 3–10 images per setting. Use it to explain, not as final figures.
9. **The investigation's scripts aren't in the repo.** They were throwaway scripts, kept outside the project.

---

## 7. Next steps

1. **Ask the authors** about the NeuralHash column and which model produced Table 1. Their answer decides how strongly the thesis can
   word §4.
2. **Optional, confirm §4.4 at scale:** run the authors' exact attack and plain 100-step small-step PGD on all 200 images, for both DinoHash
   models. This gives comparable figures to quote.
3. **Optional, strengthen NeuralHash:** rerun with BCE added, so its success rates aren't understated.
4. **Later in Phase C:** black-box attacks and a transferability matrix.

## 8. Where the data is

| What | Path |
|---|---|
| DinoHash and CertPHash results | `hashbench/results/adv_v200_kaggle/` (`raw.csv`, `summary.csv`, `summary.md`, `config.json`, log) |
| NeuralHash results | `hashbench/results/adv_v200_neuralhash/` (same files) |
| Attack code | `hashbench/hashbench/adversarial.py` |
| Sweep script | `hashbench/scripts/run_adversarial.py` |
| Tests | `hashbench/tests/test_adversarial.py` |
| Earlier pilots | `hashbench/results/adv_pilot_*` (superseded) |

**To rebuild the summary tables,** from `hashbench/`:

```
python scripts/run_adversarial.py --out results/adv_v200_kaggle --summary-only
python scripts/run_adversarial.py --out results/adv_v200_neuralhash --summary-only
```

## References

- Singhi, Gupta, Struppek. *DINOHash.* ICML 2025 CODEML workshop. arXiv 2503.11195.
- Yang et al. *CertPHash.* USENIX Security 2025.
- Struppek et al. *Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash.* FAccT 2022.
- Croce, Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks.* ICML 2020.
- Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018.
