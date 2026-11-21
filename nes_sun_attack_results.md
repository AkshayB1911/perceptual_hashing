# NES black-box attack in the setting of Sun et al. 2026 (replication)

*BTP: Perceptual Hashing. Runs 2026-11-18/19 (Kaggle T4). A second, separate black-box run next to
our own NES results (`nes_attack_results.md`, unchanged). Data: `hashbench/results/nes_sun_v200_dino/`,
`nes_sun_v200_nh-cert/` (per-job outputs and logs), merged in `nes_sun_v200_combined/`
(`nes_sun.csv`, `summary.csv`, `summary.md`).*

## 1. Why

Sun et al. 2026 (arXiv 2608.03101) report a black-box (NES) attack in which DinoHash is **more**
robust than NeuralHash (average success 15.6% vs 35.2%). Our own NES run found the opposite
(82.0% vs 5.0% at ε = 8/255). Their setting differs from ours in almost every detail, so this run
repeats their setting as closely as the paper allows and checks which ranking holds.

## 2. Setup

| Item | Sun et al. (as stated in the paper) | This run |
|---|---|---|
| Images | 100 each from ImageNet, COCO, Stable Diffusion | V200 (200 COCO val2017 images) |
| Image size | RGB, resized to 512×512, [0, 1], clipped after modification | Same (resize: PIL bilinear, not stated) |
| Budget | ℓ2 caps 40 / 90 / 180 | Same, on the 512×512×3 image |
| Search space | 64×64 grayscale, mapped to RGB | Same (bilinear upsample, gray copied to R, G, B; not stated) |
| NES | 32 antithetic directions, NES scale 4/255, step 2/255, clean start | Same; update z ← z + 2/255·sign(g), then ℓ2 projection (probes projected too) |
| Limits | ≤ 500 steps, 1,500-query limit, early stop at success | Same; 65 queries per step (64 probes + 1 check) → 23 steps |
| Success | Bit error rate > 0.2 for all hashes | Same: ≥ 20 of 96 bits (DinoHash, NeuralHash). CertPHash has no bits: its FPR 1e-3 L1 threshold |
| Seed | 2026 | 2026 (+ an offset per budget and batch) |
| Hashes | NeuralHash, DINOHash (+ 6 others) | DinoHash trained / untrained, NeuralHash, 6 CertPHash |
| Queries | float images | rounded to 8 bits (a real upload), as in all our attacks |

**The hash service is the real pipeline.** The attacker sends the 512×512 image and each hash applies
its own resize (DinoHash: PIL bilinear to 224, NeuralHash: PIL bicubic to 360, CertPHash: torchvision
bilinear-antialias to 64). The PIL resizes were reproduced on the GPU with exact integer arithmetic;
a self-check at the start of each job confirmed they equal Pillow's output pixel for pixel. Every
returned image was also re-hashed through the plain preprocessing: the attack's own success decision
agreed with it for **all 5,400 rows**.

Code (new files only; earlier scripts unchanged): `hashbench/hashbench/blackbox_sun.py`,
`scripts/run_nes_sun.py`, `tests/test_blackbox_sun.py`, Kaggle package `kaggle/nes_sun/`.
Run time on a T4: DinoHash ×2 2.3 h, NeuralHash + 6 CertPHash 0.9 h.

## 3. Results

Evasion success (%), 200 images per cell. "Average" is the mean over the three budgets, the number
Sun et al. give in their Table 1.

| Hash | ℓ2 40 | ℓ2 90 | ℓ2 180 | **Average** | Sun et al. average |
|---|---|---|---|---|---|
| DinoHash untrained | 58.0 | 53.0 | 53.0 | **54.7** | – |
| DinoHash trained | 42.0 | 40.0 | 41.5 | **41.2** | 15.6 |
| NeuralHash | 26.0 | 24.0 | 23.5 | **24.5** | 35.2 |
| CertPHash ε2 | 2.5 | 2.5 | 2.5 | 2.5 | – |
| CertPHash ε4 | 2.5 | 2.0 | 2.0 | 2.2 | – |
| CertPHash adv | 0.5 | 0.5 | 0.5 | 0.5 | – |
| CertPHash ε1 / base / ε8 | 0.0 | 0.0 | 0.0 | 0.0 | – |

95% intervals: about ±7 points for DinoHash and NeuralHash (e.g. DinoHash trained at ℓ2 40:
35.4–48.9; NeuralHash: 20.4–32.5).

Sun et al. per budget: NeuralHash 16.3 / 28.0 / 61.3, DINOHash 3.7 / 8.7 / 34.3.

Median queries of the successful attacks: about 800–1,100. Mean SSIM of the returned images: 0.955
(DinoHash, NeuralHash), 0.94 (CertPHash).

## 4. What this shows

- **Same ranking as our own NES run: DinoHash is easier to evade than NeuralHash** (41.2% vs 24.5%
  trained; the untrained DinoHash is easier still). This is the opposite of Sun et al.'s ranking.
- **The gap is much smaller than in our own setting** (82.0% vs 5.0%). Two reasons: here both hashes
  face the same 20-bit bar (in our setting NeuralHash's calibrated bar is 26 bits, DinoHash's 17),
  and the attack works on the full 512×512 image, before each hash's own resize.
- **CertPHash stays robust** (≤ 2.5%), as in every other attack.

## 5. The ℓ2 budget never mattered here

Success is flat across the three budgets, while in Sun et al. it rises steeply (NeuralHash 16 → 61%).
The reason is the query limit:

- With 32 antithetic directions, a step costs 65 queries, so 1,500 queries allow only **23 steps**.
- 23 sign steps of 2/255 move the image by at most ℓ2 ≈ 26 in practice (largest seen: 25.9). Even the
  smallest cap (40) is never reached, so the three budgets are the same attack with different random
  seeds; the 0–5 point differences between them are noise.
- Our images stay very close to the originals (SSIM 0.955). Sun et al. report SSIM 0.625 / 0.330 /
  0.171 for the three budgets, i.e. their perturbations fill the budget.

So their attack must take far more than 23 steps (their stated cap is 500). The paper does not say
how queries are counted per step; a reading where the probes are not counted against the
1,500-query limit (or far fewer probes per step) is the most likely explanation. **This run is the
literal reading of the paper.** It gives a fair comparison between the hashes, because all of them
get exactly the same attack, but its absolute numbers are not comparable with Sun et al.'s.

A variant that allows the full 500 steps was run next (§6).

## 6. 500-step variant

Same setting, but the attack runs up to the stated 500 steps (65 queries per step, so up to 32,501
queries per image) instead of stopping at 1,500 queries. First 100 images of V200 (Sun et al. use
100 per dataset). Kaggle T4, `kaggle/nes_sun/kernel/run_nes_sun500_kernel.py`; data
`hashbench/results/nes_sun500_v100_nh-all/`, `nes_sun500_v100_dino-180/`, merged in
`nes_sun500_v100_combined/`. Run time: NeuralHash (3 budgets) 2.5 h, DinoHash ℓ2 180 1.1 h.
DinoHash at ℓ2 40 and 90 was not run (§6.2).

Evasion success (%), the same 100 images in every column:

| Hash | 23 steps, ℓ2 40 / 90 / 180 | **500 steps, ℓ2 40 / 90 / 180** | Sun et al., ℓ2 40 / 90 / 180 |
|---|---|---|---|
| DinoHash trained | 43 / 40 / 42 | – / – / **84** (75.6–89.9) | 3.7 / 8.7 / 34.3 |
| NeuralHash | 28 / 25 / 23 | **63 / 69 / 63** (±9) | 16.3 / 28.0 / 61.3 |

All 400 rows were confirmed by the plain hasher.

### 6.1 What this shows

- **More steps raise both hashes a lot** (NeuralHash about 25% → 65%, DinoHash 42% → 84%), so the
  step count, not the budget, was what limited the literal run.
- **DinoHash is still easier to evade than NeuralHash** at ℓ2 180 (84% vs 63% on the same 100 images).
  The ranking is the same in all three of our settings.
- **NeuralHash at ℓ2 180 (63%) is close to Sun et al.'s 61.3%**; DinoHash (84%) is far from their 34.3%.
  So the disagreement with Sun et al. is about DinoHash only.

### 6.2 The ℓ2 cap still does not bind

- Even after 500 steps, the returned images have mean ℓ2 ≈ 22–30 and at most 59 (DinoHash, 180).
  Their SSIM is 0.92–0.94. Sun et al. report SSIM 0.625 / 0.330 / 0.171.
- The sign steps mostly cancel out: z wanders around a small region instead of growing to the cap.
  NeuralHash's three budgets therefore give the same result (63 / 69 / 63; the differences are seed
  noise), and DinoHash at ℓ2 40 and 90 would almost certainly match its ℓ2 180 number. That is why
  those two jobs were not run.
- So with the stated hyper-parameters (step 2/255, NES scale 4/255, sign update), this NES cannot
  produce perturbations as large as Sun et al.'s. Their attack must differ in a detail the paper does
  not give: for example the update rule (a gradient step scaled to the ℓ2 budget instead of a sign
  step), how far each step moves, or the definition of the reported SSIM. Without their code, we
  cannot match their absolute numbers more closely than this.

## 7. Limitations

- **Different images**: 200 COCO images instead of 300 from three sources.
- **Unstated details** filled in with standard choices (§2); the query-counting one matters (§5).
- **Which DinoHash**: we use the released ViT-S/14 96-bit model; Sun et al. do not name the version.
- **CertPHash uses its own threshold**, since a bit-error-rate bar does not apply to its byte hash.
