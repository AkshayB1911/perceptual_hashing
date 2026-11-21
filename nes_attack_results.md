# NES black-box attack results (attack #2 of `future_attacks.md`)

*BTP: Perceptual Hashing. Runs 2026-11-17/18 (Kaggle T4). Data:
`hashbench/results/nes_v200_dino-t/`, `nes_v200_dino-u/`, `nes_v200_nh-cert/` (per-job outputs and
logs), `nes_v200_classical/` (pHash / PDQ, CPU run on a separate Linux PC), all merged in
`hashbench/results/nes_v200_combined/` (`nes.csv`, `summary.csv`, `summary.md`).
DinoHash trained at NeuralHash's 26-bit threshold: `nes_v200_dino-t-thr26/` (§5.3).*

## 1. Question

Can an attacker who has **no model weights** break the hashes? They can only send an image to a
hashing service and read back its hash. This is the realistic attacker. It also tests whether the
white-box results (where DinoHash was the weakest) only hold for an attacker with the weights.

## 2. Setup

| Item | Value |
|---|---|
| Attack | NES score-based attack (Ilyas et al. 2018): antithetic Gaussian probes in a 3×32×32 space upsampled bilinearly to the input size, momentum 0.9, sign step clipped to the ℓ∞ ball, step size halved after 5 steps without improvement |
| Settings | 20 pairs per step (41 queries per step), σ = 1.0·ε, step 0.2·ε. Chosen by a 6-image pilot (`results/nes_pilot_kaggle/`) over σ ∈ {0.5, 1, 2}·ε and 20 / 50 pairs |
| What the attacker sees | The discrete hash only (96 bits; CertPHash's 144 rounded bytes). Every queried image is rounded to 8 bits |
| Budget | 10,000 queries per image. The attack stops at the first success, so one run gives the whole success-vs-queries curve |
| Settings run | Evasion ε = 8/255 and 4/255; random-target collision ε = 16/255 (target: image i + n/2) |
| Success | At each hash's calibrated FPR 1e-3 threshold, the same as all other Phase C results (DinoHash-T 17 bits, Dino-U 21, NeuralHash 26, CertPHash L1 2581–5435) |
| Images | `V200` (200 COCO val2017 images) |
| Hashes | DinoHash trained / untrained, NeuralHash, 6 CertPHash variants (Kaggle T4); pHash and PDQ (CPU, §6) |
| Code | `hashbench/hashbench/blackbox.py`, `scripts/run_nes.py`, Kaggle package `kaggle/nes/` (older scripts unchanged). NeuralHash came from a separate private Kaggle dataset |
| Run time | Kaggle T4: DinoHash trained 6.1 h, untrained 5.8 h, NeuralHash + 6 CertPHash 2.5 h. pHash + PDQ: 10.8 h on an i7-12700 with 19 worker processes (204 core-hours) |
| Check | 3 attacked Dino-T images were saved and re-hashed with the plain DinoHash pipeline: all evade (18 bits > 17) at ℓ∞ 8/255 |

## 3. Results at the calibrated thresholds

Success (%) within 10,000 queries, with 95% Wilson intervals. "Median q" = median queries of the
successful images.

| Hash | Evasion ε=8 | median q | Evasion ε=4 | median q | Collision ε=16 | median q |
|---|---|---|---|---|---|---|
| DinoHash untrained | **87.5** (82.2–91.4) | 1,477 | **47.5** (40.7–54.4) | 4,060 | 1.5 (0.5–4.3) | 8,406 |
| DinoHash trained | **82.0** (76.1–86.7) | 924 | **39.5** (33.0–46.4) | 3,240 | 0.0 (0.0–1.9) | – |
| NeuralHash | 5.0 (2.7–9.0) | 3,917 | 0.5 (0.1–2.8) | 247 | 1.0 (0.3–3.6) | 6,684 |
| CertPHash ε1 | 0.0 | – | 0.0 | – | 6.5 (3.8–10.8) | 1,026 |
| CertPHash base | 0.0 | – | 0.0 | – | 8.0 (5.0–12.6) | 3,732 |
| CertPHash adv | 0.5 | 4,388 | 0.0 | – | **13.0** (9.0–18.4) | 4,081 |
| CertPHash ε2 | 1.5 | 329 | 0.0 | – | 2.5 (1.1–5.7) | 5,249 |
| CertPHash ε4 | 1.0 | 4,655 | 0.0 | – | 1.5 (0.5–4.3) | 657 |
| CertPHash ε8 | 0.0 | – | 0.0 | – | 0.5 (0.1–2.8) | 6,069 |
| pHash | 6.5 (4.0–10.8) | 3,035 | 1.0 (0.3–3.6) | 5,946 | 13.5 (9.5–18.9) | 3,650 |
| PDQ | 5.0 (2.7–9.0) | 4,880 | 0.5 (0.1–2.8) | 5,987 | **98.5** (95.7–99.5) | 616 |

Success within smaller budgets (1k / 2k / 5k queries) is in `results/nes_v200_combined/summary.md`.
At ε=8, DinoHash trained is evaded for 43.5% of images within 1,000 queries.

Mean SSIM of the attacked images: 0.97 at ε=8, 0.99 at ε=4, 0.90–0.95 at ε=16.

## 4. Comparison with the other attacks (success %, calibrated thresholds)

| Hash | White-box evasion ε=8 | **NES evasion ε=8** | White-box evasion ε=4 | **NES evasion ε=4** | White-box collision ε=16 | **NES collision ε=16** |
|---|---|---|---|---|---|---|
| DinoHash untrained | 100 | **87.5** | 100 | **47.5** | 100 | **1.5** |
| DinoHash trained | 100 | **82.0** | 100 | **39.5** | 100 | **0.0** |
| NeuralHash | 94.5 | **5.0** | 18.0 | **0.5** | 56.5 | **1.0** |
| CertPHash ε1 | 1.0 | **0.0** | 0.0 | **0.0** | 53.0 | **6.5** |
| CertPHash base | 11.5 | **0.0** | 0.0 | **0.0** | 88.0 | **8.0** |
| CertPHash adv | 1.5 | **0.5** | 0.5 | **0.0** | 51.0 | **13.0** |
| CertPHash ε2 | 3.5 | **1.5** | 1.0 | **0.0** | 20.5 | **2.5** |
| CertPHash ε4 | 2.5 | **1.0** | 0.0 | **0.0** | 19.0 | **1.5** |
| CertPHash ε8 | 0.0 | **0.0** | 0.0 | **0.0** | 15.2 | **0.5** |
| pHash | n/a | **6.5** | n/a | **1.0** | n/a | **13.5** |
| PDQ | n/a | **5.0** | n/a | **0.5** | n/a | **98.5** |

White-box: the V200 sweep (`results/adv_v200_kaggle`, `results/adv_v200_neuralhash`). Transfer
(`transfer_attack_results.md`, the zero-query black-box attack): NeuralHash is reached by at most 3.5%
from any other hash; transfer works only inside the DinoHash pair and inside the CertPHash family.

**What this shows:**
- **DinoHash is the easiest hash to evade even without the weights.** NES evades the trained model on
  82% of images at ε=8, most of them within about 1,000 queries.
- **NeuralHash drops from 94.5% (white-box) to 5% (black-box).** Its weakness needs gradients;
  DinoHash's does not.
- **CertPHash resists black-box evasion completely** (≤ 1.5%), as it does white-box.
- **Collision is much harder without gradients.** It stays at 0–1.5% for DinoHash and NeuralHash.
  CertPHash is the exception: random-target collisions reach 6.5–13% for the non-certified-strength
  variants (base, adv, ε1). The strongest certified model (ε8) stays at 0.5%.
- **PDQ is the exception for collisions: 98.5%**, most within about 600 queries (§6).
- pHash and PDQ have no gradients, so they have no white-box column (n/a); NES is the first attack in
  this project that reaches them directly. Transfer gave them 0% evasion and ≤ 1% collision.

## 5. Is the threshold the reason? A like-for-like comparison

The calibrated thresholds differ: an evasion succeeds on DinoHash trained at more than **17** flipped
bits, on NeuralHash only at more than **26**. DinoHash gets the stricter value because different
images already land closer together in its hash space (the heavy impostor tail, finding F19 in
`benchmark_findings.md`). So an evasion needs fewer flipped bits on DinoHash. This section checks how
much of the gap comes from that.

### 5.1 White-box, without any threshold: mean bits flipped (of 96)

The white-box attack keeps the iterate with the most bits flipped, so these numbers do not depend on a
threshold.

| ε (/255) | DinoHash trained | DinoHash untrained | NeuralHash |
|---|---|---|---|
| 1 | 52.2 | 76.2 | 5.3 |
| 2 | 74.6 | 93.8 | 10.6 |
| 4 | 86.1 | 95.9 | 20.7 |
| 8 | 92.5 | 96.0 | 39.2 |

The same bar for every hash (white-box evasion success %, at > 17 / > 21 / > 26 bits):

| ε (/255) | DinoHash trained | NeuralHash |
|---|---|---|
| 2 | 100 / 99.5 / 99.0 | 9.5 / 2.0 / 0.5 |
| 4 | 100 / 100 / 100 | 65.0 / 37.5 / 18.0 |
| 8 | 100 / 100 / 100 | 100 / 99.5 / 94.5 |

White-box collision at ε=8, mean Hamming distance to the target after the attack: DinoHash trained
0.8, untrained 0.0, NeuralHash 33.1. With gradients, DinoHash can be steered to almost any hash.

### 5.2 NES at a common bar

NES stops at the first success, so for a successful image we only know that the distance passed that
hash's own threshold. Success at a **lower** bar can therefore be read from the data; success at a
higher bar cannot (§5.3).

| NES evasion, success % | > 17 bits (Dino-T's bar) | > 21 bits (Dino-U's bar) | > 26 bits (NH's bar) |
|---|---|---|---|
| ε=8: DinoHash trained | **82.0** | n/a | 41.5 (§5.3) |
| ε=8: DinoHash untrained | 95.0 | **87.5** | n/a |
| ε=8: NeuralHash | 31.5 | 17.0 | **5.0** |
| ε=4: DinoHash trained | **39.5** | n/a | 6.5 (§5.3) |
| ε=4: DinoHash untrained | 73.0 | **47.5** | n/a |
| ε=4: NeuralHash | 3.0 | 0.5 | **0.5** |

**Answer:** the threshold explains part of the black-box gap, but not all of it (at NeuralHash's
26-bit bar DinoHash trained is still evaded 41.5% vs 5.0%, §5.3). At the same 17-bit
bar, NES evades DinoHash trained on 82% of images and NeuralHash on 31.5% (ε=8); at ε=4 the
numbers are 39.5% and 3%. Without any threshold (§5.1), the white-box gap is even larger. So DinoHash
is genuinely less robust to adversarial perturbations than NeuralHash in our tests. The stricter
threshold that its weaker separation of different images forces only adds to this.

### 5.3 DinoHash trained at NeuralHash's 26-bit threshold

A rerun of the DinoHash-trained NES job with the threshold set to NeuralHash's 26 bits; everything
else (settings, images, seeds, targets) unchanged. Kaggle T4, 7.4 h
(`kaggle/nes/kernel/run_nes_thr26_kernel.py`, output `results/nes_v200_dino-t-thr26/`).

| Same 26-bit bar, success % (95% CI) | DinoHash trained | NeuralHash |
|---|---|---|
| NES evasion ε=8 | **41.5** (34.9–48.4) | 5.0 (2.7–9.0) |
| NES evasion ε=4 | **6.5** (3.8–10.8) | 0.5 (0.1–2.8) |
| NES collision ε=16 (match = ≤ 26 bits) | **5.0** (2.7–9.0) | 1.0 (0.3–3.6) |

Median queries of successful evasions at ε=8: 3,978 (vs 924 at the 17-bit bar).

**Answer:** with the same bar, DinoHash trained is still clearly easier to break without the weights:
about 8× NeuralHash's evasion rate at ε=8, and higher at ε=4 and for collisions. The threshold
explains about half of the ε=8 gap (82.0% → 41.5%), not all of it. Collisions get easier for
DinoHash at 26 bits because a looser bar is easier to reach.

## 6. pHash and PDQ

Run on a separate Linux PC (i7-12700, 20 threads, 19 worker processes; 2026-11-17 23:39 to
2026-11-18 10:26) with the package `offline/phash_pdq/` (`scripts/run_nes_classical.py` +
`hashbench/blackbox_classical.py`). Same NES settings, images, seeds, collision targets and calibrated
thresholds (pHash 16/64 bits, PDQ 100/256 bits) as the Kaggle runs; `config.json` and `env.txt` in
`results/nes_v200_classical/` record the settings and the machine. One difference: pHash and PDQ do
their own resizing, so the perturbation lives on the whole image (max side 512), not on a model input.

| Hash | Evasion ε=8 | Evasion ε=4 | Collision ε=16 | Collision: distance to target before → after (mean) |
|---|---|---|---|---|
| pHash (64 bit) | 6.5% | 1.0% | 13.5% | 30.3 → 22.3 bits (threshold 16) |
| PDQ (256 bit) | 5.0% | 0.5% | **98.5%** | 127.4 → 99.7 bits (threshold 100) |

**What this shows:**
- **Evasion:** pHash and PDQ are about as hard to evade as NeuralHash (5–6.5% at ε=8), and far
  harder than DinoHash (82%). Low-frequency DCT bits change little under ℓ∞ noise of 8/255.
- **Collision on PDQ is easy (98.5%, median 616 queries).** The cause is PDQ's loose calibrated
  threshold: at FPR 1e-3 a match is anything within 100 of 256 bits. Two unrelated images start
  about 127 bits apart, so the attack only has to remove about 28 bits. On pHash it has to go from
  30 to 16 of 64 bits, and only 13.5% succeed.
- PDQ's own recommended threshold is much stricter: 31 bits, where the benign benchmark saw zero
  false matches in 200,000 pairs. At 31 bits the attack would have to remove about 96 bits instead
  of 28, so this collision result depends strongly on the threshold. A run at 31 bits would test
  this.
- **Run time:** 204 core-hours, far more than the 37 core-hours estimated from laptop timings. Per
  query, each process took 64 ms (pHash) and 89 ms (PDQ), against 6.7 and 15.5 ms in the
  single-process laptop measurement: 6–10 times slower with 19 processes sharing 12 cores.

## 7. Comparison with the papers

| Source | Claim | Ours |
|---|---|---|
| DinoHash paper (Singhi et al., ICML 2025 CODEML workshop) | DinoHash is much more adversarially robust than NeuralHash (Table 1, white-box only) | Not supported. The paper's NeuralHash row comes from a script that attacks DinoHash (`dinohash_table1_reproduction.md` §4). Correctly attacked, NeuralHash is at least as robust under the authors' own attack, and clearly more robust under our stronger white-box attack and NES |
| DinoHash paper | Adversarial training makes DinoHash much more robust than plain DINOv2 | Reproduces under the authors' attack (68.9% vs 0.0% bit accuracy). Under NES the gain is small: 82.0% vs 87.5% evasion at ε=8 |
| Sun et al. 2026 (arXiv 2608.03101) | Black-box attack success: DINOHash 15.6%, NeuralHash 35.2% | Opposite ranking: 82.0% vs 5.0%. Their attack, budget, threshold and data differ, so the absolute numbers are not comparable. Repeating their setting (`nes_sun_attack_results.md`) gives DinoHash 41.2% vs NeuralHash 24.5%: our ranking holds there too |
| CertPHash paper | Certified training gives empirical robustness | Holds black-box too: ≤ 1.5% evasion for all variants; collision success falls with the certification strength (adv 13% → ε8 0.5%) |

## 8. Limitations

- **One NES configuration**, tuned on 6 images. A stronger black-box attack (more pairs, a different
  search space, or restarts) could raise all numbers; the ranking is what matters most here.
- **The perturbation lives at each model's input size** (224 / 360 / 64 px), as in the white-box
  sweep, not on the original image. This matches how the hashes were attacked before and keeps them
  comparable, but a real attacker would have to perturb the full image before the service resizes it.
- **Success for successful images is a lower bound on bits flipped**, because the attack stops at the
  first success (§5.2).
- **200 images.** The 95% intervals are about ±5–7 points around the DinoHash numbers.
