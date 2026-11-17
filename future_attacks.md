# Future attacks: catalogue, details and ranking

*BTP: Perceptual Hashing. Written 2026-11-16. This is the list of attacks still possible after the
Phase C white-box sweep (`adversarial_testing_report.md`) and the DinoHash Table 1 reproduction
(`dinohash_table1_reproduction.md`). Progress tracking is in `phaseC_implementation_progress.md`;
the research ideas these come from are in `thesis_research_plan.md` (Phase C and Ideas 1–7).*

**Status markers:** ✅ = done, ◐ = partly done, ⬜ = not started.

**Novelty caveat:** none of the 12 attacks is a new algorithm; they are all published techniques.
What can be new is *applying* them to these hashes, or the *comparison* (five hash families,
FPR-calibrated thresholds). The novelty judgements below come from the papers read so far and
need a literature check (planned Nov 20 – Dec 5) before anything is claimed as new.

---

## 0. Already done (for reference)

| Attack | Status | Where |
|---|---|---|
| White-box evasion, ℓ∞ (APGD + small-step PGD ensemble, warm start) | ✅ | `results/adv_v200_*`, `adversarial_testing_report.md` |
| White-box collision, random and closest target | ✅ | same |
| The DinoHash authors' APGD (Table 1 reproduction, 50 images) | ✅ | `dinohash_table1_reproduction.md` |
| Benign transforms (23 families, 88 settings) | ✅ | `benchmark_findings.md` |
| Transfer (#1): 9 sources × 11 targets, V200, with random-noise control | ✅ 2026-11-17 | `transfer_attack_results.md`, `results/transfer_v200_matrix/` |

### Terms used below

| Term | Meaning |
|---|---|
| **Evasion** | Change an image slightly so its hash **no longer matches** the original (distance > threshold). The attacker is someone hiding a known image from a scanner |
| **Collision** | Change an image slightly so its hash **matches a chosen target** hash (distance ≤ threshold). The attacker is framing an innocent image or flooding the system with false matches |
| **White-box** | The attacker has the model weights and can compute gradients |
| **Black-box (score-based)** | The attacker can only send images and see the full hash output |
| **Hard-label (decision-based)** | The attacker sees only "match / no match", as with a real scanning service |
| **ε (budget)** | The largest allowed change. ℓ∞: no pixel changes by more than ε (e.g. 8/255). ℓ2: total change energy ‖δ‖₂ ≤ ε |
| **Query** | One call to the hash function. Black-box attacks are measured by how many queries they need |
| **Threshold** | The match threshold for each hash, calibrated so every hash has the same false-match rate (FPR 1e-3) |

---

## 1. Ranking: feasibility + novelty

Scores are 1–5 (higher is better). **Feasibility** means how easy the attack is to build and run with
the existing `hashbench` code and the requested compute. **Novelty** means how likely it is to
be new for perceptual hashes.

| Rank | # | Attack | Feasibility | Novelty | Total | Effort | Source |
|---|---|---|---|---|---|---|---|
| 1 | 7 | Joint evasion of several hashes | 4 | 5 | **9** | 3–4 days (learned hashes) | Plan, Idea 4 |
| 2 | 1 | Transfer attack ✅ done 2026-11-17 (`transfer_attack_results.md`) | 5 | 3 | **8** | 1–2 days | Plan, Phase C |
| 3 | 8 | Universal adversarial perturbation | 4 | 4 | **8** | 2–3 days | Added 2026-11 |
| 4 | 9 | Hash inversion, optimisation version | 3 | 4 | **7** | a few days | Plan, Idea 1 |
| 5 | 11 | Spatial / semantic attack | 3 | 4 | **7** | ~1 week | Added 2026-11 |
| 6 | 10 | Adaptive attack on DualShield | 2 | 5 | **7** | 1–2 weeks | Plan, Idea 5 |
| 7 | 4 | Soft pHash / PDQ surrogates | 3 | 3–4 | **6.5** | pHash 2 days, PDQ ~1 week | Plan, Phase C |
| 8 | 2 | NES black-box (score-based) | 4 | 2 | **6** | 3–5 days | Plan, Phase C |
| 9 | 3 | Hard-label black-box (HopSkipJump) | 3 | 3 | **6** | ~1 week | Added 2026-11 |
| 10 | 6 | EOT (transform-surviving) | 4 | 2 | **6** | 3–5 days | Plan, Phase C |
| 11 | 5 | ℓ2 budget | 5 | 1 | **6** | hours | Plan, Phase C |
| 12 | 12 | Backdoor / poisoning | 1 | 3 | **4** | weeks | Plan, Idea 7 |

**Things to keep in mind:**
- **NES (#2) ranks low on novelty but is still needed.** Without a black-box result the thesis
  cannot answer "are these hashes only broken by white-box attackers?". Sun et al. found DinoHash
  stronger than NeuralHash under black-box attacks, so this is where our ranking could change.
- **For the cluster run** (the compute request), ℓ2 (#5) and the decoder version of inversion are
  dropped. Only one of transfer (#1) and hard-label (#3) will be done; transfer is cheaper and
  scores higher.

### Suggested order

1. **Must-haves for a complete comparison:** Transfer (#1), NES (#2).
2. **Cheap novelty:** Joint evasion (#7), Universal perturbation (#8).
3. **Main idea:** Inversion vs. robustness (#9, Idea 1).
4. **If time allows:** Soft PDQ (#4), Spatial (#11), DualShield (#10).
5. **Skip:** Backdoor (#12).

### Where each attack came from, and what could be new

| # | Attack | Source | Done on perceptual hashes before? | What could be new for us |
|---|---|---|---|---|
| 1 | Transfer | Plan | Partly (not certain) | A transfer matrix across DinoHash trained/untrained, NeuralHash, CertPHash; a gradient-masking test |
| 2 | NES | Plan | **Yes**: Jain et al. 2022 (classical hashes), Sun et al. 2026 (NeuralHash, DinoHash) | CertPHash; calibrated thresholds |
| 3 | Hard-label | Added | **Yes** on classical hashes (Prokos et al. 2023); not on DinoHash that we know of | DinoHash / CertPHash under a match-only attacker |
| 4 | Soft surrogates | Plan | Possibly (to be checked) | White-box pHash/PDQ in the same table as the neural hashes |
| 5 | ℓ2 | Plan | **Yes** (CertPHash paper, Sun et al.) | Nothing new; only for comparing with other papers |
| 6 | EOT | Plan | Technique exists (Athalye 2018); robust NeuralHash attacks exist (Struppek et al. 2022) | DinoHash / CertPHash |
| 7 | Joint evasion | Plan | Not that we know of | **Likely new:** do hash ensembles work as a defence? |
| 8 | Universal | Added | Technique exists (Moosavi-Dezfooli 2017); not on these hashes that we know of | Probably a new application |
| 9 | Inversion | Plan | **Yes** (Struppek et al., Madden et al. 2024) | **Robustness vs. leakage** under controlled ablations, likely new |
| 10 | DualShield adaptive | Plan | Defence is from Aug 2026; no attacks on it yet | **New**, if it breaks |
| 11 | Spatial | Added | Technique exists (stAdv, Xiao et al. 2018); not on these hashes that we know of | Probably a new application |
| 12 | Backdoor | Plan | Exists in general ML | Possibly new for hashes; high risk |

---

## 2. Attack details

Each entry gives the threat model, how the attack works step by step, what it shows, how it fits
into the existing code, how to measure it, cost, and risks.

### #1 Transfer attack ✅ (done 2026-11-17, results in `transfer_attack_results.md`)

- **Threat model:** the attacker does **not** have the target hash's weights, but has a similar
  public model (a "surrogate"). Black-box, zero queries.
- **How it works:**
  1. Run our existing white-box attack (evasion or collision) on a **source** hash A, e.g.
     DinoHash untrained.
  2. Save the adversarial image (8-bit PNG).
  3. Hash that image with a **different target** hash B, e.g. DinoHash trained, NeuralHash,
     CertPHash, pHash, PDQ.
  4. Check whether B is also evaded (or collides).
- **Example pairs worth running:**
  - DinoHash untrained → DinoHash trained (same backbone; tests whether the robust training only
    hides gradients);
  - NeuralHash ↔ DinoHash (different architectures);
  - CertPHash adv → CertPHash ε8;
  - any learned hash → pHash / PDQ (do neural perturbations affect classical hashes?).
- **What it shows:**
  - whether an attacker needs the exact model;
  - **gradient masking:** if attacks transfer *into* trained DinoHash better than the direct
    white-box attack works, its robustness is partly gradient hiding. Our white-box results say
    the opposite (it is easy to break directly), so this is a consistency check.
- **Code:** no new attack code.
  - Add `--save-images` to `scripts/run_adversarial.py` (write the quantised adversarial PNG per
    image/model/ε/mode).
  - A new script `scripts/transfer_matrix.py` loads saved images, hashes them with every model,
    and computes distances to the clean hash / target hash.
- **Metrics:** a source × target matrix of success rate at the calibrated threshold, per ε; also
  mean bits flipped.
- **Ensemble variant:** attack the sum of two source hashes to improve transfer (same code as #7).
- **Cost:** re-hashing is cheap (CPU is fine). The source attacks are already part of the white-box
  run.
- **Risks:** low. Transfer between unrelated hashes will probably be weak; that is still a result.

### #2 NES black-box attack (score-based) ⬜

- **Threat model:** the attacker can send images to a hashing service and see the **full hash**
  (bits or the 144-byte vector), but has no weights. A query budget (e.g. 1k / 5k / 10k queries
  per image) limits them.
- **How it works:** NES = Natural Evolution Strategies (Ilyas et al. 2018).
  1. Define a score to increase: for evasion, distance(hash(x′), hash(x)); for collision, minus
     the distance to the target. Bits are 0/1, so use the **continuous** output where the service
     gives it, or Hamming distance otherwise.
  2. **Estimate the gradient without backprop:** draw n random noise images u₁…uₙ (Gaussian). For
     each, query the hash at x′ + σuᵢ and x′ − σuᵢ ("antithetic pairs"). The estimate is
     g ≈ (1 / 2nσ) · Σ [score(x′ + σuᵢ) − score(x′ − σuᵢ)] · uᵢ. Directions that raise the score get
     positive weight.
  3. **Step:** x′ ← clip_ε(x′ + α · sign(g)), as in PGD.
  4. Repeat until success or the query budget runs out. Count every query.
- **Details:**
  - typical settings: n = 50 pairs (100 queries per step), σ = 0.001–0.01, 50–100 steps;
  - search in a smaller space (e.g. 32×32 or 64×64 noise, upsampled), which cuts the queries
    needed a lot;
  - for a hard 0/1 output, Hamming distance moves in integer steps, so use larger σ.
- **What it shows:** whether the hashes are broken by a realistic attacker without weights. It
  works on every hash, including the exact pHash and PDQ (no gradients needed). It is directly
  comparable with Sun et al., who found DinoHash **more** robust than NeuralHash here.
- **Code:** a new optimiser `_nes` next to `_pgd` / `_apgd` in `hashbench/hashbench/adversarial.py`.
  It reuses `_project`, `_results` and the threshold logic. Wrap the hash so it runs under
  `torch.no_grad()` and counts calls.
- **Metrics:** success rate against query budget (curves at 1k, 2k, 5k, 10k), at fixed ε; the
  median number of queries to success.
- **Cost:** about 5–10k forward passes per image. pHash, PDQ and CertPHash are cheap on CPU;
  DinoHash and NeuralHash need a GPU (batches of queries run in parallel).
- **Risks:** low; well understood. It may need tuning of σ and step size per hash.

### #3 Hard-label black-box attack (HopSkipJump) ⬜

- **Threat model:** the most realistic one. The attacker sees only **match / no match**, as with a
  real client-side scanner. Formally the oracle is O(x′) = [distance(hash(x′), hash(x)) ≤ threshold].
- **How it works:** HopSkipJump (Chen, Jordan, Wainwright 2020), adapted to evasion.
  1. **Start** from an image x₀ that already does **not** match, e.g. the original with strong
     noise or blur, or a different image. It is far from x, but on the correct side.
  2. **Binary search** on the line between x and x₀ to find the boundary point x_b: the closest
     image on that line that still does not match.
  3. **Estimate the boundary direction:** query O at x_b + δuᵢ for many small random uᵢ. The
     average of uᵢ weighted by ±1 (no match / match) points away from the "match" region.
  4. **Step** along that direction (geometric step-size search), then binary-search back towards x.
     Each round moves the non-matching image closer to the original.
  5. Stop at the query budget. Report the final perturbation size (ℓ∞ or ℓ2).
- **Collision version:** the same, but "success" means the hash **matches** the target, and the
  starting point is an image whose hash already matches it (e.g. the target image itself).
- **What it shows:** the smallest perturbation a match-only attacker can find, compared across
  hashes for the same query budget.
- **Code:**
  - wrap each hash as a 0/1 oracle with a query counter;
  - either use `adversarial-robustness-toolbox` (ART) `HopSkipJump` with a wrapper that presents
    the oracle as a two-class classifier, or write the ~200-line algorithm ourselves (simpler to
    control).
- **Metrics:** median perturbation size against queries; success rate at ε = 8/255 (ℓ∞) after
  1k / 5k / 10k / 20k queries.
- **Cost:** 10k+ queries per image; GPU for DinoHash / NeuralHash, CPU for the others.
- **Risks:** medium. ℓ∞ versions of HopSkipJump converge more slowly than ℓ2; a good starting
  point matters.
- **Alternative to #1:** for the cluster run only one of #1 / #3 will be done.

### #4 Soft pHash / PDQ surrogates (white-box classical hashes) ⬜

- **Threat model:** white-box on the classical hashes, which have no gradients because of
  rounding, the median and `sign`.
- **How it works:**
  1. Rewrite pHash and PDQ in PyTorch.
  2. Replace every hard step with a smooth one:
     - `bit = (x > median)` → `tanh((x − median) / T)` with a temperature T;
     - median → a soft median (e.g. a sorting-network relaxation, or simply the median treated as
       a constant during the backward pass);
     - resize and the DCT are linear, so they are already differentiable;
     - PDQ's Jarosz box filter is also linear.
  3. Run our existing PGD / APGD on the smooth version.
  4. **Check the result on the exact hash** (our bit-exact `phash` / `pdq`). Only that counts.
- **What it shows:** pHash and PDQ enter the same white-box comparison table as the neural hashes.
  Earlier work attacked them only as a black box (Prokos et al. 2023).
- **Code:** new `hashbench/hashbench/soft_hashes.py`. Test that, at a low temperature, the soft hash
  gives the same bits as the exact one on the benchmark images. Then plug it into `adversarial.py`
  as another differentiable model.
- **Metrics:** the same as the white-box sweep (success against ε at the calibrated threshold).
- **Cost:** cheap; CPU is fine.
- **Effort:** pHash about 2 days; PDQ about 1 week (more steps, all must match the exact code).
- **Risks:** medium. The soft and exact hashes can disagree near the median; anneal T during the
  attack.

### #5 ℓ2 budget ◐ (code supports it, never run)

- **Threat model:** the same white-box attacker, but with a limit on the **total** change instead
  of on each pixel. ℓ2 allows a large change on a few pixels; ℓ∞ allows a small change everywhere.
- **How it works:** the same PGD / APGD. The step follows the normalised gradient, and the
  projection scales δ back onto the ball ‖δ‖₂ ≤ ε. Already implemented in `_project`.
- **What it shows:** nothing new; it makes our numbers directly comparable with CertPHash Table 4 and
  Sun et al., who both use ℓ2.
- **Code:** `--norm l2` and a choice of ε values (e.g. ε = 0.5, 1, 2, 4 in [0, 1] pixel units on the
  model's input size; match the papers).
- **Cost:** a few GPU hours.
- **Status:** **dropped from the cluster plan.**

### #6 EOT: transform-surviving attacks ◐ (hook exists, unused)

- **Threat model:** the attacker knows the platform will re-compress, resize or crop the uploaded
  image before hashing it.
- **How it works:** Expectation over Transformation (Athalye et al. 2018).
  1. At every attack step, apply k random transforms T₁…T_k to the current adversarial image
     (JPEG quality 70–95, resize 0.5–1.5×, crop up to 10%, small noise).
  2. Compute the loss on each transformed version and **average the gradients**.
  3. Step and project as usual. The result is an image that fools the hash *after* typical
     processing.
- **Details:** JPEG is not differentiable. Use a differentiable JPEG approximation (e.g. DiffJPEG),
  or the straight-through trick: real JPEG in the forward pass, identity in the backward pass.
- **What it shows:**
  - whether the attacks survive real upload pipelines (our current images may fail after one JPEG
    save);
  - also needed for attacking randomised defences (#10).
- **Code:** the hook already exists in `adversarial.py`; add a transform sampler and the JPEG
  approximation.
- **Metrics:** success rate measured **after** applying held-out transforms to the final image.
- **Cost:** k times more per step (k = 5–20), so a GPU is needed.

### #7 Joint evasion of several hashes ⬜ (Idea 4)

- **Threat model:** the defender runs several hash families together and flags an image if
  **any** of them matches. The attacker must evade **all** of them with one perturbation.
- **How it works:**
  1. Write the combined loss as the sum (or the max) of each hash's evasion loss, normalised by each
     hash's threshold so no hash dominates: L = Σₕ Lₕ(x′) / thrₕ.
  2. Run the same APGD / PGD ensemble on L.
  3. Success = **every** hash's distance is above its threshold.
- **Combinations:**
  - DinoHash + NeuralHash (two different neural architectures);
  - + CertPHash ε8 (the most robust);
  - + PDQ (needs the soft surrogate from #4) — mixes classical and learned hashes.
- **What it shows:** whether combining hash families is a cheap, real defence. The benign benchmark
  found that failures are not correlated across families; this checks whether that also holds for
  attacks. Key number: **how much larger ε must be** to evade the ensemble than the weakest member.
- **Code:** a wrapper model in `adversarial.py` that calls several hashes and returns a combined
  loss; `_results` reports per-hash and joint success.
- **Cost:** about the sum of the individual attacks; GPU.
- **Novelty:** likely new for perceptual hashes (to be checked).

### #8 Universal adversarial perturbation ⬜

- **Threat model:** a mass-scale attacker who wants **one** noise pattern that can be added to any
  image to evade the hash, with no optimisation per image.
- **How it works:** Moosavi-Dezfooli et al. 2017, in its simpler SGD form.
  1. Start with δ = 0 (one image-sized pattern, at the model's input size).
  2. For mini-batches of **training** images xᵢ: compute the evasion loss of every xᵢ + δ, average,
     take a gradient step on δ, and project it into the ε ball.
  3. Repeat for a few epochs.
  4. **Test** on held-out images: add δ, hash, and count how many are evaded.
- **Collision version:** a universal pattern that pushes many images towards one target hash
  (a "flood the system with false matches" attack).
- **What it shows:** how cheap a large-scale attack is; whether robust hashes resist
  image-independent noise better than per-image noise.
- **Code:** a new script `scripts/universal_perturbation.py` reusing the loss functions in
  `adversarial.py`. Training images: a COCO subset disjoint from the test set (e.g. 500 train,
  V200 test).
- **Metrics:** held-out evasion rate against ε; compare with the per-image rate at the same ε.
- **Cost:** a few hundred images × a few epochs; a GPU, about an hour per hash.

### #9 Hash inversion (privacy attack) ⬜ (Idea 1, main contribution)

- **Threat model:** the attacker has only the **hash** (e.g. from a leaked database) and wants to
  recover what the image shows.
- **Two versions:**
  - **(a) Decoder:** train a network that maps hash → 64×64 image, on COCO train
    (`coco100x100.tar.gz`). The same architecture and training budget for every hash, for fairness.
    **Dropped from the cluster plan** (GPU cost).
  - **(b) Optimisation (no training):**
    1. Start from a grey or random image.
    2. Run gradient descent so its hash gets closer to the target hash (the collision loss we
       already have), plus a smoothness term (total variation) so the result looks like an image.
    3. Optionally start from a blurred average image.
- **What it shows:** whether the hash leaks the content. The thesis question is whether **more
  robust hashes leak more** (CertPHash ε sweep: 6 points on one architecture; DinoHash
  trained/untrained: 2 points).
- **Metrics:** SSIM and DINOv2 similarity between the reconstruction and the original; a human
  check of a few examples; leakage plotted against measured robustness.
- **Code:** (b) reuses the collision attack with ε = ∞ and an extra TV loss; only white-box hashes,
  or the soft surrogates for pHash / PDQ.
- **Cost:** (b) is cheap (a few minutes per image on a GPU).
- **Prior work:** Struppek et al. (NeuralHash inversion), Madden et al. 2024. The robustness-vs-leakage
  comparison is the likely new part.

### #10 Adaptive attack on DualShield ⬜ (Idea 5)

- **Threat model:** white-box against a hash protected by DualShield (Sun et al., arXiv 2608.03101,
  Aug 2026, not peer-reviewed). The defence adds Gaussian noise at matching time (σ = 0.10) and
  hardens the model.
- **How it works:**
  1. Re-implement DualShield on NeuralHash and DinoHash (or use their code, if released).
  2. Attack it with **EOT-PGD** (#6): average gradients over many noise samples, so the random
     noise does not hide the gradient.
  3. Add attacks that know the defence, e.g. optimise the expected hash under the noise, or attack
     the hardened model directly.
- **What it shows:** whether DualShield's robustness holds against an attacker who knows the
  defence. Random-noise defences have often been broken this way (Athalye et al. 2018).
- **Cost:** EOT makes it 10–50× the cost of a plain attack; GPU.
- **Risks:** high. It depends on code availability and a faithful re-implementation. Needs #6 first.
- **Novelty:** high if the attack succeeds.

### #11 Spatial / semantic attack ⬜

- **Threat model:** white-box, but instead of adding pixel noise the attacker **moves pixels or
  changes colours** slightly, which humans notice even less.
- **How it works:** stAdv (Xiao et al. 2018) style.
  1. Parametrise a small flow field (per-pixel displacement), or a global rotation / scale /
     translation, or a colour transform (brightness, contrast, hue).
  2. Apply it with `torch.nn.functional.grid_sample` (differentiable).
  3. Optimise the parameters with the evasion or collision loss plus a smoothness penalty on the
     flow.
  4. Constrain the size (e.g. at most 1–2 pixels of displacement, or ≤ 5° rotation).
- **What it shows:** how the architectures react to geometric changes. NeuralHash's instance norm
  and DinoHash's semantic features may behave very differently, and the benign benchmark already
  showed rotation and crop as weak spots for some hashes.
- **Code:** a new parametrisation in `adversarial.py` (optimise flow parameters instead of δ); the
  rest of the pipeline is reused.
- **Metrics:** success rate against flow size / rotation angle; SSIM.
- **Cost:** about the same as the white-box sweep; GPU.

### #12 Backdoor / poisoning ⬜ (Idea 7, stretch)

- **Threat model:** the attacker controls part of the hash's training data or fine-tuning.
- **How it works:** fine-tune a hash (e.g. CertPHash or the DinoHash head) on data where images with
  a small trigger pattern are trained to produce one chosen hash. After training, adding the
  trigger to any image forces that hash (a collision on demand), while normal images hash as before.
- **What it shows:** the supply-chain risk of a scanning model; has an ethical dimension.
- **Cost:** full training runs.
- **Risks:** high. **Recommendation: skip; mention as future work.**

---

## 3. References (as cited in the plan and our notes)

| Short name | Paper |
|---|---|
| Athalye et al. 2018 | *Obfuscated Gradients Give a False Sense of Security* (EOT) |
| Ilyas et al. 2018 | *Black-box Adversarial Attacks with Limited Queries and Information* (NES) |
| Chen, Jordan, Wainwright 2020 | *HopSkipJumpAttack: A Query-Efficient Decision-Based Attack* |
| Moosavi-Dezfooli et al. 2017 | *Universal Adversarial Perturbations* |
| Xiao et al. 2018 | *Spatially Transformed Adversarial Examples* (stAdv) |
| Jain et al. 2022 | *Adversarial Detection Avoidance Attacks* (black-box attacks on classical hashes) |
| Prokos et al. 2023 | *Squint Hard Enough: Attacking Perceptual Hashing with Adversarial Machine Learning* |
| Struppek et al. 2022 | *Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash* |
| Madden et al. 2024 | Hash inversion (see `thesis_research_plan.md` Idea 1) |
| Sun et al. 2026 | DualShield, arXiv 2608.03101 |
