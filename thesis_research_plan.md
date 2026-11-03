# Thesis research plan: next steps and research directions

*BTP: Perceptual Hashing. Written 2026-10-29. Builds on `learning_models_plans.md`
(feasibility study and first roadmap) and `hashbench/README.md` (what is implemented).*

---

## 0. Where we are

**Done**
- `hashbench/` contains our own implementations of 5 hash families, each checked against
  the official implementation:
  - pHash and PDQ (bit-exact);
  - NeuralHash (read directly from Apple's Espresso files);
  - DinoHash (ViT-S/96 and ViT-B/512, each in a trained and an untrained version);
  - CertPHash (certified ε = 1, 2, 4, 8 /255, plus the normal-trained and
    adversarially-trained baselines).
- Attack suite: 23 benign transformation families with 88 parameter settings in total.
  Thresholds are calibrated to the same false-positive rate (FPR) for every hash, and the
  report computes ROC/AUC/EER and plots.
- Every learned model is a differentiable PyTorch `nn.Module`, so white-box attacks can be
  plugged in directly.
- Findings so far:
  - CertPHash: the more robust the training, the further the hash drifts from real
    PhotoDNA. At ε=1/255, 96.9% of hashes are within the match threshold of real PhotoDNA;
    at ε=8/255, only 15.9%.
  - Three problems in the CertPHash authors' evaluation code: the normalization is
    missing, the evasion check (CNER) uses the wrong distance, and ε is applied after
    normalization (about 4.4x larger in pixel space).
  - DinoHash: the ViT-S release uses a different PCA head than the one in the repo.
  - NeuralHash: on low-contrast images, our instance norm is more accurate than
    onnxruntime's.

**Not done**
- No benchmark on a real dataset yet; only smoke tests on 12 images.
- No adversarial (white-box or black-box) attacks.
- No privacy attacks or defences.
- SSCD is not implemented.

**Constraints**
- Laptop: Ryzen 5 5600H, 15 GB RAM, no CUDA.
- GPU work goes to Kaggle or Colab, or an institute server if one becomes available.
- **No new downloads for now.** Every step below uses data and weights already on disk.
  Steps that would need a download are marked **[download]** and are optional.
- Runs longer than about 1 hour need approval first (marked **[long run]**).

---

## 1. Fixed decisions

| Decision | Choice | Why |
|---|---|---|
| Evaluation dataset | **COCO val2017**, 5,000 images, `hashbench/data/coco/val2017` (788 MB, verified complete) | Already on disk. No hash in the project was trained on it (CertPHash trained on COCO *train*; DinoHash on DiffusionDB). Used by CertPHash and DualShield, so results can be compared. |
| Training data (inversion decoders, smoothing σ tuning, anything learned) | **COCO train at 100×100**, 118k images, `hashbench/data/certphash/coco100x100.tar.gz` (1.1 GB, on disk, not extracted) | Disjoint from val2017. Small images are enough, because all decoders work at 64×64. |
| PhotoDNA reference | `data/certphash/data/coco-{train,val}.csv` (real PhotoDNA hashes) | The only ground truth for PhotoDNA we have. The PhotoDNA binary itself is never run. |
| Evaluation subsets | `V5k` = all 5,000 images (impostor pairs, thresholds); `V1k` = a fixed random 1,000 images (benign attack suite); `V200` = a fixed random 200 images (white-box attacks, certification) | Keeps CPU cost manageable. Fixed seeds and lists are saved in `hashbench/data/splits/` so every chapter uses the same images. |
| Main hashes | pHash, PDQ, NeuralHash, DinoHash (ViT-S trained and untrained; ViT-B where affordable), CertPHash (ep1, ep2, ep4, ep8, base, adv) | Includes two controlled ablation families (DinoHash trained vs untrained, the CertPHash variants). These pairs are what make the research ideas in §3 possible. |

Data and weights that are on disk but don't need to be used: `data/coco/val2017.zip`
(the extracted folder is enough) and the MNIST CertPHash checkpoints.

---

## 2. Engineering foundations (needed by every research direction)

### Phase A: Housekeeping (½ day)
- [ ] **Put the repo under version control** (`git init` in `hashbench/` or at the repo
  root, with a `.gitignore` for `models/`, `data/` and `results/`). Nothing is versioned
  at the moment, so one bad edit could lose weeks of work. *(Needs your OK.)*
- [ ] Freeze the environment: `pip freeze > hashbench/requirements-lock.txt`.
- [ ] Create `data/splits/{V1k,V200}.txt` with seeded random image lists, so every
  experiment uses the same images.
- [ ] Extract `coco100x100.tar.gz` locally into `data/certphash/coco100x100/` (1.1 GB; no
  download needed).

### Phase B: Full benign benchmark on COCO (1 week of calendar time, mostly unattended)
- [ ] Short timing run (20 images, every hash) to measure the cost per image of each hash.
  The current estimates are about 63 ms/image for DinoHash ViT-S, about 210 ms for ViT-B
  and about 2.5 ms for CertPHash. With 88 attacked copies per image, the attack suite on
  `V1k` costs about 1.5 h for ViT-S and about 5 h for ViT-B.
- [ ] **[long run]** `benchmark --images data/coco/val2017 --attack-limit 1000
  --max-pairs 200000` for every hash. Split it into separate runs per hash family so a crash
  doesn't lose everything.
- [ ] Report at FPR 1e-2, 1e-3 and 1e-4. With 200k impostor pairs, 1e-4 is still estimated
  from 20 pairs.
- [ ] Add bootstrap confidence intervals over images to `report.py`. This is a small
  change, but the thesis needs error bars.
- [ ] Deliverable: **Chapter "Benign robustness"**, with tables and plots for every
  hash × attack × severity combination, plus a discussion of which hash families fail on
  geometric changes and which on photometric changes.

### Phase C: Adversarial attack module, `hashbench/adversarial.py` (2–3 weeks)
This is the module everything else depends on. It must be model-agnostic and work on any
`nn.Module` that maps [0, 1] images to per-bit logits (or to CertPHash's 144 outputs).

- [ ] **Evasion** (make an image stop matching its own hash):
  - PGD and APGD (Croce & Hein 2020) under ℓ∞ and ℓ2 budgets.
  - Losses: hinge on the logit margin of each bit, as in Struppek et al.; L1 for
    CertPHash.
- [ ] **Collision** (make an image match a target hash): targeted variant of the same
  attacks. Port the Struppek et al. code from `neuralHash/adv1_*` and `adv2_*`.
- [ ] **Pixel-space budgets.** ε is always applied in [0, 1] pixel space, before the model's
  own normalization. CertPHash's code does the opposite (see §0); we must not copy that.
- [ ] **EOT option** (Athalye et al. 2018): average gradients over random transforms or
  noise. Needed to attack randomized defences (§3, ideas 2 and 5).
- [ ] **Image-quality metrics:**
  - PSNR, SSIM and ℓ∞/ℓ2 norms, implemented with numpy only.
  - **[download]** LPIPS needs a weight download; skip it for now, and use DINOv2 feature
    similarity with the original DINOv2 weights we already have as a perceptual proxy.
- [ ] **Differentiable surrogates for classical hashes:** a "soft" pHash and PDQ, made by
  replacing `sign` with `tanh(x/T)` (PDQ: luma → Jarosz filter → DCT → soft median
  threshold). This lets us run white-box attacks on PDQ, which earlier work (Prokos et al.
  2023) only attacked as a black box. A small contribution of its own.
- [ ] **Black-box baseline:** NES gradient estimation with a query counter (Jain et al.;
  DualShield). It works for any hash, including the non-differentiable exact PDQ.
- [ ] Tests:
  - the attack never exceeds its budget;
  - a sanity check that attacking an untrained DinoHash is easier than a trained one;
  - reproduce Struppek et al.'s result that NeuralHash's evasion succeeds within a few
    steps.
- [ ] Deliverable: **Chapter "Adversarial robustness"**: success rate against ε at the
  calibrated threshold, steps needed, and image quality; plus a transferability matrix
  between hashes (does an attack crafted on hash A also work on hash B?).

### Phase D: Optional extra hashes
- [ ] **[download]** SSCD (about 100 MB of weights). It is a strong learned copy-detection
  baseline, used in DualShield. Add it only if a chapter needs it.

---

## 3. Research directions (candidates for the main contribution)

Ranked by (novelty × feasibility on our compute). Each is written as a hypothesis we could
confirm or refute.
**Before committing to any of them, do a focused literature check** (Google Scholar and
arXiv, 2024–2026) using the "check" keywords given for each idea. All novelty claims below
are our current understanding, not verified facts.

### ⭐ Idea 1: Does robustness cost privacy? Robustness vs. hash-inversion leakage

**Hypothesis.** Hashes trained to be adversarially robust leak *more* information about the
image than non-robust ones. Two results from classifier research support this: robust
models have "perceptually aligned" features that are much easier to invert (Engstrom et
al. 2019; Santurkar et al. 2019), and a perceptual hash is published or sent to a server,
so its invertibility is a privacy risk. In client-side scanning the hash is exactly what
leaves the user's device.

**Why we are well placed.** We have **controlled pairs**: the same architecture with and
without robust training.
- DinoHash: untrained vs. adversarially fine-tuned (ViT-S/96 and ViT-B/512).
- CertPHash: normal → adversarial → certified with ε = 1, 2, 4 and 8 /255, a full sweep of
  how strongly the model was made robust.
- pHash, PDQ and NeuralHash as non-robust reference points.

Earlier inversion work (Struppek et al. `adv4`, Madden et al. 2024, the "Ribosome" attack on
PhotoDNA) inverted *individual* hashes. As far as we know, no one has asked whether the
robustness defences themselves increase leakage, holding everything else fixed.

**Method.**
1. For each hash H, train a decoder D_H that reconstructs a 64×64 image from H(x).
   - Train it on COCO train at 100×100 (118k images, on disk).
   - Use the same decoder architecture and training budget for every hash, for fairness.
   - The decoder is a small transposed-convolution network (about 2–5M parameters).
2. A second attack that needs no training: **optimisation-based inversion**. Start from
   noise and use gradient descent to find an image whose hash equals the target, with a
   smoothness prior. This only works for differentiable hashes; use the soft surrogates for
   pHash and PDQ.
3. Measure leakage on val2017 (`V1k`):
   - reconstruction PSNR and SSIM;
   - **re-identification rate**: is the original image the nearest neighbour of the
     reconstruction among 5,000 images, using original-DINOv2 features (weights on disk)?
   - an information measure: the effective entropy of the hash bits (bits that are nearly
     constant or correlated with each other leak less than the nominal number of bits).
4. Plot **leakage against robustness**: the CertPHash ε sweep gives 6 points on one
   architecture, and the DinoHash pairs give 2 × 2 points. Robustness here is the measured
   evasion ε from Phase C.

**Expected outcome.** A robustness–privacy frontier. If the hypothesis is refuted, i.e.
robust hashes do *not* leak more, that is also a publishable negative result, because it
contradicts the classifier intuition.

**Compute.**
- Decoder training on 64×64 images with 96–1152-dimensional inputs: about 1–3 h per hash on
  CPU, 10–20 min on a Kaggle GPU. About 12 hashes in total.
- The 118k training images must be hashed once per hash. That is cheap for CertPHash
  (about 5 min), takes about 2 h for DinoHash ViT-S on CPU, and about 7 h for ViT-B, which
  should go to Kaggle.

**Risk:** low to medium. The data, models and code (Phase C) are all available.
Check: "perceptual hash inversion robustness", "robust representations invertibility
privacy", "client-side scanning hash leakage".

### ⭐ Idea 2: Certified perceptual hashing without retraining, via randomized smoothing on hash bits

**Idea.** Apply randomized smoothing (Cohen et al. 2019) to each hash bit:
`ĥ_k(x) = majority over δ ~ N(0, σ²I) of sign(f_k(x + δ))`. For each bit, Cohen's bound gives
a certified ℓ2 radius `R_k = σ · Φ⁻¹(p_k)`, where p_k is a lower confidence bound on the
probability of the majority sign. For an attack of ℓ2 size r, only bits with R_k ≤ r can
flip, so:

```
worst-case distance to reference  ≤  d(ĥ(x), ref) + #{k : R_k ≤ r}
certified match       if this ≤ τ
certified no-collision (impostor pairs):  d(ĥ(x), ref) − #{k : R_k ≤ r} > τ
```

The per-bit failure probabilities are combined with a union bound: confidence level α/n_bits
for each bit.

**Why it's interesting.**
- CertPHash needs *special training* (and an A100) to get certificates. Smoothing works on
  **any** existing hash with no retraining: NeuralHash, DinoHash, and also **pHash and PDQ**,
  because smoothing only needs forward passes. To our knowledge, nobody has certified
  classical perceptual hashes.
- It gives a direct comparison between certified training (CertPHash) and certification at
  inference time (smoothing), at the same calibrated FPR.
- DualShield (2026) adds Gaussian noise at matching time but only evaluates it empirically.
  This idea would give it guarantees.

**Experiments.**
- Certified match rate against r for each hash and several σ.
- The cost of smoothing in benign robustness and false matches (the smoothed hash differs
  from the original hash).
- The cost in time: n Monte-Carlo samples per image.
- Comparison with CertPHash's own ℓ∞ certificates, with an honest ℓ∞ ↔ ℓ2 conversion
  (ℓ2 ≤ ε√d); note that smoothing is known to be weak against ℓ∞ attacks.

**Compute.**
- With n = 1,000 samples per image: pHash, PDQ and CertPHash are cheap (seconds per image).
- NeuralHash: about 10–30 s per image.
- DinoHash ViT-S: about 1 min per image on CPU, so `V200` takes about 3 h, or use Kaggle.

**Risk:** medium. The theory is standard. The novelty lies in applying it to multi-bit
hashes and Hamming thresholds, and has to be checked.
Check: "randomized smoothing hashing certified", "certified robustness deep hashing
retrieval", "smoothed embeddings certified", "certified Hamming distance".

### Idea 3: Re-evaluating CertPHash (a reproducibility study)

We already found three problems in the authors' evaluation (§0) and a fidelity trade-off
they did not report. A careful re-evaluation would consist of:
1. Functionality (benign ROC/AUC) with the correct ImageNet normalization, compared with
   their reported AUC of 0.98.
2. Empirical evasion with strong adaptive attacks (APGD, plus EOT where relevant) at
   **pixel-space** ε, compared with their claimed certified no-evasion rate (CNER).
3. Their CNER recomputed with the correct PhotoDNA L1 metric and worst-case L1 over the
   output box, using ℓ∞ rather than L0. This needs `auto_LiRPA` (a pip package, not a
   dataset). IBP bounds are cheap on CPU for a 2.6M-parameter network at 64×64; CROWN may
   need Kaggle.
4. The fidelity–robustness frontier: distance to real PhotoDNA against certified ε (already
   measured).

**Value:** reproducibility studies are welcome at venues such as ML Reproducibility
Challenge and USENIX/ACSAC SoK tracks. This one is low risk and builds directly on finished
work. It makes a strong **chapter**, but it is probably not enough as the whole thesis.

### Idea 4: Hash ensembles as a defence. Can an attacker evade hashes from different families at once?

**Hypothesis.** Different hash families respond to different image features:
- PDQ and pHash use low-frequency DCT;
- NeuralHash is a CNN;
- DinoHash is a ViT with global attention;
- CertPHash works at 64×64.

A perturbation that evades one of them may not transfer to the others. Combining them (flag
a match if *any* hash matches, each at its calibrated FPR, with the total FPR controlled)
might force much larger perturbations.

**Experiments.**
- Transferability matrix (from Phase C).
- Joint evasion: attack the sum of the losses of all hashes, then measure the ε and the
  image quality needed against single-hash evasion.
- Find out which pairs of hashes are the most complementary.

The soft PDQ surrogate from Phase C makes the classical hashes attackable here.
**Risk:** low. **Novelty:** medium. Ensembles have been studied for classifiers but rarely
for perceptual hashes. Check: "ensemble perceptual hashing adversarial", "multi-hash
client-side scanning evasion".

### Idea 5: Adaptive attacks on DualShield (arXiv 2608.03101, Aug 2026, not peer-reviewed)

DualShield defends deep hashes in two ways: Gaussian noise at matching time (σ = 0.10), and
a "hardening" perturbation (ℓ∞ ≤ 16/255) applied to reference images. Randomized
preprocessing defences have historically been broken by EOT attacks (Athalye et al. 2018;
Tramèr et al. 2020).

Plan:
- Reimplement DualShield on NeuralHash and DinoHash.
- Evaluate it with EOT-PGD and with attacks that take the hardening into account.
- If they hold up, Idea 2 can turn their empirical noise into a certified version.

**Risk:** medium. The preprint is very recent and may be revised, and we must reimplement it
faithfully. **Novelty:** high if the attacks succeed. Combines naturally with Idea 2.

### Idea 6: Bit-level anatomy of learned hashes (cheap analysis chapter)

- Per-bit flip probabilities under each attack.
- Bit entropy and correlations between bits, giving the **effective hash length**.
- Which bits flip first under adversarial attack?
- Extrapolate false-match rates to deployment scale (10⁹ images) by fitting the tail of the
  impostor-distance distribution, compared with the empirical 200k-pair estimate.

All of this uses data Phase B already produces. It complements "Hamming Distributions of
Popular Perceptual Hashing Techniques" (arXiv 2212.08035), which covered only classical
hashes.

### Idea 7 (stretch, high risk): A backdoor implanted while fine-tuning a perceptual hash

Adapt DarkHash (written for retrieval hashing) to DinoHash's fine-tuning: a poisoned
fine-tune makes images containing a trigger collide with, or evade, chosen targets, while
clean behaviour stays unchanged.
- Very novel for perceptual hashing.
- Needs GPU fine-tuning of a ViT (Kaggle).
- Has an ethical dimension: a backdoored scanning hash. Treat it as future work unless
  everything else is finished early.

---

## 4. Recommended thesis shape

**Working title:** *"The Price of Robustness in Learned Perceptual Hashing: Adversarial
Evasion, Certification, and Privacy Leakage"*

| Chapter | Content | Source |
|---|---|---|
| 1. Introduction | Perceptual hashing, client-side scanning, the CSAM debate, threat models | Literature (already read) |
| 2. Background | pHash, PDQ, NeuralHash, DinoHash, CertPHash; attacks; certification | Literature + `hashbench` |
| 3. hashbench | Our faithful implementations and how each was validated (bit-exactness, the bugs found in the references), plus the calibrated-FPR evaluation protocol | Done; write up |
| 4. Benign robustness | COCO val2017, 23 attack families | Phase B |
| 5. Adversarial robustness and transferability | White-box, black-box, soft-PDQ attacks, ensembles | Phase C + Idea 4 |
| 6. **Contribution A: robustness vs. privacy** | Inversion leakage across the controlled robustness ablations | **Idea 1** |
| 7. **Contribution B: certification without retraining** | Randomized smoothing for hash bits vs. CertPHash's certified training | **Idea 2** (+ Idea 3 as the CertPHash re-evaluation) |
| 8. Discussion | Trade-offs, implications for client-side scanning, limitations, ethics | |

Fallbacks if something fails:
- If Idea 1 shows nothing interesting, Ideas 3 + 4 still make a solid thesis.
- If Idea 2 turns out to be already published, use it as the defence baseline for Idea 5.
- Idea 6 is cheap insurance and can be added at any point.

---

## 5. Timeline

*Assumption: sem 7 ends in early December 2026 and the final BTP submission is around
April 2027. Adjust once the deadlines are confirmed.*

| Period | Work | Milestone |
|---|---|---|
| Oct 30 – Nov 5 | Phase A; timing run; start Phase B (long runs at night) | Repo versioned, splits frozen |
| Nov 6 – Nov 19 | Finish Phase B; Phase C core (PGD/APGD evasion and collision, quality metrics, tests) | First real results: benign tables + first adversarial curves |
| Nov 20 – Dec 5 | Literature check for Ideas 1 and 2; small pilots (decoder for CertPHash and DinoHash ViT-S; smoothing for pHash and NeuralHash on 20 images) | **Sem-7 evaluation**: benchmark results + pilot evidence for the chosen direction |
| Dec | Soft PDQ/pHash, NES, transferability matrix (Phase C complete) | Chapter 5 draft |
| Jan | Idea 1 at full scale (Kaggle for the DinoHash ViT-B hashing and decoders) | Chapter 6 results |
| Feb | Idea 2 at full scale + Idea 3 (CertPHash re-evaluation) | Chapter 7 results |
| Mar | Idea 6 analysis; optionally Idea 5; write chapters 1–4 | Full draft |
| Apr | Revisions, figures, final presentation | Submission |

---

## 6. Risks and how we handle them

| Risk | How we handle it |
|---|---|
| CPU too slow for ViT-B experiments | Use ViT-S as the main DinoHash; run ViT-B on Kaggle (about 30 GPU-h/week free) or only on `V200` |
| A research idea turns out to be already published | Do the literature check *before* scaling up (Nov 20 – Dec 5); the fallbacks are in §4 |
| Inversion decoders are too weak to show differences between hashes | Add optimisation-based inversion; increase decoder capacity equally for every hash |
| Results not reproducible | Fixed splits, seeds, `config.json` for every run (already written), version control (Phase A) |
| Ethics (CSAM-related topic, Apple's model files) | Use only COCO; never redistribute Apple's files (git-ignored); include an ethics section; reveal no operational evasion details beyond what is already published |
| An experiment runs for many hours unattended | Always do a timing pilot first, and ask before runs longer than about 1 h |

---

## 7. Open decisions for you / your supervisor

- [ ] Which contribution pair? Recommendation: **Idea 1 + Idea 2**, with Idea 3 as a
  chapter and Idea 6 as a cheap extra.
- [ ] OK to `git init` the repo (Phase A)?
- [ ] GPU source: Kaggle, Colab, or an institute server?
- [ ] Optional downloads later: LPIPS weights, SSCD weights, `auto_LiRPA` (a pip package),
  COCO annotations (only if we add attribute-inference experiments to Idea 1).
- [ ] Confirm the sem-7 evaluation date and the final submission deadline, so §5 can be
  fixed.

---

## References to read or check for the ideas above

- Cohen, Rosenfeld, Kolter. *Certified Adversarial Robustness via Randomized Smoothing.* ICML 2019.
- Salman et al. *Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers.* NeurIPS 2019.
- Engstrom et al. *Adversarial Robustness as a Prior for Learned Representations.* 2019.
- Santurkar et al. *Image Synthesis with a Single (Robust) Classifier.* NeurIPS 2019.
- Croce, Hein. *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks* (APGD). ICML 2020.
- Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security* (BPDA, EOT). ICML 2018.
- Tramèr et al. *On Adaptive Attacks to Adversarial Example Defenses.* NeurIPS 2020.
- Struppek et al. *Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash.* FAccT 2022 (`neuralHash/` in this repo).
- Madden et al. *Robustness of Practical Perceptual Hashing Algorithms to Hash-Evasion and Hash-Inversion Attacks.* arXiv 2406.00918.
- Prokos et al. *Squint Hard Enough: Attacking Perceptual Hashing with Adversarial Machine Learning.* USENIX Security 2023.
- Jain et al. *Adversarial Detection Avoidance Attacks.* USENIX Security 2022.
- Yang et al. *CertPHash.* USENIX Security 2025 (`CertPHash.pdf`).
- Singhi et al. *DinoHash.* ICML 2025 CODEML workshop (`DinoHash.pdf`).
- *Double Down on Defense (DualShield).* arXiv 2608.03101, 2026.
- *Hamming Distributions of Popular Perceptual Hashing Techniques.* arXiv 2212.08035.
- DarkHash, DHTA, UAH (retrieval hashing, in this repo).
