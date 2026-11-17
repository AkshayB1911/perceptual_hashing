# New research ideas: architecture, efficiency, probing and defences

*BTP: Perceptual Hashing. Written 2026-11-16. Adds to `thesis_research_plan.md` (Ideas 1–7) and
`future_attacks.md` (12 attacks); nothing here repeats those, except where a new paper changes
them (§5). Assumes about 4 months of work (to mid-March 2027), no local GPU, Kaggle T4 for training,
and the institute server if it arrives.*

**Status:** every idea here is a proposal. Nothing has been run. Cost figures are estimates scaled
from the timings already measured in this project, not measurements.

**Novelty caveat:** the "is it new?" lines come from a short web search on 2026-11-16 (sources in
§7), not from a full literature review. Do a proper check before claiming anything as new.

---

## 0. Summary

### The gap these ideas fill

The work so far treats each hash as a fixed black box: benchmark it, attack it. Nobody in the papers
we have read asks the *architecture* questions:

- Which part of the network (depth, head, input size, patch grid) is responsible for robustness,
  for false matches, and for cost?
- Can a learned hash be made smaller or faster without losing what it is good at?
- Does a smaller hash stay as robust, and does it still produce the *same* hash as the big one?

The search found compression-vs-robustness work for classifiers and distillation work for copy
detection embeddings, but nothing that measures either for perceptual hashes.

### The ten ideas

Scores are 1–5, higher is better. "Training" says whether any model has to be trained.

| ID | Idea | Training? | Feasibility | Novelty | Effort |
|---|---|---|---|---|---|
| **E1** | Depth study: a DinoHash from every ViT layer (early exit) | No | 5 | 4 | 1–2 weeks |
| **E2** | Head redesign without retraining (balanced bits, rotation, code length) | No | 5 | 3 | 1 week |
| **E3** | Remove the 14 px patch-grid weakness (shift averaging) | No | 4 | 3–4 | 1 week |
| **E4** | Compression vs robustness: quantisation, token merging, input size | No | 4 | 4 | 2–3 weeks |
| **E5** | TinyHash: distil a hash into a small student (also a model-stealing attack) | **Yes** | 3 | 4 | 3–4 weeks |
| **P1** | How linear is each hash? Linear surrogates for attack and inversion | Linear fit only | 5 | 4 | 1–2 weeks |
| **P2** | What does a hash reveal? Category probes on hashes | Linear fit only | 5 | 3 | 3–5 days |
| **P3** | Natural near-collisions: how pool size lowers the collision budget | No | 4 | 3 | 1 week |
| **D1** | Per-bit smoothing certificates (update to Idea 2 after DualShield) | No | 3 | 2–3 | 2 weeks |
| **D2** | A held-out "verification hash" against collisions | No | 4 | 3 | 1 week |

### Recommendation in one paragraph

Make **E1 + E2 + E4** the core ("where does a learned hash's behaviour come from, and how much of
the network does it need?"), with **E5** as the one piece that involves training and **P1 + P2** as
a cheap privacy chapter that can replace the expensive decoder version of Idea 1. E1, E2, E3 and D2
all run offline from **one saved set of features** (§1), so the first two weeks of engineering pay
for four ideas. Details and a timeline are in §6.

### One thing to know before choosing

The Phase C sweep found both DinoHash models evaded at ε = 1–2/255 at the calibrated threshold.
So for DinoHash, "keep the adversarial robustness after compression" has little to keep if success
rate is the metric. Use finer metrics for every idea below: **bit accuracy under attack** (45.6% vs
20.6% at ε = 1 for trained vs untrained, so there is a difference to track) and the **median ε
needed to evade**. NeuralHash and CertPHash have real threshold-level robustness to preserve.

---

## 1. Shared groundwork: one feature dump

Several ideas need the *internal* outputs of DinoHash, not just the 96 bits. Extract them once.

- **What to save,** per image, for both backbones (released fine-tuned and original DINOv2 ViT-S):
  - the class token after each of the 12 transformer blocks (12 × 384 values);
  - the mean of the patch tokens after each block (12 × 384 values);
  - the final 96 logits before the sign.
- **Which images:** the 5,000 clean val2017 images (for impostor pairs), plus the 88 attacked
  versions of an evaluation subset. `V200` gives 17,600 attacked images; `V1k` gives 88,000.
- **Cost:** all 12 layers come out of one forward pass, so this costs the same as the benign run:
  about 2.5 h per backbone on the desktop for `V1k` (Run 2 took 2 h 30 min), about a fifth of that
  for `V200`. Storage in float16: about 1.7 GB per backbone for `V1k`, 0.4 GB for `V200`.
  **This is a long run and needs approval first.**
- **What it buys:** every head, layer, code length, bit-balancing rule and held-out head in E1, E2
  and D2 becomes a few seconds of numpy on the laptop. It also settles the deferred mixed
  head/backbone question (F17) for free, because any head can be applied to either backbone's
  saved features.
- **A calibration set** is needed to fit new heads (PCA, medians). It must be disjoint from the
  evaluation images. Options, best first: full-resolution COCO train images on Kaggle (no laptop
  download); the `coco100x100` images upsampled (on disk, but blurry, and DinoHash is sensitive to
  blur, F14); the 4,000 val2017 images outside `V1k` (then impostor pairs must be restricted to
  `V1k`).

---

## 2. Architecture and efficiency ideas

### E1. Depth study: a DinoHash from every layer ("early exit")

**Question.** DinoHash reads its hash from the last of 12 transformer blocks. What happens if the
hash is read from block 6, 8 or 10 instead?

**Hypotheses.**
1. Most of the benign robustness is already present a few blocks before the end, so the last blocks
   can be skipped. Stopping at block 8 would cut compute by about a third.
2. Shallower layers are **less semantic**. F18 found that DinoHash's false matches are same-subject
   pairs (two zebra photos), and F10/F21 found that this is exactly why it loses at strict FPR. If
   the semantic grouping forms in the last blocks, an earlier exit could give *fewer* false matches,
   so a cheaper hash could be better at FPR 1e-4, not just "almost as good".

**Method.**
1. From the feature dump (§1), take the class token (and, separately, the mean patch token) at each
   block.
2. Fit a 96-bit head per block on the calibration set: PCA to 96 dimensions, then sign, as in the
   DinoHash repo's untrained head.
3. For each block, compute impostor distances, calibrate thresholds, and report the same tables as
   `benchmark_findings.md`: match rate at FPR 1e-2 / 1e-3 / 1e-4, by attack family.
4. Repeat the closest-pairs analysis of F18 per block: what share of the 100 closest pairs are
   same-subject? (Needs the COCO annotations, as P2 does, to do it with labels instead of by eye.)
5. For 3–4 chosen blocks, build the truncated model and run the white-box sweep on `V200`
   (shallower models are also cheaper to attack).

**What the result looks like.** A curve of match rate, false-match tail, bit accuracy under attack
and cost against depth, for both backbones. Either outcome is reportable: "the last four blocks buy
nothing a hash needs" or "depth is what makes it work".

**Cost.** Steps 1–4 are offline after the feature dump. Step 5 is a few Kaggle GPU-hours.

**Risks.** Low. Intermediate class tokens of a ViT are not trained to be used directly, so shallow
layers may be poor; mean patch tokens are the fallback and are included for that reason. The
fine-tuned backbone was trained only through its last layer's head, so its intermediate layers may
behave differently from the original's; that comparison is itself a result.

**Is it new?** Early exit and layer-wise probing are standard for classifiers. The search found no
layer-wise study of a perceptual hash, and none linking depth to semantic false matches.

### E2. Head redesign without retraining

**Question.** How much of DinoHash's weakness comes from the last step (feature → 96 bits) and can
be fixed there, with the backbone untouched?

**Why.** F16 traced the trained model's whole deficit (89.7% vs 94.3% at FPR 1e-3) to false
matches, and part of that to unbalanced bits: 7 bits lie outside 0.2–0.8 frequency, two are 1 in
about 98% of images, and the per-bit entropy sums to 89.8 of 96. F19 showed the false-match tail is
millions of times heavier than the Binomial model assumes. These are properties of the head.

**Variants to test** (all are numpy on saved logits/features):

| Variant | What changes | Expected effect |
|---|---|---|
| Median re-centring | Subtract each logit's median (from the calibration set) before the sign | Every bit becomes 50/50; should recover about 2 of the 4 threshold bits lost in F16 |
| Rotation (ITQ, Gong & Lazebnik 2011) | Rotate the 96-dim PCA space to minimise quantisation error | Fewer logits near zero, so fewer bits flip under benign edits |
| Whitening | Equalise variance across the 96 dimensions | Bits carry more equal information |
| Code length | 32, 48, 64, 96, 128, 192, 256 bits from the same features | How many bits a learned hash really needs at each FPR |
| Weighted Hamming | Weight each bit by its measured reliability | Better ROC with the same bits (server-side change only) |
| Head swap (F17) | Trained backbone + PCA head; original backbone + released head | Separates the effect of fine-tuning from the effect of the head |

**Metrics.** The benchmark tables at three FPRs; bit balance and entropy; and for the best two
heads, the white-box sweep (changing the head changes the attack surface: balanced bits have smaller
logit margins and may flip more easily, so a benign gain could be an adversarial loss).

**Important limit.** A new head produces **different hashes**, so it is a new hash function, not a
drop-in fix for a deployed database. State this.

**Cost.** Negligible after the feature dump. **Risks:** low; the gains may be small, which is still
an answer to F16/F17. **Is it new?** The techniques are old (ITQ is from 2011). Applying them to fix
a documented flaw in a released perceptual hash, with FPR-calibrated evaluation, is a contribution
of the "careful engineering" kind, not a new method.

### E3. Removing the patch-grid weakness

**Question.** F15 showed that a 1 px shift costs DinoHash about 5 bits, a third of its 17-bit
budget, while a shift by a whole 14 px patch costs far fewer. Can this be removed without retraining?

**Method.** Three variants, cheapest first:
1. **Blur before patching.** Because the patch embedding is linear, averaging it over small shifts
   equals low-pass filtering the input first (classic anti-aliasing, as in Zhang 2019 for CNNs).
   Zero extra cost. But F14 showed DinoHash loses bits under blur, so this may trade one weakness
   for another. Test small filters (σ = 0.5–2 px).
2. **Shift averaging.** Hash the image at k sub-patch offsets (for example 4 offsets on a 7 px
   grid), average the 96 logits, then take the sign. Costs k forward passes.
3. **Canonical alignment.** Choose the offset by a deterministic rule computed from the image (for
   example the offset that maximises the norm of the patch embedding), so a shifted copy picks the
   matching offset. One forward pass plus a cheap search. This is the idea behind shift-equivariant
   ViTs (Rojas-Gomez et al., CVPR 2024) [verify the reference].

**Metrics.** The patch-grid probe of F15 (bits changed against shift in px); the geometric family
of the benchmark; false matches (averaging may pull different images closer); cost.

**Link to defences.** Variant 2 with *random* offsets is a randomised defence. It must then be
attacked with EOT (attack #6 in `future_attacks.md`) before any robustness claim.

**Cost.** Needs new forward passes: about 1–2 h on the desktop per variant on `V200`.
**Risks:** medium; variant 1 may hurt, variant 3 may be unstable under crops. **Is it new?**
Anti-aliasing is known; using it on a perceptual hash, against a weakness measured in this thesis,
was not found in the search.

### E4. Compression versus robustness

**Question.** If a hash model is made smaller or faster with standard training-free tools, what
happens to (a) its hashes, (b) its benign robustness, (c) its adversarial robustness?

**Why it matters.** Client-side scanning runs on phones. NeuralHash already ships in float16.
Any deployed hash will be compressed, but the papers evaluate full-precision models.

**Compression methods** (none needs training):

| Method | Applies to | Tool |
|---|---|---|
| float16 weights | all three | PyTorch `.half()` / ONNX |
| int8 dynamic quantisation | DinoHash (a ViT is mostly linear layers) | `torch.ao.quantization.quantize_dynamic` |
| int8 static quantisation | CertPHash (ResNet), NeuralHash (ONNX copy) | ONNX Runtime quantisation, already in the venv |
| Token merging (ToMe, Bolya et al. 2023) | DinoHash | about 50 lines; merges similar tokens between blocks |
| Lower input resolution | DinoHash (224 → 182 → 154 → 112 px, position embeddings interpolated) | built into DINOv2; cost falls with the square of the side |
| Fewer blocks | DinoHash | from E1 |

**Three measurements per compressed model.**
1. **Hash compatibility:** how many bits differ from the original model's hash on the same image?
   If a phone runs the int8 model and the server's database was built with float32, every differing
   bit is spent from the matching threshold before any edit happens. Report mean and worst-case
   bits, and the match rate when the query uses the compressed model and the reference the original.
2. **Stand-alone quality:** the benchmark tables when both sides use the compressed model.
3. **Adversarial robustness:** bit accuracy and success rate on `V200`.

**Two attack questions that compression creates.**
- **Gradient masking.** An int8 model has no usable gradients, so a naive white-box attack will make
  it look robust. Evaluate with (i) attacks crafted on the float32 model and transferred, and
  (ii) a straight-through gradient. State which is reported.
- **Divergence attack.** Search for a small perturbation that makes the *compressed* and *original*
  models disagree (maximise the distance between their two hashes). An image like this passes
  server-side testing and fails on the device, or the reverse. This exists for classifiers (DIVA,
  Hao et al. 2022) [verify the reference]; the search found nothing for hashes.

**A specific check for CertPHash.** Its certificates are proved for the float32 network. Measure
how far the int8 outputs move from the float32 ones (in L1, the hash's own metric) and compare that
with the certified margin. If the gap is not small, the certificate does not cover the deployed
model, which is worth a paragraph.

**What the result looks like.** A table per hash: size (MB), latency (ms/image on the laptop CPU),
compatibility bits, match rate at FPR 1e-3, bit accuracy under attack. Plus a cost-vs-quality plot
marking which compressions are "free".

**Cost.** Benign runs on `V200` are an hour or two per variant on CPU. Adversarial runs transfer
saved float32 adversarial images (cheap) plus a few Kaggle hours.
**Risks:** medium. int8 backends differ between Windows/Linux/ARM; report the backend. Speed-ups
measured on a laptop CPU are indicative only.
**Is it new?** Found: compression-vs-robustness studies for classifiers, with mixed conclusions
(pruning tends to hurt, distillation tends to preserve). Not found: any such study for perceptual
hashes, or the hash-compatibility question.

### E5. TinyHash: distilling a hash into a small student

**Question.** Can a network several times smaller reproduce a learned hash, and does it inherit the
teacher's robustness?

This is the one idea that trains a model, and it answers two questions with one experiment, because
**a student trained only from the teacher's hashes is also a model-stealing attack.**

**Method.**
1. **Teacher labels.** Hash a training set with the teacher and keep the 96 logits. DinoHash on
   Kaggle (COCO train is available there as a public dataset). NeuralHash only locally, on images
   on disk, because Apple's files must not be uploaded.
2. **Student.** A small pretrained backbone with a 96-output layer: MobileNetV3-Small (about 2.5M
   parameters), EfficientNet-B0 (about 5M), or ViT-Tiny (about 6M), against the teacher's about 22M.
   Lower input resolution (160 or 112 px) is a second size knob.
3. **Loss.** Match the teacher's logits (scale-normalised regression) plus a hinge term on the sign
   of each bit, so the student is rewarded for agreeing on bits, not only on values.
4. **Evaluate** on the usual splits.

**Experiments.**

| # | Experiment | Answers |
|---|---|---|
| 1 | Student vs teacher: bit agreement, benchmark tables, latency | How small can the hash get? Is the student compatible with the teacher's database? |
| 2 | Students of the fine-tuned vs the original DinoHash, same recipe, clean images only | Is robustness **inherited** through distillation? (For classifiers the literature disagrees: Goldblum et al. 2020 say much of it transfers; Shao et al. say plain distillation loses it.) |
| 3 | Training-set size 1k / 10k / 100k images | How many queries does stealing a hash take? |
| 4 | Attack the student white-box, test on the teacher | A black-box attack on the teacher with **zero** attack-time queries; compare with NES (#2) at equal query budgets |
| 5 | Optional: distil with adversarial examples in the loop (ARD) | Can the student be made more robust than plain distillation gives? |

Experiment 4 gives the transfer result that `future_attacks.md` lists as a must-have (#1), in a
stronger form: the attacker does not need any similar public model.

**Cost (estimates).** Teacher labels for 118k images: under an hour on a T4. One student run of
about 20 epochs: 1–3 GPU-hours. About 8–10 runs for the table above fits in two weeks of the free
Kaggle quota. **The first run should be a 10k-image pilot.**

**Risks.** Medium to high: it is the only idea with a training loop to debug. Fallback if students
agree poorly with the teacher: report it as "these hashes are hard to steal" and keep experiments
1 and 3 only.

**Is it new?** Found: distillation for copy-detection *embeddings* (RDCD, Kim et al., WACV 2025),
and robustness-transfer studies for classifiers. Found for NeuralHash: black-box attacks using its
near-linearity (Bhatia & Meng 2022) and query-based attacks (Madden et al. 2024), but no
surrogate-training attack. Not found: distillation of a perceptual hash, or robustness inheritance
for hashes.

---

## 3. Probing ideas

### P1. How linear is each hash?

**Background.** Bhatia & Meng (2022) found that NeuralHash is approximately a *linear* function of
the image, and used that to evade, collide and leak information with no model access. Nobody has
asked the same of DinoHash or CertPHash.

**Why CertPHash is the interesting case.** Certified training (interval bound propagation) is known
to push networks towards having most ReLUs permanently on or off, which makes them closer to linear.
So the hypothesis is: **the larger the certified ε, the more linear the hash.** If true, it gives a
*mechanism* for the robustness-vs-privacy question of Idea 1: a near-linear hash is robust to small
noise, and also easy to copy and to invert.

**Method.**
1. **Global fit.** For each hash, fit a ridge regression from pixels (at the model's input size or
   a downscaled version) to the pre-threshold outputs on the 118k `coco100x100` images. Report R²
   and bit agreement on held-out images. For CertPHash this is a 12,288 → 144 linear map; minutes
   on the laptop.
2. **Local linearity.** For a perturbation δ of size ε, compare f(x + δ) with the first-order
   prediction f(x) + J·δ. Report the relative error against ε, for random and adversarial δ.
3. **Use the linear copy W as a weapon:**
   - **Attack:** the gradient of a linear model is the same for every image, so W gives a
     no-query, image-independent attack direction. This is a principled universal perturbation
     (attack #8). Measure its success against the real hash.
   - **Inversion:** apply the pseudo-inverse of W to a hash to get a low-resolution image. No
     training, no optimisation. Score it with SSIM and with the re-identification rate from Idea 1.
4. **Plot** linearity against certified ε (six CertPHash models), and against measured robustness
   for all nine learned models.

**Cost.** CPU only for CertPHash and the fits. Hashing 118k images with DinoHash needs about 2 h
(or Kaggle). **Risks:** low. If only NeuralHash is linear, that is a clean negative for the others.
**Is it new?** NeuralHash: done. DinoHash, CertPHash and the link to certified training: not found.

### P2. What does a hash reveal? Category probes

**Question.** Given only the hash of a photo, how well can an attacker tell what is in it?

This is a cheap, numeric version of Idea 1. It needs **no decoder training and no new hashing**:
the hashes of all 5,000 val2017 images for all 12 hashes are already in `hashes.csv` from Runs 1
and 2.

**Method.**
1. Get COCO's category labels for val2017. **[download]** `annotations_trainval2017.zip`, about
   240 MB. Ask first.
2. For each hash, train a logistic regression from the hash (96 bits, or CertPHash's 144 values) to
   labels, with 5-fold cross-validation over the 5,000 images:
   - "is there a person?" (about half of COCO images; balanced accuracy);
   - the 12 supercategories and 80 categories (mean average precision).
3. Reference points: random bits (chance), pHash and PDQ (expected near chance), and the full
   384-dim DINOv2 feature (upper bound).
4. Compare: trained vs untrained DinoHash; the CertPHash ε sweep; and, if E1/E2/E4/E5 are done,
   every layer, head, compressed model and student.

**What the result looks like.** One table: "from a 96-bit DinoHash, person-presence is predicted
with X% accuracy; from PDQ, Y%". If F18's reading is right, DinoHash leaks far more than the rest,
and E1 would show the leak growing with depth.

**The same labels also finish F18** (do close impostor pairs share a category?) and give a
content-dependent FPR: the false-match rate among same-category pairs versus overall.

**Cost.** Minutes on the laptop. **Risks:** very low.
**Is it new?** Struppek et al. (2022) showed NeuralHash hashes predict ImageNet categories, and
Hawkes et al. (2024) reconstruct images from four hashes with a GAN. The comparison across hash
families, robustness levels and model sizes was not found.

### P3. Natural near-collisions: pool size versus collision budget

**Question.** The Phase C sweep showed that picking the *closest* of 200 targets makes collisions
far easier than a random target (NeuralHash at ε = 2: 63.5% vs 0%). A real attacker can pick from
millions of images. How fast does the needed ε fall as the pool grows?

**Method.**
1. Hash a large pool (the 118k COCO train images) with each hash.
2. For each `V200` source and pool sizes N = 100, 1k, 10k, 118k (random subsets), find the nearest
   pool hash. Plot nearest distance against log N, per hash.
3. Run the closest-target collision attack at N = 118k and compare the success-vs-ε curve with the
   existing N = 200 curve.
4. Extrapolate the trend to 10⁸–10⁹ images, with stated uncertainty.

**Why it differs between hashes.** DinoHash groups images by subject (F18), so its nearest
neighbour should get close quickly. PDQ's impostor distances are tightly concentrated (F10), so its
curve should be flat. The same curve read from the defender's side is the false-match rate at scale.

**Cost.** Pool hashing: minutes for CertPHash and the classical hashes, about 1 h NeuralHash, about
2 h DinoHash on CPU. Attacks: a few Kaggle hours. **Risks:** low. The pool is 100×100 images, so
pool hashes are of low-resolution images; state this, or build the pool on Kaggle at full size.
**Is it new?** Leblanc-Albarel & Preneel (USENIX Security 2026) study black-box collisions on
NeuralHash and PhotoDNA; read it first. The pool-size scaling across families was not found.

---

## 4. Defence ideas

### D1. Per-bit smoothing certificates (what is left of Idea 2)

See §5 for why Idea 2 needs to change. What remains open after DualShield:

- **Collision certificates.** DualShield certifies that a near-duplicate cannot *evade*. A per-bit
  certificate (each bit has its own certified radius) bounds the hash distance in both directions,
  so it also certifies that an impostor cannot be pushed *into* a match.
- **Classical hashes.** Smoothing needs only forward passes, so pHash and PDQ can be certified.
- **A like-for-like comparison with CertPHash** at the same calibrated FPR: certification by
  training versus certification at inference time.

**Cost.** About 1,000 forward passes per image; fine for pHash, PDQ and CertPHash on CPU, Kaggle
for the others on `V200`. **Risks:** medium; the per-bit union bound may make the certificates
loose. Keep this as a secondary item.

### D2. A held-out verification hash against collisions

**Background.** Apple's 2021 design answered the NeuralHash collision attacks by saying a second,
independent hash on the server would re-check matches. Whether a forged collision on one hash
survives a second hash has not been measured for these models.

**Method.**
1. **Free version from the feature dump:** DinoHash's public hash uses PCA components 1–96. Build a
   second hash from components 97–192 of the same feature. Take the saved collision images from the
   Phase C sweep and check: do they also match the target on the second hash?
2. **Independent version:** check the same images against a different family (NeuralHash, CertPHash).
   This is the collision half of the transfer matrix (#1).
3. **Adaptive attacker:** someone who knows a second check exists attacks the whole 384-dim feature
   (make it close to the target's feature) instead of the 96 bits. Measure how much larger ε must be.
4. **Cost to the defender:** false-match and match rates when a match requires both hashes.

**What it shows.** Whether "check with a second hash" is a real defence or only raises the budget
slightly. It complements Idea 4, which looks at evasion: requiring *both* hashes to match makes
collisions harder and evasion easier, so both tables are needed together.

**Limit.** Only applies where the second hash can be kept on a server. In pure client-side scanning
the whole model is on the device.

**Cost.** Step 1–2 are re-hashing saved images (needs `--save-images` in `run_adversarial.py`, which
attack #1 needs anyway). Step 3 is one more Kaggle sweep. **Risks:** low.

---

## 5. What the search changes in the existing plan

1. **Idea 2 (smoothing) is partly taken.** DualShield (Sun et al., arXiv 2608.03101) is described
   in the plan as evaluating noise "only empirically". Its abstract says it gives a **certified ℓ2
   radius of about 0.3** from matching-time randomized smoothing, on eight hashes, with no
   retraining. That is most of Idea 2's headline. Read the full paper before spending time; D1 lists
   what is still open.
2. **Idea 1 (inversion)** has one more prior paper to cite and differentiate from: Hawkes et al.,
   *Perceptual Hash Inversion Attacks on Image-Based Sexual Abuse Removal Tools* (arXiv 2412.06056),
   GAN inversion of aHash, PDQ, NeuralHash and PhotoDNA. The controlled robustness-vs-leakage
   comparison still looks open. P1 and P2 reach the same question at a fraction of the cost.
3. **NeuralHash black-box attacks** have two more prior works: Bhatia & Meng (arXiv 2207.14258,
   near-linearity) and Leblanc-Albarel & Preneel (USENIX Security 2026, black-box collisions on
   NeuralHash and PhotoDNA). Cite both in the NES and transfer sections.
4. **Robustness Tokens** (Pulfer et al., arXiv 2503.10191) add a few learnable tokens to a frozen
   DINOv2 to make it robust to white-box attacks. This is a much lighter alternative to DinoHash's
   full fine-tuning, and a natural extension of E5 if there is time: train robustness tokens for
   the DinoHash loss and compare with the released fine-tuned model.

---

## 6. Recommended package and timeline

### Thesis shape with these ideas

*Working title:* "What a learned perceptual hash needs: depth, bits, size and robustness."

| Chapter | Content | Status |
|---|---|---|
| hashbench and benign benchmark | Five families, calibrated FPR | Done |
| Adversarial robustness | White-box sweep, Table 1 reproduction, plus transfer and NES | Mostly done; transfer and NES still needed |
| **Anatomy of a learned hash** | E1 depth, E2 head, E3 patch grid | New |
| **Smaller hashes** | E4 compression, E5 distillation and stealing | New |
| **What hashes reveal** | P1 linearity, P2 category probes | New; replaces the decoder version of Idea 1 |
| Optional | P3, D2, D1 | If time allows |

### Why this package

- **It fits the compute.** Seven of the ten ideas need no training. E1, E2 and D2 are offline after
  one feature dump. Only E5 depends on a GPU for more than a few hours.
- **It builds on findings only this project has:** the false-match explanation (F16), the semantic
  closest pairs (F18), the patch grid (F15), the head/backbone confound (F17).
- **It degrades gracefully.** Each idea produces a table on its own. If E5 fails, the thesis still
  has two complete new chapters.

### Timeline (16 Nov 2026 – mid-March 2027)

| Period | Work | Output |
|---|---|---|
| 16–30 Nov | Commit current work. Literature check for E1, E4, E5, P1 (and read DualShield in full). P2 (needs only the annotations). Write the feature-dump script; pilot on 20 images | Category-probe table; go/no-go on each idea |
| December | Feature dump (approved long run). E2, then E1 offline. E3 on `V200`. Transfer attack (#1) with saved images, which D2 reuses | "Anatomy" chapter results; F17 resolved |
| January | E4: compatibility and benign tables first, then adversarial. P1 on CertPHash, then the others | "Smaller hashes" part 1; linearity plot |
| February | E5: 10k pilot, then the experiment table. NES (#2) for the black-box comparison | "Smaller hashes" part 2 |
| 1–15 March | Buffer, optional P3 / D2 step 3, writing | Draft |

### Decisions needed

- [ ] Which ideas to take to the supervisor. Suggested: E1 + E2 + E4 + E5 + P1 + P2.
- [ ] Approval for the feature-dump run (`V200` first, about 1 h; `V1k` about 5 h for both backbones).
- [ ] OK to download the COCO annotations (about 240 MB) for P2 and F18.
- [ ] Whether full-resolution COCO train on Kaggle is acceptable as the calibration/training set.
- [ ] Whether the decoder version of Idea 1 is dropped in favour of P1 + P2.

---

## 7. Sources found in the search (2026-11-16)

Read each before citing; the summaries above come from abstracts and search snippets.

| Paper | Relevance |
|---|---|
| Sun et al., *Double Down on Defense* (DualShield), [arXiv 2608.03101](https://arxiv.org/abs/2608.03101) | Certified smoothing for deep hashes; overlaps Idea 2 |
| Bhatia & Meng, *Exploiting and Defending Against the Approximate Linearity of Apple's NeuralHash*, [arXiv 2207.14258](https://arxiv.org/abs/2207.14258) | Basis of P1 |
| Hawkes et al., *Perceptual Hash Inversion Attacks on Image-Based Sexual Abuse Removal Tools*, [arXiv 2412.06056](https://arxiv.org/html/2412.06056v1) | Prior inversion work (Idea 1, P2) |
| Madden et al., *Robustness of Practical Perceptual Hashing Algorithms to Hash-Evasion and Hash-Inversion Attacks*, [arXiv 2406.00918](https://arxiv.org/pdf/2406.00918) | Already in the plan; black-box and inversion baselines |
| Leblanc-Albarel & Preneel, [USENIX Security 2026 prepublication](https://www.usenix.org/system/files/conference/usenixsecurity26/sec26_prepub_leblanc-albarel.pdf) | Black-box collisions on NeuralHash and PhotoDNA (P3, NES) |
| Kim et al., *Relational Self-supervised Distillation with Compact Descriptors for Image Copy Detection* (RDCD), [arXiv 2405.17928](https://arxiv.org/abs/2405.17928) | Closest prior work to E5 (embeddings, not hashes) |
| Goldblum et al., *Adversarially Robust Distillation*, [arXiv 1905.09747](https://arxiv.org/pdf/1905.09747) | Robustness can transfer to a student (E5) |
| Shao et al., *How and When Adversarial Robustness Transfers in Knowledge Distillation?*, [arXiv 2110.12072](https://ar5iv.labs.arxiv.org/html/2110.12072) | The opposing finding (E5) |
| *Pruning Adversarially Robust Neural Networks without Adversarial Examples*, [arXiv 2210.04311](https://arxiv.org/pdf/2210.04311) | Compression vs robustness for classifiers (E4) |
| *Improving Adversarial Robustness in Weight-quantized Neural Networks*, [arXiv 2012.14965](https://arxiv.org/pdf/2012.14965) | Quantisation vs robustness (E4) |
| Parekh et al., adversarial transfer to compressed models, [arXiv 2209.13785](https://arxiv.org/pdf/2209.13785) | Attacks crafted on a full model, tested on its compressed versions (E4) |
| Pulfer et al., *Robustness Tokens*, [arXiv 2503.10191](https://arxiv.org/abs/2503.10191) | Light-weight robustness for DINOv2 (§5) |

Cited from memory, not checked in this search: ITQ (Gong & Lazebnik 2011), ToMe (Bolya et al.
2023), anti-aliased CNNs (Zhang 2019), shift-equivariant ViTs (Rojas-Gomez et al. 2024), DIVA (Hao
et al. 2022). Confirm titles and venues before citing.
