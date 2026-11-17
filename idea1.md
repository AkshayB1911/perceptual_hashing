# Idea 1: Does robustness cost privacy?

*BTP: Perceptual Hashing. Written 2026-11-05. Expands Idea 1 in `thesis_research_plan.md`.
Status: **proposal, not started**. Nothing here is a result yet, and the choice of direction still
needs the supervisor's agreement.*

## 1. Summary

**Question.** If a perceptual hash is trained to resist adversarial attacks, does it become easier
to recover the original image from the hash?

**Hypothesis.** More robust hashes leak more about the image.

**Approach.** For each hash version, train an attacker to rebuild images from hashes alone. Measure
how much it recovers, then plot leakage against the robustness measured in Phase C. Both outcomes
are useful:

| Result | Meaning |
|---|---|
| Leakage rises with robustness | Robustness defences have a privacy cost |
| Flat, or leakage falls | The intuition from classifiers doesn't carry over to perceptual hashes |

## 2. Why this matters

In client-side scanning, the photo stays on the device, but its hash can leave it. The hash is often
treated as safe because it is short and looks meaningless. If it can be turned back into a
recognisable image, it reveals content the system was meant to keep private.

Robustness work usually asks only whether a hash resists attack. This study asks what that resistance
costs.

### Why robust hashes might leak more

- A normal network can depend on tiny patterns that humans can't see. That makes it easy to attack,
  and those patterns may say little about what the image looks like.
- A robust network can't depend on those patterns, because small changes mustn't affect its output.
  This pushes it towards larger features such as shapes and objects.
- In classifiers, robust models are known to be easier to invert (Engstrom et al. 2019; Santurkar et
  al. 2019).

### Why they might leak less

Larger-ε CertPHash models keep different images closer together (mean impostor distance 0.155 at ε=8,
about 0.21 for other variants) and agree less with real PhotoDNA (96.9% at ε=1, 15.9% at ε=8). They
may simply carry less information. The effect could also be mixed: less information overall, but more
of what shows up in a reconstruction. The experiment has to be able to show this, not assume the
hypothesis.

### What is new

As far as we know, earlier inversion work (Struppek et al., Madden et al. 2024, the Ribosome attack on
PhotoDNA) inverts individual hashes and doesn't ask whether robustness defences change leakage with
everything else held fixed. **This has not been verified.** The literature check in Stage 0 must
happen before any large run.

## 3. Threat model

| | Assumption |
|---|---|
| **Attacker has** | A hash of an unknown image, and the ability to compute the same hash on their own images |
| **Attacker wants** | To learn what the original image looks like |
| **Decoder attack needs** | Only pairs of images and hashes, so no access to the network's internals |
| **Direct inversion needs** | White-box access to the network |
| **Out of scope** | Attacks on the server, the database or the matching protocol |

This study uses COCO images only.

## 4. Hash versions

There are 11 decoders to train. PDQ-dihedral uses the same hash as PDQ and changes only the matching
step, so it needs no decoder of its own.

| Family | Versions | Output | Role |
|---|---|---|---|
| CertPHash | base, adv, ε=1, 2, 4, 8 | 144 values, 0–255 | **Main evidence**: one architecture across a range of robust training |
| DinoHash ViT-S | untrained, trained | 96 bits | Second comparison, with caveats (§9) |
| NeuralHash | one | 96 bits | Non-robust learned reference |
| pHash | one | 64 bits | Classical reference |
| PDQ | one | 256 bits | Classical reference |

**Optional:** DinoHash ViT-B at 512 bits, trained and untrained.

**Compare within a family, not across families.** CertPHash outputs 144 values and DinoHash 96 bits, so
CertPHash can hold more information whatever its robustness. Classical and NeuralHash results are
reference points only.

## 5. Data

| Use | Data | Status |
|---|---|---|
| Decoder training and validation | COCO train at 100×100 (118,287 images) | On disk; hold out a fixed validation subset |
| Leakage testing | `data/splits/V1k.txt` (1,000 val2017 images) | On disk |
| Re-identification gallery | All 5,000 val2017 images | On disk |

The training and test images don't overlap.

### Resolution mismatch to resolve first

The plan assumes 100×100 images are enough because the decoders output 64×64. That holds for the
reconstruction target, but **not necessarily for the hash input**:

- CertPHash (64×64 input) should be little affected, and neither should pHash or PDQ.
- DinoHash (224×224) and NeuralHash (360×360) would hash an enlarged, blurry 100×100 image. DinoHash is
  sensitive to this kind of change: on rescaling to 10% of a 512 px image, it matches only 10% (trained)
  and 35% (untrained) of copies.

So a decoder trained on hashes of 100×100 images may do badly on hashes of full-size images, and that
would look like low leakage when it isn't.

**Decision:** in the main evaluation, downscale test images to 100×100 before hashing, so training and
testing use the same pipeline. Separately report results on hashes of full-size images, to measure the
effect of the mismatch. If it is large for DinoHash, consider full-resolution COCO train, which would
need a download and your approval.

## 6. Method

### 6.1 Attack 1: learned decoder (main attack)

1. **Hash** every training image with each version and save the results, so hashing runs once.
2. **Train** a decoder to map a hash to a 64×64 RGB image.
3. **Test** it on hashes of images it hasn't seen.

Fairness rules:

- **Same setup for all versions:** architecture, optimiser, training budget, and training and validation
  images. Only the input layer changes, to fit the hash size.
- **Starting architecture:** a transposed-convolution network with about 2–5M parameters.
- **Input encoding:** bits as ±1; CertPHash values scaled to [−1, 1].
- **Loss:** start with L1 pixel loss. Add a perceptual loss only if pilots need it, and then for all
  versions.
- **Model selection:** use the validation subset, never the test images.
- **Repeats:** train the final models with at least 3 random seeds, to separate real differences from
  training noise.

### 6.2 Attack 2: direct inversion (supporting attack)

Start from noise and use gradient descent to find an image whose hash matches the target, with a
smoothness penalty to keep it image-like. This can reuse the loss and optimiser ideas from
`hashbench/adversarial.py`.

- **Covers learned hashes only.** It would need differentiable pHash and PDQ, and those attacks are out
  of scope for now, so pHash and PDQ are measured by the decoder alone.
- **Why include it:** it needs no training data. It also tests whether results depend on the decoder.
- **Limit:** a hash has many possible preimages, so a matching hash doesn't guarantee a similar image.
  Report how close the hash got and how similar the image is, separately.

### 6.3 Controls

Without these, a good-looking score could mean nothing.

| Control | What it does | Why |
|---|---|---|
| **Mean image** | Outputs the average training image every time | The score a decoder must beat to show that it learned anything from the hash |
| **Shuffled hashes** | Trains on hashes paired with the wrong images | Should fall to the mean-image level; if not, something is wrong |
| **Larger decoder** | Repeats selected versions with about 3× capacity | Shows whether results depend on decoder strength |
| **Upper reference** | Encoder-decoder with the same code size, trained end to end | Shows how much a code of that size could keep if built for reconstruction |

## 7. Measuring leakage

Measure on the V1k test images, at 64×64.

| Measure | Question | Notes |
|---|---|---|
| **PSNR, SSIM** | Does it look like the original? | Pixel-level, and they reward blur |
| **Re-identification** | Can it pick out the right image from 5,000? | Top-1, top-5, top-1%; chance for top-1 is 1 in 5,000 |
| **Gain over the mean image** | How much of the score comes from the hash? | Fairer across versions than raw scores |
| **Bit entropy** | How much can the hash carry at most? | An upper bound; correlated bits reduce it further |

### Re-identification

For each reconstruction, extract features, rank all 5,000 gallery images by similarity, and record where
the true image lands.

Two things have to be stated with these results:

- **The feature extractor could favour one family.** Original DINOv2 features are semantic, which might
  favour DinoHash, whose hash is also built from DINOv2. So also report re-identification using plain
  pixel similarity.
- **This is not a realistic attack on its own.** An attacker who already has the gallery could simply
  hash it and look for a match. Here, re-identification measures whether reconstructions carry
  image-specific content. Reporting it as a privacy attack would overstate it.

### Entropy

Existing per-bit entropy sums (`benchmark_findings.md`, F12) are a starting point, all upper bounds:
NeuralHash 95.4 of 96, DinoHash untrained 93.0, DinoHash trained 89.8. CertPHash's values aren't bits,
so its measure needs defining, for example per-value entropy after a fixed binning. Compare it only
within the family.

## 8. Measuring robustness and the final plot

Each version becomes one point: robustness on the x-axis and leakage on the y-axis. Use evasion results
from `results/adv_v200_kaggle` and `results/adv_v200_neuralhash`.

**Current CertPHash numbers** (mean L1 distance moved by evasion at ε=8/255, n = 200; lower means
harder to move):

| Version | Mean L1 moved |
|---|---|
| base | 3,404 |
| adv | 2,177 |
| ε=1 | 1,914 |
| ε=2 | 1,847 |
| ε=4 | 1,576 |
| ε=8 | 937 |

**Current DinoHash numbers** (mean bits flipped by evasion at ε=1/255, n = 200): untrained 76.2, trained
52.2. ε=1 is used because both are close to fully broken at larger budgets.

Rules for the plot:

- **Normalise before comparing.** ε=8 has a narrower output range, so a raw L1 distance overstates its
  robustness. Also divide by each version's mean impostor distance, and show the result both ways.
- **Plot each family separately.**
- **Show uncertainty:** bootstrap intervals over test images, and the spread across decoder seeds.
- **Don't over-read the trend.** With 6 CertPHash points, report the rank correlation alongside the plot
  and treat it as descriptive, not as proof of a causal law.

## 9. Known threats to validity

| Threat | Why it matters | How to handle it |
|---|---|---|
| **The x-axis depends on Phase C** | Our DinoHash and NeuralHash results disagree with the DinoHash paper, and the investigation is still open | Finish it first and record the conclusion. CertPHash isn't part of that disagreement |
| **DinoHash isn't a clean pair** | Trained and untrained differ in both backbone and head | Test the two mixed models from `benchmark_findings.md` F17 before crediting fine-tuning |
| **DinoHash robustness gap looks modest** | Trained DinoHash is also close to fully broken, so its signal may be weak | Treat it as supporting evidence; rely on CertPHash |
| **One CertPHash model per setting** | ε can't be separated from run-specific quirks, such as the ε=2 and ε=4 black-pixel shortcut | State this clearly. Check whether outliers come from such quirks |
| **Information vs robustness** | Larger ε may reduce information, hiding or reversing a real effect | Report entropy and gain over the mean image, not raw scores alone |
| **Decoder strength** | Low recovery shows only what this attacker achieved | Use the larger decoder and direct inversion. Never claim that a hash is private |
| **COCO-specific results** | Decoders learn COCO's typical content | Report as COCO results; don't generalise further |
| **Resolution mismatch** | See §5 | Use the same pipeline for both, and report the mismatch |

## 10. Work plan

No run longer than about an hour starts without approval.

### Stage 0: Literature check
Search for: "perceptual hash inversion robustness", "robust representations invertibility privacy",
"client-side scanning hash leakage".

**Pass if:** nobody has already published this comparison on perceptual hashes. If somebody has, adjust
the question or fall back as in §12.

### Stage 1: Small pilot
- CertPHash base and ε=8 only, the two ends of the range. They hash quickly (about 5 minutes for the
  whole training set, per the plan's estimate).
- Train one decoder each, plus the mean-image and shuffled-hash controls.

**Pass if:** decoders clearly beat both controls, and the two versions differ by more than the noise
across seeds. Otherwise, try a stronger decoder once before changing direction.

### Stage 2: Pipeline for the other hashes
- Add one bit-hash version, such as DinoHash untrained.
- Test the resolution mismatch from §5.
- Freeze the decoder, training budget and metrics before Stage 3, so choices aren't tuned to the result.

### Stage 3: Full decoder experiment
All 11 versions, at least 3 seeds each, all metrics, and the plots from §8.

### Stage 4: Direct inversion
Run on the learned hashes for a subset of test images, sized after a timing pilot.

### Stage 5: Separate DinoHash's head and backbone
Only if the DinoHash pair shows a difference worth explaining. Test the two mixed models.

### Stage 6: Write-up
Thesis chapter: method, results, controls, limits.

## 11. Compute

Estimates from `thesis_research_plan.md`, except NeuralHash; all need a timing check.

| Task | Estimate |
|---|---|
| Hash the training set, CertPHash | About 5 min per version on CPU |
| Hash the training set, DinoHash ViT-S | About 2 h per version on CPU; better on Kaggle |
| Hash the training set, NeuralHash | Roughly 1 h on CPU, from benchmark timing |
| Hash the training set, DinoHash ViT-B | About 7 h per version on CPU; use Kaggle |
| Train one decoder | 1–3 h on CPU, or 10–20 min on a Kaggle GPU |
| Full decoder experiment (11 versions × 3 seeds) | About 6–11 GPU-hours, before controls |

**Apple's model files must not go to Kaggle.** Compute NeuralHash's hashes locally and upload only the
hash file, since the decoder needs just the hashes and images.

### Expected new files (nothing created yet)

| File | Purpose |
|---|---|
| `hashbench/hashbench/inversion.py` | Decoder model and direct-inversion attack |
| `hashbench/scripts/hash_trainset.py` | Hash the training images once per version |
| `hashbench/scripts/train_decoder.py` | Train with fixed settings and save metrics |
| `hashbench/scripts/eval_leakage.py` | Compute all §7 measures and controls |
| `hashbench/tests/test_inversion.py` | Basic checks, such as the shuffled-hash control |
| `hashbench/results/inversion/` | Results, configurations and sample reconstructions |

## 12. Possible outcomes and fallbacks

| Outcome | What it means | Thesis framing |
|---|---|---|
| Leakage clearly rises with robustness | The hypothesis holds on these models | A measured robustness–privacy trade-off |
| Flat within noise | No effect at this decoder strength | A negative result, with its limits stated |
| Leakage falls with robustness | Robust training may discard information | Robustness may help privacy here, at a cost to fidelity and matching |
| Decoders barely beat the controls | Hashes resist this attacker, or it's too weak | Try direct inversion and the larger decoder; report what was tried |
| Already published | The question isn't new | Adapt, or switch to Idea 3 + Idea 4 as in the research plan |

## 13. Deliverables

- Trained-decoder results for all 11 versions, with controls.
- A leakage-against-robustness plot for each family.
- A table of PSNR, SSIM, re-identification and entropy per version.
- Sample reconstructions from COCO.
- A thesis chapter stating what the results do and don't show.

## 14. Decisions for the supervisor

1. Is Idea 1 the right direction, alone or with Idea 2?
2. Is it acceptable for CertPHash to carry the main evidence, with DinoHash as support?
3. Should a stronger attacker, such as a diffusion-based decoder, be in scope, or is a fixed small decoder
   enough for a BTP?

## References

- Engstrom et al. *Adversarial Robustness as a Prior for Learned Representations.* 2019.
- Santurkar et al. *Image Synthesis with a Single (Robust) Classifier.* NeurIPS 2019.
- Struppek et al. *Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash.* FAccT 2022.
- Madden et al. *Robustness of Practical Perceptual Hashing Algorithms to Hash-Evasion and Hash-Inversion
  Attacks.* arXiv 2406.00918.
- Yang et al. *CertPHash.* USENIX Security 2025.
- Singhi et al. *DINOHash.* ICML 2025 CODEML workshop.
