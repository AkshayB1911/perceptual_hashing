# Reproducing Table 1 of the DinoHash paper

*BTP: Perceptual Hashing. Run on 2026-11-11. This supersedes the 5-image probe in
`adversarial_testing_report.md` §4.*

**Paper:** S. Singhi, A. Gupta, L. Struppek, *DINOHash: Learning Adversarially Robust Perceptual
Hashes from Self-Supervised Features*, ICML 2025 CODEML workshop
(`research_papers/DinoHash.pdf`; https://icml.cc/virtual/2025/48181). Table 1 there is the one
reproduced here.

*Related paper by the same first author, not the source of Table 1:* Singhi, Yadav, Gupta,
Ebrahimi, Hassanizadeh, *Provenance Detection for AI-Generated Images*, arXiv:2503.11195.

**Authors' code:** https://github.com/proteus-photos/dinohash-perceptual-hash

## 1. Summary

| | Result |
|---|---|
| DinoHash (trained) | **Reproduces.** 68.9% vs the paper's 72.2% at ε=4/255 |
| DinoHash (untrained) | **Reproduces exactly.** 0.0% vs 0.3% / 0.0% |
| NeuralHash | **Does not reproduce.** 75.6% vs the paper's 0.1% at ε=4/255 |
| Why NeuralHash differs | The authors' committed NeuralHash script attacks and scores **DinoHash**, then compares it with **NeuralHash** bits (§4). Copying that logic gives 0.0%, matching the paper |

Our thresholds and setup were not the cause of the disagreement with the paper. Under the
authors' own attack, NeuralHash is at least as robust as trained DinoHash.

## 2. Setup

| Item | Value |
|---|---|
| Attack | The authors' APGD, `apgd_attack.py` @ `c7008888b7`, vendored **unmodified** as `hashbench/scripts/third_party/dinohash_apgd_c700888.py` |
| Attack details | 100 steps, random start in the ε-box, first step 2ε (halved when stalling), BCE loss towards the flipped hash, best-loss iterate kept |
| Budget | ℓ∞, ε = 4/255 and 8/255, in [0, 1] pixel space (normalisation inside the model) |
| Input | 224×224 bilinear resize (DinoHash), 360×360 (NeuralHash) |
| Metric | Bit accuracy = share of the 96 bits that are the same as the clean hash. Higher = more robust |
| Images | First 50 images of `V200` (COCO val2017). The paper used DiffusionDB |
| Trained model | Released DinoHash ViT-S/14-reg, 96 bits. Its head is identical to the paper's `PCA_96` file, so this is the paper's model |
| Untrained model | Meta's DINOv2 ViT-S/14-reg + the same `PCA_96` head (what the authors' eval code computes) |
| Hardware | DinoHash rows on a Kaggle GPU (about 2.5 min); NeuralHash rows on the laptop CPU (about 1.5 h) |

## 3. Results (bit accuracy %, 50 images)

| Model | Paper ε=4 | **Ours ε=4** | Paper ε=8 | **Ours ε=8** |
|---|---|---|---|---|
| DinoHash trained | 72.2 | **68.9** | 64.8 | **56.1** |
| DinoHash untrained | 0.3 | **0.0** | 0.0 | **0.0** |
| NeuralHash (attacked correctly) | 0.1 | **75.6** | 0.0 | **58.4** |
| NeuralHash, the authors' script logic (§4) | 0.1 | **0.0** | 0.0 | **0.0** |

- Rounding the attacked image to 8 bits changes these numbers by at most 0.2 points.
- Likely causes of the remaining gap on trained DinoHash: different images (COCO vs DiffusionDB)
  and a small sample (50 images).

**Comparison with our own attack (the full `V200` sweep, `results/adv_v200_kaggle`).** At ε=4/255
trained DinoHash keeps only **10.3%** of its bits, and NeuralHash keeps **78.4%**. The sweep is
stronger because it adds:
- small-step PGD next to APGD;
- a warm start from the previous ε;
- best-iterate selection by bits flipped instead of by loss.

APGD's large steps stall on trained DinoHash. So the paper's 72% describes robustness against
that one attack, not against a well-tuned one with the same budget.

## 4. The problem in the authors' NeuralHash script

File: `neuralhash_attack_results.py` @ `f8c2fd5d65`:
https://github.com/proteus-photos/dinohash-perceptual-hash/blob/f8c2fd5d65/neuralhash_attack_results.py

```python
from hashes.dinohash import dinohash, dinov2                       # line 13
from apgd_attack import APGDAttack                                  # line 14
...
features = neuralhash(image_tensors[0].unsqueeze(0)).squeeze((0,1)).T   # line 66
logits = features @ seed1.T                                             # line 68: clean NeuralHash bits
adv_images, _ = apgd.attack_single_run(image_tensors, logits, 100)      # line 70
adv_logits = dinohash(adv_images, differentiable=False, logits=True, prod_output=False)  # line 71
acc = ((adv_logits>=0) == (logits>=0)).float().mean(1).cpu().numpy()   # line 72
```

| # | Line | Problem |
|---|---|---|
| 1 | 70 | The attack optimises **DinoHash**, not NeuralHash. `APGDAttack` calls `dinohash` inside it, via `from hashes.dinohash import dinohash` in `apgd_attack.py` @ `c7008888b7`: https://github.com/proteus-photos/dinohash-perceptual-hash/blob/c7008888b7d3cbe7d9fdd859b0d3abed579ece6f/apgd_attack.py. NeuralHash is never passed to the attack |
| 2 | 71–72 | The attacked image is hashed with **DinoHash** and compared with the clean **NeuralHash** bits. These are two unrelated hashes. The attack pushes DinoHash's output to the opposite of the reference bits, so the agreement is about 0% |
| 3 | 66 | Only the first image of each batch (`image_tensors[0]`) is hashed, and its bits are used as the reference for every image in the batch |

**Check:** reproducing this logic (`neuralhash_as_committed` in our script) gives 0.0% / 0.0%,
matching the paper's 0.1% / 0.0%. Attacking NeuralHash itself with the same APGD gives
75.6% / 58.4%.

**Caveat:** we cannot prove that this committed script produced the number in the paper. It is
the NeuralHash evaluation script in their repository, and it is the only way we found to obtain
their value. Worth confirming with the authors.

## 5. Other notes

- The script's docstring cites "Singhi, Gupta, Struppek, *DINOHash*, ICML 2025 CODEML". That is
  correct: Table 1 comes from that workshop paper, not from arXiv:2503.11195.
- The paper only evaluates a **white-box** APGD attack (100 steps). It trains against 10-step APGD
  and has no black-box or transfer evaluation. Black-box results for DINOHash come from Sun et al.,
  arXiv:2608.03101 (2026).
- `adversarial_testing_report.md` §4.6 lists model size (paper ViT-B vs released ViT-S) as a
  possible cause of the gap. That no longer applies: the released head equals the paper's
  `PCA_96`.
- The authors' later APGD version (`1bb6b0055e`) crashes as committed
  (`hash_loss_grad() missing 1 required positional argument: 'hasher'`), so `c7008888b7` was used.

## 6. Files and how to rerun

| File | Content |
|---|---|
| `hashbench/scripts/reproduce_dinohash_table1.py` | Reproduction script (models, metric, resumable `raw.csv`) |
| `hashbench/scripts/third_party/dinohash_apgd_c700888.py` | The authors' APGD, unmodified |
| `hashbench/scripts/third_party/hashes/dinohash.py` | Shim for the APGD's hard-coded `hashes.dinohash` import |
| `kaggle/table1/` | Separate Kaggle package (`build_table1.sh`, `push_table1.sh`, `kernel/run_table1.py`). The older `kaggle/` scripts are unchanged |
| `hashbench/results/dinohash_table1_repro_kaggle/` | DinoHash rows (GPU): `raw.csv`, `summary.md`, kernel log |
| `hashbench/results/dinohash_table1_repro_local/` | NeuralHash rows (CPU): `raw.csv`, `summary.md` |
| `hashbench/results/dinohash_table1_repro_combined/` | Merged table |

```bash
# DinoHash rows on Kaggle (needs the kaggle CLI + token; never uploads NeuralHash files)
bash kaggle/table1/push_table1.sh <kaggle_username> 50
# NeuralHash rows locally (from hashbench/)
python scripts/reproduce_dinohash_table1.py --limit 50 --models neuralhash neuralhash_as_committed \
    --device cpu --out results/dinohash_table1_repro_local
```
