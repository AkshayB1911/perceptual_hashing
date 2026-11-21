# pHash + PDQ NES black-box attack (CPU run)

This package runs one experiment from a BTP thesis on perceptual hashing: the **NES
score-based black-box attack** (Ilyas et al. 2018) against the two classical hashes,
**pHash** (64 bit) and **PDQ** (256 bit), on 200 COCO val2017 images (the "V200" split).
The learned hashes (DinoHash, NeuralHash, CertPHash) were attacked with the same code and
settings on Kaggle GPUs. pHash and PDQ need no model and no GPU, so they run here on a CPU.

Nothing here downloads data or weights. Everything needed is in this folder.

## What the attack does (short)

- The attacker can only submit an 8-bit image and read back its hash. Every submitted image
  counts as one **query**. The budget is 10,000 queries per image.
- **Evasion:** change the image (each pixel by at most ε) so that its hash moves *further*
  than the match threshold from the original's hash. That way the image is no longer
  detected.
- **Collision:** change image *i* (by at most ε per pixel) so that its hash comes *within*
  the threshold of a different target image *j*, giving a false match. The target is
  image (i + 100) mod 200 in the list.
- **Settings:** evasion ε = 8/255, collision ε = 16/255, evasion ε = 4/255.
- **Thresholds** are the calibrated false-positive rate 1e-3 thresholds from the benign
  benchmark: pHash Hamming distance 16/64, PDQ 100/256. They are read from
  `hashbench/results/coco_run*/overall.csv`.

## Layout

```
phash_pdq_black_box_run/
  README.md, AGENTS.md          this file and the instructions for an agent
  run.sh                        the whole job: setup, tests, smoke run, full run, results zip
  requirements.txt              Python packages (CPU-only torch)
  hashbench/
    pyproject.toml
    hashbench/                  the library (blackbox.py, blackbox_classical.py, hashes/, ...)
    scripts/run_nes_classical.py   the runner for this job (CPU, parallel over images)
    scripts/run_nes.py, run_adversarial.py   helpers it imports (+ the --summary-only report)
    tests/                      unit tests for the attack and the pHash/PDQ adapter
    data/coco/val2017/          the 200 images
    data/splits/V200.txt        their order (this fixes the collision targets)
    results/coco_run1|2/overall.csv   the thresholds
```

## How to run

You need Linux, Python 3.10 to 3.13, and about 2 GB of free disk space for the venv.

```bash
cd phash_pdq_black_box_run
bash run.sh                 # uses (number of CPUs - 1) worker processes
# or choose the number of workers:
WORKERS=8 bash run.sh
```

For a long run, keep it alive after you log out:

```bash
nohup bash run.sh > run_sh.out 2>&1 &
tail -f hashbench/results/nes_v200_classical/run.log
```

`run.sh` does the following:

1. Creates `.venv` and installs the requirements.
2. Runs the unit tests.
3. Runs a 2-image smoke test (output goes to `results/nes_smoke`, a separate folder).
4. Runs the full experiment into `hashbench/results/nes_v200_classical/`.
5. Packs the results into **`nes_classical_results.zip`** in this folder.

## Time

On a laptop CPU, one query takes about 7 ms for pHash and 16 ms for PDQ, per worker
process. There are 1,200 attacks in total (2 hashes × 3 settings × 200 images).

| Case | Total CPU work | Wall time with 8 workers | Wall time with 16 workers |
|---|---|---|---|
| Worst case (every attack uses all 10k queries) | about 37 core-hours | about 4.5 h | about 2.5 h |
| Expected (successful attacks stop early) | about half of the worst case | about 2–3 h | about 1–1.5 h |

The run is **resumable**: if it stops, run `bash run.sh` again with the same settings.
Images already in `nes.csv` are skipped.

## Output (what to send back)

`nes_classical_results.zip` contains:

| File | Content |
|---|---|
| `nes_v200_classical/nes.csv` | One row per (hash, setting, image), 1,200 rows. Columns: `hash, mode, eps_255, image, target, threshold, max_distance, orig_distance, distance, success, queries, steps, linf, ssim, seconds_per_image` |
| `nes_v200_classical/config.json` | All the settings, plus the CPU count and torch version |
| `nes_v200_classical/summary.csv`, `summary.md` | Success within 1k/2k/5k/10k queries, median queries, mean distance, ℓ∞, SSIM |
| `nes_v200_classical/run.log` | Full log of the run |
| `env.txt` | OS, CPU, Python and package versions |
| `smoke/` | The smoke-test output |

The columns are identical to the Kaggle GPU runs (`run_nes.py`), so the results can be
merged with them directly.

## Do not change

The attack settings are: 10,000 queries, 20 antithetic pairs, σ = 1.0·ε, step 0.2·ε,
momentum 0.9, a 32×32 search space, FPR 1e-3 thresholds, max side 512, seed 0, and the
V200 order. They must stay the same as in the GPU runs, or the results cannot be compared.
`run_nes_classical.py` refuses to append to a results folder that was created with
different settings.
