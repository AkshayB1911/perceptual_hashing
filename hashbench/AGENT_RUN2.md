# Instructions for the agent: run the DinoHash benchmark (Run 2)

You are running one fixed experiment for a bachelor thesis on perceptual image hashing. The
results will be compared with an earlier run of other hashes (Run 1), done on a different machine
with the same images, attacks and settings. **The comparison is only valid if you run exactly the
commands below**, so follow them as written.

Work through the steps in order. Run every command from this folder (the one containing
`pyproject.toml`).

## Rules

- **Do not** edit any code, image list, model or reference file.
- **Do not** change, add or remove command options. In particular, do not use `--limit` or
  `--attack-limit`, and do not reduce the images, attacks or impostor pairs to save time.
- **Do not** change package versions or replace a package with another one.
- **Do not** download any dataset or model. Everything needed is in this folder. The only
  downloads allowed are Python 3.12 (if missing) and the packages in `requirements-lock.txt`.
- **Do not** run other hashes (CertPHash, NeuralHash or the ViT-B DinoHash models). Their files are
  not in this folder.
- **If a step fails or a check does not pass, stop and report it to the user** with the exact
  output. Do not work around it, because a workaround can silently make the results incomparable.

## Step 1: Install

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e ".[neural]"
```

Keep the virtual environment active for all the steps below (run `source .venv/bin/activate` again
in any new shell).

## Step 2: Consistency check (about 5 minutes)

This reruns 20 images that were already run on the Run 1 laptop and compares the results.

```bash
mkdir -p data/crosscheck20
while read -r f; do cp "data/coco/val2017/$f" data/crosscheck20/; done < data/splits/crosscheck20.txt
python -m hashbench.benchmark --images data/crosscheck20 --out results/crosscheck_linux_dinohash \
    --algorithms dinohash dinohash_untrained --workers 1
python scripts/compare_runs.py results/crosscheck_laptop_dinohash results/crosscheck_linux_dinohash
```

Continue only if, for **both** `dinohash` and `dinohash_untrained`:

- `identical_genuine_distances` is at least 0.80, and
- `max_abs_genuine_difference` is at most 3.

A few differences of a bit or two are normal floating-point rounding between machines. Otherwise,
stop and send the user the table. Keep the table either way, for the final message.

## Step 3: Run the benchmark

Use `--workers 1` whatever the size of the machine: that one process already uses all the CPU
cores. On a 6-core laptop, this run was estimated at about 4 to 6 hours.

```bash
mkdir -p results/coco_run2
nohup python -m hashbench.benchmark --images data/coco/val2017 --out results/coco_run2 \
    --algorithms dinohash dinohash_untrained --attack-images data/splits/V1k.txt \
    --workers 1 --checkpoint-every 100 > results/coco_run2/run.log 2>&1 &
```

Check progress with `tail -c 200 results/coco_run2/run.log`.

**If the run stops early** (crash, reboot, closed session), run the same command again on the same
machine. It resumes from the last checkpoint, saved every 100 images. Change `>` to `>>` so the
earlier log is kept. Do not delete `results/coco_run2/checkpoints/`, and do not start again from
nothing unless the user says so.

The run has finished when the log ends with a `wrote ... genuine and ... impostor distances` line,
`genuine.csv`, `impostor.csv` and `hashes.csv` exist in `results/coco_run2`, and the `checkpoints`
folder is gone.

## Step 4: Produce the report and tables

These are the same commands used for Run 1, so the files have the same format.

```bash
python -m hashbench.report --run results/coco_run2 --fpr 0.001 0.01 0.0001
python scripts/analyze_run.py --run results/coco_run2 --fpr 0.001
python scripts/analyze_closest_pairs.py --run results/coco_run2 --images data/coco/val2017
python scripts/write_run_summary.py --run results/coco_run2 \
    --reference results/coco_run1_config.json --out results/coco_run2/RESULTS_SUMMARY.md
{ python --version; uname -a; lscpu; free -h; pip freeze; } > results/coco_run2/environment.txt 2>&1
```

Run 1's `scripts/probe_invariances.py` is deliberately left out: its probes target the other hashes.

## Step 5: Check the results

Open `results/coco_run2/RESULTS_SUMMARY.md`. All of these must hold:

| Section | Expected |
|---|---|
| Row counts, for each of the 2 hashes | Genuine rows 88,000; Failed attacks 0; Attacked images 1,000; Settings 88; Impostor pairs 200,000 |
| Row counts, images hashed | 5,000 |
| Settings compared with `coco_run1_config.json` | `yes` on every row |
| Files | Everything listed in Step 6 exists |

If anything differs, report it to the user in the final message and do not call the run valid.

## Step 6: Package the results

```bash
tar -czf coco_run2_results.tar.gz -C results coco_run2 crosscheck_linux_dinohash
ls -la coco_run2_results.tar.gz
```

`results/coco_run2/` must contain:

- from the benchmark: `config.json`, `genuine.csv`, `impostor.csv`, `hashes.csv`, `run.log`
- from the report: `overall.csv`, `summary.csv`, `attack_curves.png`, `mean_distance_curves.png`,
  `distance_hist.png`, `roc.png`
- `RESULTS_SUMMARY.md` and `environment.txt`
- in `analysis/`: `family_match_rates.csv`, `per_param.csv`, `zero_fpr.csv`,
  `exact_invariances.csv`, `closest_impostors.csv`, `bit_balance.csv`,
  `image_difficulty_corr.csv`, `family_heatmap.png`, `black_frame_images.csv`,
  `image_mean_luma.csv`, `black_frame_impostors.csv`, `dark_image_impostors.csv`

## Step 7: Report to the user

Give the user:

1. the path of `coco_run2_results.tar.gz`, and tell them to send this file back to the Run 1 laptop;
2. the consistency-check table from Step 2;
3. the contents of `RESULTS_SUMMARY.md`, unchanged;
4. the result of each check in Step 5;
5. the machine's CPU model, how long the run took, and any restart or problem that happened.

Report the numbers only. Do not rewrite the tables, change their format or add conclusions about
which hash is better: the user will analyse them alongside Run 1.
