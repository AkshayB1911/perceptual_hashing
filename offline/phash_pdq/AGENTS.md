# AGENTS.md: instructions for a coding agent running this package

Your goal: run the pHash + PDQ NES black-box experiment to completion, then hand
`nes_classical_results.zip` back to the user. Read `README.md` for the background.

## Steps

1. Check the environment:
   - Run `python3 --version`. It must be 3.10 to 3.13.
   - Run `nproc` and `free -g`. Each worker needs about 0.5 GB of RAM.
   - Pick `WORKERS` as the smaller of (number of CPUs - 1) and (free RAM in GB × 2).
2. Run the job in the background so it survives a closed terminal:
   ```bash
   cd phash_pdq_black_box_run
   WORKERS=<n> nohup bash run.sh > run_sh.out 2>&1 &
   ```
3. Monitor progress:
   - Watch `run_sh.out` (setup, tests, smoke run) and then
     `hashbench/results/nes_v200_classical/run.log`. Each finished attack prints one line
     with `k/1200`.
   - Expect roughly 1–5 h depending on the CPU (see the README).
   - Do not poll faster than every few minutes.
4. When `run.sh` prints `ALL DONE`, check the results:
   - `nes_classical_results.zip` exists.
   - `nes.csv` has 1,200 data rows, plus the header.
   - `summary.md` has 3 sections (evasion 8, collision 16, evasion 4) with rows for both
     `phash` and `pdq`, and each row has `n` = 200, or fewer in the collision section,
     which excludes pairs that already matched.
5. Report back to the user:
   - the path of `nes_classical_results.zip`;
   - the contents of `summary.md`;
   - the total run time;
   - anything unusual you saw.

## If something fails

- **Install errors.** Fix the environment, not the code. For example, install `python3-venv`
  or use another Python 3.10–3.13. torch must be the CPU build, from
  `https://download.pytorch.org/whl/cpu` (already set in `run.sh`).
- **A test fails or the smoke run fails.** Stop and report the full error to the user. Do not
  edit the attack code to make it pass.
- **The run stops partway** (reboot, out of memory, killed). Run `bash run.sh` again with the
  same `WORKERS` or fewer. It resumes from `nes.csv`. If it ran out of memory, lower `WORKERS`.
- **"holds a run with different settings".** Someone changed a setting. Restore the original
  command; do not delete the results folder without asking the user.

## Rules

- Do **not** change any attack setting, the thresholds, the image list, its order, or the
  seed. The results must be comparable with GPU runs made elsewhere. Only `WORKERS` may change.
- Do **not** modify files under `hashbench/hashbench/` or `hashbench/scripts/`.
- Do **not** upload the images or results anywhere. Hand the zip to the user only.
- Do not delete `hashbench/results/nes_v200_classical/` once it has rows; it is the resume state.
