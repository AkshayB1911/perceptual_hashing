"""Kaggle GPU job: Sun et al. 2026 NES setting, 500-step variant (arXiv 2608.03101).

The literal run (run_nes_sun_kernel.py) counts every NES probe against the 1,500-query limit, which
allows only 23 steps, so the l2 caps 40 / 90 / 180 are never reached. Sun et al. report SSIM
0.625 / 0.330 / 0.171 for the three budgets, i.e. their perturbations fill the caps, so their attack
must run far more steps. This variant runs the stated step cap (500 steps; 65 queries per step, so
up to 32,501 queries per image) with everything else unchanged.

JOB (filled in by kaggle/nes_sun/stage_sun500.sh): "<hash>-<budget>" or "<hash>-all".
Images: the first 100 of V200 (Sun et al. use 100 per dataset). Same seeds as the literal run.
Output: /kaggle/working/nes_sun500_<JOB>.zip (nes_sun.csv, config.json, summary.*) and the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = "__JOB__"
HASH = {"nh": "neuralhash", "dino": "dinohash"}[JOB.split("-")[0]]
BUDGETS = ["40", "90", "180"] if JOB.endswith("-all") else [JOB.split("-")[1]]
T0 = time.time()
LIMIT_H = 11.3                       # Kaggle GPU sessions stop at 12 h

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / f"nes_sun500_v100_{JOB}"

archive = next(Path("/kaggle/input").rglob("hashbench_nes_sun.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("run_nes_sun.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

if HASH == "neuralhash":
    nh = HB / "models" / "neuralhash"
    nh.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "neuralhash_128x96_seed1.dat"):
        shutil.copy(next(Path("/kaggle/input").rglob(f)), nh / f)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

left = LIMIT_H - (time.time() - T0) / 3600
cmd = [sys.executable, "scripts/run_nes_sun.py", "--list", "data/splits/V200.txt", "--limit", "100",
       "--out", str(OUT), "--device", "cuda", "--batch-size", "8", "--chunk", "128",
       "--max-steps", "500", "--max-queries", str(1 + 500 * 65),
       "--time-limit", f"{left:.2f}", "--hashes", HASH, "--budgets", *BUDGETS]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=HB, check=True)

subprocess.run([sys.executable, "scripts/run_nes_sun.py", "--out", str(OUT), "--summary-only"], cwd=HB)
shutil.make_archive(str(WORK / f"nes_sun500_{JOB}"), "zip", OUT)
shutil.rmtree(HB)  # keep only the zip + log in the output (no model files)
