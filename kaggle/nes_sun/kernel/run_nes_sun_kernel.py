"""Kaggle GPU job: NES black-box evasion in the Sun et al. 2026 setting (arXiv 2608.03101) on V200.

JOB is filled in by kaggle/nes_sun/build_sun.sh:
    dino     DinoHash trained + untrained
    nh-cert  NeuralHash (private dataset <user>/hashbench-nh-private) + 6 CertPHash
Settings: 512x512 images, l2 caps 40 / 90 / 180, 64x64 grayscale NES, 32 antithetic directions,
NES scale 4/255, step 2/255, clean start, <= 500 steps, 1,500 queries, early stop at BER > 0.2.
Output: /kaggle/working/nes_sun_<JOB>.zip (nes_sun.csv, config.json, summary.*) and the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = "__JOB__"
RUNS = {
    "dino": [["dinohash", "dinohash_untrained"]],
    "nh-cert": [["neuralhash"], ["certphash", "certphash_ep2", "certphash_ep4", "certphash_ep8",
                                 "certphash_base", "certphash_adv"]],
}[JOB]
T0 = time.time()
LIMIT_H = 11.0                       # Kaggle GPU sessions stop at 12 h

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / f"nes_sun_v200_{JOB}"

archive = next(Path("/kaggle/input").rglob("hashbench_nes_sun.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("run_nes_sun.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

if JOB == "nh-cert":
    nh = HB / "models" / "neuralhash"
    nh.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "neuralhash_128x96_seed1.dat"):
        shutil.copy(next(Path("/kaggle/input").rglob(f)), nh / f)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

for hashes in RUNS:
    left = LIMIT_H - (time.time() - T0) / 3600
    if left <= 0.1:
        print(f"no time left for {hashes}", flush=True)
        break
    cmd = [sys.executable, "scripts/run_nes_sun.py", "--list", "data/splits/V200.txt", "--out", str(OUT),
           "--device", "cuda", "--batch-size", "8", "--chunk", "128",
           "--time-limit", f"{left:.2f}", "--hashes", *hashes]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HB, check=True)

subprocess.run([sys.executable, "scripts/run_nes_sun.py", "--out", str(OUT), "--summary-only"], cwd=HB)
shutil.make_archive(str(WORK / f"nes_sun_{JOB}"), "zip", OUT)
shutil.rmtree(HB)  # keep only the zip + log in the output (no model files)
