"""Kaggle GPU job: NES black-box attack (attack #2 of future_attacks.md) on V200.

JOB is filled in by kaggle/nes/stage_kernels.sh:
    dino-t   DinoHash trained
    dino-u   DinoHash untrained
    nh-cert  NeuralHash (from the private dataset <user>/hashbench-nh-private) + 6 CertPHash
Each job: 200 V200 images, settings evasion 8/255, collision 16/255, evasion 4/255 (in that
order), 10,000 queries per image. A time guard stops starting new batches before Kaggle's
12 h limit; the run is resumable by pushing the job again with the partial nes.csv.
Output: /kaggle/working/nes_<JOB>.zip (nes.csv, config.json, summary.*) and the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = "__JOB__"
NES_ARGS = ["--max-queries", "10000", "--pairs", "20", "--sigma-frac", "1.0", "--lr-frac", "0.2",
            "--dim", "32", "--settings", "evasion:8", "collision:16", "evasion:4"]
RUNS = {
    "dino-t": [(["dinohash"], 16)],
    "dino-u": [(["dinohash_untrained"], 16)],
    "nh-cert": [(["neuralhash"], 16),
                (["certphash", "certphash_ep2", "certphash_ep4", "certphash_ep8",
                  "certphash_base", "certphash_adv"], 64)],
}[JOB]
T0 = time.time()
LIMIT_H = 11.0                       # Kaggle GPU sessions stop at 12 h

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / f"nes_v200_{JOB}"

archive = next(Path("/kaggle/input").rglob("hashbench_nes.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("run_nes.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

if JOB == "nh-cert":
    nh = HB / "models" / "neuralhash"
    nh.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "neuralhash_128x96_seed1.dat"):
        shutil.copy(next(Path("/kaggle/input").rglob(f)), nh / f)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

for hashes, batch in RUNS:
    left = LIMIT_H - (time.time() - T0) / 3600
    if left <= 0.1:
        print(f"no time left for {hashes}", flush=True)
        break
    cmd = [sys.executable, "scripts/run_nes.py", "--list", "data/splits/V200.txt", "--out", str(OUT),
           "--device", "cuda", "--batch-size", str(batch), "--time-limit", f"{left:.2f}",
           "--hashes", *hashes, *NES_ARGS]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HB, check=True)

subprocess.run([sys.executable, "scripts/run_nes.py", "--out", str(OUT), "--summary-only"], cwd=HB)
shutil.make_archive(str(WORK / f"nes_{JOB}"), "zip", OUT)
shutil.rmtree(HB)  # keep only the zip + log in the output (no model files)
