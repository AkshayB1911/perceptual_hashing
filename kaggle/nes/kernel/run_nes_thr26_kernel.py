"""Kaggle GPU job: NES on trained DinoHash at NeuralHash's match threshold (26 of 96 bits).

Like-for-like check for the black-box comparison. The main runs use each hash's own FPR 1e-3
threshold (DinoHash trained 17 bits, NeuralHash 26 bits), so DinoHash needs fewer flipped bits
to be evaded. This job repeats the DinoHash-trained NES run with the threshold set to
NeuralHash's value, so both hashes face the same 26-bit bar. Everything else (NES settings,
images, seeds, collision targets) is identical to run_nes_kernel.py.

The threshold is set by writing a thresholds file whose "dinohash" row carries NeuralHash's
FPR 1e-3 value (read from the dataset's coco_run1/overall.csv), passed with --thresholds-from.
Settings order: evasion 8, evasion 4, collision 16 (the evasion numbers matter most).
Output: /kaggle/working/nes_dino-t-thr26.zip (nes.csv, config.json, summary.*) and the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

NES_ARGS = ["--max-queries", "10000", "--pairs", "20", "--sigma-frac", "1.0", "--lr-frac", "0.2",
            "--dim", "32", "--settings", "evasion:8", "evasion:4", "collision:16"]
T0 = time.time()
LIMIT_H = 11.0

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / "nes_v200_dino-t-thr26"
THR = HB / "results" / "thr_neuralhash_for_dinohash"

archive = next(Path("/kaggle/input").rglob("hashbench_nes.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:
    src = next(p for p in Path("/kaggle/input").rglob("run_nes.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

col = "threshold_at_fpr_0.001"
run1 = pd.read_csv(HB / "results" / "coco_run1" / "overall.csv")
nh = float(run1.loc[run1["algorithm"] == "neuralhash", col].iloc[0])
THR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"algorithm": ["dinohash"], col: [nh]}).to_csv(THR / "overall.csv", index=False)
print(f"dinohash threshold set to NeuralHash's {nh:.6f} = {int(nh * 96 + 1e-6)} of 96 bits", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

left = LIMIT_H - (time.time() - T0) / 3600
cmd = [sys.executable, "scripts/run_nes.py", "--list", "data/splits/V200.txt", "--out", str(OUT),
       "--device", "cuda", "--batch-size", "16", "--time-limit", f"{left:.2f}",
       "--thresholds-from", str(THR), "--hashes", "dinohash", *NES_ARGS]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=HB, check=True)

subprocess.run([sys.executable, "scripts/run_nes.py", "--out", str(OUT), "--summary-only"], cwd=HB)
shutil.make_archive(str(WORK / "nes_dino-t-thr26"), "zip", OUT)
shutil.rmtree(HB)
