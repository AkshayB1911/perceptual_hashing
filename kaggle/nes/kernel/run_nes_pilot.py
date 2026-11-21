"""Kaggle GPU pilot for the NES attack: tune sigma / pairs and measure the real per-query time.

6 V200 images, evasion 8/255, 10,000 queries, on DinoHash trained / untrained, NeuralHash and
CertPHash, for a small grid of NES settings. Output: /kaggle/working/nes_pilot.zip + the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

GRID = [  # (sigma_frac, pairs, lr_frac)
    (0.5, 50, 0.2),
    (1.0, 50, 0.2),
    (1.0, 20, 0.2),
    (2.0, 20, 0.2),
]
HASHES = ["dinohash_untrained", "dinohash", "neuralhash", "certphash"]

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = WORK / "pilot"

archive = next(Path("/kaggle/input").rglob("hashbench_nes.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:
    src = next(p for p in Path("/kaggle/input").rglob("run_nes.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)
nh = HB / "models" / "neuralhash"
nh.mkdir(parents=True, exist_ok=True)
for f in ("model.onnx", "neuralhash_128x96_seed1.dat"):
    shutil.copy(next(Path("/kaggle/input").rglob(f)), nh / f)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

for sigma, pairs, lr in GRID:
    out = OUT / f"s{sigma:g}_n{pairs}_l{lr:g}"
    cmd = [sys.executable, "scripts/run_nes.py", "--list", "data/splits/V200.txt", "--limit", "6",
           "--out", str(out), "--device", "cuda", "--batch-size", "6", "--hashes", *HASHES,
           "--settings", "evasion:8", "--max-queries", "10000", "--sigma-frac", str(sigma),
           "--pairs", str(pairs), "--lr-frac", str(lr)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=HB, check=True)

shutil.make_archive(str(WORK / "nes_pilot"), "zip", OUT)
shutil.rmtree(HB)
