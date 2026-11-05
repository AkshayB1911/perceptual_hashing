"""Kaggle GPU job: full V200 adversarial sweep for DinoHash and CertPHash.

Reads the private dataset sushanthmb/hashbench-adv (built by kaggle/build_package.sh),
unpacks it into /kaggle/working, and runs scripts/run_adversarial.py with --device cuda.
Results land in /kaggle/working/hashbench/results/adv_v200/ (raw.csv, summary.*,
monotonicity.csv, config.json). The sweep is resumable: if a session dies, attach the
previous output's raw.csv as RESUME_FROM and push again.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / "adv_v200"

archive = next(Path("/kaggle/input").rglob("hashbench_adv.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("run_adversarial.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

# resume from an earlier partial run if one was attached as input
prev = [p for p in Path("/kaggle/input").rglob("adv_v200/raw.csv")]
if prev:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in prev[0].parent.iterdir():
        shutil.copy(f, OUT / f.name)
    print(f"resuming from {prev[0]}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

HASHES = ["dinohash", "dinohash_untrained", "certphash", "certphash_ep2", "certphash_ep4",
          "certphash_ep8", "certphash_base", "certphash_adv"]
cmd = [sys.executable, "scripts/run_adversarial.py",
       "--list", "data/splits/V200.txt", "--out", "results/adv_v200",
       "--hashes", *HASHES,
       "--modes", "evasion", "collision", "collision_closest",
       "--eps", "1", "2", "4", "8",
       "--eps-collision", "1", "2", "4", "8", "16", "32",
       "--device", "cuda", "--batch-size", "32"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=HB, check=True)

# keep the outputs small and easy to download
shutil.make_archive(str(WORK / "adv_v200_results"), "zip", OUT)
