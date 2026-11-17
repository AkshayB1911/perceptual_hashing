"""Kaggle GPU job: DinoHash Table 1 reproduction with the authors' own APGD.

Reads the private dataset <user>/hashbench-dinohash-table1 (built by
kaggle/table1/build_table1.sh), unpacks it into /kaggle/working and runs
scripts/reproduce_dinohash_table1.py on the GPU for the two DinoHash models.
The NeuralHash rows are run locally (Apple's files never leave the laptop).
Output: /kaggle/working/table1_results.zip (raw.csv, summary.csv, summary.md, config.json).
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / "dinohash_table1_repro_kaggle"

archive = next(Path("/kaggle/input").rglob("hashbench_table1.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("reproduce_dinohash_table1.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

lst = next((HB / "data" / "splits").glob("V200_first*.txt"))
cmd = [sys.executable, "scripts/reproduce_dinohash_table1.py",
       "--list", str(lst), "--out", str(OUT),
       "--models", "dinohash", "dinohash_untrained_paper",
       "--eps", "4", "8", "--steps", "100", "--batch", "25", "--device", "cuda"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=HB, check=True)
shutil.make_archive(str(WORK / "table1_results"), "zip", OUT)
