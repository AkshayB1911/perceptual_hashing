"""Kaggle GPU job: generate transfer-attack images for DinoHash and CertPHash sources.

Reads the private dataset <user>/hashbench-transfer (built by kaggle/transfer/build_transfer.sh),
unpacks it into /kaggle/working and runs scripts/generate_transfer_images.py on the GPU for the
8 non-NeuralHash sources (evasion eps 2, 4, 8; collision eps 8, 16; V200).
The NeuralHash source runs locally (Apple's files never leave the laptop).
Output: /kaggle/working/transfer_v200.zip (images/, generation.csv, config.json) and the log.
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
HB = WORK / "hashbench"
OUT = HB / "results" / "transfer_v200"

archive = next(Path("/kaggle/input").rglob("hashbench_transfer.tar.gz"), None)
if archive is not None:
    with tarfile.open(archive) as t:
        t.extractall(WORK)
else:  # Kaggle sometimes unpacks archives itself
    src = next(p for p in Path("/kaggle/input").rglob("generate_transfer_images.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
os.environ["HASHBENCH_MODELS"] = str(HB / "models")

cmd = [sys.executable, "scripts/generate_transfer_images.py",
       "--list", "data/splits/V200.txt", "--out", str(OUT), "--device", "cuda", "--batch-size", "16",
       "--hashes", "dinohash", "dinohash_untrained", "certphash", "certphash_ep2", "certphash_ep4",
       "certphash_ep8", "certphash_base", "certphash_adv",
       "--eps-evasion", "2", "4", "8", "--eps-collision", "8", "16"]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, cwd=HB, check=True)
shutil.make_archive(str(WORK / "transfer_v200"), "zip", OUT)
shutil.rmtree(HB)  # keep only the zip + log in the output
