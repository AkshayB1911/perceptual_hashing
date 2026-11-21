#!/usr/bin/env bash
# pHash + PDQ NES black-box attack: setup, tests, smoke run, full run, results zip.
#   WORKERS=8 bash run.sh        (default: number of CPUs - 1)
# Resumable: rerun the same command after an interruption.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
NPROC="$(nproc 2>/dev/null || echo 2)"
WORKERS="${WORKERS:-$(( NPROC > 1 ? NPROC - 1 : 1 ))}"
PY="${PYTHON:-python3}"
OUT="results/nes_v200_classical"

echo "== 1/5 environment (workers: $WORKERS of $NPROC CPUs)"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -r requirements.txt
fi
VPY="$ROOT/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{ echo "date: $(date -Is)"; uname -a; "$VPY" --version
  (lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)') || true
  "$VPY" -m pip freeze; } > env.txt

cd hashbench
echo "== 2/5 unit tests"
"$VPY" -m pytest -q tests/test_blackbox.py tests/test_blackbox_classical.py

echo "== 3/5 smoke run (2 images, 500 queries, separate folder)"
rm -rf results/nes_smoke
"$VPY" scripts/run_nes_classical.py --list data/splits/V200.txt --limit 2 --max-queries 500 \
  --out results/nes_smoke --workers "$(( WORKERS < 4 ? WORKERS : 4 ))" > /dev/null
test "$(($(wc -l < results/nes_smoke/nes.csv) - 1))" -eq 12

echo "== 4/5 full run -> hashbench/$OUT (progress: tail -f hashbench/$OUT/run.log)"
mkdir -p "$OUT"
"$VPY" scripts/run_nes_classical.py --list data/splits/V200.txt --out "$OUT" \
  --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log"
ROWS=$(( $(wc -l < "$OUT/nes.csv") - 1 ))
[ "$ROWS" -eq 1200 ] || { echo "expected 1200 rows in nes.csv, found $ROWS"; exit 1; }

echo "== 5/5 packing results"
cd "$ROOT"
"$VPY" - <<'EOF'
import zipfile
from pathlib import Path
with zipfile.ZipFile("nes_classical_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("hashbench/results/nes_v200_classical").iterdir():
        z.write(p, f"nes_v200_classical/{p.name}")
    for p in Path("hashbench/results/nes_smoke").iterdir():
        z.write(p, f"smoke/{p.name}")
    z.write("env.txt", "env.txt")
EOF
cat hashbench/$OUT/summary.md
echo "ALL DONE: $ROOT/nes_classical_results.zip"
