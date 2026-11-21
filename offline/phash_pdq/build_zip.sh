#!/usr/bin/env bash
# Build phash_pdq_black_box_run.zip (the offline pHash + PDQ NES job) at the repo root.
#   bash offline/phash_pdq/build_zip.sh
# Contains code, the 200 V200 images and the threshold files only: no model weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
HB="$REPO/hashbench"
STAGE="$HERE/_stage"; NAME="phash_pdq_black_box_run"; PKG="$STAGE/$NAME"
rm -rf "$STAGE"; mkdir -p "$PKG/hashbench/scripts" "$PKG/hashbench/tests" \
  "$PKG/hashbench/data/coco/val2017" "$PKG/hashbench/data/splits"

cp "$HERE/README.md" "$HERE/AGENTS.md" "$HERE/run.sh" "$HERE/requirements.txt" "$PKG/"
sed -i 's/\r$//' "$PKG/run.sh"
cd "$HB"
cp pyproject.toml "$PKG/hashbench/"
cp -r hashbench "$PKG/hashbench/"
cp scripts/run_adversarial.py scripts/run_nes.py scripts/run_nes_classical.py "$PKG/hashbench/scripts/"
cp tests/test_blackbox.py tests/test_blackbox_classical.py "$PKG/hashbench/tests/"
find "$PKG" -name '__pycache__' -type d -prune -exec rm -rf {} +
tr -d '\r' < data/splits/V200.txt | grep . > "$PKG/hashbench/data/splits/V200.txt"
while read -r f; do cp "data/coco/val2017/$f" "$PKG/hashbench/data/coco/val2017/"; done \
  < "$PKG/hashbench/data/splits/V200.txt"
for r in coco_run1 coco_run2; do
  mkdir -p "$PKG/hashbench/results/$r"; cp "results/$r/overall.csv" "$PKG/hashbench/results/$r/"
done
if find "$PKG" \( -iname '*.onnx' -o -iname '*.pth' -o -iname '*.dat' -o -iname '*espresso*' \) | grep -q .; then
  echo "model files found in the package; refusing" >&2; exit 1
fi

rm -f "$REPO/$NAME.zip"
"$HB/.venv/Scripts/python" - "$STAGE" "$NAME" "$REPO/$NAME.zip" <<'EOF'
import sys, zipfile
from pathlib import Path
stage, name, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted((stage / name).rglob("*")):
        if p.is_file():
            info = zipfile.ZipInfo.from_file(p, p.relative_to(stage).as_posix())
            if p.suffix == ".sh":
                info.external_attr = 0o100755 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
EOF
rm -rf "$STAGE"
echo "built $REPO/$NAME.zip ($(du -h "$REPO/$NAME.zip" | cut -f1))"
