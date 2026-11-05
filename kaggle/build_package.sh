#!/usr/bin/env bash
# Stage the private Kaggle dataset for the adversarial sweep (Phase C).
#
#   kaggle/build_package.sh <staging_dir>
#
# Contents: the hashbench code, DinoHash ViT-S and CertPHash model files, the 200 V200
# images, the V200 list and the two benign overall.csv threshold files.
# Apple's NeuralHash files are deliberately NOT included (not redistributable); NeuralHash
# runs locally instead.
set -euo pipefail
HB="$(cd "$(dirname "$0")/../hashbench" && pwd)"
OUT="${1:?usage: build_package.sh <staging_dir>}"
rm -rf "$OUT"; mkdir -p "$OUT/hashbench"
cd "$HB"

# code
rsync -a --exclude '__pycache__' hashbench "$OUT/hashbench/"
mkdir -p "$OUT/hashbench/scripts"
cp scripts/run_adversarial.py "$OUT/hashbench/scripts/"
cp pyproject.toml requirements.txt "$OUT/hashbench/"

# models (no neuralhash)
mkdir -p "$OUT/hashbench/models/dinohash"
cp models/dinohash/dinov2_vits14_reg_96bit_dynamic.onnx models/dinohash/dinov2_vits14_reg4_pretrain.pth \
   models/dinohash/dinov2_vits14_reg_PCA.npy models/dinohash/dinov2_vits14_reg_means.npy \
   "$OUT/hashbench/models/dinohash/"
rsync -a models/certphash "$OUT/hashbench/models/"
if find "$OUT" -ipath '*neuralhash*' -not -path '*/hashbench/hashbench/*' | grep -q .; then
  echo "NeuralHash model files found in the package; refusing" >&2; exit 1
fi

# images, list, thresholds
mkdir -p "$OUT/hashbench/data/coco/val2017" "$OUT/hashbench/data/splits"
cp data/splits/V200.txt "$OUT/hashbench/data/splits/"
tr -d '\r' < data/splits/V200.txt | while read -r f; do
  [ -n "$f" ] && cp "data/coco/val2017/$f" "$OUT/hashbench/data/coco/val2017/"
done
for r in coco_run1 coco_run2; do
  mkdir -p "$OUT/hashbench/results/$r"
  cp "results/$r/overall.csv" "$OUT/hashbench/results/$r/"
done

# one archive, so Kaggle keeps the folder layout
(cd "$OUT" && COPYFILE_DISABLE=1 tar -czf hashbench_adv.tar.gz hashbench && rm -rf hashbench)
cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "hashbench-adv", "id": "sushanthmb/hashbench-adv", "licenses": [{"name": "other"}]}
EOF
echo "staged: $(du -sh "$OUT/hashbench_adv.tar.gz" | cut -f1)  images: $(tr -d '\r' < data/splits/V200.txt | grep -c .)"
