#!/usr/bin/env bash
# Stage the private Kaggle dataset for the transfer-attack image generation.
# Separate from kaggle/build_package.sh (V200 sweep) and kaggle/table1/, which are unchanged.
#
#   kaggle/transfer/build_transfer.sh <staging_dir> <kaggle_username>
#
# Contents: the hashbench package, scripts/run_adversarial.py (helpers) +
# scripts/generate_transfer_images.py, DinoHash ViT-S and CertPHash model files, the 200 V200
# images, the V200 list and the two benign overall.csv threshold files.
# Apple's NeuralHash files are NOT included; the NeuralHash source runs locally.
set -euo pipefail
HB="$(cd "$(dirname "$0")/../../hashbench" && pwd)"
OUT="${1:?usage: build_transfer.sh <staging_dir> <kaggle_username>}"
USER_ID="${2:?kaggle username}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"   # absolute, the script cds below
rm -rf "$OUT"; mkdir -p "$OUT/hashbench/scripts" "$OUT/hashbench/models/dinohash" \
  "$OUT/hashbench/data/coco/val2017" "$OUT/hashbench/data/splits"
cd "$HB"

# code
cp -r hashbench "$OUT/hashbench/"
cp scripts/run_adversarial.py scripts/generate_transfer_images.py "$OUT/hashbench/scripts/"
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} +

# models: DinoHash ViT-S + CertPHash, no NeuralHash
cp models/dinohash/dinov2_vits14_reg_96bit_dynamic.onnx models/dinohash/dinov2_vits14_reg4_pretrain.pth \
   models/dinohash/dinov2_vits14_reg_PCA.npy models/dinohash/dinov2_vits14_reg_means.npy \
   "$OUT/hashbench/models/dinohash/"
cp -r models/certphash "$OUT/hashbench/models/"
if find "$OUT/hashbench/models" -ipath '*neuralhash*' | grep -q . || \
   find "$OUT" -iname '*espresso*' -o -iname '*seed1.dat' | grep -q .; then
  echo "NeuralHash model files found in the package; refusing" >&2; exit 1
fi

# images, list, thresholds
tr -d '\r' < data/splits/V200.txt | grep . > "$OUT/hashbench/data/splits/V200.txt"
while read -r f; do cp "data/coco/val2017/$f" "$OUT/hashbench/data/coco/val2017/"; done \
  < "$OUT/hashbench/data/splits/V200.txt"
for r in coco_run1 coco_run2; do
  mkdir -p "$OUT/hashbench/results/$r"; cp "results/$r/overall.csv" "$OUT/hashbench/results/$r/"
done

(cd "$OUT" && tar -czf hashbench_transfer.tar.gz hashbench && rm -rf hashbench)
cat > "$OUT/dataset-metadata.json" <<EOT
{"title": "hashbench-transfer", "id": "$USER_ID/hashbench-transfer", "licenses": [{"name": "other"}]}
EOT
echo "staged $OUT: $(du -sh "$OUT/hashbench_transfer.tar.gz" | cut -f1)"
