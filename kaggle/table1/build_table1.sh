#!/usr/bin/env bash
# Stage the private Kaggle dataset for the DinoHash Table 1 reproduction.
# Separate from kaggle/build_package.sh (the V200 sweep), which is left unchanged.
#
#   kaggle/table1/build_table1.sh <staging_dir> <kaggle_username> [n_images=50]
#
# Contents: the hashbench package, scripts/reproduce_dinohash_table1.py + scripts/third_party
# (the authors' APGD), DinoHash ViT-S model files, the first n images of V200 and the list.
# Apple's NeuralHash files are NOT included; the NeuralHash rows run locally.
set -euo pipefail
HB="$(cd "$(dirname "$0")/../../hashbench" && pwd)"
OUT="${1:?usage: build_table1.sh <staging_dir> <kaggle_username> [n_images]}"
USER_ID="${2:?kaggle username}"
N="${3:-50}"
rm -rf "$OUT"; mkdir -p "$OUT/hashbench/scripts" "$OUT/hashbench/models/dinohash" \
  "$OUT/hashbench/data/coco/val2017" "$OUT/hashbench/data/splits"
cd "$HB"

# code
cp -r hashbench "$OUT/hashbench/"
cp scripts/reproduce_dinohash_table1.py "$OUT/hashbench/scripts/"
cp -r scripts/third_party "$OUT/hashbench/scripts/"
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} +

# DinoHash ViT-S models only
cp models/dinohash/dinov2_vits14_reg_96bit_dynamic.onnx models/dinohash/dinov2_vits14_reg4_pretrain.pth \
   models/dinohash/dinov2_vits14_reg_PCA.npy models/dinohash/dinov2_vits14_reg_means.npy \
   "$OUT/hashbench/models/dinohash/"
if find "$OUT/hashbench/models" -ipath '*neuralhash*' | grep -q . || \
   find "$OUT" -iname '*espresso*' -o -iname '*seed1.dat' -o -iname 'model.onnx' | grep -q .; then
  echo "NeuralHash model files found in the package; refusing" >&2; exit 1
fi

# first N images of V200, and a list with just those
tr -d '\r' < data/splits/V200.txt | grep . | head -n "$N" > "$OUT/hashbench/data/splits/V200_first$N.txt"
while read -r f; do cp "data/coco/val2017/$f" "$OUT/hashbench/data/coco/val2017/"; done \
  < "$OUT/hashbench/data/splits/V200_first$N.txt"

(cd "$OUT" && tar -czf hashbench_table1.tar.gz hashbench && rm -rf hashbench)
cat > "$OUT/dataset-metadata.json" <<EOF
{"title": "hashbench-dinohash-table1", "id": "$USER_ID/hashbench-dinohash-table1", "licenses": [{"name": "other"}]}
EOF
echo "staged $OUT: $(du -sh "$OUT/hashbench_table1.tar.gz" | cut -f1), $N images"
