#!/usr/bin/env bash
# Stage the PRIVATE Kaggle dataset with the NeuralHash weights (user decision 2026-11-17:
# NeuralHash may run on Kaggle, but only from its own private dataset, never public, never in git).
#
#   kaggle/nes/build_nh_private.sh <staging_dir> <kaggle_username>
#
# Contents: model.onnx (the AppleNeuralHash2ONNX conversion; weights bitwise equal to Apple's
# Espresso files, loaded by hashes/neuralhash.py without pyliblzfse) and the hyperplane matrix
# neuralhash_128x96_seed1.dat. Keep this dataset private.
set -euo pipefail
HB="$(cd "$(dirname "$0")/../../hashbench" && pwd)"
OUT="${1:?usage: build_nh_private.sh <staging_dir> <kaggle_username>}"
USER_ID="${2:?kaggle username}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
rm -rf "$OUT"; mkdir -p "$OUT"
cp "$HB/models/neuralhash/model.onnx" "$HB/models/neuralhash/neuralhash_128x96_seed1.dat" "$OUT/"
cat > "$OUT/dataset-metadata.json" <<EOT
{"title": "hashbench-nh-private", "id": "$USER_ID/hashbench-nh-private", "licenses": [{"name": "other"}]}
EOT
echo "staged $OUT: $(du -sh "$OUT" | cut -f1) (keep this dataset PRIVATE)"
