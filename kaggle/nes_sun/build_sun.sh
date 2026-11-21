#!/usr/bin/env bash
# Stage the Kaggle datasets and kernels for the Sun et al. 2026 NES replication.
# Separate from kaggle/nes/ (our own NES setting), which is unchanged.
#   kaggle/nes_sun/build_sun.sh <kaggle_username>
# _stage/         <user>/hashbench-nes-sun : code, DinoHash + CertPHash models, V200 images, thresholds
# _stage_nh/      <user>/hashbench-nh-private : NeuralHash only, PRIVATE (kaggle/nes/build_nh_private.sh)
# _kernel_stage/  the two GPU kernels: dino (trained + untrained), nh-cert (NeuralHash + 6 CertPHash)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HB="$(cd "$HERE/../../hashbench" && pwd)"
USER_ID="${1:?kaggle username}"
OUT="$HERE/_stage"
rm -rf "$OUT"; mkdir -p "$OUT/hashbench/scripts" "$OUT/hashbench/models/dinohash" \
  "$OUT/hashbench/data/coco/val2017" "$OUT/hashbench/data/splits"
cd "$HB"
cp -r hashbench "$OUT/hashbench/"
cp scripts/run_adversarial.py scripts/run_nes_sun.py "$OUT/hashbench/scripts/"
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} +
cp models/dinohash/dinov2_vits14_reg_96bit_dynamic.onnx models/dinohash/dinov2_vits14_reg4_pretrain.pth \
   models/dinohash/dinov2_vits14_reg_PCA.npy models/dinohash/dinov2_vits14_reg_means.npy \
   "$OUT/hashbench/models/dinohash/"
cp -r models/certphash "$OUT/hashbench/models/"
if find "$OUT/hashbench/models" -ipath '*neuralhash*' | grep -q . || \
   find "$OUT" -iname '*espresso*' -o -iname '*seed1.dat' | grep -q .; then
  echo "NeuralHash model files found in the package; refusing" >&2; exit 1
fi
tr -d '\r' < data/splits/V200.txt | grep . > "$OUT/hashbench/data/splits/V200.txt"
while read -r f; do cp "data/coco/val2017/$f" "$OUT/hashbench/data/coco/val2017/"; done \
  < "$OUT/hashbench/data/splits/V200.txt"
for r in coco_run1 coco_run2; do
  mkdir -p "$OUT/hashbench/results/$r"; cp "results/$r/overall.csv" "$OUT/hashbench/results/$r/"
done
(cd "$OUT" && tar -czf hashbench_nes_sun.tar.gz hashbench && rm -rf hashbench)
cat > "$OUT/dataset-metadata.json" <<EOT
{"title": "hashbench-nes-sun", "id": "$USER_ID/hashbench-nes-sun", "licenses": [{"name": "other"}]}
EOT
echo "staged $OUT: $(du -sh "$OUT/hashbench_nes_sun.tar.gz" | cut -f1)"
bash "$HERE/../nes/build_nh_private.sh" "$HERE/_stage_nh" "$USER_ID"

for JOB in dino nh-cert; do
  K="$HERE/_kernel_stage/$JOB"; rm -rf "$K"; mkdir -p "$K"
  sed "s/__JOB__/$JOB/" "$HERE/kernel/run_nes_sun_kernel.py" > "$K/run_nes_sun_kernel.py"
  SOURCES="\"$USER_ID/hashbench-nes-sun\""
  if [ "$JOB" = nh-cert ]; then SOURCES="$SOURCES, \"$USER_ID/hashbench-nh-private\""; fi
  cat > "$K/kernel-metadata.json" <<EOT
{
  "id": "$USER_ID/hashbench-nes-sun-$JOB",
  "title": "hashbench-nes-sun-$JOB",
  "code_file": "run_nes_sun_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [$SOURCES],
  "competition_sources": [],
  "kernel_sources": []
}
EOT
done
echo "staged kernels in $HERE/_kernel_stage"
