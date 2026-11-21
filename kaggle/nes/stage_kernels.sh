#!/usr/bin/env bash
# Write kaggle/nes/_kernel_stage/<job>/ (run_nes_kernel.py with JOB filled in + kernel-metadata.json)
# for the three GPU jobs (dino-t, dino-u, nh-cert) and the tuning pilot. Only nh-cert and pilot
# attach the NeuralHash dataset.
#   kaggle/nes/stage_kernels.sh <kaggle_username>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"

for JOB in dino-t dino-u nh-cert pilot; do
  K="$HERE/_kernel_stage/$JOB"; mkdir -p "$K"; rm -f "$K"/*
  CODE=run_nes_kernel.py; [ "$JOB" = pilot ] && CODE=run_nes_pilot.py
  sed "s/__JOB__/$JOB/" "$HERE/kernel/$CODE" > "$K/$CODE"
  SOURCES="\"$USER_ID/hashbench-nes\""
  { [ "$JOB" = nh-cert ] || [ "$JOB" = pilot ]; } && SOURCES="$SOURCES, \"$USER_ID/hashbench-nh-private\""
  cat > "$K/kernel-metadata.json" <<EOT
{
  "id": "$USER_ID/hashbench-nes-$JOB",
  "title": "hashbench-nes-$JOB",
  "code_file": "$CODE",
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

# Extra job: DinoHash trained at NeuralHash's 26-bit threshold (like-for-like comparison).
K="$HERE/_kernel_stage/dino-t-thr26"; mkdir -p "$K"; rm -f "$K"/*
cp "$HERE/kernel/run_nes_thr26_kernel.py" "$K/"
cat > "$K/kernel-metadata.json" <<EOT
{
  "id": "$USER_ID/hashbench-nes-dino-t-thr26",
  "title": "hashbench-nes-dino-t-thr26",
  "code_file": "run_nes_thr26_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$USER_ID/hashbench-nes"],
  "competition_sources": [],
  "kernel_sources": []
}
EOT
echo "staged $K"
