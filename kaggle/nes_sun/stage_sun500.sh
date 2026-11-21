#!/usr/bin/env bash
# Stage the kernels of the 500-step Sun et al. variant. Uses the datasets built by build_sun.sh
# (<user>/hashbench-nes-sun, and <user>/hashbench-nh-private for NeuralHash only).
#   kaggle/nes_sun/stage_sun500.sh <kaggle_username>
# Jobs: nh-all (NeuralHash, all three budgets) and dino-40 / dino-90 / dino-180 (DinoHash trained
# is about 2.5x slower per query, so one budget per job to stay under Kaggle's 12 h limit).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"
for JOB in nh-all dino-40 dino-90 dino-180; do
  K="$HERE/_kernel_stage/sun500-$JOB"; rm -rf "$K"; mkdir -p "$K"
  sed "s/__JOB__/$JOB/" "$HERE/kernel/run_nes_sun500_kernel.py" > "$K/run_nes_sun500_kernel.py"
  SOURCES="\"$USER_ID/hashbench-nes-sun\""
  case "$JOB" in nh-*) SOURCES="$SOURCES, \"$USER_ID/hashbench-nh-private\"";; esac
  cat > "$K/kernel-metadata.json" <<EOT
{
  "id": "$USER_ID/hashbench-nes-sun500-$JOB",
  "title": "hashbench-nes-sun500-$JOB",
  "code_file": "run_nes_sun500_kernel.py",
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
echo "staged in $HERE/_kernel_stage/sun500-*"
