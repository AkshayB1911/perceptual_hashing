#!/usr/bin/env bash
# Build + upload both datasets and push the three GPU kernels. Separate from the older packages.
#   kaggle/nes/push_nes.sh <kaggle_username>
# On Windows run the kaggle commands from PowerShell inside each folder (absolute -p paths break).
# Later:
#   kaggle kernels status <user>/hashbench-nes-<job>
#   kaggle kernels output <user>/hashbench-nes-<job> -p <dir> --file-pattern 'nes_.*\.zip$|\.log$'
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"
bash "$HERE/build_nes.sh" "$HERE/_stage" "$USER_ID"
bash "$HERE/build_nh_private.sh" "$HERE/_stage_nh" "$USER_ID"
for D in _stage:hashbench-nes:tar _stage_nh:hashbench-nh-private:skip; do
  IFS=: read -r DIR NAME MODE <<< "$D"
  if kaggle datasets status "$USER_ID/$NAME" >/dev/null 2>&1; then
    (cd "$HERE/$DIR" && kaggle datasets version -p . -m "nes" -r "$MODE")
  else
    (cd "$HERE/$DIR" && kaggle datasets create -p . -r "$MODE")   # private by default
  fi
done
bash "$HERE/stage_kernels.sh" "$USER_ID"
echo "waiting for datasets to be ready..."; sleep 90
for JOB in dino-t dino-u nh-cert; do (cd "$HERE/_kernel_stage/$JOB" && kaggle kernels push -p .); done
