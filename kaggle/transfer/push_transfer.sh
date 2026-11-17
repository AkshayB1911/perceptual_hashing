#!/usr/bin/env bash
# Build + upload the transfer dataset and push the GPU kernel. Separate from kaggle/kernel/ and
# kaggle/table1/.
#   kaggle/transfer/push_transfer.sh <kaggle_username>
# Later:
#   kaggle kernels status <user>/hashbench-transfer-gen
#   kaggle kernels output <user>/hashbench-transfer-gen -p <dir> --file-pattern 'transfer_v200\.zip$|\.log$'
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"
STAGE="${STAGE:-$HERE/_stage}"; KSTAGE="$HERE/_kernel_stage"

bash "$HERE/build_transfer.sh" "$STAGE" "$USER_ID"
# run from inside the folder: the Windows kaggle CLI mangles absolute -p paths
if kaggle datasets status "$USER_ID/hashbench-transfer" >/dev/null 2>&1; then
  (cd "$STAGE" && kaggle datasets version -p . -m "transfer generation" -r tar)
else
  (cd "$STAGE" && kaggle datasets create -p . -r tar)   # private by default
fi
bash "$HERE/stage_kernel.sh" "$USER_ID"
echo "waiting for dataset to be ready..."; sleep 60
(cd "$KSTAGE" && kaggle kernels push -p .)
