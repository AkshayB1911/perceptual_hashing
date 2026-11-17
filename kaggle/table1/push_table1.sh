#!/usr/bin/env bash
# Build + upload the Table 1 dataset and push the GPU kernel. Separate from kaggle/kernel/.
#   kaggle/table1/push_table1.sh <kaggle_username> [n_images=50]
# Needs ~/.kaggle/kaggle.json and the kaggle CLI. Later:
#   kaggle kernels status <user>/dinohash-table1-repro
#   kaggle kernels output <user>/dinohash-table1-repro -p <dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"; N="${2:-50}"
STAGE="${STAGE:-$HERE/_stage}"; KSTAGE="$HERE/_kernel_stage"

bash "$HERE/build_table1.sh" "$STAGE" "$USER_ID" "$N"
# run from inside the folder: the Windows kaggle CLI mangles absolute -p paths
if kaggle datasets status "$USER_ID/hashbench-dinohash-table1" >/dev/null 2>&1; then
  (cd "$STAGE" && kaggle datasets version -p . -m "table1 repro, first $N V200 images" -r tar)
else
  (cd "$STAGE" && kaggle datasets create -p . -r tar)   # private by default
fi

rm -rf "$KSTAGE"; mkdir -p "$KSTAGE"
cp "$HERE/kernel/run_table1.py" "$KSTAGE/"
cat > "$KSTAGE/kernel-metadata.json" <<EOF
{
  "id": "$USER_ID/dinohash-table1-repro",
  "title": "dinohash-table1-repro",
  "code_file": "run_table1.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$USER_ID/hashbench-dinohash-table1"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
echo "waiting for dataset to be ready..."; sleep 60
(cd "$KSTAGE" && kaggle kernels push -p .)
