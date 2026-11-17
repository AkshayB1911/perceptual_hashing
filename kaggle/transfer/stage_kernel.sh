#!/usr/bin/env bash
# Write kaggle/transfer/_kernel_stage (run_transfer.py + kernel-metadata.json).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
USER_ID="${1:?kaggle username}"; KSTAGE="$HERE/_kernel_stage"
rm -rf "$KSTAGE"; mkdir -p "$KSTAGE"
cp "$HERE/kernel/run_transfer.py" "$KSTAGE/"
cat > "$KSTAGE/kernel-metadata.json" <<EOT
{
  "id": "$USER_ID/hashbench-transfer-gen",
  "title": "hashbench-transfer-gen",
  "code_file": "run_transfer.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$USER_ID/hashbench-transfer"],
  "competition_sources": [],
  "kernel_sources": []
}
EOT
