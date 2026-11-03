"""Prepare Apple's NeuralHash files for the AppleNeuralHash2ONNX converter (cross-check only).

hashbench reads Apple's Espresso files itself. The independent reference used by
tests/test_neuralhash.py is the community conversion: AppleNeuralHash2ONNX's TNN fork
(tools/onnx2tnn/onnx-coreml/coreml2onnx.py) + nnhash.py in onnxruntime. That converter
only reads fp32 ``blob_weights`` (the macOS build); the iOS build stores fp16
``blob_weights_f16``. This script writes an fp32 copy: the .net/.shape decompressed, fp16
blobs upcast exactly, keys renamed. No weight is otherwise changed.

Usage:
    python scripts/neuralhash_onnx_reference.py models/neuralhash <OUT_DIR>
    git clone https://github.com/AsuharietYgvar/TNN
    python TNN/tools/onnx2tnn/onnx-coreml/coreml2onnx.py <OUT_DIR>      # -> <OUT_DIR>/model.onnx
    copy <OUT_DIR>/model.onnx models/neuralhash/
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

from hashbench.hashes.neuralhash import _read_maybe_lzfse, find_espresso, read_espresso


def main(src: Path, out: Path) -> int:
    net_path = find_espresso(src)
    if net_path is None:
        print(f"no *.espresso.net in {src}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    net, blobs = read_espresso(net_path)
    shape_path = net_path.with_name(net_path.name[:-len(".net")] + ".shape")
    (out / "model.espresso.shape").write_bytes(_read_maybe_lzfse(shape_path))

    for layer in net["layers"]:
        if "blob_weights_f16" in layer:
            i = layer.pop("blob_weights_f16")
            blobs[i] = np.frombuffer(blobs[i], "<f2").astype("<f4").tobytes()
            layer["blob_weights"] = i
    net["storage"] = "model.espresso.weights"
    (out / "model.espresso.net").write_text(json.dumps(net))

    ids = sorted(blobs)
    header = struct.pack("<Q", len(ids)) + b"".join(struct.pack("<QQ", i, len(blobs[i])) for i in ids)
    (out / "model.espresso.weights").write_bytes(header + b"".join(blobs[i] for i in ids))
    print(f"wrote fp32 Espresso model to {out}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
