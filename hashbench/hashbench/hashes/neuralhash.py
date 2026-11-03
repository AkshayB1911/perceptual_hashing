"""NeuralHash: Apple's deep perceptual hash (CSAM detection, announced Aug 2021).

Apple, "CSAM Detection: Technical Summary" (2021).
Attacks: Struppek et al., "Learning to Break Deep Perceptual Hashing: The Use Case
NeuralHash", FAccT 2022 (code in ../neuralHash). Extraction and ONNX conversion:
github.com/AsuharietYgvar/AppleNeuralHash2ONNX (reference script ``nnhash.py``).

Pipeline (``nnhash.py``):
    1. Convert to RGB, resize to 360x360 (PIL bicubic, aspect ratio NOT kept).
    2. Scale to [-1, 1].
    3. MobileNetV3-like CNN (~1.8M parameters, instance norm instead of batch norm)
       -> 128-d feature vector.
    4. Project on 96 fixed hyperplanes (the 96x128 "seed" matrix) -> 96 logits.
    5. bit = logit >= 0, most significant bit first.

Apple's files are NOT distributed (not here, not by the papers' authors). Extract them
yourself from the Vision framework of macOS 11.4+ / an iOS 14.7+ firmware and put them,
unchanged, in ``models/neuralhash/``:
    NeuralHashv3b[_fp16]-current.espresso.{net,shape,weights}   the network (Apple's
        Espresso format; ``_fp16`` = the iOS build, fp16 weights)
    neuralhash_128x96_seed1.dat     the hyperplane matrix (128-byte header + 96x128 float32)
Do not redistribute them.

We read Apple's Espresso files directly: no third-party converter is involved. ``.net``
is LZFSE-compressed JSON listing the 225 layers (needs ``pyliblzfse``); ``.weights`` is a
blob table (count, then (id, nbytes) pairs, then the blobs). Conv weights are
(out, in/groups, kh, kw), fully-connected weights (out, in), both row-major; fp16 blobs are
upcast to float32 exactly. The network is re-implemented as a block table, differentiable
for white-box attacks, and every weighted Espresso layer is checked against it (type,
channels, kernel, stride, groups, padding mode, fused ReLU) before its weights are copied.
Alternatively, ``model.onnx`` converted with AppleNeuralHash2ONNX is accepted; see
tests/test_neuralhash.py for the cross-check of both against onnxruntime.

Hashes computed on an Apple device can differ by a few bits (fp16 arithmetic on the
device; onnxruntime differs from both, see the README).
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from hashbench.hashes.base import PerceptualHash

INPUT_SIZE = 360
N_FEATURES = 128
N_BITS = 96
SEED_HEADER = 128  # bytes before the float32 matrix in neuralhash_128x96_seed1.dat

MODEL_DIR = Path(os.environ.get(
    "HASHBENCH_MODELS", Path(__file__).resolve().parents[2] / "models")) / "neuralhash"
ONNX_FILE = "model.onnx"
SEED_FILE = "neuralhash_128x96_seed1.dat"


# --- architecture ---------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    c_in: int
    c_exp: int      # expanded (depthwise) channels
    c_out: int
    kernel: int     # depthwise kernel
    stride: int
    se: int         # squeeze-excitation bottleneck channels, 0 = no SE
    act: str        # "relu" / "hswish"


# Bottleneck_1 .. Bottleneck_11 of Apple's graph (stem/MobilenetV3/...).
# Residual connection whenever stride == 1 and c_in == c_out.
BLOCKS = [
    Block(16, 16, 16, 3, 2, 4, "relu"),
    Block(16, 56, 24, 3, 2, 0, "relu"),
    Block(24, 64, 24, 3, 1, 0, "relu"),
    Block(24, 72, 32, 5, 2, 18, "hswish"),
    Block(32, 184, 32, 5, 1, 46, "hswish"),
    Block(32, 184, 32, 5, 1, 46, "hswish"),
    Block(32, 88, 40, 5, 1, 22, "hswish"),
    Block(40, 112, 40, 5, 1, 28, "hswish"),
    Block(40, 216, 72, 5, 2, 54, "hswish"),
    Block(72, 432, 72, 5, 1, 108, "hswish"),
    Block(72, 432, 72, 5, 1, 108, "hswish"),
]

IN_EPS = 1e-6  # overwritten by the epsilon stored in the model files on load


def _act(name: str) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.Hardswish()


class SameConv2d(nn.Conv2d):
    """Conv with TensorFlow-style SAME padding (extra row/column at the bottom/right),
    Espresso's ``pad_mode`` 1."""

    def __init__(self, c_in, c_out, kernel, stride=1, groups=1, bias=False):
        super().__init__(c_in, c_out, kernel, stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        pads = []
        for size, k, s in zip(x.shape[:-3:-1], self.kernel_size[::-1], self.stride[::-1]):  # (W, H)
            total = max((math.ceil(size / s) - 1) * s + k - size, 0)
            pads += [total // 2, total - total // 2]
        return F.conv2d(F.pad(x, pads), self.weight, self.bias, self.stride, 0, 1, self.groups)


def _conv_in(c_in, c_out, kernel=1, stride=1, groups=1) -> list[nn.Module]:
    return [SameConv2d(c_in, c_out, kernel, stride, groups),
            nn.InstanceNorm2d(c_out, eps=IN_EPS, affine=True)]


class _SqueezeExcite(nn.Module):
    def __init__(self, c: int, c_mid: int):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c_mid, 1)
        self.fc2 = nn.Conv2d(c_mid, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * F.hardsigmoid(self.fc2(F.relu(self.fc1(s))))


class _Bottleneck(nn.Module):
    """expand 1x1 -> act -> depthwise kxk -> [SE] -> act -> project 1x1 [+ residual].

    Submodules are registered in forward order: the weight loaders rely on it."""

    def __init__(self, b: Block):
        super().__init__()
        self.residual = b.stride == 1 and b.c_in == b.c_out
        self.expand = nn.Sequential(*_conv_in(b.c_in, b.c_exp), _act(b.act))
        self.depthwise = nn.Sequential(*_conv_in(b.c_exp, b.c_exp, b.kernel, b.stride, groups=b.c_exp))
        self.se = _SqueezeExcite(b.c_exp, b.se) if b.se else nn.Identity()
        self.project = nn.Sequential(_act(b.act), *_conv_in(b.c_exp, b.c_out))

    def forward(self, x):
        y = self.project(self.se(self.depthwise(self.expand(x))))
        return x + y if self.residual else y


class NeuralHashNet(nn.Module):
    """Differentiable NeuralHash: [0, 1] RGB images (B, 3, 360, 360) -> 96 hash logits."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(*_conv_in(3, 16, 3, 2), nn.Hardswish())
        self.blocks = nn.Sequential(*(_Bottleneck(b) for b in BLOCKS))
        self.last = nn.Sequential(*_conv_in(72, 432), nn.Hardswish())
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            SameConv2d(432, 1280, 1, bias=True), nn.Hardswish(),
            nn.Conv2d(1280, 500, 1), nn.ReLU(),        # Espresso inner_product layers
            nn.Conv2d(500, N_FEATURES, 1),
        )
        self.register_buffer("hash_matrix", torch.zeros(N_BITS, N_FEATURES))

    def features(self, x01: torch.Tensor) -> torch.Tensor:
        """The 128-d feature vector (Apple's ``leaf/logits`` output)."""
        x = x01 * 2.0 - 1.0
        return self.head(self.last(self.blocks(self.stem(x)))).flatten(1)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.features(x01) @ self.hash_matrix.T


def _weighted_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.InstanceNorm2d))]


# --- Apple's files --------------------------------------------------------------------

def load_seed(path: Path) -> np.ndarray:
    """The 96x128 hyperplane matrix from ``neuralhash_128x96_seed1.dat``."""
    raw = Path(path).read_bytes()
    expected = SEED_HEADER + N_BITS * N_FEATURES * 4
    if len(raw) != expected:
        raise ValueError(f"{path}: {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw[SEED_HEADER:], dtype="<f4").reshape(N_BITS, N_FEATURES).copy()


def _read_maybe_lzfse(path: Path) -> bytes:
    """Espresso ``.net``/``.shape`` files: 28-byte ``pbze`` header + LZFSE stream, or plain JSON."""
    raw = path.read_bytes()
    if raw[:4] != b"pbze":
        return raw
    try:
        import liblzfse
    except ImportError:
        raise ImportError(f"{path.name} is LZFSE-compressed; pip install pyliblzfse") from None
    data = liblzfse.decompress(raw[28:])
    if len(data) != struct.unpack_from(">I", raw, 16)[0]:
        raise ValueError(f"{path}: decompressed size does not match the pbze header")
    return data


def find_espresso(model_dir: Path) -> Path | None:
    """Path of the ``*.espresso.net`` file in ``model_dir`` (macOS or iOS build), if any."""
    nets = sorted(model_dir.glob("*.espresso.net"))
    if len(nets) > 1:
        raise RuntimeError(f"several Espresso models in {model_dir}: {[p.name for p in nets]}")
    return nets[0] if nets else None


def read_espresso(net_path: Path) -> tuple[dict, dict[int, bytes]]:
    """(network description, raw blobs by id) of an Espresso model."""
    net = json.loads(_read_maybe_lzfse(net_path))
    w = (net_path.parent / net["storage"]).read_bytes()
    n = struct.unpack_from("<Q", w, 0)[0]
    blobs, offset = {}, 8 + 16 * n
    for i in range(n):
        blob_id, size = struct.unpack_from("<QQ", w, 8 + 16 * i)
        blobs[blob_id] = w[offset:offset + size]
        offset += size
    if offset != len(w):
        raise ValueError(f"{net['storage']}: blob table covers {offset} of {len(w)} bytes")
    return net, blobs


def _blob(blobs: dict[int, bytes], layer: dict, key: str) -> np.ndarray:
    """A layer's blob as float32; weights may be stored as fp16 (``blob_weights_f16``)."""
    if key == "weights" and "blob_weights_f16" in layer:
        return np.frombuffer(blobs[layer["blob_weights_f16"]], "<f2").astype(np.float32)
    return np.frombuffer(blobs[layer[f"blob_{key}"]], "<f4").copy()


def load_espresso_weights(model: NeuralHashNet, net_path: Path) -> None:
    """Copy the weights of Apple's Espresso model into ``model`` (in place)."""
    net, blobs = read_espresso(net_path)
    img = net["transform_params"]["image"]
    if (not math.isclose(img["scale"], 2 / 255, rel_tol=1e-6) or img["is_network_bgr"]
            or any(img[f"bias_{c}"] != -1 for c in "rgb")):
        raise RuntimeError(f"unexpected input transform {img}; expected RGB * 2/255 - 1")

    espresso = [l for l in net["layers"] if l["type"] in ("convolution", "batchnorm", "inner_product")]
    layers = _weighted_layers(model)
    if len(espresso) != len(layers):
        raise RuntimeError(f"{net_path.name}: {len(espresso)} weighted layers, expected {len(layers)}")
    fused_relu = {id(m.fc1) for m in model.modules() if isinstance(m, _SqueezeExcite)}
    fused_relu.add(id(model.head[3]))  # the 1280 -> 500 fully-connected layer

    for l, m in zip(espresso, layers):
        where = f"Espresso layer {l['name']!r} ({l['type']}) vs {m}"
        if l["type"] == "batchnorm":
            if not isinstance(m, nn.InstanceNorm2d) or l["C"] != m.num_features or not l["training_instancenorm"]:
                raise RuntimeError(f"incompatible instance norm: {where}")
            gamma, beta, mean, var = _blob(blobs, l, "batchnorm_params").reshape(m.num_features, 4).T
            if mean.any() or (var != 1).any():  # unused by instance norm; Apple stores 0 / 1
                raise RuntimeError(f"unexpected running statistics in {where}")
            m.weight.data, m.bias.data = torch.from_numpy(gamma.copy()), torch.from_numpy(beta.copy())
            m.eps = float(l["training_eps"])
            continue

        if not isinstance(m, nn.Conv2d):
            raise RuntimeError(f"layer order mismatch: {where}")
        if l["type"] == "convolution":
            ok = (l["C"] == m.out_channels and l["K"] == m.in_channels and l["n_groups"] == m.groups
                  and (l["Ny"], l["Nx"]) == m.kernel_size
                  and (l.get("stride_y", 1), l.get("stride_x", 1)) == m.stride
                  and l["pad_mode"] == (1 if isinstance(m, SameConv2d) else 2)
                  and not any(l[f"pad_{s}"] for s in "tblr")
                  and l["fused_relu"] == (id(m) in fused_relu) and not l["fused_tanh"])
        else:  # inner_product, implemented as a 1x1 conv
            ok = (l["nC"] == m.out_channels and l["nB"] == m.in_channels and m.kernel_size == (1, 1)
                  and l["has_relu"] == (id(m) in fused_relu) and not l["has_tanh"] and not l["has_prelu"])
        if not ok or bool(l["has_biases"]) != (m.bias is not None):
            raise RuntimeError(f"incompatible layer: {where}")
        m.weight.data = torch.from_numpy(_blob(blobs, l, "weights").reshape(m.weight.shape))
        if m.bias is not None:
            m.bias.data = torch.from_numpy(_blob(blobs, l, "biases"))


def load_onnx_weights(model: NeuralHashNet, onnx_path: Path) -> None:
    """Copy the weights of an AppleNeuralHash2ONNX ``model.onnx`` into ``model`` (in place)."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    nodes = [n for n in graph.node if n.op_type in ("Conv", "InstanceNormalization")]
    layers = _weighted_layers(model)
    if len(nodes) != len(layers):
        ops = sorted({n.op_type for n in graph.node})
        raise RuntimeError(f"{onnx_path}: {len(nodes)} Conv/InstanceNormalization nodes, "
                           f"expected {len(layers)}; ops in graph: {ops}")

    for node, layer in zip(nodes, layers):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        params = [inits[name] for name in node.input[1:] if name]
        where = f"node {node.name or node.output[0]!r} ({node.op_type}) vs {layer}"
        if isinstance(layer, nn.Conv2d):
            if node.op_type != "Conv":
                raise RuntimeError(f"layer order mismatch: {where}")
            if (tuple(attrs.get("strides", (1, 1))) != layer.stride
                    or attrs.get("group", 1) != layer.groups
                    or tuple(params[0].shape) != tuple(layer.weight.shape)
                    or (len(params) > 1) != (layer.bias is not None)):
                raise RuntimeError(f"incompatible conv: {where}, onnx weight {params[0].shape}, attrs {attrs}")
            layer.weight.data = torch.from_numpy(params[0].copy())
            if layer.bias is not None:
                layer.bias.data = torch.from_numpy(params[1].reshape(-1).copy())
        else:
            if node.op_type != "InstanceNormalization" or params[0].size != layer.num_features:
                raise RuntimeError(f"incompatible instance norm: {where}")
            layer.weight.data = torch.from_numpy(params[0].reshape(-1).copy())
            layer.bias.data = torch.from_numpy(params[1].reshape(-1).copy())
            layer.eps = float(attrs.get("epsilon", 1e-5))


def load_neuralhash_model(model_dir: Path = MODEL_DIR, source: str = "auto") -> NeuralHashNet:
    """NeuralHash from ``model_dir``. ``source``: "espresso" (Apple's files), "onnx"
    (``model.onnx``), or "auto" = Espresso if present, else ONNX."""
    model_dir = Path(model_dir)
    net_path = find_espresso(model_dir) if source in ("auto", "espresso") else None
    onnx_path = model_dir / ONNX_FILE
    seed_path = model_dir / SEED_FILE
    if not seed_path.exists() or (net_path is None and (source == "espresso" or not onnx_path.exists())):
        raise FileNotFoundError(
            f"NeuralHash model files missing from {model_dir}. Apple's files are not distributed: "
            "extract NeuralHashv3b[_fp16]-current.espresso.{net,shape,weights} and "
            f"{SEED_FILE} from macOS 11.4+ / iOS 14.7+ (Vision framework; see "
            "github.com/AsuharietYgvar/AppleNeuralHash2ONNX) and copy them there unchanged")
    model = NeuralHashNet()
    if net_path is not None:
        load_espresso_weights(model, net_path)
    else:
        load_onnx_weights(model, onnx_path)
    model.hash_matrix.copy_(torch.from_numpy(load_seed(seed_path)))
    return model.eval()


# --- PerceptualHash interface ---------------------------------------------------------

def preprocess(image: Image.Image) -> torch.Tensor:
    """nnhash.py preprocessing up to the [-1, 1] scaling (done inside the model):
    PIL resize to 360x360 (bicubic, Pillow's default) and scale to [0, 1]."""
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class NeuralHash(PerceptualHash):
    name = "neuralhash"
    n_bits = N_BITS

    def __init__(self, model_dir: Path = MODEL_DIR, source: str = "auto", batch_size: int = 32):
        self.batch_size = batch_size
        self.model = load_neuralhash_model(model_dir, source)

    @torch.no_grad()
    def logits(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack([preprocess(im) for im in images[i:i + self.batch_size]])
            out.append(self.model(batch).numpy())
        return np.concatenate(out)

    def hash_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self.logits(images) >= 0

    def hash(self, image: Image.Image) -> np.ndarray:
        return self.hash_batch([image])[0]
