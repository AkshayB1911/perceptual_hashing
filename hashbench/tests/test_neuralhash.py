"""NeuralHash: our PyTorch re-implementation vs onnxruntime.

Two levels:
    * synthetic: a randomly initialised network exported to ONNX checks the ONNX loader
      and the forward pass end to end (always runs);
    * real: Apple's Espresso files, read by our own loader, vs AppleNeuralHash2ONNX's
      ``model.onnx`` conversion run by its ``nnhash.py`` pipeline in onnxruntime (skipped
      unless models/neuralhash/ holds the files).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from hashbench.hashes import get_hasher, to_hex  # noqa: E402
from hashbench.hashes.neuralhash import (  # noqa: E402
    MODEL_DIR, ONNX_FILE, SEED_FILE, SEED_HEADER, NeuralHash, NeuralHashNet, find_espresso,
    load_neuralhash_model, load_onnx_weights, load_seed, preprocess,
)

ROOT = Path(__file__).parent
IMAGES = sorted((ROOT.parent / "data" / "samples").glob("*")) + sorted((ROOT / "data" / "meta_pdq").glob("*.jpg"))


def nnhash_reference(session, seed: np.ndarray, image: Image.Image) -> tuple[np.ndarray, str]:
    """AppleNeuralHash2ONNX nnhash.py, verbatim apart from returning instead of printing."""
    image = image.convert("RGB")
    image = image.resize([360, 360])
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
    outs = session.run(None, {session.get_inputs()[0].name: arr})
    hash_output = seed.dot(outs[0].flatten())
    hash_bits = "".join(["1" if it >= 0 else "0" for it in hash_output])
    return hash_output, "{:0{}x}".format(int(hash_bits, 2), len(hash_bits) // 4)


class _OnnxFeatures(torch.nn.Module):
    """Same interface as Apple's converted model: [-1, 1] input -> (B, 128, 1, 1)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net.features((x + 1) / 2)[:, :, None, None]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Random-weight network exported to ONNX + a random seed file."""
    torch.manual_seed(0)
    net = NeuralHashNet().eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.InstanceNorm2d):
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.1)
    d = tmp_path_factory.mktemp("neuralhash")
    torch.onnx.export(_OnnxFeatures(net), torch.zeros(1, 3, 360, 360), str(d / ONNX_FILE),
                      input_names=["input"], dynamo=False)
    seed = np.random.default_rng(0).standard_normal((96, 128)).astype(np.float32)
    (d / SEED_FILE).write_bytes(bytes(SEED_HEADER) + seed.tobytes())
    return d, seed


def test_preprocess_matches_nnhash():
    """nnhash.py relies on Pillow's default resize filter; we pin it to bicubic."""
    for p in IMAGES[:6]:
        img = Image.open(p).convert("RGB")
        ref = np.array(img.resize([360, 360])).astype(np.float32) / 255.0
        assert np.array_equal(preprocess(img).permute(1, 2, 0).numpy(), ref)


def test_seed_file_format(synthetic):
    d, seed = synthetic
    assert np.array_equal(load_seed(d / SEED_FILE), seed)
    bad = d / "bad.dat"
    bad.write_bytes(b"\0" * 100)
    with pytest.raises(ValueError):
        load_seed(bad)


def test_synthetic_matches_onnxruntime(synthetic):
    """Weight loader + forward pass + hash convention vs nnhash.py on the exported graph."""
    d, seed = synthetic
    sess = ort.InferenceSession(str(d / ONNX_FILE), providers=["CPUExecutionProvider"])
    h = NeuralHash(d, source="onnx")
    for p in IMAGES[:8]:
        img = Image.open(p)
        ref_logits, ref_hex = nnhash_reference(sess, seed, img)
        ours = h.logits([img])[0]
        assert np.abs(ours - ref_logits).max() < 1e-5  # observed ~4e-7
        assert to_hex(ours >= 0) == ref_hex


def test_loader_rejects_wrong_architecture(synthetic, tmp_path):
    d, _ = synthetic
    model = NeuralHashNet()
    model.blocks = model.blocks[:-1]  # drop a block: layer count no longer matches
    with pytest.raises(RuntimeError):
        load_onnx_weights(model, d / ONNX_FILE)


def test_missing_files_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="not distributed"):
        load_neuralhash_model(model_dir=tmp_path)


def test_gradients_flow(synthetic):
    """White-box attacks need d(logits)/d(pixels)."""
    d, _ = synthetic
    model = load_neuralhash_model(d, source="onnx")
    x = preprocess(Image.open(IMAGES[0]))[None].requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# --- the real Apple model ---------------------------------------------------------------

def _need_real(onnx: bool = False):
    if not (MODEL_DIR / SEED_FILE).exists() or find_espresso(MODEL_DIR) is None:
        pytest.skip(f"Apple's NeuralHash files missing from {MODEL_DIR}")
    if onnx and not (MODEL_DIR / ONNX_FILE).exists():
        pytest.skip(f"{ONNX_FILE} (AppleNeuralHash2ONNX conversion) missing from {MODEL_DIR}")


def test_espresso_weights_match_onnx_conversion():
    """Our Espresso reader and AppleNeuralHash2ONNX's converter extract identical weights."""
    _need_real(onnx=True)
    ours = load_neuralhash_model(source="espresso").state_dict()
    ref = load_neuralhash_model(source="onnx").state_dict()
    assert ours.keys() == ref.keys()
    for k in ours:
        assert torch.equal(ours[k], ref[k]), k


def test_real_matches_nnhash():
    """Same hashes as nnhash.py + onnxruntime, except bits whose logit is ~0.

    onnxruntime's instance norm is less accurate than ours on low-contrast channels (it
    loses digits when mean^2 >> variance; see test_real_float32_accuracy), so logits are
    compared relative to their scale (~500) and a bit may flip only where |logit| ~ 0."""
    _need_real(onnx=True)
    sess = ort.InferenceSession(str(MODEL_DIR / ONNX_FILE), providers=["CPUExecutionProvider"])
    seed = load_seed(MODEL_DIR / SEED_FILE)
    h = get_hasher("neuralhash")
    exact = 0
    for p in IMAGES:
        img = Image.open(p)
        ref, ref_hex = nnhash_reference(sess, seed, img)
        ours = h.logits([img])[0]
        scale = np.abs(ref).mean()
        assert np.abs(ours - ref).max() < 1e-2 * scale, p.name
        flipped = (ours >= 0) != (ref >= 0)
        assert np.all(np.abs(ref[flipped]) < 1e-3 * scale), p.name
        exact += to_hex(ours >= 0) == ref_hex
    assert exact >= len(IMAGES) - 1  # observed: all but bridge-6-flipy.jpg (one logit = 8e-5)


def test_real_float32_accuracy():
    """Our float32 forward pass vs the same network in float64."""
    _need_real()
    model = load_neuralhash_model()
    x = torch.stack([preprocess(Image.open(p)) for p in IMAGES])
    with torch.no_grad():
        f32 = model(x).numpy()
        f64 = model.double()(x.double()).numpy()
    assert np.abs(f32 - f64).max() < 1e-4 * np.abs(f64).mean()


def test_real_hasher_interface():
    _need_real()
    h = get_hasher("neuralhash")
    img = Image.open(IMAGES[0])
    bits = h.hash(img)
    assert h.name == "neuralhash" and h.n_bits == 96
    assert bits.dtype == bool and bits.shape == (96,)
    assert np.array_equal(h.hash_batch([img, img]), np.stack([bits, bits]))
    from hashbench.attacks import jpeg
    assert h.distance(bits, h.hash(jpeg(img, 70))) <= 10
