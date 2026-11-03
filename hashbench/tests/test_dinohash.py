"""DinoHash: our PyTorch re-implementation vs the officially released ONNX models."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from hashbench.hashes import get_hasher  # noqa: E402
from hashbench.hashes.dinohash import MODEL_DIR, VARIANTS, load_dinohash_model, preprocess  # noqa: E402

ROOT = Path(__file__).parent
IMAGES = sorted((ROOT.parent / "data" / "samples").glob("*")) + sorted((ROOT / "data" / "meta_pdq").glob("*.jpg"))


def _need(*files):
    missing = [f for f in files if not (MODEL_DIR / f).exists()]
    if missing:
        pytest.skip(f"model files missing ({missing}); run scripts/download_models.py")


@pytest.fixture(scope="module")
def batch():
    return torch.stack([preprocess(Image.open(p)) for p in IMAGES])


def test_preprocess_matches_official_torchvision():
    tv = pytest.importorskip("torchvision.transforms")
    official = tv.Compose([tv.Resize((224, 224)), tv.ToTensor()])
    for p in IMAGES[:6]:
        img = Image.open(p).convert("RGB")
        assert torch.equal(preprocess(img), official(img))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_matches_official_onnx(variant, batch):
    """Bit-identical hashes and logits within float tolerance of the released model."""
    onnx_file = VARIANTS[variant][2]
    _need(onnx_file)
    sess = ort.InferenceSession(str(MODEL_DIR / onnx_file), providers=["CPUExecutionProvider"])
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    ref = sess.run(None, {"input": ((batch - mean) / std).numpy()})[0]
    with torch.no_grad():
        ours = load_dinohash_model(variant, trained=True)(batch).numpy()
    assert np.abs(ours - ref).max() < 1e-4
    assert np.array_equal(ours >= 0, ref >= 0)


def test_untrained_differs_from_trained(batch):
    _need(VARIANTS["vits14_96"][2], "dinov2_vits14_reg4_pretrain.pth", "dinov2_vits14_reg_PCA.npy")
    with torch.no_grad():
        trained = load_dinohash_model("vits14_96", trained=True)(batch[:4]) >= 0
        untrained = load_dinohash_model("vits14_96", trained=False)(batch[:4]) >= 0
    # Fine-tuning changes the features (and ViT-S ships its own PCA head), so hashes differ.
    assert (trained != untrained).float().mean() > 0.2


def test_hasher_interface():
    _need(VARIANTS["vits14_96"][2])
    h = get_hasher("dinohash")
    img = Image.open(IMAGES[0])
    bits = h.hash(img)
    assert h.name == "dinohash" and h.n_bits == 96
    assert bits.dtype == bool and bits.shape == (96,)
    assert np.array_equal(h.hash_batch([img, img]), np.stack([bits, bits]))
    # robust to a mild JPEG re-encode
    from hashbench.attacks import jpeg
    assert h.distance(bits, h.hash(jpeg(img, 70))) <= 10


def test_gradients_flow():
    """White-box attacks need d(logits)/d(pixels)."""
    _need(VARIANTS["vits14_96"][2])
    model = load_dinohash_model("vits14_96", trained=True)
    x = preprocess(Image.open(IMAGES[0]))[None].requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
