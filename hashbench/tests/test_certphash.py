"""CertPHash: our re-implementation vs the authors' code (github.com/Yuchen413/CertPhash).

The reference network and preprocessing are imported from our clone in
../HASH_IMPLEMENTATIONS/CertPhash (skipped if it, or the released models, are missing).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
T = pytest.importorskip("torchvision.transforms")

from hashbench.hashes import get_hasher  # noqa: E402
from hashbench.hashes.certphash import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, MATCH_THRESHOLD, MODEL_DIR, VARIANTS, load_certphash_model,
    outputs_to_hash, preprocess,
)

ROOT = Path(__file__).parent
IMAGES = sorted((ROOT.parent / "data" / "samples").glob("*")) + sorted((ROOT / "data" / "meta_pdq").glob("*.jpg"))
REF_REPO = ROOT.parents[1] / "HASH_IMPLEMENTATIONS" / "CertPhash"


def _need(variant):
    if not (MODEL_DIR / VARIANTS[variant]).exists():
        pytest.skip(f"{MODEL_DIR / VARIANTS[variant]} missing; run scripts/download_models.py --certphash")


@pytest.fixture(scope="module")
def reference_resnet_v5():
    path = REF_REPO / "train_verify" / "models" / "resnet_v5.py"
    if not path.exists():
        pytest.skip(f"reference repo missing: {REF_REPO}")
    spec = importlib.util.spec_from_file_location("certphash_ref_resnet_v5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resnet_v5


def reference_transform(image: Image.Image) -> torch.Tensor:
    """train_verify/datasets.py ImageToHashAugmented (num_augmented=0) on a decoded image."""
    img = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
    return T.Compose([T.Resize((64, 64)), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])(img)


def test_preprocess_matches_reference():
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    for p in IMAGES:
        img = Image.open(p)
        assert torch.allclose((preprocess(img) - mean) / std, reference_transform(img), atol=1e-6), p.name


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_matches_reference_model(variant, reference_resnet_v5):
    """Same outputs as the authors' resnet_v5 + checkpoint on the authors' preprocessing."""
    _need(variant)
    ref = reference_resnet_v5(num_classes=144, bn=True, input_dim=64)
    ref.load_state_dict(torch.load(MODEL_DIR / VARIANTS[variant], map_location="cpu", weights_only=True))
    ref.eval()
    ours = load_certphash_model(variant)
    x01 = torch.stack([preprocess(Image.open(p)) for p in IMAGES])
    with torch.no_grad():
        y_ref = ref(torch.stack([reference_transform(Image.open(p)) for p in IMAGES])).numpy()
        y = ours(x01).numpy()
    assert np.abs(y - y_ref).max() < 1e-4  # observed: exactly 0 (outputs are in byte units, 0..255)
    # authors' hash: relu(round(y)); ours additionally clips at 255
    ref_hash = np.maximum(np.round(y_ref), 0)
    assert np.array_equal(outputs_to_hash(y), np.minimum(ref_hash, 255).astype(np.uint8))


def test_hasher_interface():
    _need("ep1")
    h = get_hasher("certphash")
    img = Image.open(IMAGES[0])
    v = h.hash(img)
    assert h.name == "certphash" and h.n_bits == 144 and h.max_distance == 144 * 255
    assert v.dtype == np.uint8 and v.shape == (144,)
    assert np.array_equal(h.from_string(h.to_string(v)), v)
    assert np.array_equal(h.hash_batch([img, img]), np.stack([v, v]))
    assert h.distance(v, v) == 0
    from hashbench.attacks import jpeg
    assert h.distance(v, h.hash(jpeg(img, 70))) <= MATCH_THRESHOLD
    other = h.hash(Image.open(IMAGES[1]))
    assert h.distance(v, other) == int(np.abs(v.astype(int) - other.astype(int)).sum()) > MATCH_THRESHOLD


def test_gradients_flow():
    _need("ep1")
    model = load_certphash_model("ep1")
    x = preprocess(Image.open(IMAGES[0]))[None].requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


COCO_VAL = ROOT.parent / "data" / "certphash" / "data"


def test_imitates_real_photodna_on_coco_val():
    """The released ep1 model vs the real PhotoDNA hashes of the authors' COCO validation set
    (10,000 images, coco-val.csv). Observed: 99.9% identified, 96.9% within L1 1800."""
    import base64
    import csv
    _need("ep1")
    if not (COCO_VAL / "coco-val.csv").exists():
        pytest.skip(f"authors' data archive not extracted to {COCO_VAL}")
    rows = list(csv.reader(open(COCO_VAL / "coco-val.csv")))
    true = np.stack([np.frombuffer(base64.b64decode(r[1]), dtype=np.uint8) for r in rows]).astype(np.int64)
    imgs = [Image.open(COCO_VAL / "coco100x100_val" / Path(r[0]).name).convert("RGB") for r in rows]
    ours = get_hasher("certphash").hash_batch(imgs).astype(np.int64)
    assert np.mean(np.abs(ours - true).sum(1) <= MATCH_THRESHOLD) > 0.95
    t = torch.from_numpy(true).float()
    nearest = torch.cat([torch.cdist(torch.from_numpy(ours[i:i + 500]).float(), t, p=1).argmin(1)
                         for i in range(0, len(ours), 500)])
    assert (nearest == torch.arange(len(ours))).float().mean() > 0.995
