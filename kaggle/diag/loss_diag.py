"""Which evasion loss stops trained DinoHash's attack from stalling at ~20 bits?
3 images x 3 seeds x eps 2/4/8 x loss variants, APGD/PGD 100 steps, best iterate by bits."""
import shutil, subprocess, sys, time, tarfile
from pathlib import Path
WORK = Path("/kaggle/working"); HB = WORK / "hashbench"
arc = next(Path("/kaggle/input").rglob("hashbench_adv.tar.gz"), None)
if arc: tarfile.open(arc).extractall(WORK)
else:
    src = next(p for p in Path("/kaggle/input").rglob("run_adversarial.py")).parents[1]
    shutil.copytree(src, HB, dirs_exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx"], check=True)
import os; os.environ["HASHBENCH_MODELS"] = str(HB / "models"); sys.path.insert(0, str(HB))
import torch
from hashbench.adversarial import AttackConfig, build_adapter, _apgd, _pgd, _make_objective
from hashbench.benchmark import load_image
from hashbench.hashes import get_hasher
dev = torch.device("cuda"); print(torch.cuda.get_device_name(0), flush=True)
ad = build_adapter(get_hasher("dinohash")).to(dev)
names = [l.strip() for l in (HB / "data/splits/V200.txt").read_text().splitlines() if l.strip()][:3]
for nm in names:
    x0 = ad.preprocess(load_image(HB / "data/coco/val2017" / nm, 512))[None].to(dev)
    with torch.no_grad():
        a = ad.outputs(x0); ref = torch.where(a >= 0, 1.0, -1.0); ref_h = ad.hash_t(a)
    obj = _make_objective(ad, ref_h, "evasion", True)
    V = {
      "hinge m0 apgd":     ("apgd", None, lambda o: torch.relu(o * ref).mean(1)),
      "hinge m0.3 apgd":   ("apgd", None, lambda o: torch.relu(0.3 + o * ref).mean(1)),
      "hinge m1 apgd":     ("apgd", None, lambda o: torch.relu(1.0 + o * ref).mean(1)),
      "linear apgd":       ("apgd", None, lambda o: (o * ref).mean(1)),
      "tanh/0.1 apgd":     ("apgd", None, lambda o: torch.tanh(o * ref / 0.1).mean(1)),
      "hinge m0 pgd/10":   ("pgd", 0.1, lambda o: torch.relu(o * ref).mean(1)),
      "hinge m0.3 pgd/10": ("pgd", 0.1, lambda o: torch.relu(0.3 + o * ref).mean(1)),
      "linear pgd/10":     ("pgd", 0.1, lambda o: (o * ref).mean(1)),
    }
    print(f"\n{nm}  clean |logit| median {a.abs().median().item():.3f}", flush=True)
    for name, (meth, frac, lf) in V.items():
        t = time.time(); row = []
        for e in [2, 4, 8]:
            vals = []
            for s in range(3):
                cfg = AttackConfig(eps=e/255, steps=100, method=meth, device="cuda",
                                   step_size=None if frac is None else frac * e / 255)
                adv = (_apgd if meth == "apgd" else _pgd)(ad, x0, lf, cfg, torch.Generator(device=dev).manual_seed(s), obj)
                with torch.no_grad(): vals.append(int(-obj(adv)))
            row.append(vals)
        print(f"  {name:18s} bits flipped eps2/4/8 x3 seeds: {row}  ({time.time()-t:.0f}s)", flush=True)
