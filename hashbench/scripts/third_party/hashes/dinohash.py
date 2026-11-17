"""Shim for the vendored authors' APGD (dinohash_apgd_c700888.py), which does
`from hashes.dinohash import dinohash`. Their function is replaced by whichever model
reproduce_dinohash_table1.py installs with set_model(); the signature is theirs."""

import torch
import torch.nn.functional as F

_MODEL = None


def set_model(model):
    global _MODEL
    _MODEL = model


def dinohash(x, differentiable=False, c=1, logits=False, l2_normalize=False, prod_output=True):
    ctx = torch.enable_grad if differentiable else torch.no_grad
    with ctx():
        out = _MODEL(x)
        if l2_normalize:
            out = F.normalize(out, dim=1)
        out = out * c
        if not logits:
            out = torch.sigmoid(out) if differentiable else out >= 0
    return out
