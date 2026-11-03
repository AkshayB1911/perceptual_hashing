"""Content-preserving image transformations ("attacks") used to test hash robustness.

Each attack is a function ``(image, param) -> image``. ``ATTACK_SUITE`` lists the
parameter values (severities) the benchmark sweeps for each attack. All attacks are
deterministic (noise uses a fixed seed) so results are reproducible.

These are the non-adversarial transformations; gradient-based attacks (collision,
evasion) will live in a separate module once we add the deep hashes.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

Attack = Callable[[Image.Image, float], Image.Image]


def _rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


# --- compression / resampling -------------------------------------------------

def jpeg(image: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    _rgb(image).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rescale(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return _rgb(image).resize(size, Image.Resampling.BILINEAR)


def aspect_ratio(image: Image.Image, x_scale: float) -> Image.Image:
    """Stretch the width by ``x_scale`` while keeping the height."""
    w, h = image.size
    return _rgb(image).resize((max(1, round(w * x_scale)), h), Image.Resampling.BILINEAR)


# --- filtering / noise ----------------------------------------------------------

def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return _rgb(image).filter(ImageFilter.GaussianBlur(radius=sigma))


def median_filter(image: Image.Image, size: float) -> Image.Image:
    return _rgb(image).filter(ImageFilter.MedianFilter(size=int(size)))


def gaussian_noise(image: Image.Image, std: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(_rgb(image), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0.0, std, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def salt_and_pepper(image: Image.Image, amount: float, seed: int = 0) -> Image.Image:
    arr = np.asarray(_rgb(image)).copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(arr.shape[:2])
    arr[mask < amount / 2] = 0
    arr[mask > 1 - amount / 2] = 255
    return Image.fromarray(arr)


# --- photometric ----------------------------------------------------------------

def brightness(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(_rgb(image)).enhance(factor)


def contrast(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(_rgb(image)).enhance(factor)


def gamma(image: Image.Image, g: float) -> Image.Image:
    arr = np.asarray(_rgb(image), dtype=np.float32) / 255.0
    return Image.fromarray((np.power(arr, g) * 255.0).round().astype(np.uint8))


def saturation(image: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(_rgb(image)).enhance(factor)


def hue_shift(image: Image.Image, degrees: float) -> Image.Image:
    hsv = np.asarray(_rgb(image).convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(int) + round(degrees / 360 * 256)) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def grayscale(image: Image.Image, _: float = 0) -> Image.Image:
    return _rgb(image).convert("L").convert("RGB")


# --- geometric ------------------------------------------------------------------

def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate about the centre, keep the size, fill the corners with black."""
    return _rgb(image).rotate(degrees, resample=Image.Resampling.BILINEAR, expand=False)


def rotate_crop(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate, then crop to the largest axis-aligned rectangle with no black corners."""
    img = _rgb(image)
    w, h = img.size
    rot = img.rotate(degrees, resample=Image.Resampling.BILINEAR, expand=True)
    a = np.deg2rad(abs(degrees) % 180)
    if a > np.pi / 2:
        a = np.pi - a
    sin, cos = np.sin(a), np.cos(a)
    long_side, short_side = max(w, h), min(w, h)
    if short_side <= 2 * sin * cos * long_side or abs(sin - cos) < 1e-10:
        x = 0.5 * short_side
        cw, ch = (x / sin, x / cos) if w >= h else (x / cos, x / sin)
    else:
        cos2 = cos * cos - sin * sin
        cw, ch = (w * cos - h * sin) / cos2, (h * cos - w * sin) / cos2
    rw, rh = rot.size
    left, top = (rw - cw) / 2, (rh - ch) / 2
    return rot.crop((round(left), round(top), round(left + cw), round(top + ch)))


def center_crop(image: Image.Image, keep: float) -> Image.Image:
    """Keep the central ``keep`` fraction of each side (keep=0.8 removes 10% per border)."""
    w, h = image.size
    cw, ch = round(w * keep), round(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return _rgb(image).crop((left, top, left + cw, top + ch))


def corner_crop(image: Image.Image, keep: float) -> Image.Image:
    """Keep the top-left ``keep`` fraction of each side (an off-centre crop)."""
    w, h = image.size
    return _rgb(image).crop((0, 0, round(w * keep), round(h * keep)))


def translate(image: Image.Image, fraction: float) -> Image.Image:
    """Shift right and down by ``fraction`` of the size, black fill."""
    img = _rgb(image)
    w, h = img.size
    dx, dy = round(w * fraction), round(h * fraction)
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy))


def flip_horizontal(image: Image.Image, _: float = 0) -> Image.Image:
    return _rgb(image).transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def flip_vertical(image: Image.Image, _: float = 0) -> Image.Image:
    return _rgb(image).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def border(image: Image.Image, fraction: float) -> Image.Image:
    """Add a black border of ``fraction`` of the size on every side (letterboxing)."""
    img = _rgb(image)
    w, h = img.size
    bw, bh = round(w * fraction), round(h * fraction)
    out = Image.new("RGB", (w + 2 * bw, h + 2 * bh))
    out.paste(img, (bw, bh))
    return out


# --- overlays -------------------------------------------------------------------

def text_overlay(image: Image.Image, height_fraction: float) -> Image.Image:
    """Draw a white caption bar of ``height_fraction`` of the image height at the bottom."""
    img = _rgb(image).copy()
    w, h = img.size
    bar_h = max(1, round(h * height_fraction))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, h - bar_h, w, h), fill=(255, 255, 255))
    try:
        font = ImageFont.load_default(size=max(8, int(bar_h * 0.7)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    draw.text((bar_h // 4, h - bar_h), "hashbench watermark", fill=(0, 0, 0), font=font)
    return img


def occlusion(image: Image.Image, area_fraction: float) -> Image.Image:
    """Paint a grey square covering ``area_fraction`` of the image, centred."""
    img = _rgb(image).copy()
    w, h = img.size
    side = round(np.sqrt(area_fraction * w * h))
    left, top = (w - side) // 2, (h - side) // 2
    ImageDraw.Draw(img).rectangle((left, top, left + side, top + side), fill=(128, 128, 128))
    return img


ATTACKS: dict[str, Attack] = {
    "jpeg": jpeg,
    "rescale": rescale,
    "aspect_ratio": aspect_ratio,
    "gaussian_blur": gaussian_blur,
    "median_filter": median_filter,
    "gaussian_noise": gaussian_noise,
    "salt_and_pepper": salt_and_pepper,
    "brightness": brightness,
    "contrast": contrast,
    "gamma": gamma,
    "saturation": saturation,
    "hue_shift": hue_shift,
    "grayscale": grayscale,
    "rotate": rotate,
    "rotate_crop": rotate_crop,
    "center_crop": center_crop,
    "corner_crop": corner_crop,
    "translate": translate,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "border": border,
    "text_overlay": text_overlay,
    "occlusion": occlusion,
}

#: Severity levels swept by the benchmark, roughly mild -> strong.
ATTACK_SUITE: dict[str, list[float]] = {
    "jpeg": [90, 70, 50, 30, 10],
    "rescale": [0.75, 0.5, 0.25, 0.1],
    "aspect_ratio": [0.8, 0.9, 1.1, 1.25],
    "gaussian_blur": [0.5, 1, 2, 3, 5],
    "median_filter": [3, 5, 7],
    "gaussian_noise": [2, 5, 10, 20, 40],
    "salt_and_pepper": [0.01, 0.03, 0.05, 0.1],
    "brightness": [0.5, 0.7, 1.3, 1.6],
    "contrast": [0.5, 0.7, 1.3, 1.6],
    "gamma": [0.5, 0.75, 1.5, 2.0],
    "saturation": [0.0, 0.5, 1.5, 2.0],
    "hue_shift": [15, 45, 90, 180],
    "grayscale": [0],
    "rotate": [1, 2, 5, 10, 20, 45],
    "rotate_crop": [1, 2, 5, 10, 20],
    "center_crop": [0.95, 0.9, 0.8, 0.7, 0.5],
    "corner_crop": [0.95, 0.9, 0.8, 0.7],
    "translate": [0.02, 0.05, 0.1, 0.2],
    "flip_horizontal": [0],
    "flip_vertical": [0],
    "border": [0.02, 0.05, 0.1, 0.2],
    "text_overlay": [0.05, 0.1, 0.2],
    "occlusion": [0.01, 0.05, 0.1, 0.25],
}


def apply_attack(image: Image.Image, attack: str, param: float) -> Image.Image:
    return ATTACKS[attack](image, param)
