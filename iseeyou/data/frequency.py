from __future__ import annotations

from typing import Final

import numpy as np
from PIL import Image


REPRESENTATION_CHOICES: Final[set[str]] = {
    "rgb",
    "fft",
    "rgb_fft",
    "fft_amplitude",
    "fft_phase",
    "fft_highpass",
    "fft_lowpass",
    "rgb_fft_amplitude",
    "rgb_fft_phase",
}


def _normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(np.min(x))
    denom = float(np.max(x))
    if denom <= 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    x = x / denom
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def _fft_channels(
    image: Image.Image,
    *,
    mode: str,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    radius = max(1, min(h, w) // 16)
    cy, cx = h // 2, w // 2

    out = np.zeros_like(arr, dtype=np.uint8)
    for c in range(3):
        ch = arr[..., c]
        spec = np.fft.fftshift(np.fft.fft2(ch))
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)

        if mode in {"amplitude", "highpass", "lowpass"}:
            mag = np.log1p(np.abs(spec))
            if mode == "highpass":
                mag[y1:y2, x1:x2] = 0.0
            elif mode == "lowpass":
                keep = np.zeros_like(mag)
                keep[y1:y2, x1:x2] = mag[y1:y2, x1:x2]
                mag = keep
            out[..., c] = _normalize_to_uint8(mag)
        elif mode == "phase":
            phase = np.angle(spec)
            phase = (phase + np.pi) / (2.0 * np.pi)
            out[..., c] = np.clip(phase * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported FFT mode: {mode}")

    return Image.fromarray(out, mode="RGB")


def convert_representation(image: Image.Image, representation: str) -> Image.Image:
    mode = representation.lower()
    if mode == "rgb":
        return image.convert("RGB")

    if mode == "fft":
        return _fft_channels(image, mode="highpass")

    if mode == "fft_amplitude":
        return _fft_channels(image, mode="amplitude")

    if mode == "fft_phase":
        return _fft_channels(image, mode="phase")

    if mode == "fft_highpass":
        return _fft_channels(image, mode="highpass")

    if mode == "fft_lowpass":
        return _fft_channels(image, mode="lowpass")

    if mode == "rgb_fft":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        fft = np.asarray(_fft_channels(image, mode="highpass"), dtype=np.uint8)
        mixed = np.stack([rgb[..., 0], rgb[..., 1], fft[..., 2]], axis=-1)
        return Image.fromarray(mixed, mode="RGB")

    if mode == "rgb_fft_amplitude":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        fft = np.asarray(_fft_channels(image, mode="amplitude"), dtype=np.uint8)
        mixed = np.stack([rgb[..., 0], rgb[..., 1], fft[..., 2]], axis=-1)
        return Image.fromarray(mixed, mode="RGB")

    if mode == "rgb_fft_phase":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        fft = np.asarray(_fft_channels(image, mode="phase"), dtype=np.uint8)
        mixed = np.stack([rgb[..., 0], rgb[..., 1], fft[..., 2]], axis=-1)
        return Image.fromarray(mixed, mode="RGB")

    raise ValueError(
        f"Unsupported input representation: {representation}. "
        f"Choose one of {sorted(REPRESENTATION_CHOICES)}"
    )


def validate_representation(representation: str) -> str:
    mode = representation.lower()
    if mode not in REPRESENTATION_CHOICES:
        raise ValueError(
            f"Unsupported input representation: {representation}. "
            f"Choose one of {sorted(REPRESENTATION_CHOICES)}"
        )
    return mode
