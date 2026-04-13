from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image


def apply_band_mask_np(
    image_rgb: np.ndarray,
    *,
    top_ratio: float = 0.0,
    bottom_ratio: float = 0.0,
    left_ratio: float = 0.0,
    right_ratio: float = 0.0,
    position_mode: str = "fixed",
    fill_mode: str = "median",
    blur_kernel_size: int = 31,
    inpaint_radius: int = 3,
) -> np.ndarray:
    if image_rgb.size == 0:
        return image_rgb

    masked = image_rgb.copy()
    h, w = masked.shape[:2]

    top = max(0, min(h, int(round(h * max(0.0, float(top_ratio))))))
    bottom = max(0, min(h, int(round(h * max(0.0, float(bottom_ratio))))))
    left = max(0, min(w, int(round(w * max(0.0, float(left_ratio))))))
    right = max(0, min(w, int(round(w * max(0.0, float(right_ratio))))))
    mask = np.zeros((h, w), dtype=np.uint8)

    def _band_start(length: int, band: int, side: str) -> int:
        if band <= 0:
            return 0
        if position_mode == "random":
            max_start = max(0, length - band)
            return random.randint(0, max_start) if max_start > 0 else 0
        if side in {"bottom", "right"}:
            return max(0, length - band)
        return 0

    if top > 0:
        start = _band_start(h, top, "top")
        mask[start : start + top, :] = 255
    if bottom > 0:
        start = _band_start(h, bottom, "bottom")
        mask[start : start + bottom, :] = 255
    if left > 0:
        start = _band_start(w, left, "left")
        mask[:, start : start + left] = 255
    if right > 0:
        start = _band_start(w, right, "right")
        mask[:, start : start + right] = 255

    if not np.any(mask):
        return masked

    if fill_mode == "black":
        fill_image = np.zeros_like(masked, dtype=np.uint8)
    elif fill_mode == "median":
        fill_value = np.median(masked.reshape(-1, 3), axis=0).astype(np.uint8)
        fill_image = np.broadcast_to(fill_value.reshape(1, 1, 3), masked.shape).copy()
    elif fill_mode == "blur":
        import cv2  # type: ignore

        k = max(3, int(blur_kernel_size))
        if k % 2 == 0:
            k += 1
        fill_image = cv2.GaussianBlur(masked, (k, k), 0)
    elif fill_mode == "inpaint":
        import cv2  # type: ignore

        fill_image = cv2.inpaint(masked, mask, float(max(1, inpaint_radius)), cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    masked[mask > 0] = fill_image[mask > 0]
    return masked


def apply_text_mask_np(image_rgb: np.ndarray, text_mask_cfg: dict[str, Any] | None) -> np.ndarray:
    cfg = text_mask_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return image_rgb

    return apply_band_mask_np(
        image_rgb,
        top_ratio=float(cfg.get("top_ratio", 0.0)),
        bottom_ratio=float(cfg.get("bottom_ratio", 0.0)),
        left_ratio=float(cfg.get("left_ratio", 0.0)),
        right_ratio=float(cfg.get("right_ratio", 0.0)),
        position_mode=str(cfg.get("position_mode", "fixed")),
        fill_mode=str(cfg.get("fill_mode", "median")),
        blur_kernel_size=int(cfg.get("blur_kernel_size", 31)),
        inpaint_radius=int(cfg.get("inpaint_radius", 3)),
    )


def apply_random_box_mask_np(
    image_rgb: np.ndarray,
    *,
    area_ratio_range: tuple[float, float] = (0.08, 0.2),
    aspect_ratio_range: tuple[float, float] = (0.75, 1.5),
    fill_mode: str = "median",
) -> np.ndarray:
    if image_rgb.size == 0:
        return image_rgb

    masked = image_rgb.copy()
    h, w = masked.shape[:2]
    if h <= 1 or w <= 1:
        return masked

    area_low = max(0.0, float(area_ratio_range[0]))
    area_high = max(area_low, float(area_ratio_range[1]))
    area_ratio = random.uniform(area_low, area_high) if area_high > area_low else area_low
    target_area = max(1.0, float(h * w) * area_ratio)

    ar_low = max(1e-3, float(aspect_ratio_range[0]))
    ar_high = max(ar_low, float(aspect_ratio_range[1]))
    aspect_ratio = random.uniform(ar_low, ar_high) if ar_high > ar_low else ar_low

    box_w = int(round((target_area * aspect_ratio) ** 0.5))
    box_h = int(round((target_area / aspect_ratio) ** 0.5))
    box_w = max(1, min(w, box_w))
    box_h = max(1, min(h, box_h))

    max_x = max(0, w - box_w)
    max_y = max(0, h - box_h)
    x1 = random.randint(0, max_x) if max_x > 0 else 0
    y1 = random.randint(0, max_y) if max_y > 0 else 0
    x2 = x1 + box_w
    y2 = y1 + box_h

    fill_image = masked.copy()
    if fill_mode == "black":
        fill_image[:] = 0
    elif fill_mode == "median":
        fill_value = np.median(masked.reshape(-1, 3), axis=0).astype(np.uint8)
        fill_image[:] = fill_value.reshape(1, 1, 3)
    elif fill_mode == "blur":
        import cv2  # type: ignore

        fill_image = cv2.GaussianBlur(masked, (31, 31), 0)
    elif fill_mode == "inpaint":
        import cv2  # type: ignore

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        fill_image = cv2.inpaint(masked, mask, 3.0, cv2.INPAINT_TELEA)
    else:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")

    masked[y1:y2, x1:x2] = fill_image[y1:y2, x1:x2]
    return masked


class RandomBandMask:
    def __init__(
        self,
        *,
        p: float = 0.0,
        top_ratio_range: tuple[float, float] = (0.0, 0.0),
        bottom_ratio_range: tuple[float, float] = (0.0, 0.0),
        left_ratio_range: tuple[float, float] = (0.0, 0.0),
        right_ratio_range: tuple[float, float] = (0.0, 0.0),
        position_mode: str = "fixed",
        fill_mode: str = "median",
        blur_kernel_size: int = 31,
        inpaint_radius: int = 3,
    ) -> None:
        self.p = float(p)
        self.top_ratio_range = top_ratio_range
        self.bottom_ratio_range = bottom_ratio_range
        self.left_ratio_range = left_ratio_range
        self.right_ratio_range = right_ratio_range
        self.position_mode = position_mode
        self.fill_mode = fill_mode
        self.blur_kernel_size = int(blur_kernel_size)
        self.inpaint_radius = int(inpaint_radius)

    def _sample(self, bounds: tuple[float, float]) -> float:
        low, high = float(bounds[0]), float(bounds[1])
        if high <= low:
            return max(0.0, low)
        return random.uniform(low, high)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0.0 or random.random() > self.p:
            return image

        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        masked = apply_band_mask_np(
            arr,
            top_ratio=self._sample(self.top_ratio_range),
            bottom_ratio=self._sample(self.bottom_ratio_range),
            left_ratio=self._sample(self.left_ratio_range),
            right_ratio=self._sample(self.right_ratio_range),
            position_mode=self.position_mode,
            fill_mode=self.fill_mode,
            blur_kernel_size=self.blur_kernel_size,
            inpaint_radius=self.inpaint_radius,
        )
        return Image.fromarray(masked, mode="RGB")
