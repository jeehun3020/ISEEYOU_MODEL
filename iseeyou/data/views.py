from __future__ import annotations

import random

import numpy as np

from .detectors.base import BaseFaceDetector


def crop_from_bbox(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding_ratio: float = 0.15,
) -> np.ndarray | None:
    h, w = image_rgb.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    pad_w = int(bw * padding_ratio)
    pad_h = int(bh * padding_ratio)

    xx1 = max(0, x1 - pad_w)
    yy1 = max(0, y1 - pad_h)
    xx2 = min(w, x2 + pad_w)
    yy2 = min(h, y2 + pad_h)

    if xx2 <= xx1 or yy2 <= yy1:
        return None
    return image_rgb[yy1:yy2, xx1:xx2]


def _fill_value(image_rgb: np.ndarray, fill_mode: str) -> np.ndarray:
    if str(fill_mode).lower() == "black":
        return np.zeros(3, dtype=np.uint8)
    return np.median(image_rgb.reshape(-1, 3), axis=0).astype(np.uint8)


def mask_bbox_region(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding_ratio: float = 0.15,
    fill_mode: str = "median",
) -> np.ndarray:
    masked = image_rgb.copy()
    h, w = masked.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(max(1, bw) * padding_ratio)
    pad_h = int(max(1, bh) * padding_ratio)
    xx1 = max(0, x1 - pad_w)
    yy1 = max(0, y1 - pad_h)
    xx2 = min(w, x2 + pad_w)
    yy2 = min(h, y2 + pad_h)
    masked[yy1:yy2, xx1:xx2] = _fill_value(masked, fill_mode)
    return masked


def spotlight_bbox_region(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding_ratio: float = 0.15,
    fill_mode: str = "black",
) -> np.ndarray:
    spotlight = image_rgb.copy()
    h, w = spotlight.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(max(1, bw) * padding_ratio)
    pad_h = int(max(1, bh) * padding_ratio)
    xx1 = max(0, x1 - pad_w)
    yy1 = max(0, y1 - pad_h)
    xx2 = min(w, x2 + pad_w)
    yy2 = min(h, y2 + pad_h)

    masked = np.empty_like(spotlight)
    masked[:] = _fill_value(spotlight, fill_mode)
    masked[yy1:yy2, xx1:xx2] = spotlight[yy1:yy2, xx1:xx2]
    return masked


def random_same_area_blackout(
    image_rgb: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_mode: str = "black",
) -> np.ndarray:
    masked = image_rgb.copy()
    h, w = masked.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bw >= w or bh >= h:
        return masked

    max_x = max(0, w - bw)
    max_y = max(0, h - bh)
    for _ in range(10):
        rx1 = random.randint(0, max_x) if max_x > 0 else 0
        ry1 = random.randint(0, max_y) if max_y > 0 else 0
        rx2 = rx1 + bw
        ry2 = ry1 + bh
        intersects = not (rx2 <= x1 or rx1 >= x2 or ry2 <= y1 or ry1 >= y2)
        if not intersects:
            masked[ry1:ry2, rx1:rx2] = _fill_value(masked, fill_mode)
            return masked

    masked[0:bh, 0:bw] = _fill_value(masked, fill_mode)
    return masked


def extract_frame_view(
    image_rgb: np.ndarray,
    detector: BaseFaceDetector,
    fallback_to_full_frame: bool,
    view_mode: str,
) -> np.ndarray | None:
    normalized_mode = str(view_mode or "detector_crop").lower()
    if normalized_mode == "full_frame":
        return image_rgb

    detections = detector.detect(image_rgb)
    primary = detector.select_primary(detections, image_rgb.shape)

    if normalized_mode in {"background_masked", "background_only", "face_blackout"}:
        if primary is None:
            return image_rgb if fallback_to_full_frame else None
        return mask_bbox_region(
            image_rgb,
            x1=primary.x1,
            y1=primary.y1,
            x2=primary.x2,
            y2=primary.y2,
            fill_mode="black" if normalized_mode == "face_blackout" else "median",
        )

    if normalized_mode == "background_blackout":
        if primary is None:
            return image_rgb if fallback_to_full_frame else None
        return spotlight_bbox_region(
            image_rgb,
            x1=primary.x1,
            y1=primary.y1,
            x2=primary.x2,
            y2=primary.y2,
            fill_mode="black",
        )

    if normalized_mode == "random_same_area_blackout":
        if primary is None:
            return image_rgb if fallback_to_full_frame else None
        return random_same_area_blackout(
            image_rgb,
            x1=primary.x1,
            y1=primary.y1,
            x2=primary.x2,
            y2=primary.y2,
            fill_mode="black",
        )

    if normalized_mode != "detector_crop":
        raise ValueError(f"Unsupported preprocess.view_mode: {view_mode}")

    if primary is None:
        return image_rgb if fallback_to_full_frame else None

    crop = crop_from_bbox(
        image_rgb,
        x1=primary.x1,
        y1=primary.y1,
        x2=primary.x2,
        y2=primary.y2,
    )
    if crop is None or crop.size == 0:
        return image_rgb if fallback_to_full_frame else None
    return crop
