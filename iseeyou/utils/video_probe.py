from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from iseeyou.data.adapters import RawSample
from iseeyou.data.detectors.base import BaseFaceDetector

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _safe_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def probe_media_metadata(path: Path, media_type: str) -> dict[str, float | int | str]:
    file_size = float(path.stat().st_size) if path.exists() else 0.0
    if media_type == "image":
        with Image.open(path) as img:
            width, height = img.size
        aspect_ratio = float(width) / float(height) if height else 0.0
        return {
            "width": float(width),
            "height": float(height),
            "fps": 0.0,
            "frame_count": 1.0,
            "duration": 0.0,
            "aspect_ratio": aspect_ratio,
            "bitrate_kbps": 0.0,
            "file_size_bytes": file_size,
            "resolution": f"{width}x{height}",
        }

    cap = cv2.VideoCapture(str(path))
    width = _safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = _safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    aspect_ratio = width / height if height > 0 else 0.0
    bitrate_kbps = (file_size * 8.0 / 1000.0) / duration if duration > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "bitrate_kbps": bitrate_kbps,
        "file_size_bytes": file_size,
        "resolution": f"{int(width)}x{int(height)}" if width and height else "",
    }


def sample_uniform_frame_indices(frame_count: int, num_samples: int) -> list[int]:
    if frame_count <= 0:
        return [0]
    if num_samples <= 1:
        return [0 if frame_count == 1 else frame_count // 2]
    positions = np.linspace(0, max(frame_count - 1, 0), num=min(num_samples, frame_count), dtype=int)
    unique = sorted({int(x) for x in positions.tolist()})
    return unique or [0]


def read_video_frames_by_indices(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    if not frame_indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_sample_frames(sample: RawSample, frame_indices: list[int]) -> list[np.ndarray]:
    if sample.media_type == "image":
        return [np.array(Image.open(sample.path).convert("RGB"))]
    return read_video_frames_by_indices(sample.path, frame_indices)


def estimate_motion_score(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = []
    prev = frames[0].astype(np.float32) / 255.0
    for frame in frames[1:]:
        cur = frame.astype(np.float32) / 255.0
        diffs.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    return float(np.mean(diffs)) if diffs else 0.0


def estimate_text_mask_map_np(image_rgb: np.ndarray, strip_ratio: float = 0.22) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))

    candidate = np.zeros((h, w), dtype=np.uint8)
    band = max(1, int(round(h * strip_ratio)))
    candidate[:band, :] = 1
    candidate[h - band :, :] = 1

    grad_vals = grad[candidate > 0]
    if grad_vals.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    threshold = max(12, int(np.percentile(grad_vals, 88)))
    mask = ((grad >= threshold) & (candidate > 0)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 11), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 3), dtype=np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    img_area = float(h * w)
    for idx in range(1, num_labels):
        x, y, ww, hh, area = stats[idx]
        if area < max(12, int(0.00005 * img_area)):
            continue
        if hh > max(8, int(0.12 * h)):
            continue
        if ww < max(10, int(0.03 * w)):
            continue
        filtered[labels == idx] = 255
    return filtered


def estimate_text_area_ratio(frames: list[np.ndarray]) -> float:
    if not frames:
        return 0.0
    ratios = []
    for frame in frames:
        mask = estimate_text_mask_map_np(frame)
        ratios.append(float(mask.mean() / 255.0))
    return float(np.mean(ratios)) if ratios else 0.0


def estimate_face_count(frames: list[np.ndarray], detector: BaseFaceDetector | None) -> float:
    if detector is None or not frames:
        return 0.0
    counts = []
    for frame in frames:
        try:
            counts.append(float(len(detector.detect(frame))))
        except Exception:
            counts.append(0.0)
    return float(np.mean(counts)) if counts else 0.0


def summarize_probe(
    sample: RawSample,
    sampled_frame_indices: list[int],
    detector: BaseFaceDetector | None,
) -> dict[str, float | str]:
    meta = probe_media_metadata(sample.path, sample.media_type)
    frames = load_sample_frames(sample, sampled_frame_indices)
    text_ratio = estimate_text_area_ratio(frames)
    motion = estimate_motion_score(frames)
    face_count = estimate_face_count(frames, detector)
    return {
        **meta,
        "face_count_estimate": face_count,
        "text_area_ratio_estimate": text_ratio,
        "motion_score": motion,
        "sampled_frame_indices": ";".join(str(x) for x in sampled_frame_indices),
    }
