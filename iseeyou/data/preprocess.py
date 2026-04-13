from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from iseeyou.config import ensure_dir
from iseeyou.constants import build_task_spec
from iseeyou.utils.masking import apply_text_mask_np
from iseeyou.utils.video import iter_video_frames, resize_image

from .adapters import RawSample, collect_samples_from_config
from .detectors.base import BaseFaceDetector
from .detectors.factory import build_face_detector
from .manifest import write_manifest
from .split import create_group_splits
from .views import extract_frame_view


@dataclass
class PreprocessStats:
    total_raw_samples: int = 0
    used_raw_samples: int = 0
    skipped_unknown_class: int = 0
    skipped_no_face: int = 0
    skipped_failed_decode: int = 0
    extracted_frames: int = 0



def _crop_from_bbox(
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


def _mask_bbox_region(
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
    if str(fill_mode).lower() == "black":
        fill_value = np.zeros(3, dtype=np.uint8)
    else:
        fill_value = np.median(masked.reshape(-1, 3), axis=0).astype(np.uint8)
    masked[yy1:yy2, xx1:xx2] = fill_value
    return masked


def _spotlight_bbox_region(
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

    if str(fill_mode).lower() == "median":
        fill_value = np.median(spotlight.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        fill_value = np.zeros(3, dtype=np.uint8)

    masked = np.empty_like(spotlight)
    masked[:] = fill_value
    masked[yy1:yy2, xx1:xx2] = spotlight[yy1:yy2, xx1:xx2]
    return masked


def _extract_frame_view(
    image_rgb: np.ndarray,
    detector: BaseFaceDetector,
    fallback_to_full_frame: bool,
    view_mode: str,
) -> np.ndarray | None:
    return extract_frame_view(
        image_rgb=image_rgb,
        detector=detector,
        fallback_to_full_frame=fallback_to_full_frame,
        view_mode=view_mode,
    )


def _extract_face_or_full_frame(
    image_rgb: np.ndarray,
    detector: BaseFaceDetector,
    fallback_to_full_frame: bool,
) -> np.ndarray | None:
    return _extract_frame_view(
        image_rgb=image_rgb,
        detector=detector,
        fallback_to_full_frame=fallback_to_full_frame,
        view_mode="detector_crop",
    )



def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)



def _process_video_sample(
    sample: RawSample,
    split_name: str,
    faces_root: Path,
    detector: BaseFaceDetector,
    preprocess_cfg: dict,
    stats: PreprocessStats,
) -> list[dict]:
    rows: list[dict] = []

    target_fps = preprocess_cfg["target_fps"]
    max_frames = preprocess_cfg.get("max_frames_per_video")
    image_size = preprocess_cfg["image_size"]
    fallback_to_full_frame = preprocess_cfg.get("fallback_to_full_frame", False)
    view_mode = preprocess_cfg.get("view_mode", "detector_crop")
    text_mask_cfg = preprocess_cfg.get("text_mask", {})

    for frame_idx, frame_rgb in iter_video_frames(sample.path, target_fps, max_frames):
        frame_view = _extract_frame_view(frame_rgb, detector, fallback_to_full_frame, view_mode=view_mode)
        if frame_view is None:
            stats.skipped_no_face += 1
            continue

        frame_view = apply_text_mask_np(frame_view, text_mask_cfg)
        frame_view = resize_image(frame_view, image_size)

        out_path = (
            faces_root
            / split_name
            / sample.class_name
            / sample.dataset
            / f"{sample.sample_id}_{frame_idx:05d}.jpg"
        )
        _save_rgb_image(out_path, frame_view)

        rows.append(
            {
                "split": split_name,
                "split_tag": split_name,
                "dataset": sample.dataset,
                "class_name": sample.class_name,
                "frame_path": str(out_path),
                "video_id": sample.video_id,
                "sample_id": sample.sample_id,
                "frame_idx": frame_idx,
                "identity_id": sample.identity_id,
                "source_id": sample.source_id,
                "original_id": sample.original_id,
                "platform_id": sample.platform_id,
                "creator_account": sample.creator_account,
                "generator_family": sample.generator_family,
                "template_id": sample.template_id,
                "prompt_id": sample.prompt_id,
                "scene_id": sample.scene_id,
                "source_url": sample.source_url,
                "source_family": sample.source_family,
                "raw_asset_group": sample.raw_asset_group,
                "upload_pipeline": sample.upload_pipeline,
            }
        )
        stats.extracted_frames += 1

    return rows



def _process_image_sample(
    sample: RawSample,
    split_name: str,
    faces_root: Path,
    detector: BaseFaceDetector,
    preprocess_cfg: dict,
    stats: PreprocessStats,
) -> list[dict]:
    image_size = preprocess_cfg["image_size"]
    fallback_to_full_frame = preprocess_cfg.get("fallback_to_full_frame", False)
    view_mode = preprocess_cfg.get("view_mode", "detector_crop")
    text_mask_cfg = preprocess_cfg.get("text_mask", {})

    image_rgb = np.array(Image.open(sample.path).convert("RGB"))
    face_crop = _extract_frame_view(image_rgb, detector, fallback_to_full_frame, view_mode=view_mode)

    if face_crop is None:
        stats.skipped_no_face += 1
        return []

    face_crop = apply_text_mask_np(face_crop, text_mask_cfg)
    face_crop = resize_image(face_crop, image_size)

    out_path = (
        faces_root
        / split_name
        / sample.class_name
        / sample.dataset
        / f"{sample.sample_id}_00000.jpg"
    )
    _save_rgb_image(out_path, face_crop)

    stats.extracted_frames += 1
    return [
        {
            "split": split_name,
            "split_tag": split_name,
            "dataset": sample.dataset,
            "class_name": sample.class_name,
            "frame_path": str(out_path),
            "video_id": sample.video_id,
            "sample_id": sample.sample_id,
            "frame_idx": 0,
            "identity_id": sample.identity_id,
            "source_id": sample.source_id,
            "original_id": sample.original_id,
            "platform_id": sample.platform_id,
            "creator_account": sample.creator_account,
            "generator_family": sample.generator_family,
            "template_id": sample.template_id,
            "prompt_id": sample.prompt_id,
            "scene_id": sample.scene_id,
            "source_url": sample.source_url,
            "source_family": sample.source_family,
            "raw_asset_group": sample.raw_asset_group,
            "upload_pipeline": sample.upload_pipeline,
        }
    ]



def run_preprocessing(config: dict) -> None:
    task_spec = build_task_spec(config["task"])
    allowed_classes = set(task_spec.classes)

    paths_cfg = config["paths"]
    faces_root = ensure_dir(paths_cfg["processed_faces_dir"])
    manifests_root = ensure_dir(paths_cfg["manifests_dir"])

    preprocess_cfg = config["preprocess"]
    split_cfg = config["split"]

    samples = collect_samples_from_config(config["datasets"])
    stats = PreprocessStats(total_raw_samples=len(samples))

    filtered_samples: list[RawSample] = []
    for sample in samples:
        if sample.class_name not in allowed_classes:
            stats.skipped_unknown_class += 1
            continue
        filtered_samples.append(sample)

    stats.used_raw_samples = len(filtered_samples)

    if len(filtered_samples) == 0:
        # TODO: consider raising an explicit error if empty input should fail CI.
        for split_name in ["train", "val", "test"]:
            write_manifest([], manifests_root / f"{split_name}.csv")
        write_manifest([], manifests_root / "all.csv")

        report = {
            "stats": asdict(stats),
            "splits": {
                "train": {"num_rows": 0, "num_videos": 0},
                "val": {"num_rows": 0, "num_videos": 0},
                "test": {"num_rows": 0, "num_videos": 0},
            },
        }
        report_path = manifests_root / "preprocess_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[WARN] no valid samples found. empty manifests were generated.")
        return

    split_map = create_group_splits(
        filtered_samples,
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=config["seed"],
        group_priority=split_cfg["group_priority"],
    )

    detector = build_face_detector(preprocess_cfg["detector"])

    manifest_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_name in ["train", "val", "test"]:
        indices = split_map[split_name]
        print(f"[INFO] preprocessing {split_name}: {len(indices)} raw samples")

        for idx in tqdm(indices, desc=f"preprocess-{split_name}"):
            sample = filtered_samples[idx]
            try:
                if sample.media_type == "video":
                    rows = _process_video_sample(
                        sample=sample,
                        split_name=split_name,
                        faces_root=faces_root,
                        detector=detector,
                        preprocess_cfg=preprocess_cfg,
                        stats=stats,
                    )
                elif sample.media_type == "image":
                    rows = _process_image_sample(
                        sample=sample,
                        split_name=split_name,
                        faces_root=faces_root,
                        detector=detector,
                        preprocess_cfg=preprocess_cfg,
                        stats=stats,
                    )
                else:
                    # TODO: handle mixed media types if an external dataset needs both image and video.
                    rows = []
            except Exception as exc:
                stats.skipped_failed_decode += 1
                print(
                    f"[WARN] skipping unreadable sample: dataset={sample.dataset} "
                    f"sample_id={sample.sample_id} path={sample.path} error={exc}"
                )
                rows = []

            manifest_rows[split_name].extend(rows)

    for split_name, rows in manifest_rows.items():
        write_manifest(rows, manifests_root / f"{split_name}.csv")

    all_rows = manifest_rows["train"] + manifest_rows["val"] + manifest_rows["test"]
    write_manifest(all_rows, manifests_root / "all.csv")

    split_summary = {
        split: {
            "num_rows": len(rows),
            "num_videos": len({row["video_id"] for row in rows}),
        }
        for split, rows in manifest_rows.items()
    }

    report = {
        "stats": asdict(stats),
        "splits": split_summary,
    }
    report_path = manifests_root / "preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] preprocessing done. report: {report_path}")
