from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from iseeyou.constants import LabelMapper, TaskSpec
from iseeyou.utils.masking import apply_text_mask_np
from iseeyou.utils.video import resize_image
from iseeyou.utils.video_probe import read_video_frames_by_indices

from .detectors.factory import build_face_detector
from .video_manifest import read_video_manifest
from .views import extract_frame_view, mask_bbox_region, random_same_area_blackout
from iseeyou.utils.masking import apply_random_box_mask_np


def _parse_indices(raw: str, fallback: int = 0) -> list[int]:
    if not raw:
        return [fallback]
    out = []
    for part in str(raw).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out or [fallback]


class VideoManifestFrameDataset(Dataset):
    def __init__(
        self,
        video_manifest_path: str | Path,
        task_spec: TaskSpec,
        split_tags: tuple[str, ...] = ("train",),
        preprocess_cfg: dict | None = None,
        augmentation_cfg: dict | None = None,
        train_mode: bool = False,
        transform=None,
    ):
        self.rows = [
            row
            for row in read_video_manifest(video_manifest_path)
            if row.get("split_tag", "") in set(split_tags)
        ]
        self.label_mapper = LabelMapper(task_spec)
        self.transform = transform
        self.train_mode = bool(train_mode)
        self.preprocess_cfg = preprocess_cfg or {}
        self.augmentation_cfg = augmentation_cfg or {}
        self.image_size = int(self.preprocess_cfg.get("image_size", 224))
        self.frame_sampling_mode = str(self.preprocess_cfg.get("frame_sampling_mode", "all")).lower()
        self.view_mode = self.preprocess_cfg.get("view_mode", "full_frame")
        self.fallback_to_full_frame = bool(self.preprocess_cfg.get("fallback_to_full_frame", True))
        self.text_mask_cfg = self.preprocess_cfg.get("text_mask", {})
        self.detector = build_face_detector(self.preprocess_cfg.get("detector", {"name": "none"}))
        self.region_dropout_cfg = self.augmentation_cfg.get("region_dropout", {}) or {}

        self.items: list[dict] = []
        for row in self.rows:
            indices = _parse_indices(row.get("sampled_frame_indices", "0"))
            if self.frame_sampling_mode in {"anchor", "first", "single"}:
                indices = indices[:1]
            for frame_idx in indices:
                self.items.append({"row": row, "frame_idx": frame_idx})
        self.labels = [self.label_mapper.to_index(item["row"]["label"]) for item in self.items]
        self._decode_fail_warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.items)

    def _fallback_frame(self, row: dict[str, str]) -> np.ndarray:
        width = int(float(row.get("width", 0.0) or 0.0))
        height = int(float(row.get("height", 0.0) or 0.0))
        if width <= 0 or height <= 0:
            width = self.image_size
            height = self.image_size
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _apply_region_dropout(self, image_rgb: np.ndarray) -> np.ndarray:
        cfg = self.region_dropout_cfg
        if not self.train_mode or not bool(cfg.get("enabled", False)):
            return image_rgb
        if random.random() > float(cfg.get("p", 0.0)):
            return image_rgb

        mode = str(cfg.get("mode", "random_box")).lower()
        fill_mode = str(cfg.get("fill_mode", "median"))

        if mode in {"same_area_blackout", "face_blackout"}:
            detections = self.detector.detect(image_rgb)
            primary = self.detector.select_primary(detections, image_rgb.shape)
            if primary is not None:
                if mode == "same_area_blackout":
                    return random_same_area_blackout(
                        image_rgb,
                        x1=primary.x1,
                        y1=primary.y1,
                        x2=primary.x2,
                        y2=primary.y2,
                        fill_mode=fill_mode,
                    )
                return mask_bbox_region(
                    image_rgb,
                    x1=primary.x1,
                    y1=primary.y1,
                    x2=primary.x2,
                    y2=primary.y2,
                    fill_mode=fill_mode,
                )
            if not bool(cfg.get("fallback_to_random_box", True)):
                return image_rgb

        return apply_random_box_mask_np(
            image_rgb,
            area_ratio_range=tuple(cfg.get("area_ratio_range", [0.08, 0.2])),
            aspect_ratio_range=tuple(cfg.get("aspect_ratio_range", [0.75, 1.5])),
            fill_mode=fill_mode,
        )

    def _load_frame(self, row: dict[str, str], frame_idx: int) -> np.ndarray:
        path = Path(row["path"])
        media_type = row.get("media_type", "video")
        if media_type == "image":
            try:
                image = np.array(Image.open(path).convert("RGB"))
            except Exception:
                image = self._fallback_frame(row)
        else:
            candidate_indices = [frame_idx]
            if frame_idx != 0:
                candidate_indices.append(0)
            frame_count = int(float(row.get("frame_count", 0.0) or 0.0))
            if frame_count > 1:
                candidate_indices.append(max(0, frame_count // 2))
            try:
                frames = read_video_frames_by_indices(path, candidate_indices)
            except Exception:
                frames = []
            if not frames:
                path_key = str(path)
                if path_key not in self._decode_fail_warned:
                    print(f"[WARN] decode fallback used for {path}")
                    self._decode_fail_warned.add(path_key)
                image = self._fallback_frame(row)
            else:
                image = frames[0]
        image = extract_frame_view(
            image_rgb=image,
            detector=self.detector,
            fallback_to_full_frame=self.fallback_to_full_frame,
            view_mode=self.view_mode,
        )
        if image is None:
            raise RuntimeError(f"No valid view for {path} frame {frame_idx}")
        image = self._apply_region_dropout(image)
        image = apply_text_mask_np(image, self.text_mask_cfg)
        image = resize_image(image, self.image_size)
        return image

    def __getitem__(self, index: int):
        item = self.items[index]
        row = item["row"]
        image_np = self._load_frame(row, int(item["frame_idx"]))
        image = Image.fromarray(image_np)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        label = self.label_mapper.to_index(row["label"])
        return {
            "image": image,
            "label": label,
            "video_id": row["video_id"],
            "frame_idx": int(item["frame_idx"]),
            "class_name": row["label"],
            "split_tag": row.get("split_tag", ""),
        }

    def get_labels(self) -> list[int]:
        return self.labels


class VideoManifestSequenceDataset(Dataset):
    def __init__(
        self,
        video_manifest_path: str | Path,
        task_spec: TaskSpec,
        split_tags: tuple[str, ...] = ("train",),
        sequence_length: int = 8,
        sampling: str = "uniform",
        frame_mode: str = "rgb",
        order_mode: str = "preserve",
        train_mode: bool = False,
        preprocess_cfg: dict | None = None,
        transform=None,
    ):
        self.rows = [
            row
            for row in read_video_manifest(video_manifest_path)
            if row.get("split_tag", "") in set(split_tags)
        ]
        self.label_mapper = LabelMapper(task_spec)
        self.transform = transform
        self.sequence_length = int(sequence_length)
        self.sampling = sampling
        self.frame_mode = frame_mode
        self.order_mode = order_mode
        self.train_mode = train_mode
        self.preprocess_cfg = preprocess_cfg or {}
        self.image_size = int(self.preprocess_cfg.get("image_size", 224))
        self.view_mode = self.preprocess_cfg.get("view_mode", "full_frame")
        self.fallback_to_full_frame = bool(self.preprocess_cfg.get("fallback_to_full_frame", True))
        self.text_mask_cfg = self.preprocess_cfg.get("text_mask", {})
        self.detector = build_face_detector(self.preprocess_cfg.get("detector", {"name": "none"}))

        if self.frame_mode not in {"rgb", "frame_diff"}:
            raise ValueError("frame_mode must be one of: rgb, frame_diff")
        if self.order_mode not in {"preserve", "shuffle", "reverse"}:
            raise ValueError("order_mode must be one of: preserve, shuffle, reverse")

        self.samples = []
        for row in self.rows:
            indices = _parse_indices(row.get("sampled_frame_indices", "0"))
            self.samples.append({"row": row, "indices": indices})
        self.labels = [self.label_mapper.to_index(sample["row"]["label"]) for sample in self.samples]
        self._decode_fail_warned: set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _fallback_frame(self, row: dict[str, str]) -> np.ndarray:
        width = int(float(row.get("width", 0.0) or 0.0))
        height = int(float(row.get("height", 0.0) or 0.0))
        if width <= 0 or height <= 0:
            width = self.image_size
            height = self.image_size
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _select_indices(self, indices: list[int]) -> list[int]:
        n_frames = len(indices)
        if n_frames >= self.sequence_length:
            if self.sampling == "random" and self.train_mode:
                chosen = np.sort(np.random.choice(n_frames, size=self.sequence_length, replace=False))
                selected = [indices[int(i)] for i in chosen.tolist()]
            elif self.sampling == "head":
                selected = indices[: self.sequence_length]
            else:
                chosen = np.linspace(0, n_frames - 1, num=self.sequence_length)
                selected = [indices[int(round(i))] for i in chosen.tolist()]
        else:
            selected = list(indices) + [indices[-1]] * (self.sequence_length - n_frames)

        if self.order_mode == "reverse":
            selected = list(reversed(selected))
        elif self.order_mode == "shuffle" and len(selected) > 1:
            rng = np.random.default_rng()
            selected = rng.permutation(selected).tolist()
        return selected

    def _load_frames(self, row: dict[str, str], frame_indices: list[int]) -> list[np.ndarray]:
        path = Path(row["path"])
        media_type = row.get("media_type", "video")
        if media_type == "image":
            try:
                image = np.array(Image.open(path).convert("RGB"))
            except Exception:
                image = self._fallback_frame(row)
            frames = [image for _ in frame_indices]
        else:
            try:
                frames = read_video_frames_by_indices(path, frame_indices)
            except Exception:
                frames = []
            if not frames:
                path_key = str(path)
                if path_key not in self._decode_fail_warned:
                    print(f"[WARN] sequence decode fallback used for {path}")
                    self._decode_fail_warned.add(path_key)
                frames = [self._fallback_frame(row) for _ in frame_indices]
            elif len(frames) < len(frame_indices):
                frames.extend([frames[-1]] * (len(frame_indices) - len(frames)))
        processed = []
        for frame in frames:
            view = extract_frame_view(
                image_rgb=frame,
                detector=self.detector,
                fallback_to_full_frame=self.fallback_to_full_frame,
                view_mode=self.view_mode,
            )
            if view is None:
                view = frame
            view = apply_text_mask_np(view, self.text_mask_cfg)
            processed.append(resize_image(view, self.image_size))
        return processed

    def __getitem__(self, index: int):
        sample = self.samples[index]
        row = sample["row"]
        selected = self._select_indices(sample["indices"])
        frames_np = self._load_frames(row, selected)
        valid_length = min(len(sample["indices"]), self.sequence_length)

        if self.frame_mode == "frame_diff":
            diff_frames = []
            prev = None
            for current in frames_np:
                if prev is None:
                    diff = np.zeros_like(current, dtype=np.uint8)
                else:
                    diff = np.abs(current.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
                diff_frames.append(diff)
                prev = current
            frames_np = diff_frames

        frames = []
        for image_np in frames_np:
            image = Image.fromarray(image_np)
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            frames.append(image)
        video_tensor = torch.stack(frames, dim=0)
        label = self.label_mapper.to_index(row["label"])
        return {
            "video": video_tensor,
            "label": label,
            "video_id": row["video_id"],
            "length": valid_length,
            "class_name": row["label"],
            "split_tag": row.get("split_tag", ""),
        }

    def get_labels(self) -> list[int]:
        return self.labels
