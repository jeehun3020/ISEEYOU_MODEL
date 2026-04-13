from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from iseeyou.constants import LabelMapper, TaskSpec

from .manifest import read_manifest


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        task_spec: TaskSpec,
        sequence_length: int = 8,
        sampling: str = "uniform",
        min_frames_per_video: int = 1,
        frame_mode: str = "rgb",
        order_mode: str = "preserve",
        train_mode: bool = False,
        transform=None,
    ):
        self.rows = read_manifest(manifest_path)
        self.transform = transform
        self.train_mode = train_mode
        self.sequence_length = int(sequence_length)
        self.sampling = sampling
        self.min_frames_per_video = int(min_frames_per_video)
        self.frame_mode = frame_mode
        self.order_mode = order_mode
        self.label_mapper = LabelMapper(task_spec)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.frame_mode not in {"rgb", "frame_diff"}:
            raise ValueError("frame_mode must be one of: rgb, frame_diff")
        if self.order_mode not in {"preserve", "shuffle", "reverse"}:
            raise ValueError("order_mode must be one of: preserve, shuffle, reverse")

        grouped: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            grouped.setdefault(row["video_id"], []).append(row)

        self.samples: list[dict] = []
        for video_id, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: int(x.get("frame_idx", 0) or 0))
            if len(items_sorted) < self.min_frames_per_video:
                continue

            class_name = items_sorted[0]["class_name"]
            if any(item["class_name"] != class_name for item in items_sorted):
                # TODO: handle noisy/mixed labels in same video manifest if needed.
                continue

            self.samples.append(
                {
                    "video_id": video_id,
                    "class_name": class_name,
                    "frame_paths": [item["frame_path"] for item in items_sorted],
                }
            )

        self.labels = [self.label_mapper.to_index(sample["class_name"]) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _select_indices(self, n_frames: int) -> np.ndarray:
        if n_frames >= self.sequence_length:
            if self.sampling == "random" and self.train_mode:
                idx = np.sort(np.random.choice(n_frames, size=self.sequence_length, replace=False))
                return idx.astype(np.int64)

            if self.sampling == "head":
                return np.arange(self.sequence_length, dtype=np.int64)

            # default: uniform
            idx = np.linspace(0, n_frames - 1, num=self.sequence_length)
            return np.round(idx).astype(np.int64)

        # Pad short sequences by repeating the last frame index.
        idx = list(range(n_frames))
        idx += [n_frames - 1] * (self.sequence_length - n_frames)
        return np.array(idx, dtype=np.int64)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        frame_paths = sample["frame_paths"]
        n_frames = len(frame_paths)

        selected_idx = self._select_indices(n_frames)
        if self.order_mode == "reverse":
            selected_idx = selected_idx[::-1].copy()
        elif self.order_mode == "shuffle" and len(selected_idx) > 1:
            seed = int(hashlib.md5(sample["video_id"].encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            selected_idx = rng.permutation(selected_idx)

        raw_images = [np.array(Image.open(frame_paths[int(i)]).convert("RGB")) for i in selected_idx]
        if self.frame_mode == "frame_diff":
            processed_images = []
            prev = None
            for current in raw_images:
                if prev is None:
                    diff = np.zeros_like(current, dtype=np.uint8)
                else:
                    diff = np.abs(current.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
                processed_images.append(diff)
                prev = current
        else:
            processed_images = raw_images

        frames = []
        for image_np in processed_images:
            image = Image.fromarray(image_np)
            if self.transform is not None:
                image = self.transform(image)
            else:
                # Fallback to tensor if transform is missing.
                image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            frames.append(image)

        video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
        label = self.label_mapper.to_index(sample["class_name"])
        valid_length = min(n_frames, self.sequence_length)

        return {
            "video": video_tensor,
            "label": label,
            "video_id": sample["video_id"],
            "length": valid_length,
            "class_name": sample["class_name"],
        }

    def get_labels(self) -> list[int]:
        return self.labels
