from __future__ import annotations

import csv
from pathlib import Path

VIDEO_MANIFEST_COLUMNS = [
    "video_id",
    "dataset",
    "label",
    "class_name",
    "media_type",
    "path",
    "source_url",
    "creator_id",
    "creator_account",
    "channel_id",
    "source_family",
    "generator_family",
    "raw_asset_group",
    "upload_pipeline",
    "platform_id",
    "resolution",
    "width",
    "height",
    "fps",
    "duration",
    "aspect_ratio",
    "bitrate_kbps",
    "file_size_bytes",
    "face_count_estimate",
    "text_area_ratio_estimate",
    "motion_score",
    "frame_count",
    "sampled_frame_indices",
    "identity_id",
    "source_id",
    "original_id",
    "template_id",
    "prompt_id",
    "scene_id",
    "split_tag",
    "slice_tags",
]


def write_video_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in VIDEO_MANIFEST_COLUMNS})


def read_video_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{col: row.get(col, "") for col in VIDEO_MANIFEST_COLUMNS} for row in reader]


def filter_video_manifest(rows: list[dict[str, str]], split_tags: set[str] | None = None) -> list[dict[str, str]]:
    if not split_tags:
        return list(rows)
    return [row for row in rows if row.get("split_tag", "") in split_tags]
