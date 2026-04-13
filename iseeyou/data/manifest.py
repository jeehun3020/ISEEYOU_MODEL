from __future__ import annotations

import csv
from pathlib import Path

MANIFEST_COLUMNS = [
    "split",
    "split_tag",
    "dataset",
    "class_name",
    "frame_path",
    "video_id",
    "sample_id",
    "frame_idx",
    "identity_id",
    "source_id",
    "original_id",
    "platform_id",
    "creator_account",
    "generator_family",
    "template_id",
    "prompt_id",
    "scene_id",
    "source_url",
    "source_family",
    "raw_asset_group",
    "upload_pipeline",
]


def write_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            out_row = {column: row.get(column, "") for column in MANIFEST_COLUMNS}
            writer.writerow(out_row)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    return rows
