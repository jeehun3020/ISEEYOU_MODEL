#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build download-ready CSVs from youtube_dataset packaged manifests."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to all_videos_with_split.csv inside youtube_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where download-ready CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        "video_id",
        "shorts_url",
        "webpage_url",
        "suggested_label",
        "split",
        "source_group",
        "package_category",
        "index_id",
        "source_path",
        "packaged_video_path",
        "source_value",
        "note",
    ]

    converted: list[dict[str, str]] = []
    for row in rows:
        video_id = (row.get("\ufeffsource_id") or row.get("source_id") or "").strip()
        final_label = (row.get("final_label") or "").strip().lower()
        if final_label == "real":
            suggested_label = "real"
        elif final_label == "fake":
            suggested_label = "generated"
        else:
            suggested_label = ""

        split = (row.get("split") or "").strip()
        package_category = (row.get("package_category") or "").strip()
        source_group = (row.get("source_group") or "").strip()

        converted.append(
            {
                "video_id": video_id,
                "shorts_url": f"https://www.youtube.com/shorts/{video_id}" if video_id else "",
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                "suggested_label": suggested_label,
                "split": split,
                "source_group": source_group,
                "package_category": package_category,
                "index_id": (row.get("index_id") or "").strip(),
                "source_path": (row.get("source_path") or "").strip(),
                "packaged_video_path": (row.get("packaged_video_path") or "").strip(),
                "source_value": package_category or source_group or video_id,
                "note": "derived_from_youtube_dataset_all_videos_with_split",
            }
        )

    all_csv = args.output_dir / "youtube_dataset_all_download_ready.csv"
    with all_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted)

    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in converted if row["split"] == split_name]
        split_csv = args.output_dir / f"youtube_dataset_{split_name}_download_ready.csv"
        with split_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"[INFO] wrote {split_csv} rows={len(split_rows)}")

    print(f"[INFO] wrote {all_csv} rows={len(converted)}")


if __name__ == "__main__":
    main()
