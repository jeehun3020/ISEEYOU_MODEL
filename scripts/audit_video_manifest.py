#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.data.video_manifest import read_video_manifest


KEYS_TO_AUDIT = [
    "video_id",
    "source_url",
    "creator_account",
    "source_id",
    "original_id",
    "identity_id",
    "generator_family",
    "raw_asset_group",
    "upload_pipeline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strict protocol on video_manifest.csv")
    parser.add_argument("--config", required=True, help="Path to protocol yaml")
    parser.add_argument("--video-manifest", default="", help="Override video_manifest.csv path")
    parser.add_argument("--output-json", default="", help="Optional output path")
    return parser.parse_args()


def pairwise_overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def composite_group_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            f"creator_account={row.get('creator_account', '')}",
            f"raw_asset_group={row.get('raw_asset_group', '')}",
            f"generator_family={row.get('generator_family', '')}",
            f"upload_pipeline={row.get('upload_pipeline', '')}",
        ]
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = (
        Path(args.video_manifest)
        if args.video_manifest
        else Path(config["paths"]["video_manifest_path"])
    )
    rows = read_video_manifest(manifest_path)

    split_rows = {
        split: [row for row in rows if row.get("split_tag", "") == split]
        for split in ["train", "val", "test", "stress_real", "stress_generated"]
    }
    main_splits = ["train", "val", "test"]

    split_stats = {}
    slice_counts = {}
    for split, items in split_rows.items():
        split_stats[split] = {
            "num_videos": len(items),
            "label_counts": dict(Counter(row.get("label", "") for row in items)),
            "dataset_counts": dict(Counter(row.get("dataset", "") for row in items)),
        }
        tag_counter = Counter()
        for row in items:
            for tag in filter(None, str(row.get("slice_tags", "")).split(";")):
                tag_counter[tag] += 1
        slice_counts[split] = dict(tag_counter)

    overlap_report: dict[str, dict[str, int]] = {}
    for key in KEYS_TO_AUDIT:
        values = {
            split: {row.get(key, "") for row in split_rows[split] if row.get(key, "")}
            for split in main_splits
        }
        overlap_report[key] = {
            "train_val": pairwise_overlap(values["train"], values["val"]),
            "train_test": pairwise_overlap(values["train"], values["test"]),
            "val_test": pairwise_overlap(values["val"], values["test"]),
        }

    composite_values = {
        split: {composite_group_key(row) for row in split_rows[split] if composite_group_key(row)}
        for split in main_splits
    }
    overlap_report["strict_group_key"] = {
        "train_val": pairwise_overlap(composite_values["train"], composite_values["val"]),
        "train_test": pairwise_overlap(composite_values["train"], composite_values["test"]),
        "val_test": pairwise_overlap(composite_values["val"], composite_values["test"]),
    }

    report = {
        "config": args.config,
        "video_manifest": str(manifest_path),
        "split_stats": split_stats,
        "slice_counts": slice_counts,
        "overlap_report": overlap_report,
    }

    output_json = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(config["paths"].get("protocol_report_dir", "outputs/protocol"))
        / f"audit_video_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overlap_report"], indent=2))
    print(f"[INFO] saved video manifest audit: {output_json}")


if __name__ == "__main__":
    main()
