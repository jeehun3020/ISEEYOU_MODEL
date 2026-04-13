#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.utils.youtube import extract_video_id, validate_youtube_url

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
VALID_SPLITS = {"train", "val", "test"}
LABEL_MAP = {"real": "real", "fake": "generated", "generated": "generated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a redownloadable YouTube-only manifest from a mixed final manifest.")
    parser.add_argument("--input-csv", required=True, help="Path to final_experiment_manifest.csv")
    parser.add_argument("--output-clean-csv", required=True, help="Path to cleaned YouTube-only manifest CSV")
    parser.add_argument("--output-reject-csv", required=True, help="Path to rejected rows CSV with reject_reason")
    parser.add_argument("--output-summary-json", required=True, help="Path to summary JSON")
    parser.add_argument(
        "--existing-root",
        default="",
        help="Optional youtube_dataset_split root. If set, writes split-wise missing-only CSVs next to the clean CSV.",
    )
    return parser.parse_args()


def scan_existing(existing_root: Path) -> dict[tuple[str, str], set[str]]:
    existing: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in VALID_SPLITS:
        for label in ("real", "generated"):
            root = existing_root / split / label
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.name.startswith("._"):
                    continue
                if path.suffix.lower() not in VIDEO_EXTS:
                    continue
                existing[(split, label)].add(path.stem)
    return existing


def row_label(row: dict[str, str]) -> str:
    raw = (row.get("final_label") or row.get("suggested_label") or "").strip().lower()
    return LABEL_MAP.get(raw, "")


def build_candidate(row: dict[str, str]) -> tuple[dict[str, str] | None, str]:
    split = (row.get("split") or "").strip().lower()
    if split not in VALID_SPLITS:
        return None, "invalid_split"

    label = row_label(row)
    if label not in {"real", "generated"}:
        return None, "invalid_label"

    url = (row.get("shorts_url") or row.get("webpage_url") or "").strip()
    if not url:
        return None, "missing_url"

    try:
        resolved_url = validate_youtube_url(url)
    except Exception as exc:
        return None, f"invalid_youtube_url:{exc}"

    video_id = extract_video_id(resolved_url).strip()
    if not video_id:
        return None, "missing_video_id"

    return (
        {
            "video_id": video_id,
            "shorts_url": (row.get("shorts_url") or f"https://www.youtube.com/shorts/{video_id}").strip(),
            "webpage_url": (row.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}").strip(),
            "suggested_label": label,
            "split": split,
            "source_type": row.get("source_type", ""),
            "source_value": row.get("source_value", ""),
            "source_name": row.get("source_name", ""),
            "source_group": row.get("source_group", ""),
            "package_category": row.get("package_category", ""),
            "final_label": row.get("final_label", ""),
            "title": row.get("title", ""),
            "uploader": row.get("uploader", ""),
            "channel_url": row.get("channel_url", ""),
            "dataset_name": row.get("dataset_name", "youtube_dataset"),
            "original_video_id_field": row.get("video_id", ""),
            "source_path": row.get("source_path", ""),
            "packaged_video_path": row.get("packaged_video_path", ""),
        },
        "",
    )


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_clean = Path(args.output_clean_csv)
    output_reject = Path(args.output_reject_csv)
    output_summary = Path(args.output_summary_json)
    existing_root = Path(args.existing_root) if args.existing_root else None
    existing = scan_existing(existing_root) if existing_root else {}

    clean_rows: list[dict[str, str]] = []
    reject_rows: list[dict[str, str]] = []
    missing_rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    stats = Counter()
    seen_keys: set[tuple[str, str]] = set()

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows"] += 1
            candidate, reason = build_candidate(row)
            if candidate is None:
                stats[f"reject::{reason}"] += 1
                reject_rows.append({**row, "reject_reason": reason})
                continue

            key = (candidate["split"], candidate["video_id"])
            if key in seen_keys:
                stats["reject::duplicate_video_id_in_split"] += 1
                reject_rows.append({**row, "reject_reason": "duplicate_video_id_in_split"})
                continue
            seen_keys.add(key)

            if existing_root:
                present = candidate["video_id"] in existing.get((candidate["split"], candidate["suggested_label"]), set())
                candidate["already_present"] = "1" if present else "0"
                if not present:
                    missing_rows_by_split[candidate["split"]].append(candidate)
                    stats[f"missing::{candidate['split']}"] += 1
                else:
                    stats[f"present::{candidate['split']}"] += 1

            clean_rows.append(candidate)
            stats["clean_rows"] += 1
            stats[f"clean::{candidate['split']}"] += 1
            stats[f"clean::{candidate['suggested_label']}"] += 1

    clean_fields = [
        "video_id",
        "shorts_url",
        "webpage_url",
        "suggested_label",
        "split",
        "source_type",
        "source_value",
        "source_name",
        "source_group",
        "package_category",
        "final_label",
        "title",
        "uploader",
        "channel_url",
        "dataset_name",
        "original_video_id_field",
        "source_path",
        "packaged_video_path",
    ]
    if existing_root:
        clean_fields.append("already_present")

    reject_fields = sorted({k for row in reject_rows for k in row.keys()} | {"reject_reason"})
    write_csv(output_clean, clean_rows, clean_fields)
    write_csv(output_reject, reject_rows, reject_fields)

    missing_paths: dict[str, str] = {}
    if existing_root:
        for split, rows in missing_rows_by_split.items():
            path = output_clean.parent / f"{output_clean.stem}_{split}_missing_only.csv"
            write_csv(path, rows, clean_fields)
            missing_paths[split] = str(path)

    summary = {
        "input_csv": str(input_csv),
        "output_clean_csv": str(output_clean),
        "output_reject_csv": str(output_reject),
        "existing_root": str(existing_root) if existing_root else "",
        "missing_paths": missing_paths,
        "stats": dict(stats),
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
