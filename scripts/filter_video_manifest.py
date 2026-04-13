#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an existing video_manifest.csv")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--blacklist-file", default="")
    parser.add_argument("--exclude-dataset", action="append", default=[])
    return parser.parse_args()


def normalize_source_url(value: str) -> str:
    return str(value or "").strip().lower()


def load_blacklist(path_str: str) -> set[str]:
    if not path_str:
        return set()
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {path}")
    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        entries.add(value)
        entries.add(str(Path(value).expanduser()))
        entries.add(str(Path(value).expanduser().resolve()))
        entries.add(normalize_source_url(value))
    return entries


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json else None

    blacklist = load_blacklist(args.blacklist_file)
    excluded_datasets = set(args.exclude_dataset)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No header found in {input_csv}")

        kept_rows: list[dict[str, str]] = []
        dropped_dataset = Counter()
        dropped_blacklist = Counter()
        for row in reader:
            dataset = row.get("dataset", "")
            if dataset in excluded_datasets:
                dropped_dataset[dataset] += 1
                continue
            candidates = {
                row.get("path", ""),
                str(Path(row.get("path", "")).expanduser()),
                normalize_source_url(row.get("source_url", "")),
            }
            if blacklist and any(candidate in blacklist for candidate in candidates if candidate):
                dropped_blacklist[dataset] += 1
                continue
            kept_rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    split_counts = Counter(row.get("split_tag", "") for row in kept_rows)
    label_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dataset_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in kept_rows:
        split = row.get("split_tag", "")
        label_by_split[split][row.get("label", "")] += 1
        dataset_by_split[split][row.get("dataset", "")] += 1

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "input_rows": sum(split_counts.values()) + sum(dropped_dataset.values()) + sum(dropped_blacklist.values()),
        "output_rows": len(kept_rows),
        "excluded_datasets": sorted(excluded_datasets),
        "blacklist_entries": len(blacklist),
        "dropped_dataset": dict(dropped_dataset),
        "dropped_blacklist": dict(dropped_blacklist),
        "split_counts": dict(split_counts),
        "label_by_split": {k: dict(v) for k, v in label_by_split.items()},
        "dataset_by_split": {k: dict(v) for k, v in dataset_by_split.items()},
    }
    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
