#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MAIN_SPLITS = {"train", "val", "test"}
BALANCE_FEATURES = ["width", "aspect_ratio", "duration", "bitrate_kbps"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a metadata-balanced video manifest subset")
    parser.add_argument("--input-csv", required=True, help="Input video_manifest.csv")
    parser.add_argument("--output-csv", required=True, help="Balanced output csv")
    parser.add_argument("--summary-json", default="", help="Optional summary json path")
    parser.add_argument("--bins", type=int, default=4, help="Quantile bins per feature")
    parser.add_argument("--seed", type=int, default=42, help="Stable seed for deterministic selection")
    return parser.parse_args()


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_edges(rows: list[dict], feature: str, bins: int) -> list[float]:
    values = np.array([safe_float(row.get(feature, 0.0)) for row in rows], dtype=np.float64)
    if values.size == 0:
        return [0.0, 1.0]
    quantiles = np.linspace(0.0, 1.0, max(2, bins + 1))
    edges = np.quantile(values, quantiles).tolist()
    cleaned = [float(edges[0])]
    for edge in edges[1:]:
        edge = float(edge)
        if edge > cleaned[-1]:
            cleaned.append(edge)
    if len(cleaned) == 1:
        cleaned.append(cleaned[0] + 1.0)
    return cleaned


def bucketize(value: float, edges: list[float]) -> int:
    if len(edges) <= 1:
        return 0
    for idx in range(len(edges) - 1):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == len(edges) - 2:
            if lower <= value <= upper:
                return idx
        if lower <= value < upper:
            return idx
    return len(edges) - 2


def row_bucket_key(row: dict, feature_edges: dict[str, list[float]]) -> tuple[int, ...]:
    return tuple(
        bucketize(safe_float(row.get(feature, 0.0)), feature_edges[feature])
        for feature in BALANCE_FEATURES
    )


def stable_rank(row: dict, seed: int) -> str:
    key = "||".join(
        [
            str(seed),
            str(row.get("video_id", "")),
            str(row.get("source_url", "")),
            str(row.get("path", "")),
        ]
    )
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def split_label_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("split_tag", ""))][str(row.get("label", ""))] += 1
    return {split: dict(counter) for split, counter in counts.items()}


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json) if args.summary_json else None

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    main_rows = [row for row in rows if row.get("split_tag") in MAIN_SPLITS]
    aux_rows = [row for row in rows if row.get("split_tag") not in MAIN_SPLITS]

    feature_edges = {feature: build_edges(main_rows, feature, args.bins) for feature in BALANCE_FEATURES}

    selected_main: list[dict] = []
    bucket_summary: dict[str, dict[str, dict[str, int]]] = {}

    for split in sorted(MAIN_SPLITS):
        split_rows = [row for row in main_rows if row.get("split_tag") == split]
        groups: dict[tuple[int, ...], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

        for row in split_rows:
            label = str(row.get("label", ""))
            bucket = row_bucket_key(row, feature_edges)
            groups[bucket][label].append(row)

        bucket_summary[split] = {}
        for bucket, label_rows in groups.items():
            real_rows = sorted(label_rows.get("real", []), key=lambda row: stable_rank(row, args.seed))
            generated_rows = sorted(label_rows.get("generated", []), key=lambda row: stable_rank(row, args.seed))
            keep = min(len(real_rows), len(generated_rows))
            if keep <= 0:
                continue
            selected_main.extend(real_rows[:keep])
            selected_main.extend(generated_rows[:keep])
            bucket_summary[split][str(bucket)] = {
                "real": keep,
                "generated": keep,
                "dropped_real": max(0, len(real_rows) - keep),
                "dropped_generated": max(0, len(generated_rows) - keep),
            }

    final_rows = selected_main + aux_rows

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "bins": int(args.bins),
        "features": list(BALANCE_FEATURES),
        "input_counts": split_label_counts(rows),
        "output_counts": split_label_counts(final_rows),
        "bucket_summary": bucket_summary,
        "kept_main_rows": len(selected_main),
        "kept_aux_rows": len(aux_rows),
        "dropped_main_rows": max(0, len(main_rows) - len(selected_main)),
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
