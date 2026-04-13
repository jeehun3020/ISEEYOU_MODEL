#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config


FEATURES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit metadata distribution shift in video_manifest.csv")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-manifest", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def feature_stats(rows: list[dict[str, str]], feature: str) -> dict[str, float]:
    values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in rows], dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def smd(real_values: np.ndarray, fake_values: np.ndarray) -> float:
    if len(real_values) == 0 or len(fake_values) == 0:
        return float("nan")
    real_mean = float(np.mean(real_values))
    fake_mean = float(np.mean(fake_values))
    pooled = float(np.sqrt((np.var(real_values) + np.var(fake_values)) / 2.0))
    if pooled < 1e-12:
        return 0.0
    return (fake_mean - real_mean) / pooled


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = (
        Path(args.video_manifest)
        if args.video_manifest
        else Path(config["paths"]["video_manifest_path"])
    )
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    rows = [row for row in rows if row.get("split_tag", "") in {"train", "val", "test"}]

    report = {
        "config": args.config,
        "video_manifest": str(manifest_path),
        "by_split": {},
    }

    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row.get("split_tag", "") == split]
        real_rows = [row for row in split_rows if row.get("label", "") == "real"]
        fake_rows = [row for row in split_rows if row.get("label", "") == "generated"]
        split_payload = {
            "num_rows": len(split_rows),
            "num_real": len(real_rows),
            "num_generated": len(fake_rows),
            "feature_stats": {},
            "sorted_smd": [],
        }

        smd_rows = []
        for feature in FEATURES:
            real_values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in real_rows], dtype=np.float64)
            fake_values = np.asarray([float(row.get(feature, 0.0) or 0.0) for row in fake_rows], dtype=np.float64)
            feature_payload = {
                "real": feature_stats(real_rows, feature),
                "generated": feature_stats(fake_rows, feature),
                "smd": float(smd(real_values, fake_values)),
            }
            split_payload["feature_stats"][feature] = feature_payload
            smd_rows.append({"feature": feature, "smd": feature_payload["smd"]})

        split_payload["sorted_smd"] = sorted(
            smd_rows,
            key=lambda row: abs(float(row["smd"])) if not np.isnan(row["smd"]) else -1.0,
            reverse=True,
        )
        report["by_split"][split] = split_payload

    protocol_dir = ensure_dir(config["paths"].get("protocol_report_dir", "outputs/protocol"))
    output_json = (
        Path(args.output_json)
        if args.output_json
        else protocol_dir / f"metadata_shift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({split: payload["sorted_smd"][:5] for split, payload in report["by_split"].items()}, indent=2))
    print(f"[INFO] saved metadata shift audit: {output_json}")


if __name__ == "__main__":
    main()
