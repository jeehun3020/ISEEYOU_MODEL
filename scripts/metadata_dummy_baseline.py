#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import LabelMapper, build_task_spec
from iseeyou.data.adapters import collect_samples_from_config
from iseeyou.data.manifest import read_manifest
from iseeyou.utils.metrics import compute_classification_metrics


FEATURE_NAMES = [
    "is_video",
    "is_image",
    "width",
    "height",
    "aspect_ratio",
    "log_file_size",
    "duration_sec",
    "fps",
    "frame_count",
    "bitrate_kbps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata-only dummy baseline to detect protocol leakage")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--output-json", default="", help="Optional output path")
    parser.add_argument("--cache-csv", default="", help="Optional metadata cache CSV path")
    return parser.parse_args()


def _safe_float(value: float | int) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def extract_media_metadata(path: Path, media_type: str) -> dict[str, float]:
    file_size = float(path.stat().st_size) if path.exists() else 0.0

    if media_type == "image":
        with Image.open(path) as img:
            width, height = img.size
        aspect_ratio = float(width) / float(height) if height else 0.0
        return {
            "is_video": 0.0,
            "is_image": 1.0,
            "width": float(width),
            "height": float(height),
            "aspect_ratio": aspect_ratio,
            "log_file_size": math.log1p(file_size),
            "duration_sec": 0.0,
            "fps": 0.0,
            "frame_count": 1.0,
            "bitrate_kbps": 0.0,
        }

    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    width = _safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = _safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = frame_count / fps if fps > 0.0 else 0.0
    bitrate_kbps = (file_size * 8.0 / 1000.0) / duration_sec if duration_sec > 0.0 else 0.0
    aspect_ratio = width / height if height > 0.0 else 0.0

    return {
        "is_video": 1.0,
        "is_image": 0.0,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "log_file_size": math.log1p(file_size),
        "duration_sec": duration_sec,
        "fps": fps,
        "frame_count": frame_count,
        "bitrate_kbps": bitrate_kbps,
    }


def build_split_lookup(manifests_dir: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        rows = read_manifest(manifests_dir / f"{split}.csv")
        for row in rows:
            lookup.setdefault(row["video_id"], split)
    return lookup


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    task_spec = build_task_spec(config["task"])
    label_mapper = LabelMapper(task_spec)

    manifests_dir = Path(config["paths"]["manifests_dir"])
    split_lookup = build_split_lookup(manifests_dir)
    raw_samples = collect_samples_from_config(config["datasets"])

    rows = []
    for sample in raw_samples:
        split = split_lookup.get(sample.video_id)
        if not split:
            continue
        try:
            meta = extract_media_metadata(sample.path, sample.media_type)
        except Exception:
            continue
        row = {
            "split": split,
            "video_id": sample.video_id,
            "dataset": sample.dataset,
            "class_name": sample.class_name,
            "path": str(sample.path),
        }
        row.update(meta)
        rows.append(row)

    if not rows:
        raise RuntimeError("No metadata rows extracted")

    if args.cache_csv:
        cache_path = Path(args.cache_csv)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "video_id", "dataset", "class_name", "path"] + FEATURE_NAMES,
            )
            writer.writeheader()
            writer.writerows(rows)

    X = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.array([label_mapper.to_index(row["class_name"]) for row in rows], dtype=np.int64)
    splits = np.array([row["split"] for row in rows])

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(config.get("seed", 42)),
                ),
            ),
        ]
    )
    clf.fit(X[train_mask], y[train_mask])

    result = {
        "config": args.config,
        "feature_names": FEATURE_NAMES,
        "num_rows": int(len(rows)),
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
    }

    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        if not np.any(mask):
            continue
        probs = clf.predict_proba(X[mask])
        metrics = compute_classification_metrics(y_true=y[mask], y_prob=probs, num_classes=task_spec.num_classes)
        result[split_name] = metrics

    out_path = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
        / f"metadata_dummy_{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[INFO] saved metadata baseline: {out_path}")


if __name__ == "__main__":
    main()
