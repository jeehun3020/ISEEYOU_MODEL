#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.data.adapters import RawSample, collect_samples_from_config
from iseeyou.data.detectors.factory import build_face_detector
from iseeyou.data.split import create_group_splits
from iseeyou.data.video_manifest import write_video_manifest
from iseeyou.utils.video_probe import probe_media_metadata, sample_uniform_frame_indices, summarize_probe

GENERATOR_KEYWORDS = [
    "sora",
    "veo",
    "hailuo",
    "runway",
    "midjourney",
    "kling",
    "minimax",
    "luma",
    "dreamina",
    "pika",
    "genmo",
    "face_swap",
    "faceswap",
    "digital_human",
    "virtual_human",
    "virtual_actor",
    "virtual_influencer",
    "ai_avatar",
    "ai_movie",
    "ai_story",
    "ai_human",
    "generated_human",
]

GENERIC_CREATOR_PATTERNS = [
    re.compile(r"^channel_[0-9a-f]{6,}$"),
    re.compile(r"^source_[0-9a-f]{6,}$"),
    re.compile(r"^identity_[0-9a-f]{6,}$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build protocol-first video_manifest.csv")
    parser.add_argument("--config", required=True, help="Path to protocol yaml")
    parser.add_argument("--output-csv", default="", help="Override output video_manifest.csv")
    parser.add_argument("--output-json", default="", help="Optional summary json path")
    return parser.parse_args()


def normalize_source_url(value: str) -> str:
    return str(value or "").strip().lower()


def load_blacklist_entries(protocol_cfg: dict) -> set[str]:
    blacklist_path = str(protocol_cfg.get("blacklist_paths_file", "")).strip()
    if not blacklist_path:
        return set()
    path = Path(blacklist_path).expanduser()
    if not path.exists():
        print(f"[WARN] blacklist file not found: {path}")
        return set()

    entries: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        entries.add(value)
        entries.add(str(Path(value).expanduser()))
        entries.add(str(Path(value).expanduser().resolve()))
        entries.add(normalize_source_url(value))
    print(f"[INFO] loaded blacklist entries: {len(entries)} from {path}")
    return entries


def infer_split_role(dataset_name: str, cfg: dict) -> str:
    explicit = str(cfg.get("split_role", "")).strip().lower()
    if explicit:
        return explicit
    name = dataset_name.lower()
    if "hardnegative" in name:
        return "stress_real"
    if "hardcase" in name:
        return "stress_generated"
    return "main"


def infer_generator_family(sample: RawSample) -> str:
    if sample.generator_family:
        return sample.generator_family

    tokens = " ".join(
        [
            str(sample.creator_account or ""),
            str(sample.source_id or ""),
            str(sample.original_id or ""),
            str(sample.rel_path or ""),
        ]
    ).lower()
    for keyword in GENERATOR_KEYWORDS:
        if keyword in tokens:
            return keyword
    if sample.class_name == "generated":
        return "unknown_generated"
    return ""


def is_generic_creator(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    return any(pattern.match(normalized) for pattern in GENERIC_CREATOR_PATTERNS)


def canonical_creator(sample: RawSample) -> str:
    creator = str(sample.creator_account or sample.source_id or sample.identity_id or "").strip()
    if not creator:
        return ""
    if is_generic_creator(creator):
        return f"{sample.dataset}::{creator}"
    return creator


def canonical_source(sample: RawSample) -> str:
    source = str(sample.source_id or "").strip()
    if not source:
        return ""
    if is_generic_creator(source):
        return f"{sample.dataset}::{source}"
    return source


def canonical_identity(sample: RawSample) -> str:
    identity = str(sample.identity_id or "").strip()
    if not identity:
        return ""
    if is_generic_creator(identity):
        return f"{sample.dataset}::{identity}"
    return identity


def dedupe_and_resolve_roles(samples: list[RawSample], datasets_cfg: dict) -> tuple[list[tuple[RawSample, str]], list[dict]]:
    buckets: dict[str, list[tuple[RawSample, str]]] = defaultdict(list)
    for sample in samples:
        cfg = datasets_cfg.get(sample.dataset, {})
        role = infer_split_role(sample.dataset, cfg)
        dedupe_key = normalize_source_url(sample.source_url) or f"path::{sample.path.resolve()}"
        buckets[dedupe_key].append((sample, role))

    resolved: list[tuple[RawSample, str]] = []
    conflicts: list[dict] = []
    role_priority = {"stress_real": 0, "stress_generated": 1, "main": 2}

    for dedupe_key, items in buckets.items():
        labels = sorted({sample.class_name for sample, _ in items})
        if len(labels) > 1:
            conflicts.append(
                {
                    "dedupe_key": dedupe_key,
                    "labels": labels,
                    "members": [
                        {
                            "dataset": sample.dataset,
                            "path": str(sample.path),
                            "source_url": sample.source_url,
                            "role": role,
                        }
                        for sample, role in items
                    ],
                }
            )
            continue

        chosen_sample, chosen_role = sorted(items, key=lambda item: role_priority.get(item[1], 9))[0]
        resolved.append((chosen_sample, chosen_role))

    return resolved, conflicts


def enrich_slice_tags(rows: list[dict], protocol_cfg: dict) -> None:
    if not rows:
        return

    main_rows = [row for row in rows if row.get("split_tag") in {"train", "val", "test"}]
    if not main_rows:
        main_rows = rows

    text_values = np.array([float(row.get("text_area_ratio_estimate", 0.0) or 0.0) for row in main_rows], dtype=np.float64)
    motion_values = np.array([float(row.get("motion_score", 0.0) or 0.0) for row in main_rows], dtype=np.float64)

    high_text_threshold = float(np.quantile(text_values, float(protocol_cfg.get("high_text_quantile", 0.67)))) if len(text_values) else 0.0
    low_text_threshold = float(np.quantile(text_values, float(protocol_cfg.get("low_text_quantile", 0.33)))) if len(text_values) else 0.0
    high_motion_threshold = float(np.quantile(motion_values, float(protocol_cfg.get("high_motion_quantile", 0.67)))) if len(motion_values) else 0.0
    low_motion_threshold = float(np.quantile(motion_values, float(protocol_cfg.get("low_motion_quantile", 0.33)))) if len(motion_values) else 0.0
    low_res_shorter_edge = int(protocol_cfg.get("low_res_shorter_edge", 480))
    overlay_text_ratio = float(protocol_cfg.get("overlay_text_ratio", 0.03))

    train_creators = {row.get("creator_account", "") for row in rows if row.get("split_tag") == "train" and row.get("creator_account", "")}
    train_generators = {row.get("generator_family", "") for row in rows if row.get("split_tag") == "train" and row.get("generator_family", "")}
    train_platforms = {row.get("platform_id", "") for row in rows if row.get("split_tag") == "train" and row.get("platform_id", "")}
    train_identity_scene = {
        f"{row.get('identity_id', '')}::{row.get('scene_id', '')}"
        for row in rows
        if row.get("split_tag") == "train"
    }

    for row in rows:
        tags: set[str] = set()
        text_ratio = float(row.get("text_area_ratio_estimate", 0.0) or 0.0)
        motion = float(row.get("motion_score", 0.0) or 0.0)
        face_count = float(row.get("face_count_estimate", 0.0) or 0.0)
        width = int(float(row.get("width", 0.0) or 0.0))
        height = int(float(row.get("height", 0.0) or 0.0))
        shorter = min(width, height) if width and height else 0
        upload_pipeline = str(row.get("upload_pipeline", "")).lower()

        if high_text_threshold > low_text_threshold:
            if text_ratio >= high_text_threshold:
                tags.add("high-text")
            elif text_ratio <= low_text_threshold:
                tags.add("low-text")
        if face_count >= 2.0:
            tags.add("multi-face")
        if high_motion_threshold > low_motion_threshold:
            if motion >= high_motion_threshold:
                tags.add("high-motion")
            elif motion <= low_motion_threshold:
                tags.add("low-motion")
        if shorter and shorter <= low_res_shorter_edge:
            tags.add("low-res")
        if text_ratio >= overlay_text_ratio:
            tags.add("overlayed")
        if any(token in upload_pipeline for token in ["reencode", "recapture", "screen", "re-encode"]):
            tags.add("re-encoded")

        split_tag = row.get("split_tag", "")
        if split_tag in {"val", "test", "stress_real", "stress_generated"}:
            creator = row.get("creator_account", "")
            if creator and creator not in train_creators:
                tags.add("creator-heldout")
            generator = row.get("generator_family", "")
            if generator and generator not in train_generators:
                tags.add("generator-heldout")
            platform = row.get("platform_id", "")
            if platform and platform not in train_platforms:
                tags.add("platform-heldout")
            identity_scene = f"{row.get('identity_id', '')}::{row.get('scene_id', '')}"
            if identity_scene not in train_identity_scene:
                tags.add("identity-scene-heldout")

        row["slice_tags"] = ";".join(sorted(tags))


def build_row(sample: RawSample, split_tag: str, sampled_frames: int, detector, fast_manifest: bool = False) -> dict:
    metadata = probe_media_metadata(sample.path, sample.media_type)
    if sample.media_type == "image":
        indices = [0]
    else:
        frame_count = int(round(float(metadata.get("frame_count", 0.0) or 0.0)))
        indices = sample_uniform_frame_indices(frame_count, sampled_frames)
    if fast_manifest:
        width = int(float(metadata.get("width", 0.0) or 0.0))
        height = int(float(metadata.get("height", 0.0) or 0.0))
        resolution = f"{width}x{height}" if width and height else ""
        summary = {
            "resolution": resolution,
            "width": width,
            "height": height,
            "fps": float(metadata.get("fps", 0.0) or 0.0),
            "duration": float(metadata.get("duration", 0.0) or 0.0),
            "aspect_ratio": float(metadata.get("aspect_ratio", 0.0) or 0.0),
            "bitrate_kbps": float(metadata.get("bitrate_kbps", 0.0) or 0.0),
            "file_size_bytes": int(float(metadata.get("file_size_bytes", 0.0) or 0.0)),
            "frame_count": int(float(metadata.get("frame_count", 0.0) or 0.0)),
            "sampled_frame_indices": ",".join(str(i) for i in indices),
            "face_count_estimate": 0.0,
            "text_area_ratio_estimate": 0.0,
            "motion_score": 0.0,
        }
    else:
        try:
            summary = summarize_probe(sample, indices, detector)
        except Exception:
            width = int(float(metadata.get("width", 0.0) or 0.0))
            height = int(float(metadata.get("height", 0.0) or 0.0))
            resolution = f"{width}x{height}" if width and height else ""
            summary = {
                "resolution": resolution,
                "width": width,
                "height": height,
                "fps": float(metadata.get("fps", 0.0) or 0.0),
                "duration": float(metadata.get("duration", 0.0) or 0.0),
                "aspect_ratio": float(metadata.get("aspect_ratio", 0.0) or 0.0),
                "bitrate_kbps": float(metadata.get("bitrate_kbps", 0.0) or 0.0),
                "file_size_bytes": int(float(metadata.get("file_size_bytes", 0.0) or 0.0)),
                "frame_count": int(float(metadata.get("frame_count", 0.0) or 0.0)),
                "sampled_frame_indices": ",".join(str(i) for i in indices),
                "face_count_estimate": 0.0,
                "text_area_ratio_estimate": 0.0,
                "motion_score": 0.0,
            }

    creator = canonical_creator(sample)
    channel_id = creator
    creator_id = creator
    source_id = canonical_source(sample)
    identity_id = canonical_identity(sample)
    generator_family = infer_generator_family(sample)

    return {
        "video_id": sample.video_id,
        "dataset": sample.dataset,
        "label": sample.class_name,
        "class_name": sample.class_name,
        "media_type": sample.media_type,
        "path": str(sample.path),
        "source_url": sample.source_url,
        "creator_id": creator_id,
        "creator_account": creator,
        "channel_id": channel_id,
        "source_family": sample.source_family,
        "generator_family": generator_family,
        "raw_asset_group": sample.raw_asset_group,
        "upload_pipeline": sample.upload_pipeline,
        "platform_id": sample.platform_id,
        "resolution": summary.get("resolution", ""),
        "width": int(float(summary.get("width", 0.0) or 0.0)),
        "height": int(float(summary.get("height", 0.0) or 0.0)),
        "fps": float(summary.get("fps", 0.0) or 0.0),
        "duration": float(summary.get("duration", 0.0) or 0.0),
        "aspect_ratio": float(summary.get("aspect_ratio", 0.0) or 0.0),
        "bitrate_kbps": float(summary.get("bitrate_kbps", 0.0) or 0.0),
        "file_size_bytes": int(float(summary.get("file_size_bytes", 0.0) or 0.0)),
        "face_count_estimate": float(summary.get("face_count_estimate", 0.0) or 0.0),
        "text_area_ratio_estimate": float(summary.get("text_area_ratio_estimate", 0.0) or 0.0),
        "motion_score": float(summary.get("motion_score", 0.0) or 0.0),
        "frame_count": int(float(summary.get("frame_count", 0.0) or 0.0)),
        "sampled_frame_indices": str(summary.get("sampled_frame_indices", "")),
        "identity_id": identity_id,
        "source_id": source_id,
        "original_id": sample.original_id,
        "template_id": sample.template_id,
        "prompt_id": sample.prompt_id,
        "scene_id": sample.scene_id or identity_id,
        "split_tag": split_tag,
        "slice_tags": "",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets_cfg = config["datasets"]
    sampled_frames = int(config.get("protocol", {}).get("sampled_frames", config.get("preprocess", {}).get("max_frames_per_video", 6)))
    fast_manifest = bool(config.get("protocol", {}).get("fast_manifest", False))
    blacklist_entries = load_blacklist_entries(config.get("protocol", {}))

    raw_samples = collect_samples_from_config(datasets_cfg)
    if blacklist_entries:
        filtered_samples: list[RawSample] = []
        skipped_blacklist: list[dict] = []
        for sample in raw_samples:
            candidates = {
                str(sample.path),
                str(sample.path.resolve()),
                normalize_source_url(sample.source_url),
            }
            if any(candidate in blacklist_entries for candidate in candidates if candidate):
                skipped_blacklist.append(
                    {
                        "dataset": sample.dataset,
                        "path": str(sample.path),
                        "source_url": sample.source_url,
                    }
                )
                continue
            filtered_samples.append(sample)
        raw_samples = filtered_samples
    else:
        skipped_blacklist = []
    resolved_samples, conflicts = dedupe_and_resolve_roles(raw_samples, datasets_cfg)

    main_samples = [sample for sample, role in resolved_samples if role == "main"]
    stress_samples = [(sample, role) for sample, role in resolved_samples if role != "main"]

    split_cfg = config.get("split", {})
    split_indices = create_group_splits(
        samples=main_samples,
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
        group_priority=split_cfg.get(
            "group_priority",
            ["creator_account+raw_asset_group+generator_family+upload_pipeline", "source_id", "video_id"],
        ),
    )

    detector_cfg = (config.get("preprocess", {}) or {}).get("detector", {"name": "none"})
    detector = build_face_detector(detector_cfg)

    rows: list[dict] = []
    skipped_failures: list[dict] = []
    for split_name, indices in split_indices.items():
        for idx in indices:
            sample = main_samples[idx]
            try:
                rows.append(build_row(sample, split_name, sampled_frames, detector, fast_manifest=fast_manifest))
            except Exception as exc:
                skipped_failures.append(
                    {
                        "video_id": sample.video_id,
                        "dataset": sample.dataset,
                        "path": str(sample.path),
                        "split_tag": split_name,
                        "error": str(exc),
                    }
                )
    for sample, role in stress_samples:
        try:
            rows.append(build_row(sample, role, sampled_frames, detector, fast_manifest=fast_manifest))
        except Exception as exc:
            skipped_failures.append(
                {
                    "video_id": sample.video_id,
                    "dataset": sample.dataset,
                    "path": str(sample.path),
                    "split_tag": role,
                    "error": str(exc),
                }
            )

    enrich_slice_tags(rows, config.get("protocol", {}))

    output_csv = Path(args.output_csv) if args.output_csv else Path(config["paths"]["video_manifest_path"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_video_manifest(rows, output_csv)

    split_counts = Counter(row["split_tag"] for row in rows)
    label_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dataset_by_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        label_by_split[row["split_tag"]][row["label"]] += 1
        dataset_by_split[row["split_tag"]][row["dataset"]] += 1

    summary = {
        "config": args.config,
        "output_csv": str(output_csv),
        "total_input_samples": len(raw_samples),
        "total_resolved_samples": len(resolved_samples),
        "total_conflicts": len(conflicts),
        "total_skipped_failures": len(skipped_failures),
        "total_blacklisted": len(skipped_blacklist),
        "split_counts": dict(split_counts),
        "label_by_split": {k: dict(v) for k, v in label_by_split.items()},
        "dataset_by_split": {k: dict(v) for k, v in dataset_by_split.items()},
        "conflicts": conflicts,
        "blacklisted": skipped_blacklist[:200],
        "skipped_failures": skipped_failures[:200],
    }

    output_json = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(config["paths"].get("protocol_report_dir", "outputs/protocol"))
        / f"video_manifest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "split_counts": summary["split_counts"], "total_conflicts": len(conflicts)}, indent=2))
    print(f"[INFO] saved video manifest: {output_csv}")
    print(f"[INFO] saved manifest summary: {output_json}")


if __name__ == "__main__":
    main()
