#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.data.manifest import read_manifest


KEYS_TO_AUDIT = [
    "video_id",
    "sample_id",
    "original_id",
    "identity_id",
    "source_id",
    "platform_id",
    "creator_account",
    "generator_family",
    "template_id",
    "prompt_id",
    "scene_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit split leakage and protocol hygiene")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--output-json", default="", help="Optional output path")
    return parser.parse_args()


def _unique_video_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = row.get("video_id", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _pairwise_overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifests_dir = Path(config["paths"]["manifests_dir"])

    split_rows = {
        split: _unique_video_rows(read_manifest(manifests_dir / f"{split}.csv"))
        for split in ["train", "val", "test"]
    }

    split_stats = {}
    key_sets: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for split, rows in split_rows.items():
        split_stats[split] = {
            "num_videos": len(rows),
            "class_counts": dict(Counter(row.get("class_name", "") for row in rows)),
            "dataset_counts": dict(Counter(row.get("dataset", "") for row in rows)),
        }
        for key in KEYS_TO_AUDIT:
            key_sets[key][split] = {row.get(key, "") for row in rows if row.get(key, "")}

    overlap_report: dict[str, dict[str, int]] = {}
    for key in KEYS_TO_AUDIT:
        train_values = key_sets[key]["train"]
        val_values = key_sets[key]["val"]
        test_values = key_sets[key]["test"]
        overlap_report[key] = {
            "train_val": _pairwise_overlap(train_values, val_values),
            "train_test": _pairwise_overlap(train_values, test_values),
            "val_test": _pairwise_overlap(val_values, test_values),
        }

    group_priority = config.get("split", {}).get("group_priority", [])
    if group_priority:
        split_group_sets: dict[str, set[str]] = {}
        for split, rows in split_rows.items():
            chosen_groups = set()
            for row in rows:
                for field in group_priority:
                    value = row.get(field, "")
                    if value:
                        chosen_groups.add(f"{row.get('dataset','')}|{field}|{value}")
                        break
            split_group_sets[split] = chosen_groups
        overlap_report["resolved_group_priority"] = {
            "train_val": _pairwise_overlap(split_group_sets["train"], split_group_sets["val"]),
            "train_test": _pairwise_overlap(split_group_sets["train"], split_group_sets["test"]),
            "val_test": _pairwise_overlap(split_group_sets["val"], split_group_sets["test"]),
        }

    unavailable_protocol_fields = [
        key
        for key in ["platform_id", "creator_account", "generator_family", "template_id", "prompt_id", "scene_id"]
        if all(not row.get(key, "") for rows in split_rows.values() for row in rows)
    ]

    report = {
        "config": args.config,
        "manifests_dir": str(manifests_dir),
        "group_priority": group_priority,
        "split_stats": split_stats,
        "overlap_report": overlap_report,
        "note": (
            "This audit checks currently stored manifest keys. "
            "Protocol fields such as prompt/template/generator family/creator account "
            "need to be added to adapters/manifests before they can be audited."
        ),
        "unavailable_protocol_fields": unavailable_protocol_fields,
    }

    out_path = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
        / f"split_audit_{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overlap_report"], indent=2))
    print(f"[INFO] saved split audit: {out_path}")


if __name__ == "__main__":
    main()
