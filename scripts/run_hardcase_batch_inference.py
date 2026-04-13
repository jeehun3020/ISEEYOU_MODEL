#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_ensemble import (
    VIDEO_EXTENSIONS,
    apply_decision_policy,
    apply_uncertainty_policy,
    extract_face_crops,
    normalize_component_weights,
    resolve_device,
    run_frame_model,
    run_temporal_model,
)
from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import LabelMapper, build_task_spec
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.engine.temporal import load_temporal_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for held-out hard-case Shorts videos")
    parser.add_argument("--input-root", type=str, required=True, help="Root directory containing video files")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--frame-checkpoint", type=str, default="", help="Path to frame model checkpoint")
    parser.add_argument("--temporal-checkpoint", type=str, default="", help="Path to temporal model checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Max number of videos to evaluate")
    parser.add_argument(
        "--expected-label",
        type=str,
        default="",
        help="Optional expected label for rough accuracy reporting",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output summary json path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional per-video result csv path",
    )
    return parser.parse_args()


def list_videos(input_root: Path, limit: int) -> list[Path]:
    videos = sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if limit > 0:
        videos = videos[:limit]
    return videos


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"[ERROR] input root not found: {input_root}")

    task_spec = build_task_spec(config["task"])
    label_mapper = LabelMapper(task_spec)
    training_cfg = config["training"]
    temporal_cfg = config.get("temporal", {})
    temporal_training_cfg = temporal_cfg.get("training", training_cfg)
    temporal_model_cfg = temporal_cfg.get("model", {})
    temporal_dataset_cfg = temporal_cfg.get("dataset", {})
    ensemble_cfg = config.get("ensemble", {})
    inference_cfg = config.get("inference", {})

    device = resolve_device(training_cfg.get("device", "auto"))

    frame_ckpt = (
        Path(args.frame_checkpoint)
        if args.frame_checkpoint
        else Path(config["paths"].get("checkpoints_dir", "outputs/checkpoints")) / "best.pt"
    )
    temporal_ckpt = (
        Path(args.temporal_checkpoint)
        if args.temporal_checkpoint
        else Path(config["paths"].get("temporal_checkpoints_dir", "outputs/checkpoints_temporal"))
        / "best.pt"
    )

    frame_model, _ = load_model_from_checkpoint(
        checkpoint_path=frame_ckpt,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )
    temporal_model, _ = load_temporal_model_from_checkpoint(
        checkpoint_path=temporal_ckpt,
        model_cfg=temporal_model_cfg,
        num_classes=task_spec.num_classes,
        device=device,
    )

    image_size = config["preprocess"]["image_size"]
    frame_transform = build_eval_transform(
        image_size,
        input_representation=ensemble_cfg.get(
            "frame_input_representation",
            inference_cfg.get("input_representation", training_cfg.get("input_representation", "rgb")),
        ),
    )
    temporal_transform = build_eval_transform(
        image_size,
        input_representation=temporal_cfg.get(
            "input_representation",
            temporal_training_cfg.get("input_representation", training_cfg.get("input_representation", "rgb")),
        ),
    )

    frame_aggregation = ensemble_cfg.get("frame_aggregation", inference_cfg.get("aggregation", "confidence_mean"))
    min_conf = float(ensemble_cfg.get("min_confidence", inference_cfg.get("min_confidence", 0.0)))
    topk_ratio = float(ensemble_cfg.get("topk_ratio", inference_cfg.get("topk_ratio", 0.5)))
    conf_power = float(ensemble_cfg.get("conf_power", inference_cfg.get("conf_power", 2.0)))
    temporal_frame_mode = temporal_dataset_cfg.get("frame_mode", "rgb")

    component_weights = normalize_component_weights(
        {
            "frame": float(ensemble_cfg.get("frame_weight", 0.5)),
            "temporal": float(ensemble_cfg.get("temporal_weight", 0.5)),
        }
    )
    decision_policy = ensemble_cfg.get("decision_policy", "argmax")
    fake_threshold = float(ensemble_cfg.get("fake_threshold", 0.45))
    adaptive_temporal_min_prob = float(ensemble_cfg.get("adaptive_temporal_min_prob", 0.35))
    adaptive_ensemble_min_prob = float(ensemble_cfg.get("adaptive_ensemble_min_prob", 0.27))
    adaptive_temporal_frame_gap = float(ensemble_cfg.get("adaptive_temporal_frame_gap", 0.16))
    adaptive_temporal_direct_prob = float(ensemble_cfg.get("adaptive_temporal_direct_prob", 0.48))
    real_idx = label_mapper.class_to_idx.get("real", 0)
    uncertainty_cfg = ensemble_cfg.get("uncertainty", {})
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False))
    uncertain_lower = float(uncertainty_cfg.get("lower_fake_prob", 0.4))
    uncertain_upper = float(uncertainty_cfg.get("upper_fake_prob", 0.6))
    uncertain_margin = float(uncertainty_cfg.get("min_margin", 0.12))

    videos = list_videos(input_root, args.limit)
    if not videos:
        raise SystemExit(f"[ERROR] no video files found under: {input_root}")

    results: list[dict[str, object]] = []
    pred_counter: Counter[str] = Counter()
    success_probs: list[np.ndarray] = []
    failures = 0

    for idx, video_path in enumerate(videos, start=1):
        try:
            crops = extract_face_crops(video_path, config)
            if len(crops) == 0:
                raise RuntimeError("No face crops extracted from video")

            frame_prob, _, _, _, _ = run_frame_model(
                frame_model=frame_model,
                crops=crops,
                transform=frame_transform,
                device=device,
                batch_size=int(inference_cfg.get("batch_size", 32)),
                aggregation=frame_aggregation,
                min_confidence=min_conf,
                topk_ratio=topk_ratio,
                conf_power=conf_power,
            )
            temporal_prob, temporal_indices = run_temporal_model(
                temporal_model=temporal_model,
                crops=crops,
                transform=temporal_transform,
                device=device,
                sequence_length=int(temporal_dataset_cfg.get("sequence_length", 8)),
                frame_mode=temporal_frame_mode,
            )

            ensemble_prob = (
                float(component_weights["frame"]) * frame_prob
                + float(component_weights["temporal"]) * temporal_prob
            )
            ensemble_prob = ensemble_prob / max(1e-8, float(np.sum(ensemble_prob)))
            pred_idx, decision_reason = apply_decision_policy(
                ensemble_prob=ensemble_prob,
                frame_prob=frame_prob,
                temporal_prob=temporal_prob,
                freq_prob=None,
                real_idx=real_idx,
                decision_policy=decision_policy,
                fake_threshold=fake_threshold,
                adaptive_temporal_min_prob=adaptive_temporal_min_prob,
                adaptive_ensemble_min_prob=adaptive_ensemble_min_prob,
                adaptive_temporal_frame_gap=adaptive_temporal_frame_gap,
                adaptive_temporal_direct_prob=adaptive_temporal_direct_prob,
            )
            pred_label = label_mapper.index_to_name(pred_idx)
            uncertain_label, uncertainty_reason = apply_uncertainty_policy(
                ensemble_prob=ensemble_prob,
                real_idx=real_idx,
                enabled=uncertainty_enabled,
                lower_fake_prob=uncertain_lower,
                upper_fake_prob=uncertain_upper,
                min_margin=uncertain_margin,
            )
            if uncertain_label is not None:
                pred_label = uncertain_label
                decision_reason = uncertainty_reason or decision_reason
            pred_counter[pred_label] += 1
            success_probs.append(ensemble_prob)

            result = {
                "video_path": str(video_path),
                "relative_path": str(video_path.relative_to(input_root)),
                "prediction": pred_label,
                "authenticity_score": float(ensemble_prob[real_idx]),
                "fake_score": float(1.0 - ensemble_prob[real_idx]),
                "real_prob": float(ensemble_prob[label_mapper.class_to_idx["real"]]),
                "generated_prob": float(ensemble_prob[label_mapper.class_to_idx["generated"]]),
                "frame_real_prob": float(frame_prob[label_mapper.class_to_idx["real"]]),
                "frame_generated_prob": float(frame_prob[label_mapper.class_to_idx["generated"]]),
                "temporal_real_prob": float(temporal_prob[label_mapper.class_to_idx["real"]]),
                "temporal_generated_prob": float(temporal_prob[label_mapper.class_to_idx["generated"]]),
                "num_face_frames": int(len(crops)),
                "temporal_indices": temporal_indices,
                "decision_reason": decision_reason,
                "status": "ok",
            }
        except Exception as exc:
            failures += 1
            result = {
                "video_path": str(video_path),
                "relative_path": str(video_path.relative_to(input_root)),
                "prediction": "error",
                "authenticity_score": None,
                "fake_score": None,
                "real_prob": None,
                "generated_prob": None,
                "frame_real_prob": None,
                "frame_generated_prob": None,
                "temporal_real_prob": None,
                "temporal_generated_prob": None,
                "num_face_frames": 0,
                "temporal_indices": [],
                "decision_reason": str(exc),
                "status": "error",
            }

        results.append(result)
        if idx % 10 == 0 or idx == len(videos):
            print(f"[INFO] processed {idx}/{len(videos)} videos")

    successful = [r for r in results if r["status"] == "ok"]
    mean_prob = (
        np.mean(np.stack(success_probs, axis=0), axis=0).tolist()
        if success_probs
        else [None] * task_spec.num_classes
    )

    summary: dict[str, object] = {
        "input_root": str(input_root),
        "num_videos": len(videos),
        "num_success": len(successful),
        "num_failures": failures,
        "prediction_counts": dict(pred_counter),
        "prediction_rate": {
            label: (pred_counter[label] / max(1, len(successful)))
            for label in [*task_spec.classes, "uncertain"]
        },
        "mean_probability": {
            task_spec.classes[i]: mean_prob[i] for i in range(task_spec.num_classes)
        },
        "weights": component_weights,
        "uncertainty": {
            "enabled": uncertainty_enabled,
            "lower_fake_prob": uncertain_lower,
            "upper_fake_prob": uncertain_upper,
            "min_margin": uncertain_margin,
        },
        "results": results,
    }

    if args.expected_label:
        expected_label = args.expected_label.strip().lower()
        if expected_label not in [*task_spec.classes, "uncertain"]:
            raise SystemExit(f"[ERROR] invalid expected label: {expected_label}")
        expected_hits = sum(1 for r in successful if r["prediction"] == expected_label)
        summary["expected_label"] = expected_label
        summary["expected_label_rate"] = expected_hits / max(1, len(successful))

    eval_dir = ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.output_json) if args.output_json else eval_dir / f"hardcase_batch_{timestamp}.json"
    out_csv = Path(args.output_csv) if args.output_csv else eval_dir / f"hardcase_batch_{timestamp}.csv"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "relative_path",
        "prediction",
        "authenticity_score",
        "fake_score",
        "real_prob",
        "generated_prob",
        "frame_real_prob",
        "frame_generated_prob",
        "temporal_real_prob",
        "temporal_generated_prob",
        "num_face_frames",
        "decision_reason",
        "status",
        "video_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[INFO] hardcase summary: {summary['prediction_counts']}")
    print(f"[INFO] mean_probability: {summary['mean_probability']}")
    if args.expected_label:
        print(f"[INFO] expected_label_rate({args.expected_label})={summary['expected_label_rate']:.4f}")
    print(f"[INFO] saved hardcase summary json: {out_json}")
    print(f"[INFO] saved hardcase summary csv: {out_csv}")


if __name__ == "__main__":
    main()
