#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.dataset import FaceFrameDataset
from iseeyou.data.protocol_dataset import VideoManifestFrameDataset, VideoManifestSequenceDataset
from iseeyou.data.sequence_dataset import VideoSequenceDataset
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.engine.temporal import load_temporal_model_from_checkpoint
from iseeyou.utils.aggregation import build_video_level_predictions
from iseeyou.utils.dataloader import resolve_num_workers
from iseeyou.utils.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune frame/temporal ensemble weights on a validation split.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--frame-checkpoint", type=str, required=True, help="Path to frame model checkpoint")
    parser.add_argument("--temporal-checkpoint", type=str, required=True, help="Path to temporal model checkpoint")
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--step", type=float, default=0.05, help="Weight sweep step size")
    parser.add_argument(
        "--monitor",
        type=str,
        default="f1",
        choices=["f1", "auc", "accuracy"],
        help="Metric to optimize on the validation split",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional output path")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_frame_predictions(
    config: dict,
    task_spec,
    split: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_size = config["preprocess"]["image_size"]
    eval_cfg = config.get("evaluation", {})
    training_cfg = config["training"]
    input_representation = eval_cfg.get(
        "input_representation",
        training_cfg.get("input_representation", "rgb"),
    )
    num_workers = resolve_num_workers(training_cfg["num_workers"])
    video_manifest_path = config.get("paths", {}).get("video_manifest_path", "")
    if video_manifest_path:
        dataset = VideoManifestFrameDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=(split,),
            preprocess_cfg=config["preprocess"],
            augmentation_cfg=None,
            train_mode=False,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
    else:
        manifests_dir = Path(config["paths"]["manifests_dir"])
        manifest_path = manifests_dir / f"{split}.csv"
        dataset = FaceFrameDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
    loader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )

    model.eval()
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    video_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            video_ids.extend(list(batch["video_id"]))

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    return y_true, y_prob, video_ids


def collect_temporal_predictions(
    config: dict,
    task_spec,
    split: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_size = config["preprocess"]["image_size"]
    temporal_cfg = config.get("temporal", {})
    dataset_cfg = temporal_cfg.get("dataset", {})
    model_cfg = temporal_cfg.get("model", {})
    training_cfg = temporal_cfg.get("training", config["training"])
    input_representation = temporal_cfg.get(
        "input_representation",
        training_cfg.get("input_representation", config["training"].get("input_representation", "rgb")),
    )
    num_workers = resolve_num_workers(int(training_cfg.get("num_workers", 2)))

    video_manifest_path = config.get("paths", {}).get("video_manifest_path", "")
    if video_manifest_path:
        dataset = VideoManifestSequenceDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=(split,),
            sequence_length=int(dataset_cfg.get("sequence_length", 8)),
            sampling="uniform",
            frame_mode=dataset_cfg.get("frame_mode", "rgb"),
            order_mode=dataset_cfg.get("order_mode", "preserve"),
            train_mode=False,
            preprocess_cfg=config["preprocess"],
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
    else:
        manifests_dir = Path(config["paths"]["manifests_dir"])
        manifest_path = manifests_dir / f"{split}.csv"
        dataset = VideoSequenceDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            sequence_length=int(dataset_cfg.get("sequence_length", 8)),
            sampling="uniform",
            min_frames_per_video=int(dataset_cfg.get("min_frames_per_video", 1)),
            frame_mode=dataset_cfg.get("frame_mode", "rgb"),
            train_mode=False,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, _ = load_temporal_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        num_classes=task_spec.num_classes,
        device=device,
    )

    model.eval()
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    video_ids: list[str] = []

    with torch.no_grad():
        for batch in loader:
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            logits = model(videos, lengths=lengths)
            probs = torch.softmax(logits, dim=1)

            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            video_ids.extend(list(batch["video_id"]))

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    return y_true, y_prob, video_ids


def build_frame_video_probs(config: dict, y_true: np.ndarray, y_prob: np.ndarray, video_ids: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    eval_cfg = config.get("evaluation", {})
    method = eval_cfg.get("video_aggregation", "mean")
    topk_ratio = float(eval_cfg.get("topk_ratio", 0.5))
    conf_power = float(eval_cfg.get("conf_power", 2.0))
    y_true_video, y_prob_video = build_video_level_predictions(
        video_ids=video_ids,
        y_true=y_true,
        y_prob=y_prob,
        method=method,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )
    video_ids_sorted = sorted(set(video_ids))
    return y_true_video, y_prob_video, video_ids_sorted


def align_modalities(
    frame_true: np.ndarray,
    frame_prob: np.ndarray,
    frame_video_ids: list[str],
    temporal_true: np.ndarray,
    temporal_prob: np.ndarray,
    temporal_video_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    frame_map = {
        vid: (int(label), prob)
        for vid, label, prob in zip(frame_video_ids, frame_true, frame_prob)
    }
    temporal_map = {
        vid: (int(label), prob)
        for vid, label, prob in zip(temporal_video_ids, temporal_true, temporal_prob)
    }

    common_ids = sorted(set(frame_map) & set(temporal_map))
    if not common_ids:
        raise ValueError("No overlapping video ids between frame and temporal predictions")

    y_true = []
    frame_aligned = []
    temporal_aligned = []
    for vid in common_ids:
        frame_label, frame_probs = frame_map[vid]
        temporal_label, temporal_probs = temporal_map[vid]
        if frame_label != temporal_label:
            raise ValueError(f"Mismatched labels for video_id={vid}: frame={frame_label}, temporal={temporal_label}")
        y_true.append(frame_label)
        frame_aligned.append(frame_probs)
        temporal_aligned.append(temporal_probs)

    return (
        np.array(y_true, dtype=np.int64),
        np.array(frame_aligned, dtype=np.float64),
        np.array(temporal_aligned, dtype=np.float64),
        common_ids,
    )


def evaluate_weights(
    y_true: np.ndarray,
    frame_prob: np.ndarray,
    temporal_prob: np.ndarray,
    num_classes: int,
    frame_weight: float,
) -> dict:
    temporal_weight = 1.0 - frame_weight
    ensemble_prob = frame_weight * frame_prob + temporal_weight * temporal_prob
    ensemble_prob = ensemble_prob / np.clip(ensemble_prob.sum(axis=1, keepdims=True), 1e-8, None)
    metrics = compute_classification_metrics(y_true=y_true, y_prob=ensemble_prob, num_classes=num_classes)
    return {
        "frame_weight": frame_weight,
        "temporal_weight": temporal_weight,
        "metrics": metrics,
        "ensemble_prob": ensemble_prob,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    task_spec = build_task_spec(config["task"])
    device = resolve_device(config["training"].get("device", "auto"))

    frame_ckpt = Path(args.frame_checkpoint)
    temporal_ckpt = Path(args.temporal_checkpoint)

    frame_true_val, frame_prob_val_frame, frame_ids_val_frame = collect_frame_predictions(
        config, task_spec, args.val_split, frame_ckpt, device
    )
    frame_true_val_video, frame_prob_val_video, frame_video_ids_val = build_frame_video_probs(
        config, frame_true_val, frame_prob_val_frame, frame_ids_val_frame
    )
    temporal_true_val, temporal_prob_val, temporal_video_ids_val = collect_temporal_predictions(
        config, task_spec, args.val_split, temporal_ckpt, device
    )
    y_true_val, frame_val_aligned, temporal_val_aligned, _ = align_modalities(
        frame_true_val_video,
        frame_prob_val_video,
        frame_video_ids_val,
        temporal_true_val,
        temporal_prob_val,
        temporal_video_ids_val,
    )

    frame_true_test, frame_prob_test_frame, frame_ids_test_frame = collect_frame_predictions(
        config, task_spec, args.test_split, frame_ckpt, device
    )
    frame_true_test_video, frame_prob_test_video, frame_video_ids_test = build_frame_video_probs(
        config, frame_true_test, frame_prob_test_frame, frame_ids_test_frame
    )
    temporal_true_test, temporal_prob_test, temporal_video_ids_test = collect_temporal_predictions(
        config, task_spec, args.test_split, temporal_ckpt, device
    )
    y_true_test, frame_test_aligned, temporal_test_aligned, _ = align_modalities(
        frame_true_test_video,
        frame_prob_test_video,
        frame_video_ids_test,
        temporal_true_test,
        temporal_prob_test,
        temporal_video_ids_test,
    )

    candidates = []
    step = max(1e-4, float(args.step))
    weights = np.arange(0.0, 1.0 + step * 0.5, step)
    for frame_weight in weights:
        val_row = evaluate_weights(
            y_true=y_true_val,
            frame_prob=frame_val_aligned,
            temporal_prob=temporal_val_aligned,
            num_classes=task_spec.num_classes,
            frame_weight=float(round(frame_weight, 4)),
        )
        test_row = evaluate_weights(
            y_true=y_true_test,
            frame_prob=frame_test_aligned,
            temporal_prob=temporal_test_aligned,
            num_classes=task_spec.num_classes,
            frame_weight=float(round(frame_weight, 4)),
        )
        candidates.append(
            {
                "frame_weight": val_row["frame_weight"],
                "temporal_weight": val_row["temporal_weight"],
                "val": val_row["metrics"],
                "test": test_row["metrics"],
            }
        )

    candidates.sort(
        key=lambda row: (
            float(row["val"].get(args.monitor, float("-inf"))),
            float(row["val"].get("auc", float("-inf"))),
            float(row["test"].get("f1", float("-inf"))),
        ),
        reverse=True,
    )
    best = candidates[0]

    summary = {
        "monitor": args.monitor,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "frame_checkpoint": str(frame_ckpt),
        "temporal_checkpoint": str(temporal_ckpt),
        "best": best,
        "top5": candidates[:5],
        "frame_only": next(row for row in candidates if row["frame_weight"] == 1.0),
        "temporal_only": next(row for row in candidates if row["frame_weight"] == 0.0),
        "current_config": {
            "frame_weight": float(config.get("ensemble", {}).get("frame_weight", 0.5)),
            "temporal_weight": float(config.get("ensemble", {}).get("temporal_weight", 0.5)),
        },
    }

    output_json = (
        Path(args.output_json)
        if args.output_json
        else ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
        / f"ensemble_weight_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] best ensemble weight: frame={best['frame_weight']:.2f}, temporal={best['temporal_weight']:.2f}")
    print(f"[INFO] best val metrics: {best['val']}")
    print(f"[INFO] best test metrics: {best['test']}")
    print(f"[INFO] saved tuning summary: {output_json}")


if __name__ == "__main__":
    main()
