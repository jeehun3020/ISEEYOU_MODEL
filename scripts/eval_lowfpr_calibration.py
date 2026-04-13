#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_curve
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.dataset import FaceFrameDataset
from iseeyou.data.protocol_dataset import VideoManifestFrameDataset
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.utils.aggregation import build_video_level_predictions
from iseeyou.utils.dataloader import resolve_num_workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-FPR operating points and calibration")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--video-aggregation", type=str, default="")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--video-manifest",
        type=str,
        default="",
        help="Optional protocol video_manifest.csv override",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    y_true_list = []
    y_prob_list = []
    video_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1)
            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            video_ids.extend(list(batch["video_id"]))

    if not y_true_list:
        raise ValueError("No samples found in evaluation loader")

    return np.concatenate(y_true_list), np.concatenate(y_prob_list), video_ids


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    eligible = np.where(fpr <= target_fpr)[0]
    if len(eligible) == 0:
        return 0.0
    return float(np.max(tpr[eligible]))


def compute_calibration(y_true: np.ndarray, y_score: np.ndarray, bins: int) -> dict:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    payload_bins = []

    for idx in range(bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        count = int(mask.sum())
        if count == 0:
            payload_bins.append(
                {
                    "bin_index": idx,
                    "lower": float(lo),
                    "upper": float(hi),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_accuracy": None,
                }
            )
            continue
        conf = float(np.mean(y_score[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(acc - conf) * (count / len(y_true))
        payload_bins.append(
            {
                "bin_index": idx,
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_confidence": conf,
                "empirical_accuracy": acc,
            }
        )

    return {
        "ece": float(ece),
        "brier": float(brier_score_loss(y_true, y_score)),
        "reliability_bins": payload_bins,
    }


def summarize_binary_operating_points(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> dict:
    pos_prob = y_prob[:, 1]
    calibration = compute_calibration(y_true, pos_prob, bins=bins)
    return {
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, pos_prob, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, pos_prob, 0.001),
        **calibration,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    task_spec = build_task_spec(config["task"])
    if task_spec.num_classes != 2:
        raise SystemExit("[ERROR] This script currently supports binary classification only.")

    eval_cfg = config.get("evaluation", {})
    image_size = config["preprocess"]["image_size"]
    input_representation = eval_cfg.get(
        "input_representation",
        config["training"].get("input_representation", "rgb"),
    )

    video_manifest_path = args.video_manifest or config.get("paths", {}).get("video_manifest_path", "")
    if video_manifest_path:
        dataset = VideoManifestFrameDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=(args.split,),
            preprocess_cfg=config["preprocess"],
            augmentation_cfg=None,
            train_mode=False,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        dataset_hint = str(video_manifest_path)
    else:
        manifest_path = Path(config["paths"]["manifests_dir"]) / f"{args.split}.csv"
        dataset = FaceFrameDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        dataset_hint = str(manifest_path)
    if len(dataset) == 0:
        raise SystemExit(f"[ERROR] Empty dataset: {dataset_hint}")

    training_cfg = config["training"]
    device = resolve_device(training_cfg.get("device", "auto"))
    num_workers = resolve_num_workers(training_cfg["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["paths"]["checkpoints_dir"]) / "best.pt"
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

    y_true, y_prob, video_ids = collect_predictions(model, loader, device)

    video_agg = args.video_aggregation or eval_cfg.get("video_aggregation", "mean")
    y_true_video, y_prob_video = build_video_level_predictions(
        video_ids=video_ids,
        y_true=y_true,
        y_prob=y_prob,
        method=video_agg,
        topk_ratio=float(eval_cfg.get("topk_ratio", 0.5)),
        conf_power=float(eval_cfg.get("conf_power", 2.0)),
    )

    payload = {
        "split": args.split,
        "video_aggregation": video_agg,
        "num_frames": int(len(y_true)),
        "num_videos": int(len(y_true_video)),
        "frame": summarize_binary_operating_points(y_true, y_prob, bins=args.bins),
        "video": summarize_binary_operating_points(y_true_video, y_prob_video, bins=args.bins),
    }

    eval_dir = ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"operating_{args.split}_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[INFO] saved operating-point report: {out_path}")


if __name__ == "__main__":
    main()
