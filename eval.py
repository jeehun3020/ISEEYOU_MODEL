from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.dataset import FaceFrameDataset
from iseeyou.data.protocol_dataset import VideoManifestFrameDataset
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.utils.aggregation import build_video_level_predictions
from iseeyou.utils.dataloader import resolve_num_workers
from iseeyou.utils.metrics import compute_classification_metrics


AGG_CHOICES = ["mean", "vote", "confidence_mean", "topk_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint (.pt). Default: paths.checkpoints_dir/best.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "stress_real", "stress_generated"],
    )
    parser.add_argument(
        "--video-aggregation",
        type=str,
        default="",
        choices=[""] + AGG_CHOICES,
        help="Video-level aggregation method. Default uses config.evaluation.video_aggregation",
    )
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
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    model.eval()

    y_true_list = []
    y_prob_list = []
    video_ids: list[str] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            video_ids.extend(list(batch["video_id"]))

    if not y_true_list:
        raise ValueError("No samples found in evaluation loader")

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    avg_loss = float(np.mean(losses)) if losses else float("nan")

    return y_true, y_prob, video_ids, avg_loss


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task_spec = build_task_spec(config["task"])
    image_size = config["preprocess"]["image_size"]
    eval_cfg = config.get("evaluation", {})
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
        manifests_dir = Path(config["paths"]["manifests_dir"])
        manifest_path = manifests_dir / f"{args.split}.csv"
        dataset = FaceFrameDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        dataset_hint = str(manifest_path)

    if len(dataset) == 0:
        raise ValueError(f"Empty evaluation dataset: {dataset_hint}")

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
        else Path(config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints_protocol_frame")) / "best.pt"
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

    y_true, y_prob, video_ids, frame_loss = collect_predictions(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
    )

    frame_metrics = compute_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=task_spec.num_classes,
    )
    frame_metrics["loss"] = frame_loss

    video_agg = args.video_aggregation or eval_cfg.get("video_aggregation", "mean")
    y_true_video, y_prob_video = build_video_level_predictions(
        video_ids=video_ids,
        y_true=y_true,
        y_prob=y_prob,
        method=video_agg,
        topk_ratio=float(eval_cfg.get("topk_ratio", 0.5)),
        conf_power=float(eval_cfg.get("conf_power", 2.0)),
    )
    video_metrics = compute_classification_metrics(
        y_true=y_true_video,
        y_prob=y_prob_video,
        num_classes=task_spec.num_classes,
    )

    metrics = {
        "split": args.split,
        "video_aggregation": video_agg,
        "num_frames": int(len(y_true)),
        "num_videos": int(len(y_true_video)),
        "frame": frame_metrics,
        "video": video_metrics,
    }

    print(f"[INFO] {args.split} frame metrics: {frame_metrics}")
    print(f"[INFO] {args.split} video metrics ({video_agg}): {video_metrics}")

    eval_dir = ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"eval_{args.split}_{timestamp}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[INFO] saved evaluation: {out_path}")


if __name__ == "__main__":
    main()
