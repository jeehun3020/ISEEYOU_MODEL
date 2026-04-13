from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.protocol_dataset import VideoManifestSequenceDataset
from iseeyou.data.sequence_dataset import VideoSequenceDataset
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.temporal import evaluate_temporal_loader, load_temporal_model_from_checkpoint
from iseeyou.utils.dataloader import resolve_num_workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint (.pt). Default: paths.temporal_checkpoints_dir/best.pt",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--order-mode",
        type=str,
        default="",
        choices=["", "preserve", "shuffle", "reverse"],
        help="Override temporal.dataset.order_mode for protocol tests.",
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task_spec = build_task_spec(config["task"])
    image_size = config["preprocess"]["image_size"]
    temporal_cfg = config.get("temporal", {})
    dataset_cfg = temporal_cfg.get("dataset", {})
    model_cfg = temporal_cfg.get("model", {})
    training_cfg = temporal_cfg.get("training", config["training"])
    input_representation = temporal_cfg.get(
        "input_representation",
        training_cfg.get("input_representation", config["training"].get("input_representation", "rgb")),
    )
    order_mode = args.order_mode if args.order_mode else dataset_cfg.get("order_mode", "preserve")

    video_manifest_path = args.video_manifest or config.get("paths", {}).get("video_manifest_path", "")
    if video_manifest_path:
        dataset = VideoManifestSequenceDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=(args.split,),
            sequence_length=int(dataset_cfg.get("sequence_length", 8)),
            sampling="uniform",
            frame_mode=dataset_cfg.get("frame_mode", "rgb"),
            order_mode=order_mode,
            train_mode=False,
            preprocess_cfg=config["preprocess"],
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        dataset_hint = str(video_manifest_path)
    else:
        manifests_dir = Path(config["paths"]["manifests_dir"])
        manifest_path = manifests_dir / f"{args.split}.csv"
        dataset = VideoSequenceDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            sequence_length=int(dataset_cfg.get("sequence_length", 8)),
            sampling="uniform",
            min_frames_per_video=int(dataset_cfg.get("min_frames_per_video", 1)),
            frame_mode=dataset_cfg.get("frame_mode", "rgb"),
            order_mode=order_mode,
            train_mode=False,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        dataset_hint = str(manifest_path)
    if len(dataset) == 0:
        raise ValueError(f"Empty temporal evaluation dataset: {dataset_hint}")

    device = resolve_device(training_cfg.get("device", "auto"))
    num_workers = resolve_num_workers(int(training_cfg.get("num_workers", 2)))
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(config["paths"].get("temporal_checkpoints_dir", "outputs/checkpoints_temporal")) / "best.pt"
    )

    model, _ = load_temporal_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        num_classes=task_spec.num_classes,
        device=device,
    )

    metrics = evaluate_temporal_loader(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        num_classes=task_spec.num_classes,
    )

    print(f"[INFO] temporal {args.split} metrics: {metrics}")

    eval_dir = ensure_dir(config["paths"].get("eval_dir", "outputs/eval"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = eval_dir / f"eval_temporal_{args.split}_{timestamp}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[INFO] saved temporal evaluation: {out_path}")


if __name__ == "__main__":
    main()
