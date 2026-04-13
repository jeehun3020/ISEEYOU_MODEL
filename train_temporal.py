from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import build_task_spec
from iseeyou.data.protocol_dataset import VideoManifestSequenceDataset
from iseeyou.data.sequence_dataset import VideoSequenceDataset
from iseeyou.data.transforms import build_eval_transform, build_train_transform
from iseeyou.engine.temporal import fit_temporal_model
from iseeyou.models.temporal import build_temporal_model
from iseeyou.utils.dataloader import resolve_num_workers
from iseeyou.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal video classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
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
    set_seed(config["seed"])

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

    sequence_length = int(dataset_cfg.get("sequence_length", 8))
    sampling = dataset_cfg.get("sampling", "random")
    min_frames = int(dataset_cfg.get("min_frames_per_video", 1))
    frame_mode = dataset_cfg.get("frame_mode", "rgb")
    order_mode = dataset_cfg.get("order_mode", "preserve")
    video_manifest_path = args.video_manifest or config.get("paths", {}).get("video_manifest_path", "")
    train_aug = training_cfg.get("augmentation", {})
    eval_transform = build_eval_transform(image_size, input_representation=input_representation)

    if video_manifest_path:
        train_dataset = VideoManifestSequenceDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=("train",),
            sequence_length=sequence_length,
            sampling=sampling,
            frame_mode=frame_mode,
            order_mode=order_mode,
            train_mode=True,
            preprocess_cfg=config["preprocess"],
            transform=build_train_transform(
                image_size,
                aug_cfg=train_aug,
                input_representation=input_representation,
            ),
        )
        val_dataset = VideoManifestSequenceDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=("val",),
            sequence_length=sequence_length,
            sampling="uniform",
            frame_mode=frame_mode,
            order_mode=order_mode,
            train_mode=False,
            preprocess_cfg=config["preprocess"],
            transform=eval_transform,
        )
        dataset_hint = f"video manifest: {video_manifest_path}"
    else:
        manifests_dir = Path(config["paths"]["manifests_dir"])
        train_manifest = manifests_dir / "train.csv"
        val_manifest = manifests_dir / "val.csv"
        train_dataset = VideoSequenceDataset(
            manifest_path=train_manifest,
            task_spec=task_spec,
            sequence_length=sequence_length,
            sampling=sampling,
            min_frames_per_video=min_frames,
            frame_mode=frame_mode,
            order_mode=order_mode,
            train_mode=True,
            transform=build_train_transform(
                image_size,
                aug_cfg=train_aug,
                input_representation=input_representation,
            ),
        )
        val_dataset = VideoSequenceDataset(
            manifest_path=val_manifest,
            task_spec=task_spec,
            sequence_length=sequence_length,
            sampling="uniform",
            min_frames_per_video=min_frames,
            frame_mode=frame_mode,
            order_mode=order_mode,
            train_mode=False,
            transform=eval_transform,
        )
        dataset_hint = f"manifests dir: {manifests_dir}"

    if len(train_dataset) == 0:
        raise ValueError(
            f"Empty temporal train dataset from {dataset_hint}. "
            "Check video manifest / split tags / dataset roots."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"Empty temporal val dataset from {dataset_hint}. "
            "Check video manifest / split tags / dataset roots."
        )

    device = resolve_device(training_cfg.get("device", "auto"))
    num_workers = resolve_num_workers(int(training_cfg.get("num_workers", 2)))

    pin_memory = device.type == "cuda"
    sampler = None
    shuffle = True
    sampling_cfg = training_cfg.get("sampling", {})
    if bool(sampling_cfg.get("balanced_sampler", True)):
        labels = np.array(train_dataset.get_labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=task_spec.num_classes)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts.astype(np.float64)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_temporal_model(model_cfg=model_cfg, num_classes=task_spec.num_classes).to(device)

    loss_cfg = training_cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    class_weight_tensor = None
    if bool(loss_cfg.get("use_class_weights", True)):
        labels = np.array(train_dataset.get_labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=task_spec.num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        class_weight_values = class_counts.sum() / (task_spec.num_classes * class_counts)
        class_weight_tensor = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
        print(f"[INFO] temporal class weights: {class_weight_values.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        weight=class_weight_tensor,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(training_cfg.get("epochs", 10)))

    output_dir = ensure_dir(config["paths"].get("temporal_checkpoints_dir", "outputs/checkpoints_temporal"))

    result = fit_temporal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_spec=task_spec,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        output_dir=output_dir,
    )

    print(f"[INFO] temporal training done: {result}")


if __name__ == "__main__":
    main()
