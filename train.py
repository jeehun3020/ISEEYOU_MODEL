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
from iseeyou.data.dataset import FaceFrameDataset
from iseeyou.data.protocol_dataset import VideoManifestFrameDataset
from iseeyou.data.transforms import build_eval_transform, build_train_transform
from iseeyou.engine.trainer import fit_model
from iseeyou.models.builder import build_model
from iseeyou.utils.dataloader import resolve_num_workers
from iseeyou.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frame-based classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--video-manifest",
        type=str,
        default="",
        help="Optional protocol video_manifest.csv override",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint path to resume or warm-start from",
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

    training_cfg = config["training"]

    image_size = config["preprocess"]["image_size"]
    augmentation_cfg = training_cfg.get("augmentation", {})
    input_representation = training_cfg.get("input_representation", "rgb")
    video_manifest_path = args.video_manifest or config.get("paths", {}).get("video_manifest_path", "")

    if video_manifest_path:
        train_dataset = VideoManifestFrameDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=("train",),
            preprocess_cfg=config["preprocess"],
            augmentation_cfg=augmentation_cfg,
            train_mode=True,
            transform=build_train_transform(
                image_size,
                aug_cfg=augmentation_cfg,
                input_representation=input_representation,
            ),
        )
        val_dataset = VideoManifestFrameDataset(
            video_manifest_path=video_manifest_path,
            task_spec=task_spec,
            split_tags=("val",),
            preprocess_cfg=config["preprocess"],
            augmentation_cfg=None,
            train_mode=False,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        empty_hint = f"video manifest: {video_manifest_path}"
    else:
        manifests_dir = Path(config["paths"]["manifests_dir"])
        train_manifest = manifests_dir / "train.csv"
        val_manifest = manifests_dir / "val.csv"

        train_dataset = FaceFrameDataset(
            manifest_path=train_manifest,
            task_spec=task_spec,
            transform=build_train_transform(
                image_size,
                aug_cfg=augmentation_cfg,
                input_representation=input_representation,
            ),
        )
        val_dataset = FaceFrameDataset(
            manifest_path=val_manifest,
            task_spec=task_spec,
            transform=build_eval_transform(image_size, input_representation=input_representation),
        )
        empty_hint = f"manifests dir: {manifests_dir}"

    if len(train_dataset) == 0:
        raise ValueError(
            f"Empty training dataset from {empty_hint}. "
            "Check split tags / manifest path / dataset roots."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"Empty validation dataset from {empty_hint}. "
            "Check split tags / manifest path / dataset roots."
        )

    device = resolve_device(training_cfg.get("device", "auto"))
    num_workers = resolve_num_workers(training_cfg["num_workers"])

    pin_memory = device.type == "cuda"
    sampler = None
    shuffle = True
    sampling_cfg = training_cfg.get("sampling", {})
    if bool(sampling_cfg.get("balanced_sampler", False)):
        labels = np.array(train_dataset.get_labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=task_spec.num_classes)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts.astype(np.float64)
        sample_weights = class_weights[labels]
        if len(sample_weights) > 0:
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=int(len(sample_weights)),
                replacement=True,
            )
            shuffle = False
        else:
            print("[WARN] balanced sampler skipped because train sample count is 0.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        pretrained=training_cfg["pretrained"],
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
    ).to(device)

    loss_cfg = training_cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    class_weight_tensor = None
    if bool(loss_cfg.get("use_class_weights", False)):
        labels = np.array(train_dataset.get_labels(), dtype=np.int64)
        class_counts = np.bincount(labels, minlength=task_spec.num_classes).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        class_weight_values = class_counts.sum() / (task_spec.num_classes * class_counts)
        class_weight_tensor = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
        print(f"[INFO] class weights: {class_weight_values.tolist()}")

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        weight=class_weight_tensor,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(training_cfg["epochs"]))

    output_dir = ensure_dir(
        config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints_protocol_frame")
    )
    resume_checkpoint = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(resume_path, map_location=device)
        print(f"[INFO] resuming from checkpoint: {resume_path}")

    result = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_spec=task_spec,
        training_cfg=training_cfg,
        output_dir=output_dir,
        resume_checkpoint=resume_checkpoint,
    )

    print(f"[INFO] training done: {result}")


if __name__ == "__main__":
    main()
