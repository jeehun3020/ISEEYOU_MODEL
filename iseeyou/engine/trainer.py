from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from iseeyou.constants import TaskSpec
from iseeyou.utils.metrics import compute_classification_metrics


def _amp_enabled(device: torch.device, requested_amp: bool) -> bool:
    return bool(requested_amp and device.type == "cuda")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip_norm: float = 0.0,
) -> float:
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.nan


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()

    losses = []
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    if not y_true_list:
        return {"loss": math.nan, "accuracy": math.nan, "f1": math.nan, "auc": math.nan}

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)

    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else math.nan
    return metrics


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    task_spec: TaskSpec,
    training_cfg: dict,
    output_dir: str | Path,
    resume_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"

    epochs = int(training_cfg["epochs"])
    monitor_name = training_cfg.get("monitor", "f1")
    amp = _amp_enabled(device, bool(training_cfg.get("amp", False)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch = 1
    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    patience = int(early_stopping_cfg.get("patience", 0) or 0)
    min_delta = float(early_stopping_cfg.get("min_delta", 0.0) or 0.0)
    stale_epochs = 0
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0) or 0.0)

    if resume_checkpoint:
        if "model_state_dict" in resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
        if resume_checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler_state = resume_checkpoint.get("scheduler_state_dict")
            resumed_epoch = int(resume_checkpoint.get("epoch", 0) or 0)
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
            elif resumed_epoch > 0:
                try:
                    scheduler.step(resumed_epoch)
                except Exception:
                    for _ in range(resumed_epoch):
                        scheduler.step()
        if amp and resume_checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

        history = list(resume_checkpoint.get("history", []) or [])
        best_metric = float(resume_checkpoint.get("best_metric", best_metric))
        best_epoch = int(resume_checkpoint.get("best_epoch", best_epoch) or 0)
        stale_epochs = int(resume_checkpoint.get("stale_epochs", stale_epochs) or 0)
        start_epoch = int(resume_checkpoint.get("epoch", 0) or 0) + 1

        if (not np.isfinite(best_metric)) and (output_dir / "best.pt").exists():
            current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
            if "model_state_dict" in best_checkpoint:
                model.load_state_dict(best_checkpoint["model_state_dict"])
                best_val_metrics = evaluate_loader(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=task_spec.num_classes,
                )
                hydrated_best_metric = float(best_val_metrics.get(monitor_name, float("nan")))
                if np.isfinite(hydrated_best_metric):
                    best_metric = hydrated_best_metric
                best_epoch = int(best_checkpoint.get("epoch", best_epoch) or best_epoch)
                print(
                    f"[INFO] hydrated best checkpoint metric from {output_dir / 'best.pt'}: "
                    f"best_epoch={best_epoch} best_{monitor_name}={best_metric:.4f}"
                )
            model.load_state_dict(current_state)

        print(
            f"[INFO] resume enabled: start_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_{monitor_name}={best_metric:.4f} "
            f"stale_epochs={stale_epochs}"
        )

    if start_epoch > epochs:
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history_path": str(history_path),
            "checkpoint_dir": str(output_dir),
        }

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp=amp,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=task_spec.num_classes,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        current_metric = float(val_metrics.get(monitor_name, float("nan")))
        is_improved = not math.isnan(current_metric) and current_metric > (best_metric + min_delta)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if is_improved:
            best_metric = current_metric
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if amp else None,
            "task_spec": asdict(task_spec),
            "training_cfg": training_cfg,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
        }

        torch.save(checkpoint, output_dir / "last.pt")
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if is_improved:
            torch.save(checkpoint, output_dir / "best.pt")

        if patience > 0 and stale_epochs >= patience:
            print(
                f"[INFO] early stopping at epoch={epoch} "
                f"(best_epoch={best_epoch}, best_{monitor_name}={best_metric:.4f})"
            )
            break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history_path": str(history_path),
        "checkpoint_dir": str(output_dir),
    }
