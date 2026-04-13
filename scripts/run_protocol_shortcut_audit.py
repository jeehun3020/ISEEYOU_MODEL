#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

cv2.setNumThreads(0)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

from iseeyou.config import ensure_dir, load_config
from iseeyou.data.video_manifest import read_video_manifest
from iseeyou.utils.video_probe import estimate_text_mask_map_np, read_video_frames_by_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shortcut audits on video_manifest.csv")
    parser.add_argument("--config", required=True, help="Path to protocol yaml")
    parser.add_argument("--video-manifest", default="", help="Override video_manifest.csv path")
    parser.add_argument("--output-json", default="", help="Optional output path")
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    out = []
    for chunk in str(raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out or [0]


def load_anchor_frame(row: dict[str, str]) -> np.ndarray:
    path = Path(row["path"])
    media_type = row.get("media_type", "video")
    if media_type == "image":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    indices = parse_indices(row.get("sampled_frame_indices", "0"))
    frames = read_video_frames_by_indices(path, [indices[0]])
    if not frames:
        raise RuntimeError(f"Could not decode anchor frame from {path}")
    return frames[0]


def downsample_gray(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(-1) / 255.0


def extract_text_crop_feature(image_rgb: np.ndarray) -> np.ndarray:
    mask = estimate_text_mask_map_np(image_rgb)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(32 * 32, dtype=np.float32)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(32 * 32, dtype=np.float32)
    return downsample_gray(crop, (32, 32))


def extract_mask_feature(image_rgb: np.ndarray) -> np.ndarray:
    mask = estimate_text_mask_map_np(image_rgb)
    resized = cv2.resize(mask, (16, 16), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(-1) / 255.0


def train_and_eval(features: np.ndarray, labels: np.ndarray, splits: np.ndarray) -> dict:
    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    train_x = features[train_mask].astype(np.float32)
    train_y = labels[train_mask].astype(np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    x_train = torch.tensor((train_x - mean) / std, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)

    model = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=1e-4)
    pos_count = float(max(1.0, float(train_y.sum())))
    neg_count = float(max(1.0, float(len(train_y) - train_y.sum())))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    out = {}
    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        if not np.any(mask):
            continue
        x_eval = ((features[mask].astype(np.float32) - mean) / std).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.tensor(x_eval, dtype=torch.float32)).squeeze(1).numpy()
        pos_prob = sigmoid_np(logits)
        probs = np.stack([1.0 - pos_prob, pos_prob], axis=1)
        metrics = compute_binary_metrics(labels[mask], probs[:, 1])
        metrics.update(compute_operating_metrics(labels[mask], probs[:, 1]))
        out[split_name] = metrics
    return out


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr = roc_curve_np(y_true, y_score)
    eligible = np.where(fpr <= target_fpr)[0]
    if len(eligible) == 0:
        return 0.0
    return float(np.max(tpr[eligible]))


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(np.mean(y_score[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(acc - conf) * (count / len(y_true))
    return float(ece)


def compute_operating_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "tpr_at_fpr_1pct": tpr_at_fpr(y_true, y_score, 0.01),
        "tpr_at_fpr_0_1pct": tpr_at_fpr(y_true, y_score, 0.001),
        "brier": float(np.mean((y_true.astype(np.float64) - y_score.astype(np.float64)) ** 2)),
        "ece": compute_ece(y_true, y_score),
    }


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = float((tp + tn) / max(1, len(y_true)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2 * precision * recall) / max(1e-12, precision + recall))
    auc = roc_auc_np(y_true, y_score)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true.astype(np.int64)[order]
    y_score = y_score.astype(np.float64)[order]
    pos_total = max(1, int(np.sum(y_true == 1)))
    neg_total = max(1, int(np.sum(y_true == 0)))

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    prev_score = None
    for label, score in zip(y_true, y_score):
        if prev_score is not None and score != prev_score:
            tpr.append(tp / pos_total)
            fpr.append(fp / neg_total)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    tpr.append(tp / pos_total)
    fpr.append(fp / neg_total)
    return np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)


def chance_warning(metrics: dict, leakage_cfg: dict) -> bool:
    auc_threshold = float(leakage_cfg.get("auc_threshold", 0.65))
    acc_threshold = float(leakage_cfg.get("accuracy_threshold", 0.65))
    auc = float(metrics.get("auc", float("nan")))
    acc = float(metrics.get("accuracy", float("nan")))
    return (not np.isnan(auc) and auc >= auc_threshold) or (not np.isnan(acc) and acc >= acc_threshold)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = (
        Path(args.video_manifest)
        if args.video_manifest
        else Path(config["paths"]["video_manifest_path"])
    )
    rows = [row for row in read_video_manifest(manifest_path) if row.get("split_tag", "") in {"train", "val", "test"}]
    if not rows:
        raise SystemExit("[ERROR] No train/val/test rows in video manifest")

    labels = np.array([1 if row.get("label") == "generated" else 0 for row in rows], dtype=np.int64)
    splits = np.array([row.get("split_tag", "") for row in rows])

    metadata_features = []
    mask_features = []
    text_area_features = []
    failures = []

    for row in rows:
        metadata_features.append(
            [
                float(row.get("width", 0.0) or 0.0),
                float(row.get("height", 0.0) or 0.0),
                float(row.get("fps", 0.0) or 0.0),
                float(row.get("duration", 0.0) or 0.0),
                float(row.get("aspect_ratio", 0.0) or 0.0),
                float(row.get("bitrate_kbps", 0.0) or 0.0),
                float(row.get("file_size_bytes", 0.0) or 0.0),
                float(row.get("face_count_estimate", 0.0) or 0.0),
                float(row.get("text_area_ratio_estimate", 0.0) or 0.0),
            ]
        )
        try:
            frame = load_anchor_frame(row)
            mask_features.append(extract_mask_feature(frame))
            text_area_features.append(extract_text_crop_feature(frame))
        except Exception as exc:
            failures.append({"video_id": row.get("video_id", ""), "error": str(exc)})
            mask_features.append(np.zeros(16 * 16, dtype=np.float32))
            text_area_features.append(np.zeros(32 * 32, dtype=np.float32))

    metadata_array = np.asarray(metadata_features, dtype=np.float64)
    mask_array = np.asarray(mask_features, dtype=np.float64)
    text_area_array = np.asarray(text_area_features, dtype=np.float64)

    results = {
        "config": args.config,
        "video_manifest": str(manifest_path),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "feature_shapes": {
            "metadata": list(metadata_array.shape),
            "mask_only": list(mask_array.shape),
            "text_area_only": list(text_area_array.shape),
        },
        "baselines": {},
        "failures": failures[:100],
    }

    leakage_cfg = config.get("protocol", {}).get("leakage_warning", {})
    for name, feature_array in [
        ("metadata_only", metadata_array),
        ("mask_only", mask_array),
        ("text_area_only", text_area_array),
    ]:
        baseline_result = train_and_eval(feature_array, labels, splits)
        baseline_result["leakage_warning"] = any(
            chance_warning(split_metrics, leakage_cfg)
            for split_metrics in baseline_result.values()
            if isinstance(split_metrics, dict)
        )
        results["baselines"][name] = baseline_result

    protocol_dir = ensure_dir(config["paths"].get("protocol_report_dir", "outputs/protocol"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = (
        Path(args.output_json)
        if args.output_json
        else protocol_dir / f"shortcut_audit_{timestamp}.json"
    )
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metrics_csv = protocol_dir / "metrics_summary.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "baseline",
                "split",
                "accuracy",
                "f1",
                "auc",
                "tpr_at_fpr_1pct",
                "tpr_at_fpr_0_1pct",
                "brier",
                "ece",
                "leakage_warning",
            ]
        )
        for baseline_name, baseline_result in results["baselines"].items():
            for split_name in ["val", "test"]:
                split_metrics = baseline_result.get(split_name, {})
                writer.writerow(
                    [
                        baseline_name,
                        split_name,
                        split_metrics.get("accuracy", ""),
                        split_metrics.get("f1", ""),
                        split_metrics.get("auc", ""),
                        split_metrics.get("tpr_at_fpr_1pct", ""),
                        split_metrics.get("tpr_at_fpr_0_1pct", ""),
                        split_metrics.get("brier", ""),
                        split_metrics.get("ece", ""),
                        baseline_result.get("leakage_warning", False),
                    ]
                )

    leakage_md = protocol_dir / "leakage_audit.md"
    leakage_lines = [
        "# Leakage Audit",
        "",
        f"- video_manifest: `{manifest_path}`",
        f"- rows: `{len(rows)}`",
        f"- decode_failures: `{len(failures)}`",
        "",
        "## Shortcut Baselines",
        "",
    ]
    for baseline_name, baseline_result in results["baselines"].items():
        leakage_lines.append(f"### {baseline_name}")
        warning = baseline_result.get("leakage_warning", False)
        leakage_lines.append(f"- leakage_warning: `{warning}`")
        for split_name in ["val", "test"]:
            split_metrics = baseline_result.get(split_name, {})
            if not split_metrics:
                continue
            leakage_lines.append(
                f"- {split_name}: accuracy={split_metrics.get('accuracy', float('nan')):.4f}, "
                f"f1={split_metrics.get('f1', float('nan')):.4f}, "
                f"auc={split_metrics.get('auc', float('nan')):.4f}, "
                f"tpr@1%={split_metrics.get('tpr_at_fpr_1pct', float('nan')):.4f}, "
                f"ece={split_metrics.get('ece', float('nan')):.4f}"
            )
        leakage_lines.append("")
    leakage_md.write_text("\n".join(leakage_lines), encoding="utf-8")

    recommendation_md = protocol_dir / "recommendation.md"
    suspicious = [
        name
        for name, baseline_result in results["baselines"].items()
        if baseline_result.get("leakage_warning", False)
    ]
    recommendation_lines = [
        "# Recommendation",
        "",
        f"- suspicious_shortcuts: `{', '.join(suspicious) if suspicious else 'none'}`",
    ]
    if suspicious:
        recommendation_lines.extend(
            [
                "- decision: `suspicious shortcut`",
                "- next_step: `split / masking / normalization 수정 후에만 학습 재진입`",
            ]
        )
    else:
        recommendation_lines.extend(
            [
                "- decision: `keep`",
                "- next_step: `strict protocol 위에서 ablation 진행 가능`",
            ]
        )
    recommendation_md.write_text("\n".join(recommendation_lines), encoding="utf-8")

    print(json.dumps(results["baselines"], indent=2))
    print(f"[INFO] saved shortcut audit: {output_json}")
    print(f"[INFO] saved metrics summary: {metrics_csv}")
    print(f"[INFO] saved leakage audit: {leakage_md}")
    print(f"[INFO] saved recommendation: {recommendation_md}")


if __name__ == "__main__":
    main()
