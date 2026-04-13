#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import load_config
from iseeyou.constants import build_task_spec
from scripts.tune_ensemble_weights import build_frame_video_probs, collect_frame_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune final dual-frame video pipeline (primary + verifier + uncertainty).")
    parser.add_argument("--primary-config", required=True)
    parser.add_argument("--primary-checkpoint", required=True)
    parser.add_argument("--verifier-config", required=True)
    parser.add_argument("--verifier-checkpoint", required=True)
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--test-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def collect_video_probs(config_path: str, checkpoint_path: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = load_config(config_path)
    task_spec = build_task_spec(cfg["task"])
    device = torch.device("cpu")
    y_true_f, y_prob_f, frame_video_ids = collect_frame_predictions(
        config=cfg,
        task_spec=task_spec,
        split=split,
        checkpoint_path=Path(checkpoint_path),
        device=device,
    )
    y_true_v, y_prob_v, video_ids_v = build_frame_video_probs(
        config=cfg,
        y_true=y_true_f,
        y_prob=y_prob_f,
        video_ids=frame_video_ids,
    )
    return y_true_v.astype(int), y_prob_v[:, 1].astype(np.float64), video_ids_v


def align_components(
    y_true_a: np.ndarray,
    prob_a: np.ndarray,
    ids_a: list[str],
    y_true_b: np.ndarray,
    prob_b: np.ndarray,
    ids_b: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    map_a = {vid: (int(y), float(p)) for vid, y, p in zip(ids_a, y_true_a, prob_a)}
    map_b = {vid: (int(y), float(p)) for vid, y, p in zip(ids_b, y_true_b, prob_b)}
    common_ids = sorted(set(map_a) & set(map_b))
    if not common_ids:
        raise ValueError("No common video ids between primary and verifier predictions")

    y_true = []
    prob_primary = []
    prob_verifier = []
    for vid in common_ids:
        ya, pa = map_a[vid]
        yb, pb = map_b[vid]
        if ya != yb:
            raise ValueError(f"Mismatched labels for video_id={vid}: {ya} vs {yb}")
        y_true.append(ya)
        prob_primary.append(pa)
        prob_verifier.append(pb)
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(prob_primary, dtype=np.float64),
        np.asarray(prob_verifier, dtype=np.float64),
        common_ids,
    )


def apply_policy(
    y_true: np.ndarray,
    primary_fake: np.ndarray,
    verifier_fake: np.ndarray,
    primary_weight: float,
    lower_fake_prob: float,
    upper_fake_prob: float,
    disagreement_gap: float,
) -> dict:
    verifier_weight = 1.0 - primary_weight
    fake_score = (primary_weight * primary_fake) + (verifier_weight * verifier_fake)
    disagreement = np.abs(primary_fake - verifier_fake)

    pred = np.full(shape=y_true.shape, fill_value=-1, dtype=np.int64)
    agreement_mask = disagreement <= disagreement_gap
    pred[(fake_score <= lower_fake_prob) & agreement_mask] = 0
    pred[(fake_score >= upper_fake_prob) & agreement_mask] = 1

    confident_mask = pred >= 0
    coverage = float(np.mean(confident_mask)) if len(pred) else 0.0

    if np.any(confident_mask):
        conf_true = y_true[confident_mask]
        conf_pred = pred[confident_mask]
        f1_confident = float(f1_score(conf_true, conf_pred, average="macro"))
        acc_confident = float(accuracy_score(conf_true, conf_pred))
    else:
        f1_confident = 0.0
        acc_confident = 0.0

    neg_mask = y_true == 0
    pos_mask = y_true == 1
    false_positive_rate = float(np.mean(pred[neg_mask] == 1)) if np.any(neg_mask) else 0.0
    true_positive_rate = float(np.mean(pred[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    uncertain_rate = float(np.mean(pred < 0)) if len(pred) else 1.0
    try:
        auc_all = float(roc_auc_score(y_true, fake_score))
    except ValueError:
        auc_all = float("nan")

    score = (
        (0.45 * f1_confident)
        + (0.15 * acc_confident)
        + (0.15 * coverage)
        + (0.15 * (0.0 if np.isnan(auc_all) else auc_all))
        + (0.10 * true_positive_rate)
        - (0.25 * false_positive_rate)
    )
    return {
        "primary_weight": round(float(primary_weight), 4),
        "verifier_weight": round(float(verifier_weight), 4),
        "lower_fake_prob": round(float(lower_fake_prob), 4),
        "upper_fake_prob": round(float(upper_fake_prob), 4),
        "disagreement_gap": round(float(disagreement_gap), 4),
        "coverage": round(float(coverage), 4),
        "uncertain_rate": round(float(uncertain_rate), 4),
        "f1_confident": round(float(f1_confident), 4),
        "accuracy_confident": round(float(acc_confident), 4),
        "auc_all": round(float(auc_all), 4) if not np.isnan(auc_all) else None,
        "false_positive_rate": round(float(false_positive_rate), 4),
        "true_positive_rate": round(float(true_positive_rate), 4),
        "score": round(float(score), 4),
    }


def main() -> None:
    args = parse_args()

    val_true_p, val_prob_p, val_ids_p = collect_video_probs(args.primary_config, args.primary_checkpoint, args.val_split)
    val_true_v, val_prob_v, val_ids_v = collect_video_probs(args.verifier_config, args.verifier_checkpoint, args.val_split)
    test_true_p, test_prob_p, test_ids_p = collect_video_probs(args.primary_config, args.primary_checkpoint, args.test_split)
    test_true_v, test_prob_v, test_ids_v = collect_video_probs(args.verifier_config, args.verifier_checkpoint, args.test_split)

    y_true_val, primary_val, verifier_val, common_val_ids = align_components(
        val_true_p, val_prob_p, val_ids_p, val_true_v, val_prob_v, val_ids_v
    )
    y_true_test, primary_test, verifier_test, common_test_ids = align_components(
        test_true_p, test_prob_p, test_ids_p, test_true_v, test_prob_v, test_ids_v
    )

    trials = []
    best = None
    for primary_weight in np.arange(0.55, 1.001, 0.05):
        for lower in np.arange(0.15, 0.46, 0.05):
            for upper in np.arange(0.55, 0.86, 0.05):
                if lower >= upper:
                    continue
                for gap in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 1.00]:
                    val_metrics = apply_policy(
                        y_true=y_true_val,
                        primary_fake=primary_val,
                        verifier_fake=verifier_val,
                        primary_weight=float(primary_weight),
                        lower_fake_prob=float(lower),
                        upper_fake_prob=float(upper),
                        disagreement_gap=float(gap),
                    )
                    test_metrics = apply_policy(
                        y_true=y_true_test,
                        primary_fake=primary_test,
                        verifier_fake=verifier_test,
                        primary_weight=float(primary_weight),
                        lower_fake_prob=float(lower),
                        upper_fake_prob=float(upper),
                        disagreement_gap=float(gap),
                    )
                    row = {"val": val_metrics, "test": test_metrics}
                    trials.append(row)
                    if best is None or float(val_metrics["score"]) > float(best["val"]["score"]):
                        best = row

    output = {
        "primary": {
            "config": args.primary_config,
            "checkpoint": args.primary_checkpoint,
        },
        "verifier": {
            "config": args.verifier_config,
            "checkpoint": args.verifier_checkpoint,
        },
        "common_val_videos": len(common_val_ids),
        "common_test_videos": len(common_test_ids),
        "best": best,
        "top_trials": sorted(trials, key=lambda x: float(x["val"]["score"]), reverse=True)[:20],
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["best"], indent=2))
    print(f"[INFO] saved final pipeline tuning: {out_path}")


if __name__ == "__main__":
    main()
