#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import LabelMapper, build_task_spec
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from inference_ensemble import extract_face_crops, resolve_device, resolve_input_video, run_frame_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final dual-frame video pipeline inference.")
    parser.add_argument("--config", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-path", default="")
    input_group.add_argument("--youtube-url", default="")
    parser.add_argument("--download-dir", default="")
    parser.add_argument("--save-frame-csv", action="store_true")
    parser.add_argument("--policy-json", default="")
    return parser.parse_args()


def load_policy(config: dict, policy_json: str) -> dict:
    final_cfg = config.get("final_pipeline", {})
    policy = dict(final_cfg.get("policy", {}))
    policy_path = policy_json or final_cfg.get("policy_json", "")
    if policy_path:
        path = Path(policy_path)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            best = payload.get("best", {})
            if best:
                val = best.get("val", {})
                policy.update(
                    {
                        "primary_weight": float(val.get("primary_weight", policy.get("primary_weight", 0.75))),
                        "verifier_weight": float(val.get("verifier_weight", policy.get("verifier_weight", 0.25))),
                        "lower_fake_prob": float(val.get("lower_fake_prob", policy.get("lower_fake_prob", 0.20))),
                        "upper_fake_prob": float(val.get("upper_fake_prob", policy.get("upper_fake_prob", 0.80))),
                        "disagreement_gap": float(val.get("disagreement_gap", policy.get("disagreement_gap", 0.25))),
                    }
                )
    policy.setdefault("primary_weight", 0.75)
    policy.setdefault("verifier_weight", 0.25)
    policy.setdefault("lower_fake_prob", 0.20)
    policy.setdefault("upper_fake_prob", 0.80)
    policy.setdefault("disagreement_gap", 0.25)
    return policy


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    final_cfg = config.get("final_pipeline", {})
    primary_cfg = load_config(final_cfg["primary"]["config"])
    verifier_cfg = load_config(final_cfg["verifier"]["config"])
    task_spec = build_task_spec(primary_cfg["task"])
    label_mapper = LabelMapper(task_spec)
    device = resolve_device(primary_cfg.get("training", {}).get("device", "auto"))

    policy = load_policy(config, args.policy_json)
    input_video_path, input_type, source_url = resolve_input_video(args, primary_cfg)
    crops = extract_face_crops(input_video_path, primary_cfg)
    if len(crops) == 0:
        raise RuntimeError("No valid frames extracted from video")

    primary_transform = build_eval_transform(
        primary_cfg["preprocess"]["image_size"],
        input_representation=primary_cfg.get("inference", {}).get(
            "input_representation",
            primary_cfg.get("training", {}).get("input_representation", "rgb"),
        ),
    )
    verifier_transform = build_eval_transform(
        verifier_cfg["preprocess"]["image_size"],
        input_representation=verifier_cfg.get("inference", {}).get(
            "input_representation",
            verifier_cfg.get("training", {}).get("input_representation", "rgb"),
        ),
    )

    primary_model, _ = load_model_from_checkpoint(
        checkpoint_path=Path(final_cfg["primary"]["checkpoint"]),
        backbone=primary_cfg["training"]["backbone"],
        num_classes=task_spec.num_classes,
        dropout=primary_cfg["training"].get("dropout", 0.0),
        freeze_backbone=bool(primary_cfg["training"].get("freeze_backbone", False)),
        hidden_dim=int(primary_cfg["training"].get("hidden_dim", 0) or 0),
        device=device,
    )
    verifier_model, _ = load_model_from_checkpoint(
        checkpoint_path=Path(final_cfg["verifier"]["checkpoint"]),
        backbone=verifier_cfg["training"]["backbone"],
        num_classes=task_spec.num_classes,
        dropout=verifier_cfg["training"].get("dropout", 0.0),
        freeze_backbone=bool(verifier_cfg["training"].get("freeze_backbone", False)),
        hidden_dim=int(verifier_cfg["training"].get("hidden_dim", 0) or 0),
        device=device,
    )

    inference_cfg = primary_cfg.get("inference", {})
    aggregation = inference_cfg.get("aggregation", "confidence_mean")
    min_confidence = float(inference_cfg.get("min_confidence", 0.55))
    topk_ratio = float(inference_cfg.get("topk_ratio", 0.5))
    conf_power = float(inference_cfg.get("conf_power", 2.0))
    batch_size = int(inference_cfg.get("batch_size", 32))

    primary_prob, primary_frame_probs, primary_frame_conf, frame_indices, keep_mask = run_frame_model(
        frame_model=primary_model,
        crops=crops,
        transform=primary_transform,
        device=device,
        batch_size=batch_size,
        aggregation=aggregation,
        min_confidence=min_confidence,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )
    verifier_prob, verifier_frame_probs, verifier_frame_conf, _, _ = run_frame_model(
        frame_model=verifier_model,
        crops=crops,
        transform=verifier_transform,
        device=device,
        batch_size=batch_size,
        aggregation=aggregation,
        min_confidence=min_confidence,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )

    fake_idx = label_mapper.class_to_idx.get("generated", 1)
    real_idx = label_mapper.class_to_idx.get("real", 0)
    primary_fake = float(primary_prob[fake_idx])
    verifier_fake = float(verifier_prob[fake_idx])
    final_fake = (
        float(policy["primary_weight"]) * primary_fake
        + float(policy["verifier_weight"]) * verifier_fake
    )
    final_prob = np.zeros_like(primary_prob, dtype=np.float64)
    final_prob[real_idx] = 1.0 - final_fake
    final_prob[fake_idx] = final_fake

    disagreement = abs(primary_fake - verifier_fake)
    if disagreement > float(policy["disagreement_gap"]):
        final_label = "uncertain"
        decision_reason = "model_disagreement"
    elif final_fake >= float(policy["upper_fake_prob"]):
        final_label = "generated"
        decision_reason = "high_fake_score"
    elif final_fake <= float(policy["lower_fake_prob"]):
        final_label = "real"
        decision_reason = "low_fake_score"
    else:
        final_label = "uncertain"
        decision_reason = "midband_uncertainty"

    output = {
        "input": {
            "type": input_type,
            "source_url": source_url,
            "resolved_video_path": str(input_video_path),
        },
        "video_path": str(input_video_path),
        "num_frames": int(len(crops)),
        "policy": {
            "primary_weight": float(policy["primary_weight"]),
            "verifier_weight": float(policy["verifier_weight"]),
            "lower_fake_prob": float(policy["lower_fake_prob"]),
            "upper_fake_prob": float(policy["upper_fake_prob"]),
            "disagreement_gap": float(policy["disagreement_gap"]),
            "decision_reason": decision_reason,
        },
        "components": {
            "primary": {
                "name": final_cfg["primary"].get("name", "primary"),
                "probabilities": primary_prob.tolist(),
                "fake_score": primary_fake,
                "confidence_mean": float(np.mean(primary_frame_conf)),
            },
            "verifier": {
                "name": final_cfg["verifier"].get("name", "verifier"),
                "probabilities": verifier_prob.tolist(),
                "fake_score": verifier_fake,
                "confidence_mean": float(np.mean(verifier_frame_conf)),
            },
        },
        "prediction": {
            "label": final_label,
            "probabilities": final_prob.tolist(),
        },
        "authenticity_score": float(final_prob[real_idx]),
        "fake_score": float(final_fake),
        "disagreement": float(disagreement),
    }

    infer_dir = ensure_dir(primary_cfg["paths"].get("inference_dir", "outputs/inference"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_video_path).stem
    out_json = infer_dir / f"final_pipeline_{stem}_{timestamp}.json"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[INFO] final pipeline prediction: {output['prediction']}")
    print(f"[INFO] saved final inference: {out_json}")

    if args.save_frame_csv:
        out_csv = infer_dir / f"final_pipeline_{stem}_{timestamp}_frames.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_idx",
                    "kept",
                    "primary_real",
                    "primary_generated",
                    "primary_confidence",
                    "verifier_real",
                    "verifier_generated",
                    "verifier_confidence",
                ]
            )
            for idx, frame_idx in enumerate(frame_indices):
                writer.writerow(
                    [
                        frame_idx,
                        int(bool(keep_mask[idx])),
                        float(primary_frame_probs[idx, real_idx]),
                        float(primary_frame_probs[idx, fake_idx]),
                        float(primary_frame_conf[idx]),
                        float(verifier_frame_probs[idx, real_idx]),
                        float(verifier_frame_probs[idx, fake_idx]),
                        float(verifier_frame_conf[idx]),
                    ]
                )
        print(f"[INFO] saved frame csv: {out_csv}")


if __name__ == "__main__":
    main()
