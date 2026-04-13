from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import LabelMapper, build_task_spec
from iseeyou.data.detectors.factory import build_face_detector
from iseeyou.data.preprocess import _extract_frame_view
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
from iseeyou.engine.temporal import load_temporal_model_from_checkpoint
from iseeyou.utils.aggregation import aggregate_probs
from iseeyou.utils.masking import apply_text_mask_np
from iseeyou.utils.video import iter_video_frames, resize_image
from iseeyou.utils.youtube import (
    find_downloaded_video_by_url,
    resolve_downloaded_video_path,
    validate_youtube_url,
)


AGG_CHOICES = ["mean", "vote", "confidence_mean", "topk_mean"]
DEFAULT_YT_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


class FrameInferenceDataset(Dataset):
    def __init__(self, face_crops: list[tuple[int, np.ndarray]], transform):
        self.face_crops = face_crops
        self.transform = transform

    def __len__(self) -> int:
        return len(self.face_crops)

    def __getitem__(self, idx: int):
        frame_idx, crop = self.face_crops[idx]
        image = Image.fromarray(crop)
        image = self.transform(image)
        return {"image": image, "frame_idx": frame_idx}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble inference (frame + frequency + temporal)")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-path", type=str, default="", help="Input local video path")
    input_group.add_argument("--youtube-url", type=str, default="", help="Input YouTube URL")

    parser.add_argument("--frame-checkpoint", type=str, default="", help="Path to frame model checkpoint")
    parser.add_argument("--temporal-checkpoint", type=str, default="", help="Path to temporal model checkpoint")
    parser.add_argument("--freq-checkpoint", type=str, default="", help="Path to frequency frame model checkpoint")
    parser.add_argument("--download-dir", type=str, default="", help="YouTube download directory")
    parser.add_argument("--youtube-format", type=str, default=DEFAULT_YT_FORMAT)

    parser.add_argument(
        "--frame-aggregation",
        type=str,
        default="",
        choices=[""] + AGG_CHOICES,
        help="Frame model aggregation method",
    )
    parser.add_argument("--frame-weight", type=float, default=-1.0)
    parser.add_argument("--temporal-weight", type=float, default=-1.0)
    parser.add_argument("--freq-weight", type=float, default=-1.0)
    parser.add_argument("--min-confidence", type=float, default=-1.0)
    parser.add_argument("--topk-ratio", type=float, default=-1.0)
    parser.add_argument("--conf-power", type=float, default=-1.0)
    parser.add_argument("--frame-input-representation", type=str, default="")
    parser.add_argument("--temporal-input-representation", type=str, default="")
    parser.add_argument("--freq-input-representation", type=str, default="")
    parser.add_argument("--temporal-frame-mode", type=str, default="", choices=["", "rgb", "frame_diff"])
    parser.add_argument(
        "--decision-policy",
        type=str,
        default="",
        choices=["", "argmax", "conservative_fake", "adaptive_auto"],
        help="Final decision policy. Default uses config.ensemble.decision_policy",
    )
    parser.add_argument(
        "--fake-threshold",
        type=float,
        default=-1.0,
        help="Used by conservative_fake policy. If max fake prob exceeds this, force fake decision.",
    )
    parser.add_argument("--uncertain-lower", type=float, default=-1.0, help="Lower fake-probability bound for uncertain output")
    parser.add_argument("--uncertain-upper", type=float, default=-1.0, help="Upper fake-probability bound for uncertain output")
    parser.add_argument("--uncertain-margin", type=float, default=-1.0, help="If top1-top2 margin is smaller than this, output uncertain")
    parser.add_argument("--save-frame-csv", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _latest_downloaded_video(download_dir: Path) -> Path:
    candidates = [
        p
        for p in download_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not candidates:
        raise RuntimeError(f"No downloaded video found in: {download_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def download_youtube_video(url: str, download_dir: Path, yt_format: str) -> Path:
    download_dir = ensure_dir(download_dir)

    try:
        from yt_dlp import YoutubeDL  # type: ignore

        ydl_opts = {
            "format": yt_format,
            "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            resolved = resolve_downloaded_video_path(
                info=info,
                ydl=ydl,
                download_dir=download_dir,
                video_extensions=VIDEO_EXTENSIONS,
            )
            if resolved is not None:
                return resolved
            return _latest_downloaded_video(download_dir)
    except Exception:
        pass

    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        raise RuntimeError("yt-dlp not found. Install with `pip install yt-dlp`.")

    cmd = [
        yt_dlp_bin,
        "-f",
        yt_format,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-o",
        str(download_dir / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download YouTube video via yt-dlp. "
            "Check the URL and use a real link (not `VIDEO_ID`)."
        ) from exc
    matched = find_downloaded_video_by_url(url, download_dir, VIDEO_EXTENSIONS)
    if matched is not None:
        return matched
    return _latest_downloaded_video(download_dir)


def resolve_input_video(args: argparse.Namespace, config: dict) -> tuple[Path, str, str | None]:
    if args.video_path:
        path = Path(args.video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path, "local_file", None

    infer_dir = ensure_dir(config["paths"].get("inference_dir", "outputs/inference"))
    inference_cfg = config.get("inference", {})
    default_download_dir = infer_dir / "downloads"
    configured = inference_cfg.get("youtube_download_dir", str(default_download_dir))
    download_dir = Path(args.download_dir) if args.download_dir else Path(configured)

    try:
        youtube_url = validate_youtube_url(args.youtube_url)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(f"[INFO] downloading youtube video: {youtube_url}")
    try:
        downloaded_path = download_youtube_video(youtube_url, download_dir, args.youtube_format)
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(f"[INFO] downloaded video path: {downloaded_path}")
    return downloaded_path, "youtube_url", youtube_url


def extract_face_crops(
    video_path: str | Path,
    config: dict,
    view_mode_override: str | None = None,
) -> list[tuple[int, np.ndarray]]:
    preprocess_cfg = config["preprocess"]
    detector_cfg = preprocess_cfg["detector"]
    detector = build_face_detector(detector_cfg)

    target_fps = preprocess_cfg["target_fps"]
    max_frames = preprocess_cfg.get("max_frames_per_video")
    image_size = preprocess_cfg["image_size"]
    fallback_to_full_frame = preprocess_cfg.get("fallback_to_full_frame", False)
    view_mode = view_mode_override or preprocess_cfg.get("view_mode", "detector_crop")
    text_mask_cfg = preprocess_cfg.get("text_mask", {})

    crops: list[tuple[int, np.ndarray]] = []
    for frame_idx, frame_rgb in iter_video_frames(video_path, target_fps, max_frames):
        crop = _extract_frame_view(
            image_rgb=frame_rgb,
            detector=detector,
            fallback_to_full_frame=fallback_to_full_frame,
            view_mode=view_mode,
        )
        if crop is None:
            continue
        crop = apply_text_mask_np(crop, text_mask_cfg)
        crop = resize_image(crop, image_size)
        crops.append((frame_idx, crop))

    return crops


def select_sequence_indices(n_frames: int, sequence_length: int) -> np.ndarray:
    if n_frames >= sequence_length:
        idx = np.linspace(0, n_frames - 1, num=sequence_length)
        return np.round(idx).astype(np.int64)

    base = list(range(n_frames))
    base += [n_frames - 1] * (sequence_length - n_frames)
    return np.array(base, dtype=np.int64)


def run_frame_model(
    frame_model: torch.nn.Module,
    crops: list[tuple[int, np.ndarray]],
    transform,
    device: torch.device,
    batch_size: int,
    aggregation: str,
    min_confidence: float,
    topk_ratio: float,
    conf_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    dataset = FrameInferenceDataset(crops, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    frame_probs_list = []
    frame_indices: list[int] = []

    frame_model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = frame_model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            frame_probs_list.append(probs)
            frame_indices.extend([int(x) for x in batch["frame_idx"]])

    frame_probs = np.concatenate(frame_probs_list, axis=0)
    frame_conf = np.max(frame_probs, axis=1)
    keep_mask = frame_conf >= min_confidence
    if not np.any(keep_mask):
        keep_mask = np.ones_like(frame_conf, dtype=bool)

    filtered_probs = frame_probs[keep_mask]
    video_prob = aggregate_probs(
        probs=filtered_probs,
        method=aggregation,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )
    return video_prob, frame_probs, frame_conf, frame_indices, keep_mask


def run_temporal_model(
    temporal_model: torch.nn.Module,
    crops: list[tuple[int, np.ndarray]],
    transform,
    device: torch.device,
    sequence_length: int,
    frame_mode: str,
) -> tuple[np.ndarray, list[int]]:
    indices = select_sequence_indices(len(crops), sequence_length)

    selected_crops = []
    selected_frame_idx: list[int] = []
    for i in indices:
        frame_idx, crop = crops[int(i)]
        selected_crops.append(crop)
        selected_frame_idx.append(int(frame_idx))

    if frame_mode == "frame_diff":
        processed_crops = []
        prev = None
        for current in selected_crops:
            if prev is None:
                diff = np.zeros_like(current, dtype=np.uint8)
            else:
                diff = np.abs(current.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
            processed_crops.append(diff)
            prev = current
    else:
        processed_crops = selected_crops

    selected = [transform(Image.fromarray(crop)) for crop in processed_crops]

    video_tensor = torch.stack(selected, dim=0).unsqueeze(0).to(device)
    lengths = torch.tensor([min(len(crops), sequence_length)], dtype=torch.long, device=device)

    temporal_model.eval()
    with torch.no_grad():
        logits = temporal_model(video_tensor, lengths=lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs, selected_frame_idx


def normalize_component_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in raw_weights.items()}
    total = float(sum(clipped.values()))
    if total <= 0:
        equal = 1.0 / max(1, len(clipped))
        return {k: equal for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def apply_decision_policy(
    ensemble_prob: np.ndarray,
    frame_prob: np.ndarray,
    temporal_prob: np.ndarray,
    freq_prob: np.ndarray | None,
    real_idx: int,
    decision_policy: str,
    fake_threshold: float,
    adaptive_temporal_min_prob: float = 0.35,
    adaptive_ensemble_min_prob: float = 0.27,
    adaptive_temporal_frame_gap: float = 0.16,
    adaptive_temporal_direct_prob: float = 0.48,
) -> tuple[int, str]:
    pred_idx = int(np.argmax(ensemble_prob))
    decision_reason = "argmax"

    if decision_policy == "adaptive_auto":
        if pred_idx != real_idx:
            return pred_idx, "argmax_already_fake"

        ensemble_fake = float(1.0 - ensemble_prob[real_idx])
        frame_fake = float(1.0 - frame_prob[real_idx])
        temporal_fake = float(1.0 - temporal_prob[real_idx])
        temporal_gap = temporal_fake - frame_fake

        if (
            temporal_fake >= adaptive_temporal_direct_prob
            and ensemble_fake >= adaptive_ensemble_min_prob
        ):
            fake_indices = [i for i in range(len(ensemble_prob)) if i != real_idx]
            forced_idx = int(max(fake_indices, key=lambda i: float(ensemble_prob[i])))
            return forced_idx, (
                "adaptive_auto_temporal_direct("
                f"temporal_fake={temporal_fake:.4f},ensemble_fake={ensemble_fake:.4f})"
            )

        if (
            temporal_fake >= adaptive_temporal_min_prob
            and ensemble_fake >= adaptive_ensemble_min_prob
            and temporal_gap >= adaptive_temporal_frame_gap
        ):
            fake_indices = [i for i in range(len(ensemble_prob)) if i != real_idx]
            forced_idx = int(max(fake_indices, key=lambda i: float(ensemble_prob[i])))
            return forced_idx, (
                "adaptive_auto_temporal_gap("
                f"temporal_fake={temporal_fake:.4f},ensemble_fake={ensemble_fake:.4f},"
                f"gap={temporal_gap:.4f})"
            )

        return pred_idx, "adaptive_auto_not_triggered"

    if decision_policy != "conservative_fake":
        return pred_idx, decision_reason

    if pred_idx != real_idx:
        return pred_idx, "argmax_already_fake"

    fake_prob_ensemble = float(1.0 - ensemble_prob[real_idx])
    fake_prob_frame = float(1.0 - frame_prob[real_idx])
    fake_prob_temporal = float(1.0 - temporal_prob[real_idx])
    fake_candidates = [fake_prob_ensemble, fake_prob_frame, fake_prob_temporal]
    if freq_prob is not None:
        fake_candidates.append(float(1.0 - freq_prob[real_idx]))
    max_fake_prob = max(fake_candidates)

    if max_fake_prob >= fake_threshold:
        fake_indices = [i for i in range(len(ensemble_prob)) if i != real_idx]
        forced_idx = int(max(fake_indices, key=lambda i: float(ensemble_prob[i])))
        return forced_idx, f"conservative_fake(max_fake_prob={max_fake_prob:.4f})"

    return pred_idx, "conservative_fake_not_triggered"


def apply_uncertainty_policy(
    ensemble_prob: np.ndarray,
    real_idx: int,
    enabled: bool,
    lower_fake_prob: float,
    upper_fake_prob: float,
    min_margin: float,
) -> tuple[str | None, str | None]:
    if not enabled:
        return None, None

    fake_prob = float(1.0 - ensemble_prob[real_idx])
    sorted_prob = np.sort(ensemble_prob)
    top_margin = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else 1.0

    in_band = lower_fake_prob <= fake_prob <= upper_fake_prob
    low_margin = top_margin <= min_margin
    if not in_band and not low_margin:
        return None, None

    reasons = []
    if in_band:
        reasons.append(
            f"fake_prob_band(fake_prob={fake_prob:.4f},range=[{lower_fake_prob:.2f},{upper_fake_prob:.2f}])"
        )
    if low_margin:
        reasons.append(f"low_margin(top_margin={top_margin:.4f},min_margin={min_margin:.2f})")
    return "uncertain", "uncertainty:" + ",".join(reasons)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task_spec = build_task_spec(config["task"])
    label_mapper = LabelMapper(task_spec)

    training_cfg = config["training"]
    temporal_cfg = config.get("temporal", {})
    temporal_training_cfg = temporal_cfg.get("training", training_cfg)
    temporal_model_cfg = temporal_cfg.get("model", {})
    temporal_dataset_cfg = temporal_cfg.get("dataset", {})
    ensemble_cfg = config.get("ensemble", {})
    inference_cfg = config.get("inference", {})
    frame_input_representation = (
        args.frame_input_representation
        if args.frame_input_representation
        else ensemble_cfg.get(
            "frame_input_representation",
            inference_cfg.get("input_representation", training_cfg.get("input_representation", "rgb")),
        )
    )
    temporal_input_representation = (
        args.temporal_input_representation
        if args.temporal_input_representation
        else temporal_cfg.get(
            "input_representation",
            temporal_training_cfg.get("input_representation", training_cfg.get("input_representation", "rgb")),
        )
    )
    freq_input_representation = (
        args.freq_input_representation
        if args.freq_input_representation
        else ensemble_cfg.get("freq_input_representation", "fft")
    )
    temporal_frame_mode = (
        args.temporal_frame_mode
        if args.temporal_frame_mode
        else temporal_dataset_cfg.get("frame_mode", "rgb")
    )
    if temporal_frame_mode not in {"rgb", "frame_diff"}:
        raise ValueError("temporal_frame_mode must be one of: rgb, frame_diff")

    device = resolve_device(training_cfg.get("device", "auto"))
    input_video_path, input_type, source_url = resolve_input_video(args, config)

    frame_ckpt = (
        Path(args.frame_checkpoint)
        if args.frame_checkpoint
        else Path(config["paths"].get("checkpoints_dir", "outputs/checkpoints")) / "best.pt"
    )
    temporal_ckpt = (
        Path(args.temporal_checkpoint)
        if args.temporal_checkpoint
        else Path(config["paths"].get("temporal_checkpoints_dir", "outputs/checkpoints_temporal"))
        / "best.pt"
    )
    freq_ckpt_raw = args.freq_checkpoint or ensemble_cfg.get("freq_checkpoint", "")
    freq_ckpt = Path(freq_ckpt_raw) if freq_ckpt_raw else None
    if freq_ckpt is not None and not freq_ckpt.exists():
        raise FileNotFoundError(freq_ckpt)

    frame_model, _ = load_model_from_checkpoint(
        checkpoint_path=frame_ckpt,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )

    temporal_model, _ = load_temporal_model_from_checkpoint(
        checkpoint_path=temporal_ckpt,
        model_cfg=temporal_model_cfg,
        num_classes=task_spec.num_classes,
        device=device,
    )
    freq_model = None
    if freq_ckpt is not None:
        freq_model, _ = load_model_from_checkpoint(
            checkpoint_path=freq_ckpt,
            backbone=ensemble_cfg.get("freq_backbone", training_cfg["backbone"]),
            num_classes=task_spec.num_classes,
            dropout=float(ensemble_cfg.get("freq_dropout", training_cfg.get("dropout", 0.0))),
            freeze_backbone=bool(ensemble_cfg.get("freq_freeze_backbone", training_cfg.get("freeze_backbone", False))),
            hidden_dim=int(ensemble_cfg.get("freq_hidden_dim", training_cfg.get("hidden_dim", 0)) or 0),
            device=device,
        )

    crops = extract_face_crops(input_video_path, config)
    if len(crops) == 0:
        raise RuntimeError("No face crops extracted from video")

    frame_transform = build_eval_transform(
        config["preprocess"]["image_size"],
        input_representation=frame_input_representation,
    )

    aggregation = args.frame_aggregation or ensemble_cfg.get("frame_aggregation", inference_cfg.get("aggregation", "confidence_mean"))
    min_conf = (
        float(args.min_confidence)
        if args.min_confidence >= 0
        else float(ensemble_cfg.get("min_confidence", inference_cfg.get("min_confidence", 0.0)))
    )
    topk_ratio = (
        float(args.topk_ratio)
        if args.topk_ratio > 0
        else float(ensemble_cfg.get("topk_ratio", inference_cfg.get("topk_ratio", 0.5)))
    )
    conf_power = (
        float(args.conf_power)
        if args.conf_power > 0
        else float(ensemble_cfg.get("conf_power", inference_cfg.get("conf_power", 2.0)))
    )

    frame_prob, frame_probs, frame_conf, frame_indices, keep_mask = run_frame_model(
        frame_model=frame_model,
        crops=crops,
        transform=frame_transform,
        device=device,
        batch_size=int(inference_cfg.get("batch_size", 32)),
        aggregation=aggregation,
        min_confidence=min_conf,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )

    freq_prob = None
    if freq_model is not None:
        freq_transform = build_eval_transform(
            config["preprocess"]["image_size"],
            input_representation=freq_input_representation,
        )
        freq_prob, _, _, _, _ = run_frame_model(
            frame_model=freq_model,
            crops=crops,
            transform=freq_transform,
            device=device,
            batch_size=int(inference_cfg.get("batch_size", 32)),
            aggregation=aggregation,
            min_confidence=min_conf,
            topk_ratio=topk_ratio,
            conf_power=conf_power,
        )

    temporal_transform = build_eval_transform(
        config["preprocess"]["image_size"],
        input_representation=temporal_input_representation,
    )
    temporal_prob, temporal_indices = run_temporal_model(
        temporal_model=temporal_model,
        crops=crops,
        transform=temporal_transform,
        device=device,
        sequence_length=int(temporal_dataset_cfg.get("sequence_length", 8)),
        frame_mode=temporal_frame_mode,
    )

    component_probs = {
        "frame": frame_prob,
        "temporal": temporal_prob,
    }
    raw_weights = {
        "frame": float(args.frame_weight) if args.frame_weight >= 0 else float(ensemble_cfg.get("frame_weight", 0.4)),
        "temporal": float(args.temporal_weight) if args.temporal_weight >= 0 else float(ensemble_cfg.get("temporal_weight", 0.6)),
    }
    if freq_prob is not None:
        component_probs["freq"] = freq_prob
        raw_weights["freq"] = (
            float(args.freq_weight) if args.freq_weight >= 0 else float(ensemble_cfg.get("freq_weight", 0.2))
        )

    normalized_weights = normalize_component_weights(raw_weights)
    ensemble_prob = np.zeros_like(frame_prob, dtype=np.float64)
    for name, prob in component_probs.items():
        ensemble_prob += float(normalized_weights[name]) * prob
    ensemble_prob = ensemble_prob / max(1e-8, float(np.sum(ensemble_prob)))
    real_idx = label_mapper.class_to_idx.get("real", 0)
    decision_policy = args.decision_policy or ensemble_cfg.get("decision_policy", "argmax")
    fake_threshold = (
        float(args.fake_threshold)
        if args.fake_threshold >= 0
        else float(ensemble_cfg.get("fake_threshold", 0.45))
    )
    uncertainty_cfg = ensemble_cfg.get("uncertainty", {})
    uncertainty_enabled = bool(uncertainty_cfg.get("enabled", False))
    uncertain_lower = (
        float(args.uncertain_lower)
        if args.uncertain_lower >= 0
        else float(uncertainty_cfg.get("lower_fake_prob", 0.4))
    )
    uncertain_upper = (
        float(args.uncertain_upper)
        if args.uncertain_upper >= 0
        else float(uncertainty_cfg.get("upper_fake_prob", 0.6))
    )
    uncertain_margin = (
        float(args.uncertain_margin)
        if args.uncertain_margin >= 0
        else float(uncertainty_cfg.get("min_margin", 0.12))
    )
    raw_pred_idx = int(np.argmax(ensemble_prob))
    pred_idx, decision_reason = apply_decision_policy(
        ensemble_prob=ensemble_prob,
        frame_prob=frame_prob,
        temporal_prob=temporal_prob,
        freq_prob=freq_prob,
        real_idx=real_idx,
        decision_policy=decision_policy,
        fake_threshold=fake_threshold,
        adaptive_temporal_min_prob=float(ensemble_cfg.get("adaptive_temporal_min_prob", 0.35)),
        adaptive_ensemble_min_prob=float(ensemble_cfg.get("adaptive_ensemble_min_prob", 0.27)),
        adaptive_temporal_frame_gap=float(ensemble_cfg.get("adaptive_temporal_frame_gap", 0.16)),
        adaptive_temporal_direct_prob=float(ensemble_cfg.get("adaptive_temporal_direct_prob", 0.48)),
    )
    final_label = label_mapper.index_to_name(pred_idx)
    uncertain_label, uncertainty_reason = apply_uncertainty_policy(
        ensemble_prob=ensemble_prob,
        real_idx=real_idx,
        enabled=uncertainty_enabled,
        lower_fake_prob=uncertain_lower,
        upper_fake_prob=uncertain_upper,
        min_margin=uncertain_margin,
    )
    if uncertain_label is not None:
        final_label = uncertain_label
        decision_reason = uncertainty_reason or decision_reason

    output = {
        "input": {
            "type": input_type,
            "source_url": source_url,
            "resolved_video_path": str(input_video_path),
        },
        "video_path": str(input_video_path),
        "num_face_frames": int(len(crops)),
        "ensemble": {
            "weights": {k: float(v) for k, v in normalized_weights.items()},
            "frame_aggregation": aggregation,
            "min_confidence": min_conf,
            "topk_ratio": topk_ratio,
            "conf_power": conf_power,
            "decision_policy": decision_policy,
            "fake_threshold": fake_threshold,
            "adaptive_temporal_min_prob": float(ensemble_cfg.get("adaptive_temporal_min_prob", 0.35)),
            "adaptive_ensemble_min_prob": float(ensemble_cfg.get("adaptive_ensemble_min_prob", 0.27)),
            "adaptive_temporal_frame_gap": float(ensemble_cfg.get("adaptive_temporal_frame_gap", 0.16)),
            "adaptive_temporal_direct_prob": float(ensemble_cfg.get("adaptive_temporal_direct_prob", 0.48)),
            "uncertainty_enabled": uncertainty_enabled,
            "uncertain_lower_fake_prob": uncertain_lower,
            "uncertain_upper_fake_prob": uncertain_upper,
            "uncertain_min_margin": uncertain_margin,
            "decision_reason": decision_reason,
            "frame_input_representation": frame_input_representation,
            "temporal_input_representation": temporal_input_representation,
            "temporal_frame_mode": temporal_frame_mode,
        },
        "components": {
            "frame": {
                "probabilities": frame_prob.tolist(),
                "used_frames": int(np.sum(keep_mask)),
                "total_frames": int(len(frame_probs)),
                "confidence_mean": float(np.mean(frame_conf)),
            },
            "temporal": {
                "probabilities": temporal_prob.tolist(),
                "selected_frame_indices": temporal_indices,
            },
        },
        "prediction": {
            "label": final_label,
            "index": pred_idx if final_label != "uncertain" else -1,
            "probabilities": ensemble_prob.tolist(),
        },
        "prediction_raw_argmax": {
            "label": label_mapper.index_to_name(raw_pred_idx),
            "index": raw_pred_idx,
            "probabilities": ensemble_prob.tolist(),
        },
        "authenticity_score": float(ensemble_prob[real_idx]),
        "fake_score": float(1.0 - ensemble_prob[real_idx]),
    }
    if freq_prob is not None:
        output["ensemble"]["freq_input_representation"] = freq_input_representation
        output["components"]["freq"] = {"probabilities": freq_prob.tolist()}

    infer_dir = ensure_dir(config["paths"].get("inference_dir", "outputs/inference"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_video_path).stem

    out_json = infer_dir / f"ensemble_{stem}_{timestamp}.json"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if pred_idx != raw_pred_idx:
        print(
            "[WARN] decision policy overrode argmax: "
            f"{label_mapper.index_to_name(raw_pred_idx)} -> {label_mapper.index_to_name(pred_idx)} "
            f"({decision_reason})"
        )
    print(f"[INFO] ensemble prediction: {output['prediction']}")
    print(f"[INFO] authenticity_score={output['authenticity_score']:.6f}")
    print(f"[INFO] saved ensemble inference: {out_json}")

    if args.save_frame_csv:
        out_csv = infer_dir / f"ensemble_{stem}_{timestamp}_frames.csv"
        frame_pred_idx = np.argmax(frame_probs, axis=1)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["frame_idx", "pred_label", "pred_index", "confidence", "used_for_frame_component"] + [
                f"prob_{i}" for i in range(frame_probs.shape[1])
            ]
            writer.writerow(header)
            for frame_idx, pred_idx_i, probs_i, used_i in zip(frame_indices, frame_pred_idx, frame_probs, keep_mask):
                writer.writerow(
                    [
                        int(frame_idx),
                        label_mapper.index_to_name(int(pred_idx_i)),
                        int(pred_idx_i),
                        float(np.max(probs_i)),
                        bool(used_i),
                    ]
                    + [float(x) for x in probs_i]
                )
        print(f"[INFO] saved ensemble frame csv: {out_csv}")


if __name__ == "__main__":
    main()
