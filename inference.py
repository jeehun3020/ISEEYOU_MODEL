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
from iseeyou.data.preprocess import _extract_face_or_full_frame
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.evaluator import load_model_from_checkpoint
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


class InferenceFaceDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Video-level inference from single video")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-path", type=str, default="", help="Input local video path")
    input_group.add_argument("--youtube-url", type=str, default="", help="Input YouTube URL")

    parser.add_argument(
        "--download-dir",
        type=str,
        default="",
        help="Directory for downloaded videos when --youtube-url is used",
    )
    parser.add_argument(
        "--youtube-format",
        type=str,
        default=DEFAULT_YT_FORMAT,
        help="yt-dlp format selector for YouTube downloads",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="",
        choices=[""] + AGG_CHOICES,
        help="Aggregation method. Default uses config.inference.aggregation",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=-1.0,
        help="Drop frame predictions below this max-prob confidence. Default uses config.",
    )
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=-1.0,
        help="Used when aggregation=topk_mean. Default uses config.",
    )
    parser.add_argument(
        "--conf-power",
        type=float,
        default=-1.0,
        help="Used when aggregation=confidence_mean. Default uses config.",
    )
    parser.add_argument(
        "--save-frame-csv",
        action="store_true",
        help="If set, save per-frame probabilities to CSV",
    )
    parser.add_argument(
        "--input-representation",
        type=str,
        default="",
        help="Input representation: rgb | fft | rgb_fft. Default uses config.",
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


def _latest_downloaded_video(download_dir: Path) -> Path:
    candidates = [
        p
        for p in download_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not candidates:
        raise RuntimeError(f"No downloaded file found in: {download_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def download_youtube_video(url: str, download_dir: Path, yt_format: str) -> Path:
    download_dir = ensure_dir(download_dir)

    # Try Python API first for better portability.
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
        # TODO: add verbose debug logging option for download failures.
        pass

    yt_dlp_bin = shutil.which("yt-dlp")
    if yt_dlp_bin is None:
        raise RuntimeError(
            "YouTube download failed. Install yt-dlp (`pip install yt-dlp` or brew install yt-dlp)."
        )

    outtmpl = str(download_dir / "%(id)s.%(ext)s")
    cmd = [
        yt_dlp_bin,
        "-f",
        yt_format,
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-o",
        outtmpl,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download YouTube video via yt-dlp. "
            "Check the URL and try a real video link such as "
            "`https://www.youtube.com/shorts/<11-char-id>`."
        ) from exc
    matched = find_downloaded_video_by_url(url, download_dir, VIDEO_EXTENSIONS)
    if matched is not None:
        return matched
    return _latest_downloaded_video(download_dir)


def resolve_input_video(args: argparse.Namespace, config: dict) -> tuple[Path, str, str | None]:
    if args.video_path:
        path = Path(args.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        return path, "local_file", None

    inference_cfg = config.get("inference", {})
    infer_dir = ensure_dir(config["paths"].get("inference_dir", "outputs/inference"))
    default_download_dir = infer_dir / "downloads"
    configured_dir = inference_cfg.get("youtube_download_dir", str(default_download_dir))
    download_dir = Path(args.download_dir) if args.download_dir else Path(configured_dir)

    try:
        youtube_url = validate_youtube_url(args.youtube_url)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(f"[INFO] downloading youtube video: {youtube_url}")
    try:
        downloaded_path = download_youtube_video(
            url=youtube_url,
            download_dir=download_dir,
            yt_format=args.youtube_format,
        )
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(f"[INFO] downloaded video path: {downloaded_path}")
    return downloaded_path, "youtube_url", youtube_url


def extract_faces_from_video(video_path: str | Path, config: dict) -> list[tuple[int, np.ndarray]]:
    preprocess_cfg = config["preprocess"]
    detector_cfg = preprocess_cfg["detector"]

    detector = build_face_detector(detector_cfg)

    target_fps = preprocess_cfg["target_fps"]
    max_frames = preprocess_cfg.get("max_frames_per_video")
    image_size = preprocess_cfg["image_size"]
    fallback_to_full_frame = preprocess_cfg.get("fallback_to_full_frame", False)
    text_mask_cfg = preprocess_cfg.get("text_mask", {})

    face_crops: list[tuple[int, np.ndarray]] = []
    for frame_idx, frame_rgb in iter_video_frames(video_path, target_fps, max_frames):
        crop = _extract_face_or_full_frame(frame_rgb, detector, fallback_to_full_frame)
        if crop is None:
            continue

        crop = apply_text_mask_np(crop, text_mask_cfg)
        crop = resize_image(crop, image_size)
        face_crops.append((frame_idx, crop))

    return face_crops


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    task_spec = build_task_spec(config["task"])
    label_mapper = LabelMapper(task_spec)

    training_cfg = config["training"]
    inference_cfg = config.get("inference", {})
    device = resolve_device(training_cfg.get("device", "auto"))
    input_representation = (
        args.input_representation
        if args.input_representation
        else inference_cfg.get("input_representation", training_cfg.get("input_representation", "rgb"))
    )

    input_video_path, input_type, source_url = resolve_input_video(args, config)

    model, _ = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        backbone=training_cfg["backbone"],
        num_classes=task_spec.num_classes,
        dropout=training_cfg.get("dropout", 0.0),
        freeze_backbone=bool(training_cfg.get("freeze_backbone", False)),
        hidden_dim=int(training_cfg.get("hidden_dim", 0) or 0),
        device=device,
    )

    face_crops = extract_faces_from_video(input_video_path, config)
    if len(face_crops) == 0:
        raise RuntimeError(
            "No face crops extracted from input video. Check detector settings or fallback_to_full_frame."
        )

    dataset = InferenceFaceDataset(
        face_crops=face_crops,
        transform=build_eval_transform(
            config["preprocess"]["image_size"],
            input_representation=input_representation,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=inference_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    frame_indices = []
    frame_probs_list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            frame_probs_list.append(probs)
            frame_indices.extend([int(x) for x in batch["frame_idx"]])

    frame_probs = np.concatenate(frame_probs_list, axis=0)
    frame_conf = np.max(frame_probs, axis=1)

    min_conf = (
        float(args.min_confidence)
        if args.min_confidence >= 0
        else float(inference_cfg.get("min_confidence", 0.0))
    )
    keep_mask = frame_conf >= min_conf
    if not np.any(keep_mask):
        keep_mask = np.ones_like(frame_conf, dtype=bool)

    filtered_probs = frame_probs[keep_mask]

    aggregation = args.aggregation or inference_cfg.get("aggregation", "mean")
    topk_ratio = float(args.topk_ratio) if args.topk_ratio > 0 else float(inference_cfg.get("topk_ratio", 0.5))
    conf_power = float(args.conf_power) if args.conf_power > 0 else float(inference_cfg.get("conf_power", 2.0))

    video_prob = aggregate_probs(
        probs=filtered_probs,
        method=aggregation,
        topk_ratio=topk_ratio,
        conf_power=conf_power,
    )
    video_pred_idx = int(np.argmax(video_prob))

    frame_pred_idx = np.argmax(frame_probs, axis=1)
    real_idx = label_mapper.class_to_idx.get("real", 0)
    authenticity_score = float(video_prob[real_idx])
    fake_score = float(1.0 - authenticity_score)

    output = {
        "input": {
            "type": input_type,
            "source_url": source_url,
            "resolved_video_path": str(input_video_path),
        },
        "video_path": str(input_video_path),
        "num_face_frames": int(len(face_crops)),
        "num_used_frames": int(len(filtered_probs)),
        "aggregation": aggregation,
        "min_confidence": min_conf,
        "topk_ratio": topk_ratio,
        "conf_power": conf_power,
        "input_representation": input_representation,
        "authenticity_score": authenticity_score,
        "fake_score": fake_score,
        "video_prediction": {
            "label": label_mapper.index_to_name(video_pred_idx),
            "index": video_pred_idx,
            "probabilities": video_prob.tolist(),
        },
        "frame_confidence": {
            "mean": float(np.mean(frame_conf)),
            "std": float(np.std(frame_conf)),
            "min": float(np.min(frame_conf)),
            "max": float(np.max(frame_conf)),
        },
        "frame_predictions": [
            {
                "frame_idx": int(frame_idx),
                "label": label_mapper.index_to_name(int(pred_idx)),
                "index": int(pred_idx),
                "confidence": float(np.max(prob)),
                "used_for_video": bool(used),
                "probabilities": prob.tolist(),
            }
            for frame_idx, pred_idx, prob, used in zip(frame_indices, frame_pred_idx, frame_probs, keep_mask)
        ],
    }

    infer_dir = ensure_dir(config["paths"].get("inference_dir", "outputs/inference"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(input_video_path).stem

    json_path = infer_dir / f"{video_stem}_{timestamp}.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[INFO] saved inference json: {json_path}")

    if args.save_frame_csv:
        csv_path = infer_dir / f"{video_stem}_{timestamp}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["frame_idx", "pred_label", "pred_index", "confidence", "used_for_video"] + [
                f"prob_{i}" for i in range(frame_probs.shape[1])
            ]
            writer.writerow(header)
            for frame_idx, pred_idx, probs, used in zip(frame_indices, frame_pred_idx, frame_probs, keep_mask):
                writer.writerow(
                    [
                        frame_idx,
                        label_mapper.index_to_name(int(pred_idx)),
                        int(pred_idx),
                        float(np.max(probs)),
                        bool(used),
                    ]
                    + [float(p) for p in probs]
                )
        print(f"[INFO] saved frame csv: {csv_path}")


if __name__ == "__main__":
    main()
