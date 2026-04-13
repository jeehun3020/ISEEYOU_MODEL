from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from iseeyou.config import ensure_dir, load_config
from iseeyou.constants import LabelMapper, build_task_spec
from iseeyou.data.detectors.factory import build_face_detector
from iseeyou.data.preprocess import _extract_face_or_full_frame
from iseeyou.data.transforms import build_eval_transform
from iseeyou.engine.temporal import load_temporal_model_from_checkpoint
from iseeyou.utils.masking import apply_text_mask_np
from iseeyou.utils.video import iter_video_frames, resize_image
from iseeyou.utils.youtube import (
    find_downloaded_video_by_url,
    resolve_downloaded_video_path,
    validate_youtube_url,
)


DEFAULT_YT_FORMAT = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal video-level inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to temporal checkpoint")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video-path", type=str, default="", help="Input local video path")
    input_group.add_argument("--youtube-url", type=str, default="", help="Input YouTube URL")

    parser.add_argument("--download-dir", type=str, default="", help="YouTube download directory")
    parser.add_argument("--youtube-format", type=str, default=DEFAULT_YT_FORMAT)
    parser.add_argument(
        "--input-representation",
        type=str,
        default="",
        help="Input representation: rgb | fft | rgb_fft. Default uses config.",
    )
    parser.add_argument(
        "--frame-mode",
        type=str,
        default="",
        choices=["", "rgb", "frame_diff"],
        help="Temporal frame mode. Default uses config.temporal.dataset.frame_mode",
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
            "Check the URL and use a real video link instead of placeholders."
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


def extract_face_crops(video_path: str | Path, config: dict) -> list[tuple[int, np.ndarray]]:
    preprocess_cfg = config["preprocess"]
    detector_cfg = preprocess_cfg["detector"]
    detector = build_face_detector(detector_cfg)

    target_fps = preprocess_cfg["target_fps"]
    max_frames = preprocess_cfg.get("max_frames_per_video")
    image_size = preprocess_cfg["image_size"]
    fallback_to_full_frame = preprocess_cfg.get("fallback_to_full_frame", False)
    text_mask_cfg = preprocess_cfg.get("text_mask", {})

    crops: list[tuple[int, np.ndarray]] = []
    for frame_idx, frame_rgb in iter_video_frames(video_path, target_fps, max_frames):
        crop = _extract_face_or_full_frame(frame_rgb, detector, fallback_to_full_frame)
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task_spec = build_task_spec(config["task"])
    label_mapper = LabelMapper(task_spec)

    temporal_cfg = config.get("temporal", {})
    dataset_cfg = temporal_cfg.get("dataset", {})
    model_cfg = temporal_cfg.get("model", {})
    training_cfg = temporal_cfg.get("training", config["training"])
    input_representation = (
        args.input_representation
        if args.input_representation
        else temporal_cfg.get(
            "input_representation",
            training_cfg.get("input_representation", config["training"].get("input_representation", "rgb")),
        )
    )
    frame_mode = args.frame_mode if args.frame_mode else dataset_cfg.get("frame_mode", "rgb")
    if frame_mode not in {"rgb", "frame_diff"}:
        raise ValueError("frame_mode must be one of: rgb, frame_diff")

    device = resolve_device(training_cfg.get("device", "auto"))
    input_video_path, input_type, source_url = resolve_input_video(args, config)

    model, _ = load_temporal_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_cfg=model_cfg,
        num_classes=task_spec.num_classes,
        device=device,
    )

    crops = extract_face_crops(input_video_path, config)
    if len(crops) == 0:
        raise RuntimeError("No face crops extracted from video")

    sequence_length = int(dataset_cfg.get("sequence_length", 8))
    indices = select_sequence_indices(len(crops), sequence_length)

    transform = build_eval_transform(
        config["preprocess"]["image_size"],
        input_representation=input_representation,
    )
    selected_crops = []
    selected_frame_idx = []
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

    selected_frames = [transform(Image.fromarray(crop)) for crop in processed_crops]

    video_tensor = torch.stack(selected_frames, dim=0).unsqueeze(0).to(device)  # [1,T,C,H,W]
    lengths = torch.tensor([min(len(crops), sequence_length)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(video_tensor, lengths=lengths)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    real_idx = label_mapper.class_to_idx.get("real", 0)

    output = {
        "input": {
            "type": input_type,
            "source_url": source_url,
            "resolved_video_path": str(input_video_path),
        },
        "video_path": str(input_video_path),
        "num_face_frames": len(crops),
        "sequence_length": sequence_length,
        "frame_mode": frame_mode,
        "input_representation": input_representation,
        "selected_frame_indices": selected_frame_idx,
        "prediction": {
            "label": label_mapper.index_to_name(pred_idx),
            "index": pred_idx,
            "probabilities": probs.tolist(),
        },
        "authenticity_score": float(probs[real_idx]),
        "fake_score": float(1.0 - probs[real_idx]),
    }

    infer_dir = ensure_dir(config["paths"].get("inference_dir", "outputs/inference"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_video_path).stem
    out_path = infer_dir / f"temporal_{stem}_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"[INFO] temporal prediction: {output['prediction']}")
    print(f"[INFO] authenticity_score={output['authenticity_score']:.6f}")
    print(f"[INFO] saved temporal inference: {out_path}")


if __name__ == "__main__":
    main()
