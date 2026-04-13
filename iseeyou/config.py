from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "task": {
        "mode": "multiclass",
        "classes": ["real", "generated", "deepfake"],
        "positive_classes": ["generated", "deepfake"],
    },
    "split": {
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "group_priority": ["original_id", "identity_id", "source_id", "video_id"],
    },
    "preprocess": {
        "target_fps": 2,
        "max_frames_per_video": 64,
        "image_size": 224,
        "frame_sampling_mode": "all",
        "view_mode": "detector_crop",
        "fallback_to_full_frame": False,
        "text_mask": {
            "enabled": False,
            "top_ratio": 0.0,
            "bottom_ratio": 0.0,
            "left_ratio": 0.0,
            "right_ratio": 0.0,
            "position_mode": "fixed",
            "fill_mode": "median",
            "blur_kernel_size": 31,
            "inpaint_radius": 3,
        },
        "detector": {
            "name": "mtcnn",
            "device": "auto",
            "min_face_size": 40,
            "keep_all": True,
        },
    },
    "training": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "dropout": 0.2,
        "input_representation": "rgb",
        "grad_clip_norm": 1.0,
        "batch_size": 32,
        "num_workers": 4,
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "amp": True,
        "monitor": "f1",
        "loss": {
            "label_smoothing": 0.05,
            "use_class_weights": True,
        },
        "sampling": {
            "balanced_sampler": True,
        },
        "early_stopping": {
            "patience": 3,
            "min_delta": 0.001,
        },
        "augmentation": {
            "hflip_p": 0.5,
            "color_jitter": True,
            "color_jitter_strength": 0.12,
            "random_erasing": True,
            "random_erasing_p": 0.25,
            "region_dropout": {
                "enabled": False,
                "p": 0.0,
                "mode": "random_box",
                "fill_mode": "median",
                "area_ratio_range": [0.08, 0.2],
                "aspect_ratio_range": [0.75, 1.5],
                "fallback_to_random_box": True,
            },
            "text_mask_aug": {
                "enabled": False,
                "p": 0.0,
                "top_ratio_range": [0.0, 0.0],
                "bottom_ratio_range": [0.0, 0.0],
                "left_ratio_range": [0.0, 0.0],
                "right_ratio_range": [0.0, 0.0],
                "position_mode": "fixed",
                "fill_mode": "median",
                "blur_kernel_size": 31,
                "inpaint_radius": 3,
            },
            "robustness_aug": {
                "enabled": False,
                "jpeg_p": 0.0,
                "jpeg_quality_range": [45, 95],
                "gaussian_blur_p": 0.0,
                "gaussian_blur_radius_range": [0.2, 1.2],
                "noise_p": 0.0,
                "noise_std_range": [2.0, 10.0],
                "color_jitter_p": 0.0,
                "contrast_range": [0.85, 1.15],
                "brightness_range": [0.9, 1.1],
                "saturation_range": [0.85, 1.15],
                "subtitle_overlay_p": 0.0,
                "corner_watermark_p": 0.0,
                "overlay_opacity_range": [0.45, 0.8],
                "subtitle_height_ratio_range": [0.08, 0.16],
                "watermark_width_ratio_range": [0.08, 0.16],
                "watermark_height_ratio_range": [0.04, 0.08],
            },
        },
    },
    "inference": {
        "batch_size": 32,
        "aggregation": "mean",
        "input_representation": "rgb",
        "min_confidence": 0.0,
        "topk_ratio": 0.5,
        "conf_power": 2.0,
        "youtube_download_dir": "outputs/inference/downloads",
    },
    "evaluation": {
        "video_aggregation": "mean",
        "input_representation": "rgb",
        "topk_ratio": 0.5,
        "conf_power": 2.0,
    },
    "ensemble": {
        "frame_weight": 0.4,
        "temporal_weight": 0.6,
        "frame_aggregation": "confidence_mean",
        "frame_input_representation": "rgb",
        "freq_input_representation": "fft",
        "min_confidence": 0.55,
        "topk_ratio": 0.5,
        "conf_power": 2.0,
        "freq_weight": 0.2,
        "decision_policy": "conservative_fake",
        "fake_threshold": 0.45,
        "uncertainty": {
            "enabled": False,
            "lower_fake_prob": 0.4,
            "upper_fake_prob": 0.6,
            "min_margin": 0.12,
        },
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), user_config)

    # TODO: add strict schema validation for config values.
    return config


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
