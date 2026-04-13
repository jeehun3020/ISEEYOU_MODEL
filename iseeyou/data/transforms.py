from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision import transforms

from .frequency import convert_representation, validate_representation
from iseeyou.utils.masking import RandomBandMask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RepresentationTransform:
    def __init__(self, input_representation: str):
        self.input_representation = validate_representation(input_representation)

    def __call__(self, image):
        return convert_representation(image, self.input_representation)


class RobustnessAugment:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def __call__(self, image: Image.Image) -> Image.Image:
        if not bool(self.cfg.get("enabled", False)):
            return image

        out = image.convert("RGB")

        if random.random() < float(self.cfg.get("jpeg_p", 0.0)):
            lo, hi = self.cfg.get("jpeg_quality_range", [45, 95])
            quality = int(random.randint(int(lo), int(hi)))
            buf = io.BytesIO()
            out.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            out = Image.open(buf).convert("RGB")

        if random.random() < float(self.cfg.get("gaussian_blur_p", 0.0)):
            lo, hi = self.cfg.get("gaussian_blur_radius_range", [0.2, 1.2])
            radius = random.uniform(float(lo), float(hi))
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))

        if random.random() < float(self.cfg.get("noise_p", 0.0)):
            lo, hi = self.cfg.get("noise_std_range", [2.0, 10.0])
            std = random.uniform(float(lo), float(hi))
            arr = np.asarray(out, dtype=np.float32)
            noise = np.random.normal(0.0, std, size=arr.shape)
            arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
            out = Image.fromarray(arr, mode="RGB")

        if random.random() < float(self.cfg.get("color_jitter_p", 0.0)):
            lo, hi = self.cfg.get("contrast_range", [0.85, 1.15])
            out = ImageEnhance.Contrast(out).enhance(random.uniform(float(lo), float(hi)))
            lo, hi = self.cfg.get("brightness_range", [0.9, 1.1])
            out = ImageEnhance.Brightness(out).enhance(random.uniform(float(lo), float(hi)))
            lo, hi = self.cfg.get("saturation_range", [0.85, 1.15])
            out = ImageEnhance.Color(out).enhance(random.uniform(float(lo), float(hi)))

        if random.random() < float(self.cfg.get("subtitle_overlay_p", 0.0)):
            out = _draw_subtitle_like_overlay(out, self.cfg)

        if random.random() < float(self.cfg.get("corner_watermark_p", 0.0)):
            out = _draw_corner_watermark_overlay(out, self.cfg)

        return out


def _draw_subtitle_like_overlay(image: Image.Image, cfg: dict) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    band_h = max(8, int(h * random.uniform(*cfg.get("subtitle_height_ratio_range", [0.08, 0.16]))))
    y1 = h - band_h - int(h * random.uniform(0.0, 0.04))
    y2 = min(h, y1 + band_h)
    opacity = int(255 * random.uniform(*cfg.get("overlay_opacity_range", [0.45, 0.8])))
    draw.rectangle([0, y1, w, y2], fill=(0, 0, 0, opacity))
    n_boxes = random.randint(2, 6)
    cursor = int(w * 0.08)
    max_w = int(w * 0.84)
    for _ in range(n_boxes):
        bw = random.randint(max(12, w // 20), max(14, w // 8))
        bh = random.randint(max(4, band_h // 8), max(6, band_h // 4))
        if cursor + bw > max_w:
            break
        by = random.randint(y1 + 2, max(y1 + 3, y2 - bh - 2))
        draw.rounded_rectangle([cursor, by, cursor + bw, by + bh], radius=2, fill=(255, 255, 255, min(255, opacity + 40)))
        cursor += bw + random.randint(6, 16)
    return out.convert("RGB")


def _draw_corner_watermark_overlay(image: Image.Image, cfg: dict) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    box_w = max(16, int(w * random.uniform(*cfg.get("watermark_width_ratio_range", [0.08, 0.16]))))
    box_h = max(10, int(h * random.uniform(*cfg.get("watermark_height_ratio_range", [0.04, 0.08]))))
    margin = max(4, int(min(w, h) * 0.02))
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        x1, y1 = margin, margin
    elif corner == "tr":
        x1, y1 = w - margin - box_w, margin
    elif corner == "bl":
        x1, y1 = margin, h - margin - box_h
    else:
        x1, y1 = w - margin - box_w, h - margin - box_h
    opacity = int(255 * random.uniform(*cfg.get("overlay_opacity_range", [0.45, 0.8])))
    draw.rounded_rectangle([x1, y1, x1 + box_w, y1 + box_h], radius=3, fill=(255, 255, 255, opacity))
    return out.convert("RGB")


def build_train_transform(
    image_size: int,
    aug_cfg: dict | None = None,
    input_representation: str = "rgb",
):
    aug_cfg = aug_cfg or {}
    input_representation = validate_representation(input_representation)
    hflip_p = float(aug_cfg.get("hflip_p", 0.5))
    color_jitter = bool(aug_cfg.get("color_jitter", True))
    color_jitter_strength = float(aug_cfg.get("color_jitter_strength", 0.1))
    random_erasing = bool(aug_cfg.get("random_erasing", True))
    random_erasing_p = float(aug_cfg.get("random_erasing_p", 0.25))
    text_mask_aug = aug_cfg.get("text_mask_aug", {}) or {}
    robustness_aug = aug_cfg.get("robustness_aug", {}) or {}

    transform_steps = [
        transforms.Resize((image_size, image_size)),
        RepresentationTransform(input_representation),
        transforms.RandomHorizontalFlip(p=hflip_p),
    ]

    if color_jitter and input_representation in {"rgb", "rgb_fft"}:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(0.5 * color_jitter_strength, 0.05),
            )
        )

    if bool(text_mask_aug.get("enabled", False)):
        transform_steps.append(
            RandomBandMask(
                p=float(text_mask_aug.get("p", 0.0)),
                top_ratio_range=tuple(text_mask_aug.get("top_ratio_range", [0.0, 0.0])),
                bottom_ratio_range=tuple(text_mask_aug.get("bottom_ratio_range", [0.0, 0.0])),
                left_ratio_range=tuple(text_mask_aug.get("left_ratio_range", [0.0, 0.0])),
                right_ratio_range=tuple(text_mask_aug.get("right_ratio_range", [0.0, 0.0])),
                position_mode=str(text_mask_aug.get("position_mode", "fixed")),
                fill_mode=str(text_mask_aug.get("fill_mode", "median")),
                blur_kernel_size=int(text_mask_aug.get("blur_kernel_size", 31)),
                inpaint_radius=int(text_mask_aug.get("inpaint_radius", 3)),
            )
        )

    if bool(robustness_aug.get("enabled", False)) and input_representation in {"rgb", "rgb_fft", "rgb_fft_amplitude", "rgb_fft_phase"}:
        transform_steps.append(RobustnessAugment(robustness_aug))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if random_erasing:
        transform_steps.append(transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.15)))

    return transforms.Compose(
        transform_steps
    )


def build_eval_transform(image_size: int, input_representation: str = "rgb"):
    input_representation = validate_representation(input_representation)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            RepresentationTransform(input_representation),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
