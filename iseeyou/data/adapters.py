from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class RawSample:
    dataset: str
    path: Path
    rel_path: str
    media_type: str  # video | image
    class_name: str
    video_id: str
    identity_id: str
    source_id: str
    original_id: str
    platform_id: str = ""
    creator_account: str = ""
    generator_family: str = ""
    template_id: str = ""
    prompt_id: str = ""
    scene_id: str = ""
    source_url: str = ""
    source_family: str = ""
    raw_asset_group: str = ""
    upload_pipeline: str = ""

    @property
    def sample_id(self) -> str:
        raw = f"{self.dataset}::{self.rel_path}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _scan_files(root: Path, exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("._") or name.startswith("."):
            continue
        if path.suffix.lower() in exts:
            files.append(path)
    return sorted(files)


def _identity_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        if part.startswith("id") and len(part) >= 3:
            return part
    return parts[0] if parts else ""


def _cfg_text(cfg: dict, key: str, default: str) -> str:
    return str(cfg.get(key, default))


def _cfg_limit(cfg: dict) -> int:
    return int(cfg.get("max_samples", 0) or 0)


def parse_ucf101(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "ucf101")
    creator_account = _cfg_text(cfg, "creator_account", "ucf101")
    generator_family = _cfg_text(cfg, "generator_family", "")
    source_family = _cfg_text(cfg, "source_family", "benchmark_video")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "benchmark_native")

    for path in _scan_files(root, VIDEO_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id="",
                source_id=stem,
                original_id=stem,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=generator_family,
                scene_id=rel.split("/")[0] if "/" in rel else stem,
                source_family=source_family,
                raw_asset_group=stem,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_voxceleb2(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "voxceleb2")
    creator_account = _cfg_text(cfg, "creator_account", "voxceleb2")
    generator_family = _cfg_text(cfg, "generator_family", "")
    source_family = _cfg_text(cfg, "source_family", "benchmark_video")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "benchmark_native")

    for path in _scan_files(root, VIDEO_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        identity = _identity_from_parts(rel_path.parts)

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity,
                source_id=rel_path.parent.as_posix(),
                original_id=path.stem,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=generator_family,
                scene_id=rel_path.parts[0] if rel_path.parts else "",
                source_family=source_family,
                raw_asset_group=identity,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_stylegan(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "generated")
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "stylegan")
    creator_account = _cfg_text(cfg, "creator_account", "stylegan")
    generator_family = _cfg_text(cfg, "generator_family", "stylegan")
    source_family = _cfg_text(cfg, "source_family", "synthetic_image")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "synthetic_native")

    for path in _scan_files(root, IMAGE_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        identity = stem.split("_")[0]

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="image",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity,
                source_id=identity,
                original_id=stem,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=generator_family,
                source_family=source_family,
                raw_asset_group=identity,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_faceforensicspp(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    include_original_sequences = cfg.get("include_original_sequences", False)
    deepfake_dirs = {
        "deepfakedetection",
        "deepfakes",
        "faceswap",
        "face2face",
        "faceshifter",
        "neuraltextures",
    }
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "faceforensicspp")
    creator_account = _cfg_text(cfg, "creator_account", "faceforensicspp")
    source_family = _cfg_text(cfg, "source_family", "benchmark_manipulated")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "benchmark_native")

    for path in _scan_files(root, VIDEO_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        rel_lower = rel.lower()
        top_dir_lower = rel_path.parts[0].lower() if rel_path.parts else ""

        if "manipulated_sequences" in rel_lower or top_dir_lower in deepfake_dirs:
            class_name = cfg.get("class_name", "deepfake")
        elif include_original_sequences and (
            "original_sequences" in rel_lower or top_dir_lower == "original"
        ):
            class_name = "real"
        else:
            continue

        stem = path.stem
        match = re.search(r"(\d{2,3})_(\d{2,3})", stem)
        if match:
            a, b = match.group(1), match.group(2)
            original_id = "_".join(sorted([a, b]))
            identity_id = f"{a}_{b}"
        else:
            original_id = stem
            identity_id = stem

        parts = rel_path.parts
        method = "unknown"
        if "manipulated_sequences" in parts:
            idx = parts.index("manipulated_sequences")
            if idx + 1 < len(parts):
                method = parts[idx + 1]
        elif len(parts) > 0:
            method = parts[0]

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity_id,
                source_id=f"{method}:{original_id}",
                original_id=original_id,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=method if class_name != "real" else "",
                scene_id=original_id,
                source_family=source_family,
                raw_asset_group=original_id,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_celebdf(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    include_real = cfg.get("include_real", False)
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "celebdf")
    creator_account = _cfg_text(cfg, "creator_account", "celebdf")
    generator_family = _cfg_text(cfg, "generator_family", "celebdf")
    source_family = _cfg_text(cfg, "source_family", "benchmark_manipulated")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "benchmark_native")

    for path in _scan_files(root, VIDEO_EXTS | IMAGE_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        rel_lower = rel.lower()
        media_type = "video" if path.suffix.lower() in VIDEO_EXTS else "image"

        if (
            "synthesis" in rel_lower
            or "/fake/" in rel_lower
            or rel_lower.startswith("fake/")
            or rel_lower.startswith("train/fake/")
            or rel_lower.startswith("val/fake/")
            or rel_lower.startswith("test/fake/")
        ):
            class_name = cfg.get("class_name", "deepfake")
        elif include_real and (
            "/real/" in rel_lower
            or rel_lower.startswith("real/")
            or rel_lower.startswith("train/real/")
            or rel_lower.startswith("val/real/")
            or rel_lower.startswith("test/real/")
        ):
            class_name = "real"
        else:
            continue

        stem = path.stem
        parts = stem.split("_")
        id_tokens = [p for p in parts if p.startswith("id")]
        if len(id_tokens) >= 2:
            pair = sorted(id_tokens[:2])
            identity_id = f"{pair[0]}_{pair[1]}"
        elif len(id_tokens) == 1:
            identity_id = id_tokens[0]
        else:
            identity_id = parts[0] if parts else stem

        original_id = identity_id if media_type == "image" else stem

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type=media_type,
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=identity_id,
                source_id=rel_path.parent.as_posix(),
                original_id=original_id,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=generator_family if class_name != "real" else "",
                scene_id=identity_id,
                source_family=source_family,
                raw_asset_group=identity_id,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_generic(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    media_type = cfg.get("media_type", "video")
    class_name = cfg.get("class_name", "real")
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", dataset)
    creator_account = _cfg_text(cfg, "creator_account", dataset)
    generator_family = _cfg_text(cfg, "generator_family", "")
    template_id = _cfg_text(cfg, "template_id", "")
    prompt_id = _cfg_text(cfg, "prompt_id", "")
    scene_id = _cfg_text(cfg, "scene_id", "")
    source_url = _cfg_text(cfg, "source_url", "")
    source_family = _cfg_text(cfg, "source_family", dataset)
    raw_asset_group = _cfg_text(cfg, "raw_asset_group", "")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", platform_id)

    exts = VIDEO_EXTS if media_type == "video" else IMAGE_EXTS
    samples: list[RawSample] = []

    for path in _scan_files(root, exts):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel = path.relative_to(root).as_posix()
        stem = path.stem
        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type=media_type,
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id="",
                source_id=stem,
                original_id=stem,
                platform_id=platform_id,
                creator_account=creator_account,
                generator_family=generator_family,
                template_id=template_id,
                prompt_id=prompt_id,
                scene_id=scene_id,
                source_url=source_url,
                source_family=source_family,
                raw_asset_group=raw_asset_group,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_youtube_shorts(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "youtube")
    generator_family = _cfg_text(cfg, "generator_family", "")
    source_family = _cfg_text(cfg, "source_family", "youtube_shorts")
    upload_pipeline = _cfg_text(cfg, "upload_pipeline", "youtube_native")
    raw_asset_group = _cfg_text(cfg, "raw_asset_group", "")

    for path in _scan_files(root, VIDEO_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        parts = rel_path.parts
        channel_slug = parts[0] if parts else "unknown_channel"
        stem = path.stem

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=f"{dataset}::{rel}",
                identity_id=channel_slug,
                source_id=channel_slug,
                original_id=stem,
                platform_id=platform_id,
                creator_account=channel_slug,
                generator_family=generator_family,
                scene_id=channel_slug,
                source_url=f"https://www.youtube.com/shorts/{stem}",
                source_family=source_family,
                raw_asset_group=raw_asset_group,
                upload_pipeline=upload_pipeline,
            )
        )

    return samples


def parse_youtube_dataset_downloaded(dataset: str, root: Path, cfg: dict) -> list[RawSample]:
    default_class_name = cfg.get("class_name", "real")
    samples: list[RawSample] = []
    max_samples = _cfg_limit(cfg)
    platform_id = _cfg_text(cfg, "platform_id", "youtube")
    source_family = _cfg_text(cfg, "source_family", "youtube_dataset")
    default_upload_pipeline = _cfg_text(cfg, "upload_pipeline", "youtube_dataset_packaged")

    for path in _scan_files(root, VIDEO_EXTS):
        if max_samples > 0 and len(samples) >= max_samples:
            break
        rel_path = path.relative_to(root)
        rel = rel_path.as_posix()
        parts = rel_path.parts
        bucket_slug = parts[0] if parts else "unknown_bucket"
        stem = path.stem

        sidecar_path = path.with_suffix(".json")
        sidecar_row: dict[str, str] = {}
        if sidecar_path.exists():
            try:
                import json

                payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                raw_row = payload.get("row", {})
                if isinstance(raw_row, dict):
                    sidecar_row = {str(k): str(v) for k, v in raw_row.items()}
            except Exception:
                sidecar_row = {}

        class_name = str(sidecar_row.get("resolved_label") or sidecar_row.get("suggested_label") or default_class_name).strip().lower()
        if class_name == "fake":
            class_name = "generated"

        source_group = str(sidecar_row.get("source_group") or "").strip()
        package_category = str(sidecar_row.get("package_category") or "").strip()
        source_value = str(sidecar_row.get("source_value") or "").strip()
        split_value = str(sidecar_row.get("split") or "").strip().lower()
        generator_family = _cfg_text(cfg, "generator_family", "")
        if not generator_family and class_name == "generated":
            generator_family = (source_value or package_category or source_group or "youtube_dataset_generated").strip().lower()

        source_url = (
            str(sidecar_row.get("resolved_url") or "").strip()
            or str(sidecar_row.get("shorts_url") or "").strip()
            or str(sidecar_row.get("webpage_url") or "").strip()
            or f"https://www.youtube.com/shorts/{stem}"
        )
        raw_asset_group = (source_group or package_category or bucket_slug).strip().lower()
        upload_pipeline = str(sidecar_row.get("note") or default_upload_pipeline).strip().lower()
        index_id = str(sidecar_row.get("index_id") or "").strip()

        samples.append(
            RawSample(
                dataset=dataset,
                path=path,
                rel_path=rel,
                media_type="video",
                class_name=class_name,
                video_id=str(sidecar_row.get("video_id") or stem).strip(),
                identity_id=bucket_slug,
                source_id=(source_value or bucket_slug).strip().lower(),
                original_id=str(sidecar_row.get("video_id") or stem).strip(),
                platform_id=platform_id,
                creator_account=bucket_slug,
                generator_family=generator_family,
                template_id=package_category.lower(),
                prompt_id=index_id,
                scene_id=(source_group or bucket_slug).strip().lower(),
                source_url=source_url,
                source_family=source_family,
                raw_asset_group=raw_asset_group,
                upload_pipeline=upload_pipeline if upload_pipeline else default_upload_pipeline,
            )
        )

    return samples


PARSERS = {
    "ucf101": parse_ucf101,
    "voxceleb2": parse_voxceleb2,
    "stylegan": parse_stylegan,
    "faceforensicspp": parse_faceforensicspp,
    "celebdf": parse_celebdf,
    "generic": parse_generic,
    "youtube_shorts": parse_youtube_shorts,
    "youtube_dataset_downloaded": parse_youtube_dataset_downloaded,
}


def collect_samples_from_config(datasets_cfg: dict) -> list[RawSample]:
    all_samples: list[RawSample] = []

    for dataset_name, cfg in datasets_cfg.items():
        if not cfg.get("enabled", True):
            continue

        root = Path(cfg["root"]).expanduser()
        if not root.exists():
            print(f"[WARN] dataset root not found, skipping: {dataset_name} -> {root}")
            continue

        parser_name = cfg.get("parser", dataset_name).lower()
        parser_fn = PARSERS.get(parser_name)
        if parser_fn is None:
            print(f"[WARN] unknown parser={parser_name}, falling back to generic")
            parser_fn = parse_generic

        dataset_samples = parser_fn(dataset_name, root, cfg)
        max_samples = int(cfg.get("max_samples", 0) or 0)
        if max_samples > 0 and len(dataset_samples) > max_samples:
            dataset_samples = dataset_samples[:max_samples]
        print(f"[INFO] {dataset_name}: {len(dataset_samples)} samples")
        all_samples.extend(dataset_samples)

    return all_samples
