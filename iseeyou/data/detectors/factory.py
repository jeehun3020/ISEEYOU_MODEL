from __future__ import annotations

import numpy as np

from .base import BaseFaceDetector


class NoOpFaceDetector(BaseFaceDetector):
    def detect(self, image_rgb: np.ndarray):
        return []


def build_face_detector(detector_cfg: dict) -> BaseFaceDetector:
    name = detector_cfg.get("name", "mtcnn").lower()

    if name in {"none", "noop", "disabled"}:
        return NoOpFaceDetector()

    if name == "mtcnn":
        from .mtcnn_detector import MTCNNFaceDetector

        return MTCNNFaceDetector(
            device=detector_cfg.get("device", "auto"),
            min_face_size=detector_cfg.get("min_face_size", 40),
            keep_all=detector_cfg.get("keep_all", True),
            thresholds=tuple(detector_cfg.get("thresholds", [0.6, 0.7, 0.7])),
        )

    if name == "retinaface":
        from .mtcnn_detector import RetinaFaceDetectorPlaceholder

        return RetinaFaceDetectorPlaceholder()

    raise ValueError(f"Unsupported detector backend: {name}")
