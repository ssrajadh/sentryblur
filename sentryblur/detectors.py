"""Detector implementations. Each returns Nx4 xyxy boxes for one frame."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    """A detector takes a BGR frame and returns an (N, 4) or (N, 5) ndarray.
    First four columns are xyxy boxes; an optional 5th column is per-box
    confidence in [0, 1] (used only by the preview renderer)."""

    name: str

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray: ...


class SCRFDFaceDetector:
    """insightface SCRFD. Auto-downloads weights to ~/.insightface/ on first use."""

    name = "scrfd"

    def __init__(self, *, det_size: int = 640, conf: float = 0.25, use_gpu: bool = False):
        from insightface.app import FaceAnalysis

        self._conf = conf
        self._app = FaceAnalysis(allowed_modules=["detection"])
        self._app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        faces = self._app.get(frame_bgr)
        if not faces:
            return np.empty((0, 5), dtype=np.float32)
        rows = [(*f.bbox, float(f.det_score)) for f in faces if f.det_score >= self._conf]
        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
