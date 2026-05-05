"""Detector implementations. Each returns Nx4 xyxy boxes for one frame."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np


class Detector(Protocol):
    """A detector takes a BGR frame and returns an (N, 4) or (N, 5) ndarray.
    First four columns are xyxy boxes; an optional 5th column is per-box
    confidence in [0, 1] (used only by the preview renderer)."""

    name: str

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray: ...


class MissingPlatesExtra(ImportError):
    """Raised when sentryblur[plates] isn't installed but is needed."""

    def __init__(self) -> None:
        super().__init__(
            "License plate detection requires the [plates] extra.\n"
            "Install with one of:\n"
            "  uv tool install '.[plates]'                  # if installed via uv tool\n"
            "  pip install 'sentryblur[plates]'             # if installed via pip"
        )


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


class OpenImagePlateDetector:
    """License plate detector via the `open-image-models` package (MIT).

    Uses YOLOv9-T ONNX weights (~5 MB) downloaded by the library on first
    use. The library handles its own cache (typically under
    ~/.cache/open-image-models/); we route via HF_HUB_CACHE when applicable
    so weights land under ~/.cache/sentryblur/."""

    name = "open-image-yolov9-plate"

    DEFAULT_MODEL = "yolo-v9-t-384-license-plate-end2end"

    def __init__(self, *, conf: float = 0.25, model_name: str | None = None,
                 use_gpu: bool = False, cache_dir: Path | None = None):
        self._conf = conf
        self._model_name = model_name or self.DEFAULT_MODEL
        self._use_gpu = use_gpu
        self._cache_dir = (
            cache_dir
            or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "sentryblur" / self._model_name
        )
        self._detector = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return

        # Best-effort redirect of HF Hub downloads (open-image-models may or
        # may not use HF under the hood depending on version). Set the env
        # var BEFORE importing so module-level constants pick it up.
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(self._cache_dir))

        try:
            from open_image_models import LicensePlateDetector as _LPD
        except ImportError as e:
            raise MissingPlatesExtra() from e

        print(
            f"sentryblur: loading {self._model_name} "
            f"(downloads ~5 MB on first run)...",
            file=sys.stderr,
        )

        # Newer versions accept onnxruntime providers; fall back gracefully.
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_gpu else ["CPUExecutionProvider"]
        )
        try:
            self._detector = _LPD(
                detection_model=self._model_name, providers=providers,
            )
        except TypeError:
            self._detector = _LPD(detection_model=self._model_name)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        assert self._detector is not None
        detections = self._detector.predict(frame_bgr)
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        rows = []
        for d in detections:
            score = float(d.confidence)
            if score < self._conf:
                continue
            bb = d.bounding_box
            rows.append((float(bb.x1), float(bb.y1),
                         float(bb.x2), float(bb.y2), score))
        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
