"""Natural-language prompted detector via SAM 3.1.

SAM 3 unifies the detector and tracker behind one text-prompted video API,
so a single call produces per-frame masks with cross-frame object identity.
Unlike the previous DINO + SAM 2 pipeline, targets can first appear at
any point in the clip — there is no frame-0 dependency.

Heavy dependencies (torch, transformers) are imported lazily in __init__
so users without the [prompt] extra get a clean ImportError at construction
time, not at module import.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


_SAM3_MODEL_ID = "facebook/sam3"


class NLDetectionFailure(Exception):
    """Raised when SAM 3 finds no targets anywhere in the clip."""


def _cache_root() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "sentryblur"
    )


class NLDetector:
    """Promptable detector. Operates on a frame directory (SAM 3's video
    session API ingests frames as a sequence) instead of single-frame
    inputs, so it does NOT implement the standard `Detector` Protocol.
    """

    name = "nl"

    def __init__(self, text_prompt: str, device: str = "auto"):
        if not text_prompt or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        self._text_prompt = text_prompt.strip()

        # Surface missing extras at construction time.
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            Sam3VideoModel,
            Sam3VideoProcessor,
        )

        self._device = self._resolve_device(device)
        # Model + processor load lazily on the first process_video()/
        # detect_boxes() call so __init__ stays cheap.
        self._model = None
        self._processor = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        cache = _cache_root() / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = Sam3VideoModel.from_pretrained(
            _SAM3_MODEL_ID, dtype=dtype, cache_dir=str(cache),
        ).to(self._device).eval()
        self._processor = Sam3VideoProcessor.from_pretrained(
            _SAM3_MODEL_ID, cache_dir=str(cache),
        )

    def _run_session(self, frames):
        """Yield (frame_idx, processed_outputs) for each frame in `frames`."""
        session = self._processor.init_video_session(
            video=frames,
            inference_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
        )
        self._processor.add_text_prompt(session, self._text_prompt)
        for model_outputs in self._model.propagate_in_video_iterator(
            inference_session=session,
        ):
            processed = self._processor.postprocess_outputs(
                session, model_outputs,
            )
            yield model_outputs.frame_idx, processed

    def detect_boxes(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Single-frame detection. Returns (N, 5) xyxy + score.
        Used by the preview path."""
        self._ensure_model()
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        import torch
        with torch.inference_mode():
            for _idx, processed in self._run_session([img]):
                boxes = processed["boxes"]
                scores = processed["scores"]
                if boxes is None or len(boxes) == 0:
                    return np.empty((0, 5), dtype=np.float32)
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy().reshape(-1, 1)
                return np.hstack([boxes, scores]).astype(np.float32)

        return np.empty((0, 5), dtype=np.float32)

    def process_video(self, frame_dir: Path) -> list[np.ndarray]:
        """Return one boolean (H, W) mask per JPG frame in `frame_dir`,
        ordered by filename."""
        frame_dir = Path(frame_dir)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise ValueError(f"No JPG frames in {frame_dir}")

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise ValueError(f"Could not read first frame: {frame_paths[0]}")
        h, w = first.shape[:2]
        n = len(frame_paths)

        from PIL import Image
        frames = [Image.open(p).convert("RGB") for p in frame_paths]

        self._ensure_model()

        import torch
        masks: list[np.ndarray] = [
            np.zeros((h, w), dtype=bool) for _ in range(n)
        ]
        found_any = False
        with torch.inference_mode():
            for frame_idx, processed in self._run_session(frames):
                if not (0 <= frame_idx < n):
                    continue
                obj_masks = processed.get("masks")
                if obj_masks is None or obj_masks.numel() == 0:
                    continue
                merged = obj_masks.any(dim=0).cpu().numpy().astype(bool)
                masks[frame_idx] = merged
                if merged.any():
                    found_any = True

        if not found_any:
            raise NLDetectionFailure(
                f"No targets found for prompt {self._text_prompt!r} anywhere "
                "in the clip. Try a more specific prompt, or verify the "
                "target is visible at some point in the video."
            )
        return masks


def make_nl_detector(
    text_prompt: str,
    backend: str = "sam3",
    device: str = "auto",
):
    """Construct the prompt-path detector for the requested backend.

    backend="sam3" (default): SAM 3.1 via HuggingFace transformers. Gated
    weights at facebook/sam3 — requires accepting Meta's SAM License and
    `huggingface-cli login` before first use.

    backend="sam2": legacy Grounding DINO + SAM 2 path. No gating, but
    requires the sam2 git install (`--with git+https://github.com/...sam2.git`)
    and has a frame-0 dependency that SAM 3 doesn't.
    """
    if backend == "sam3":
        return NLDetector(text_prompt, device=device)
    if backend == "sam2":
        from sentryblur.nl_detector_sam2 import NLDetectorSam2
        return NLDetectorSam2(text_prompt, device=device)
    raise ValueError(
        f"unknown backend {backend!r}; expected 'sam3' or 'sam2'",
    )
