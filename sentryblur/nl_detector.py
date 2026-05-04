"""Natural-language prompted detector via Grounding DINO + SAM 2.

DINO finds the target boxes from a text prompt on frame 0; SAM 2's video
predictor then propagates those targets across every frame, producing a
precise polygonal mask per frame. The masks are returned directly (not
boxes) so the downstream blur pipeline can apply tight redaction.

Heavy dependencies (torch, transformers, sam2) are imported lazily in
__init__ so users without the [prompt] extra get a clean ImportError at
construction time, not at module import.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


_SAM2_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"
_SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_tiny.pt"
)
_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_DINO_BOX_THRESHOLD = 0.35
_DINO_TEXT_THRESHOLD = 0.25


class NLDetectionFailure(Exception):
    """Raised when DINO finds no targets in frame 0."""


def _cache_root() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "sentryblur"
    )


class NLDetector:
    """Promptable detector. Operates on a frame directory (SAM 2's video
    predictor requirement) instead of single-frame inputs, so it does
    NOT implement the standard `Detector` Protocol.
    """

    name = "nl"

    def __init__(self, text_prompt: str, device: str = "auto"):
        if not text_prompt or not text_prompt.strip():
            raise ValueError("text_prompt must be a non-empty string")
        prompt = text_prompt.strip()
        if not prompt.endswith("."):
            prompt += "."
        self._text_prompt = prompt

        # Surface missing extras at construction time.
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401

        self._device = self._resolve_device(device)
        # Models are loaded lazily on the first process_video() call so
        # __init__ stays cheap and import-error-only.
        self._dino_processor = None
        self._dino_model = None
        self._sam_predictor = None

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

    def _ensure_dino(self) -> None:
        if self._dino_model is not None:
            return
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
        cache = _cache_root() / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))
        self._dino_processor = AutoProcessor.from_pretrained(
            _DINO_MODEL_ID, cache_dir=str(cache),
        )
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            _DINO_MODEL_ID, cache_dir=str(cache),
        ).to(self._device)

    def _ensure_sam(self) -> None:
        if self._sam_predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor
        cache = _cache_root() / "sam2"
        cache.mkdir(parents=True, exist_ok=True)
        ckpt = cache / _SAM2_CHECKPOINT_NAME
        if not ckpt.exists():
            print(
                f"sentryblur: downloading SAM 2 checkpoint ({_SAM2_CHECKPOINT_NAME}, ~149 MB)...",
                file=sys.stderr,
            )
            tmp = ckpt.with_suffix(ckpt.suffix + ".part")
            urllib.request.urlretrieve(_SAM2_CHECKPOINT_URL, str(tmp))
            tmp.replace(ckpt)
        self._sam_predictor = build_sam2_video_predictor(
            _SAM2_CONFIG, str(ckpt), device=self._device,
        )

    def detect_boxes(self, frame_bgr: np.ndarray) -> np.ndarray:
        """DINO-only single-frame detection. Returns (N, 5) xyxy + score.
        Used by the preview path; does not invoke SAM 2."""
        self._ensure_dino()
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self._dino_processor(
            images=image, text=self._text_prompt, return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            outputs = self._dino_model(**inputs)
        h, w = frame_bgr.shape[:2]
        results = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=_DINO_BOX_THRESHOLD,
            text_threshold=_DINO_TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )
        if not results:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.column_stack([boxes, scores]).astype(np.float32)

    def process_video(self, frame_dir: Path) -> list[np.ndarray]:
        """Return one boolean (H, W) mask per JPG frame in `frame_dir`,
        ordered by filename."""
        frame_dir = Path(frame_dir)
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise ValueError(f"No JPG frames in {frame_dir}")

        first = cv2.imread(str(frames[0]))
        if first is None:
            raise ValueError(f"Could not read first frame: {frames[0]}")
        detections = self.detect_boxes(first)  # (N, 5) xyxy + score
        if len(detections) == 0:
            raise NLDetectionFailure(
                f"No targets found for prompt {self._text_prompt!r} in the "
                "first frame. Try a more specific prompt, or verify the "
                "target is visible at the start of the clip."
            )
        boxes = detections[:, :4]

        self._ensure_sam()
        h, w = first.shape[:2]
        n = len(frames)

        import torch
        masks: list[np.ndarray] = [
            np.zeros((h, w), dtype=bool) for _ in range(n)
        ]
        with torch.inference_mode():
            # Required on MPS — without these flags, unified memory blows
            # up on clips longer than ~20 seconds.
            state = self._sam_predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            self._sam_predictor.reset_state(state)
            for obj_id in range(boxes.shape[0]):
                self._sam_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.asarray(boxes[obj_id], dtype=np.float32),
                )
            for frame_idx, _obj_ids, mask_logits in (
                self._sam_predictor.propagate_in_video(state)
            ):
                if not (0 <= frame_idx < n):
                    continue
                merged = np.zeros((h, w), dtype=bool)
                for k in range(mask_logits.shape[0]):
                    merged |= (mask_logits[k, 0] > 0).cpu().numpy()
                masks[frame_idx] = merged
        return masks