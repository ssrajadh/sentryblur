"""Pipeline tests — temporal smoothing, dilation, mask plumbing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sentryblur.pipeline import blur_video


# Target region: a hard black/white edge so blur produces detectable change
# despite h264 quantization noise on the output.
_TARGET = (16, 16, 32, 32)  # x0, y0, x1, y1
_FRAME_HW = (64, 64)


def _make_high_contrast_video(path: Path, frames: int = 10) -> None:
    h, w = _FRAME_HW
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    try:
        for _ in range(frames):
            frame = np.full((h, w, 3), 255, dtype=np.uint8)
            x0, y0, x1, y1 = _TARGET
            frame[y0:y1, x0:x1] = 0  # solid black square on white
            writer.write(frame)
    finally:
        writer.release()


class _DropoutOnFrameFive:
    """Detects target on every frame except frame 5."""

    name = "dropout"

    def __init__(self):
        self._idx = 0

    def detect(self, _frame_bgr: np.ndarray) -> np.ndarray:
        i = self._idx
        self._idx += 1
        if i == 5:
            return np.empty((0, 4), dtype=np.float32)
        x0, y0, x1, y1 = _TARGET
        return np.array([[x0, y0, x1, y1]], dtype=np.float32)


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def test_temporal_window_fills_single_frame_dropout(tmp_path: Path):
    """Frame 5's detector returns nothing. With temporal_window=3 the
    smoothed mask should still cover the target (filled by frames 4 and 6),
    so frame 5's target region ends up blurred. With temporal_window=0 the
    target on frame 5 must remain unblurred."""
    pytest.importorskip("cv2")

    inp = tmp_path / "in.mp4"
    _make_high_contrast_video(inp, frames=10)

    out_smooth = tmp_path / "smooth.mp4"
    out_no_smooth = tmp_path / "no_smooth.mp4"

    blur_video(inp, out_smooth, _DropoutOnFrameFive(),
               dilation_px=0, temporal_window=3,
               blur_mode="gaussian", blur_strength=51)
    blur_video(inp, out_no_smooth, _DropoutOnFrameFive(),
               dilation_px=0, temporal_window=0,
               blur_mode="gaussian", blur_strength=51)

    f5_smooth = _read_frame(out_smooth, 5).astype(np.int16)
    f5_no = _read_frame(out_no_smooth, 5).astype(np.int16)

    x0, y0, x1, y1 = _TARGET
    region_smooth = f5_smooth[y0:y1, x0:x1]
    region_no = f5_no[y0:y1, x0:x1]

    # Without smoothing: target on frame 5 was untouched (sharp black square).
    # Most pixels stay near 0; stddev is small.
    # With smoothing: 51-px gaussian on the black-white edge bleeds white in,
    # raising mean and stddev across the region substantially.
    assert region_smooth.mean() > region_no.mean() + 20, (
        f"smoothed frame 5 should be visibly lighter than unsmoothed; "
        f"got smooth.mean={region_smooth.mean():.1f}, "
        f"no_smooth.mean={region_no.mean():.1f}"
    )


def test_dilation_expands_mask_beyond_box(tmp_path: Path):
    """A 1-pixel dilation kernel should still leak blur slightly outside
    the raw box. Verifies dilation is wired into the pipeline at all."""
    pytest.importorskip("cv2")

    inp = tmp_path / "in.mp4"
    _make_high_contrast_video(inp, frames=5)

    class _AlwaysDetect:
        name = "always"
        def detect(self, _frame_bgr):
            x0, y0, x1, y1 = _TARGET
            return np.array([[x0, y0, x1, y1]], dtype=np.float32)

    out_no_dil = tmp_path / "no_dil.mp4"
    out_dil = tmp_path / "dil.mp4"

    blur_video(inp, out_no_dil, _AlwaysDetect(),
               dilation_px=0, temporal_window=0,
               blur_mode="gaussian", blur_strength=21)
    blur_video(inp, out_dil, _AlwaysDetect(),
               dilation_px=10, temporal_window=0,
               blur_mode="gaussian", blur_strength=21)

    f0_no = _read_frame(out_no_dil, 0).astype(np.int16)
    f0_dil = _read_frame(out_dil, 0).astype(np.int16)

    # Sample a pixel just outside the raw box but inside the dilated mask.
    x0, y0, _, _ = _TARGET
    probe = (y0 - 3, x0 - 3)  # 3 px outside the raw box
    # Without dilation, this pixel is untouched white background (~255).
    # With dilation_px=10, it falls inside the mask and gets blurred toward
    # the dark target → noticeably darker.
    assert f0_dil[probe].mean() < f0_no[probe].mean() - 10, (
        f"dilation didn't expand the blur region as expected; "
        f"no_dil={f0_no[probe].mean():.1f}, dil={f0_dil[probe].mean():.1f}"
    )
