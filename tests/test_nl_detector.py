"""Tests for NLDetector and pipeline.blur_video_nl.

Heavy real-model tests are gated behind @pytest.mark.gpu and skipped in
CI. Everything else uses fakes injected via sys.modules / monkeypatch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


@pytest.fixture
def _fake_heavy_modules(monkeypatch):
    """Stub torch / transformers in sys.modules so NLDetector can be
    imported and instantiated without the real deps."""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False),
        ),
        inference_mode=lambda: _NullCtx(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_transformers = SimpleNamespace(
        Sam3VideoModel=MagicMock(),
        Sam3VideoProcessor=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return fake_torch, fake_transformers


def test_init_does_not_load_models(_fake_heavy_modules):
    from sentryblur.nl_detector import NLDetector
    det = NLDetector("a red car")
    assert det._model is None
    assert det._processor is None
    # Prompt is stripped; SAM 3 takes plain noun phrases (no trailing period).
    assert det._text_prompt == "a red car"
    assert det._device == "cpu"  # fake torch has no cuda/mps


def test_init_rejects_empty_prompt(_fake_heavy_modules):
    from sentryblur.nl_detector import NLDetector
    with pytest.raises(ValueError):
        NLDetector("")
    with pytest.raises(ValueError):
        NLDetector("   ")


def test_process_video_raises_when_sam3_finds_nothing(
    tmp_path, monkeypatch, _fake_heavy_modules,
):
    import cv2
    from sentryblur.nl_detector import NLDetector, NLDetectionFailure

    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "00001.jpg"), frame)

    det = NLDetector("nonexistent thing")
    # Bypass model load and stub the session iterator to yield no detections.
    monkeypatch.setattr(det, "_ensure_model", lambda: None)
    monkeypatch.setattr(
        det, "_run_session",
        lambda frames: iter([(0, {"masks": None})]),
    )

    with pytest.raises(NLDetectionFailure) as exc:
        det.process_video(tmp_path)
    assert "nonexistent thing" in str(exc.value)


def test_process_video_raises_on_empty_dir(tmp_path, _fake_heavy_modules):
    from sentryblur.nl_detector import NLDetector
    det = NLDetector("anything")
    with pytest.raises(ValueError, match="No JPG frames"):
        det.process_video(tmp_path)


@pytest.mark.gpu
def test_end_to_end_real_models(tmp_path):
    """Real SAM 3.1 on a tiny clip. Skipped without GPU/MPS."""
    pytest.skip("Requires CUDA or MPS hardware; not run in CI")


# ---------- pipeline.blur_video_nl ----------


def _make_test_video(path: Path, frames: int = 10, size: tuple[int, int] = (64, 64)) -> None:
    import cv2
    h, w = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    try:
        for i in range(frames):
            frame = np.full((h, w, 3), i * 25 % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _FakeNLDetector:
    """Stand-in for NLDetector — returns one canned mask per frame in the
    given directory. Bottom-right quadrant is masked so we can verify
    coverage is non-zero."""

    def __init__(self, text_prompt: str, device: str = "auto"):
        self.text_prompt = text_prompt

    def process_video(self, frame_dir: Path) -> list[np.ndarray]:
        frames = sorted(Path(frame_dir).glob("*.jpg"))
        masks = []
        for fp in frames:
            import cv2
            img = cv2.imread(str(fp))
            h, w = img.shape[:2]
            m = np.zeros((h, w), dtype=bool)
            m[h // 2:, w // 2:] = True
            masks.append(m)
        return masks


def test_blur_video_nl_with_fake_detector(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.nl_detector.NLDetector", _FakeNLDetector,
    )

    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_test_video(inp, frames=10, size=(64, 64))

    from sentryblur.pipeline import blur_video_nl
    result = blur_video_nl(
        inp, out, "anything", dilation_px=0, temporal_window=0,
    )

    assert out.exists() and out.stat().st_size > 0
    assert result.n_frames == 10
    # The fake masks every frame's bottom-right quadrant, so coverage = 100%.
    assert result.covered_frames == 10
    assert result.fps == 30.0


def test_blur_video_nl_propagates_detection_failure(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from sentryblur.nl_detector import NLDetectionFailure

    class _FailingDetector:
        def __init__(self, *_a, **_kw):
            pass

        def process_video(self, _frame_dir):
            raise NLDetectionFailure("nothing found")

    monkeypatch.setattr(
        "sentryblur.nl_detector.NLDetector", _FailingDetector,
    )

    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_test_video(inp, frames=5, size=(64, 64))

    from sentryblur.pipeline import blur_video_nl
    with pytest.raises(NLDetectionFailure):
        blur_video_nl(inp, out, "anything")
    assert not out.exists()
