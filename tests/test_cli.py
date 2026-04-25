"""Smoke tests for the CLI. Heavy detector tests live elsewhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sentryblur.cli import cli, _default_output
from sentryblur.pipeline import blur_video


class _DummyDetector:
    """Detects nothing — verifies the pipeline runs end-to-end without ML deps."""
    name = "dummy"

    def detect(self, _frame_bgr: np.ndarray) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float32)


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


def test_default_output():
    assert _default_output(Path("foo.mp4")) == Path("foo_blurred.mp4")
    assert _default_output(Path("/x/y/clip.mov")) == Path("/x/y/clip_blurred.mov")


def test_cli_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "faces" in result.output


def test_faces_help():
    result = CliRunner().invoke(cli, ["faces", "--help"])
    assert result.exit_code == 0
    assert "INPUT_PATH" in result.output


def test_faces_missing_input():
    result = CliRunner().invoke(cli, ["faces", "/nonexistent/path.mp4"])
    assert result.exit_code != 0


def test_pipeline_runs_with_dummy_detector(tmp_path: Path):
    """End-to-end pipeline without any ML deps. Verifies frame extract,
    mask plumbing, blur application, and ffmpeg reassembly."""
    pytest.importorskip("cv2")
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_test_video(inp, frames=15)

    result = blur_video(inp, out, _DummyDetector(), dilation_px=0, temporal_window=0)

    assert out.exists()
    assert out.stat().st_size > 0
    assert result.n_frames == 15
    assert result.covered_frames == 0  # dummy detector finds nothing
    assert result.fps == 30.0


class _StubFaceDetector:
    """Stand-in for SCRFDFaceDetector — accepts the same kwargs and returns
    one fixed Nx5 detection per frame (xyxy + confidence)."""
    name = "stub"

    def __init__(self, **_kwargs):
        pass

    def detect(self, _frame_bgr: np.ndarray) -> np.ndarray:
        return np.array([[10.0, 10.0, 30.0, 30.0, 0.91]], dtype=np.float32)


def test_faces_preview_writes_jpg(tmp_path: Path, monkeypatch):
    """--preview should call generate_preview, write a JPG, and not require
    ffmpeg or any actual blur rendering."""
    pytest.importorskip("cv2")

    monkeypatch.setattr(
        "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector
    )

    inp = tmp_path / "clip.mp4"
    expected_out = tmp_path / "clip_preview.jpg"
    _make_test_video(inp, frames=12, size=(64, 64))

    result = CliRunner().invoke(cli, ["faces", str(inp), "--preview"])

    assert result.exit_code == 0, result.output
    assert "Preview saved to" in result.output
    assert expected_out.exists()
    assert expected_out.stat().st_size > 0


def test_faces_preview_explicit_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector
    )

    inp = tmp_path / "clip.mp4"
    out = tmp_path / "custom.jpg"
    _make_test_video(inp, frames=12, size=(64, 64))

    result = CliRunner().invoke(
        cli, ["faces", str(inp), "--preview", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


class _StubPlateDetector:
    """Stand-in for OpenImagePlateDetector — accepts the same kwargs and
    returns a fixed Nx5 detection per frame."""
    name = "stub-plate"

    def __init__(self, **_kwargs):
        pass

    def detect(self, _frame_bgr: np.ndarray) -> np.ndarray:
        return np.array([[5.0, 5.0, 25.0, 25.0, 0.85]], dtype=np.float32)


def test_plates_help():
    result = CliRunner().invoke(cli, ["plates", "--help"])
    assert result.exit_code == 0
    assert "INPUT_PATH" in result.output


def test_plates_blur_smoke(tmp_path: Path, monkeypatch):
    """End-to-end `plates` command with a stub detector. Verifies the
    plates wiring through _run_blur_command + blur_video."""
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.detectors.OpenImagePlateDetector", _StubPlateDetector
    )

    inp = tmp_path / "clip.mp4"
    expected_out = tmp_path / "clip_blurred.mp4"
    _make_test_video(inp, frames=10, size=(64, 64))

    result = CliRunner().invoke(cli, ["plates", str(inp)])
    assert result.exit_code == 0, result.output
    assert expected_out.exists()
    assert expected_out.stat().st_size > 0


def test_plates_preview_smoke(tmp_path: Path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.detectors.OpenImagePlateDetector", _StubPlateDetector
    )

    inp = tmp_path / "clip.mp4"
    expected_out = tmp_path / "clip_preview.jpg"
    _make_test_video(inp, frames=10, size=(64, 64))

    result = CliRunner().invoke(cli, ["plates", str(inp), "--preview"])
    assert result.exit_code == 0, result.output
    assert "Preview saved to" in result.output
    assert expected_out.exists()


def test_plates_missing_extra_clean_error(tmp_path: Path, monkeypatch):
    """Missing [plates] extra should produce a friendly install hint, not
    a stack trace."""
    pytest.importorskip("cv2")
    from sentryblur.detectors import MissingPlatesExtra

    def _raises_missing(**_kwargs):
        raise MissingPlatesExtra()

    monkeypatch.setattr(
        "sentryblur.detectors.OpenImagePlateDetector", _raises_missing
    )

    inp = tmp_path / "clip.mp4"
    _make_test_video(inp, frames=5, size=(64, 64))

    result = CliRunner().invoke(cli, ["plates", str(inp)])
    assert result.exit_code != 0
    assert "pip install 'sentryblur[plates]'" in result.output
    # Verify no traceback dumped — friendly error only.
    assert "Traceback" not in result.output
