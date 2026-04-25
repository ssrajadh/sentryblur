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
