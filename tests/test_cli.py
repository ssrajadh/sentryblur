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


class TestLastFlag:
    """--last resolves the input path from the sentry-toolkit cache."""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache" / "last_clip.json"
        monkeypatch.setattr(
            "sentryblur._toolkit_cache._cache_path", lambda: cache_file,
        )
        return cache_file

    def _seed_cache(self, cache_file, *, path, age_seconds=0, saved_by="sentrysearch"):
        import json
        from datetime import datetime, timedelta, timezone
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        saved_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
        cache_file.write_text(json.dumps({
            "version": 1,
            "path": str(path),
            "saved_at": saved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "saved_by": saved_by,
        }))

    def test_last_with_empty_cache_errors(self):
        result = CliRunner().invoke(cli, ["prompt", "anything", "--last"])
        assert result.exit_code == 1
        assert "No cached clip found" in result.output

    def test_last_with_valid_cache_confirms_then_proceeds(self, tmp_path, _isolated_cache):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_isolated_cache, path=clip)

        result = CliRunner().invoke(
            cli, ["prompt", "anything", "--last"], input="y\n",
        )
        # Confirm shown, then prompt stub's not-implemented exit (2)
        assert "Process this clip?" in result.output
        assert "not yet implemented" in result.output
        assert str(clip) in result.output
        assert result.exit_code == 2

    def test_last_with_valid_cache_and_yes_skips_confirm(self, tmp_path, _isolated_cache):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_isolated_cache, path=clip)

        result = CliRunner().invoke(
            cli, ["prompt", "anything", "--last", "--yes"],
        )
        assert "Process this clip?" not in result.output
        assert "not yet implemented" in result.output
        assert result.exit_code == 2

    def test_last_with_missing_file_errors(self, tmp_path, _isolated_cache):
        gone = tmp_path / "deleted.mp4"  # never created
        self._seed_cache(_isolated_cache, path=gone)

        result = CliRunner().invoke(cli, ["prompt", "anything", "--last", "--yes"])
        assert result.exit_code == 1
        assert "no longer exists" in result.output
        assert str(gone) in result.output

    def test_last_with_too_old_cache_errors(self, tmp_path, _isolated_cache):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_isolated_cache, path=clip, age_seconds=8 * 24 * 3600)

        result = CliRunner().invoke(cli, ["prompt", "anything", "--last", "--yes"])
        assert result.exit_code == 1
        assert "more than 7 days old" in result.output

    def test_last_with_stale_cache_warns_and_proceeds(self, tmp_path, _isolated_cache):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_isolated_cache, path=clip, age_seconds=2 * 3600)

        result = CliRunner().invoke(
            cli, ["prompt", "anything", "--last", "--yes"],
        )
        assert "cached clip is from" in result.output
        assert "ago" in result.output
        # Stub still runs, so exit 2 (not implemented)
        assert result.exit_code == 2

    def test_last_and_input_both_provided_is_usage_error(self, tmp_path, _isolated_cache):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_isolated_cache, path=clip)

        result = CliRunner().invoke(
            cli, ["prompt", "anything", str(clip), "--last"],
        )
        assert result.exit_code == 2  # click.UsageError
        assert "mutually exclusive" in result.output

    def test_neither_last_nor_input_is_missing_argument(self):
        result = CliRunner().invoke(cli, ["prompt", "anything"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output

    def test_faces_supports_last(self, tmp_path, monkeypatch, _isolated_cache):
        """--last works on faces too, not just prompt."""
        pytest.importorskip("cv2")
        monkeypatch.setattr(
            "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector,
        )
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=10, size=(64, 64))
        self._seed_cache(_isolated_cache, path=clip)

        result = CliRunner().invoke(
            cli, ["faces", "--last", "--yes", "--preview"],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "clip_preview.jpg").exists()


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
