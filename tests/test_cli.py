"""Smoke tests for the CLI. Heavy detector tests live elsewhere."""

from __future__ import annotations

from pathlib import Path

import click
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


def test_successful_preview_opens_output(tmp_path: Path, monkeypatch):
    """After a preview render succeeds, the CLI hands the output to the
    system file opener."""
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector,
    )
    opened: list[str] = []
    monkeypatch.setattr(
        "sentryblur.cli._open_file", lambda p: opened.append(p),
    )

    inp = tmp_path / "clip.mp4"
    expected_out = tmp_path / "clip_preview.jpg"
    _make_test_video(inp, frames=8, size=(64, 64))

    result = CliRunner().invoke(cli, ["faces", str(inp), "--preview"])
    assert result.exit_code == 0, result.output
    assert opened == [str(expected_out)]


class TestLastFlag:
    """--last resolves the input path from the sentry-toolkit cache.
    Uses `faces --preview` as the test harness — `prompt` does not
    support --last by design."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache" / "last_clip.json"
        monkeypatch.setattr(
            "sentryblur._toolkit_cache._cache_path", lambda: cache_file,
        )
        monkeypatch.setattr(
            "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector,
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
        result = CliRunner().invoke(cli, ["faces", "--last"])
        assert result.exit_code == 1
        assert "No cached clip found" in result.output

    def test_last_with_valid_cache_confirms_then_proceeds(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=8, size=(64, 64))
        self._seed_cache(_setup, path=clip)

        result = CliRunner().invoke(
            cli, ["faces", "--last", "--preview"], input="y\n",
        )
        assert "Process this clip?" in result.output
        assert str(clip) in result.output
        assert result.exit_code == 0, result.output
        assert (tmp_path / "clip_preview.jpg").exists()

    def test_last_with_valid_cache_and_yes_skips_confirm(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=8, size=(64, 64))
        self._seed_cache(_setup, path=clip)

        result = CliRunner().invoke(
            cli, ["faces", "--last", "--yes", "--preview"],
        )
        assert "Process this clip?" not in result.output
        assert result.exit_code == 0, result.output

    def test_last_with_missing_file_errors(self, tmp_path, _setup):
        gone = tmp_path / "deleted.mp4"  # never created
        self._seed_cache(_setup, path=gone)

        result = CliRunner().invoke(cli, ["faces", "--last", "--yes"])
        assert result.exit_code == 1
        assert "no longer exists" in result.output
        assert str(gone) in result.output

    def test_last_with_too_old_cache_errors(self, tmp_path, _setup):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")  # never read — resolver rejects on age
        self._seed_cache(_setup, path=clip, age_seconds=8 * 24 * 3600)

        result = CliRunner().invoke(cli, ["faces", "--last", "--yes"])
        assert result.exit_code == 1
        assert "more than 7 days old" in result.output

    def test_last_with_stale_cache_warns_and_proceeds(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=8, size=(64, 64))
        self._seed_cache(_setup, path=clip, age_seconds=2 * 3600)

        result = CliRunner().invoke(
            cli, ["faces", "--last", "--yes", "--preview"],
        )
        assert "cached clip is from" in result.output
        assert "ago" in result.output
        assert result.exit_code == 0, result.output

    def test_last_and_input_both_provided_is_usage_error(self, tmp_path, _setup):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        self._seed_cache(_setup, path=clip)

        result = CliRunner().invoke(
            cli, ["faces", str(clip), "--last"],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_neither_last_nor_input_is_missing_argument(self):
        result = CliRunner().invoke(cli, ["faces"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output


class TestPromptCommand:
    """Tests for `sentryblur prompt INPUT TEXT_PROMPT`."""

    @pytest.fixture
    def _force_tier(self, monkeypatch):
        from sentryblur.limits import HardwareTier
        def _set(tier: HardwareTier):
            monkeypatch.setattr(
                "sentryblur.limits.detect_hardware_tier", lambda: tier,
            )
            # Bypass duration gate for these tests; gating is covered in
            # test_limits.py.
            monkeypatch.setattr(
                "sentryblur.limits.check_clip_length_for_prompt",
                lambda *a, **kw: None,
            )
        return _set

    def test_help_does_not_load_torch(self, monkeypatch):
        # If `prompt --help` triggered the torch import, that would
        # blow context for users without [prompt] installed. Verify by
        # asserting torch is NOT in sys.modules after --help (best
        # effort: only meaningful if torch wasn't already imported).
        import sys as _sys
        torch_was_loaded = "torch" in _sys.modules
        result = CliRunner().invoke(cli, ["prompt", "--help"])
        assert result.exit_code == 0
        assert "TEXT_PROMPT" in result.output
        assert "natural language" in result.output.lower()
        if not torch_was_loaded:
            assert "torch" not in _sys.modules, (
                "prompt --help triggered a torch import"
            )

    def test_cpu_tier_errors(self, tmp_path, _force_tier):
        from sentryblur.limits import HardwareTier
        _force_tier(HardwareTier.CPU)
        inp = tmp_path / "clip.mp4"
        inp.write_bytes(b"fake")  # validation passes; CPU check fires first

        result = CliRunner().invoke(cli, ["prompt", str(inp), "face"])
        assert result.exit_code == 1
        assert "CPU is not supported" in result.output
        assert "sentryblur faces" in result.output

    def test_missing_text_prompt_is_usage_error(self, tmp_path):
        inp = tmp_path / "clip.mp4"
        inp.write_bytes(b"fake")
        result = CliRunner().invoke(cli, ["prompt", str(inp)])
        assert result.exit_code == 2
        assert "Missing argument" in result.output

    def test_full_path_runs_with_mocked_detector(
        self, tmp_path, monkeypatch, _force_tier,
    ):
        pytest.importorskip("cv2")
        from sentryblur.limits import HardwareTier
        _force_tier(HardwareTier.MPS_MID)

        inp = tmp_path / "clip.mp4"
        out = tmp_path / "out.mp4"
        _make_test_video(inp, frames=8, size=(64, 64))

        class _FakeNL:
            def __init__(self, text_prompt, device="auto"):
                pass
            def process_video(self, frame_dir):
                from pathlib import Path as _P
                frames = sorted(_P(frame_dir).glob("*.jpg"))
                return [np.ones((64, 64), dtype=bool) for _ in frames]

        monkeypatch.setattr("sentryblur.nl_detector.NLDetector", _FakeNL)

        result = CliRunner().invoke(
            cli, ["prompt", str(inp), "anything", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert "Done" in result.output

    def test_detection_failure_clean_error(
        self, tmp_path, monkeypatch, _force_tier,
    ):
        pytest.importorskip("cv2")
        from sentryblur.limits import HardwareTier
        from sentryblur.nl_detector import NLDetectionFailure
        _force_tier(HardwareTier.MPS_MID)

        inp = tmp_path / "clip.mp4"
        _make_test_video(inp, frames=5, size=(64, 64))

        class _FailingNL:
            def __init__(self, *_a, **_kw):
                pass
            def process_video(self, _frame_dir):
                raise NLDetectionFailure("inner detail")

        monkeypatch.setattr("sentryblur.nl_detector.NLDetector", _FailingNL)

        result = CliRunner().invoke(cli, ["prompt", str(inp), "unicorn"])
        assert result.exit_code == 1
        assert "Could not find 'unicorn' anywhere in the clip" in result.output
        assert "more specific or different prompt" in result.output
        assert "Traceback" not in result.output

    def test_preview_calls_generate_preview_nl(
        self, tmp_path, monkeypatch, _force_tier,
    ):
        pytest.importorskip("cv2")
        from sentryblur.limits import HardwareTier
        _force_tier(HardwareTier.MPS_MID)

        inp = tmp_path / "clip.mp4"
        _make_test_video(inp, frames=8, size=(64, 64))

        called = {"preview": 0, "blur": 0}

        def _fake_preview(input_path, output_path, text_prompt, **_kw):
            called["preview"] += 1
            output_path.write_bytes(b"fake-jpg")
            return output_path

        def _fake_blur(*_a, **_kw):
            called["blur"] += 1
            raise AssertionError("blur_video_nl should not be called with --preview")

        monkeypatch.setattr(
            "sentryblur.pipeline.generate_preview_nl", _fake_preview,
        )
        monkeypatch.setattr(
            "sentryblur.pipeline.blur_video_nl", _fake_blur,
        )

        result = CliRunner().invoke(
            cli, ["prompt", str(inp), "phone screen", "--preview"],
        )
        assert result.exit_code == 0, result.output
        assert called["preview"] == 1
        assert called["blur"] == 0
        assert (tmp_path / "clip_preview.jpg").exists()

    def test_preview_skips_duration_check(self, tmp_path, monkeypatch):
        """--preview only runs SAM 3 on 9 keyframes (no full propagation), so
        runtime is ~constant in clip length. The duration confirmation prompt
        — calibrated for full propagation — must not fire in preview mode.
        Regression test: previously check_clip_length_for_prompt ran
        unconditionally, before the `if not preview` guard."""
        pytest.importorskip("cv2")
        from sentryblur.limits import HardwareTier

        monkeypatch.setattr(
            "sentryblur.limits.detect_hardware_tier",
            lambda: HardwareTier.MPS_MID,
        )

        check_calls: list[tuple] = []

        def _spy_check(*args, **kwargs):
            check_calls.append((args, kwargs))

        monkeypatch.setattr(
            "sentryblur.limits.check_clip_length_for_prompt", _spy_check,
        )

        def _fake_preview(input_path, output_path, text_prompt, **_kw):
            output_path.write_bytes(b"fake-jpg")
            return output_path

        monkeypatch.setattr(
            "sentryblur.pipeline.generate_preview_nl", _fake_preview,
        )

        inp = tmp_path / "clip.mp4"
        _make_test_video(inp, frames=8, size=(64, 64))

        result = CliRunner().invoke(
            cli, ["prompt", str(inp), "phone screen", "--preview"],
        )
        assert result.exit_code == 0, result.output
        assert check_calls == [], (
            f"check_clip_length_for_prompt was called in --preview mode: "
            f"{check_calls!r}"
        )


class TestPromptLastFlag:
    """--last support on the prompt subcommand. Mirrors TestLastFlag's
    coverage but for the prompt path; mocks NLDetector + hardware tier
    so torch isn't required."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cache" / "last_clip.json"
        monkeypatch.setattr(
            "sentryblur._toolkit_cache._cache_path", lambda: cache_file,
        )
        from sentryblur.limits import HardwareTier
        monkeypatch.setattr(
            "sentryblur.limits.detect_hardware_tier",
            lambda: HardwareTier.MPS_MID,
        )
        monkeypatch.setattr(
            "sentryblur.limits.check_clip_length_for_prompt",
            lambda *a, **kw: None,
        )

        class _FakeNL:
            def __init__(self, text_prompt, device="auto"):
                pass
            def process_video(self, frame_dir):
                from pathlib import Path as _P
                frames = sorted(_P(frame_dir).glob("*.jpg"))
                return [np.ones((64, 64), dtype=bool) for _ in frames]

        monkeypatch.setattr(
            "sentryblur.nl_detector.NLDetector", _FakeNL,
        )
        return cache_file

    def _seed_cache(self, cache_file, *, path, age_seconds=0,
                    saved_by="sentrysearch"):
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

    def test_empty_cache_errors(self):
        result = CliRunner().invoke(cli, ["prompt", "--last", "face"])
        assert result.exit_code == 1
        assert "No cached clip found" in result.output

    def test_valid_cache_with_yes_proceeds(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=8, size=(64, 64))
        self._seed_cache(_setup, path=clip)

        result = CliRunner().invoke(
            cli, ["prompt", "--last", "--yes", "face"],
        )
        assert result.exit_code == 0, result.output
        assert "Done" in result.output

    def test_stale_cache_warns_and_proceeds(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=8, size=(64, 64))
        self._seed_cache(_setup, path=clip, age_seconds=2 * 3600)

        result = CliRunner().invoke(
            cli, ["prompt", "--last", "--yes", "face"],
        )
        assert "cached clip is from" in result.output
        assert "ago" in result.output
        assert result.exit_code == 0, result.output

    def test_too_old_cache_errors(self, tmp_path, _setup):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")  # exists, but never read — age check fires first
        self._seed_cache(_setup, path=clip, age_seconds=8 * 24 * 3600)

        result = CliRunner().invoke(
            cli, ["prompt", "--last", "--yes", "face"],
        )
        assert result.exit_code == 1
        assert "more than 7 days old" in result.output

    def test_input_and_last_mutually_exclusive(self, tmp_path, _setup):
        pytest.importorskip("cv2")
        clip = tmp_path / "clip.mp4"
        _make_test_video(clip, frames=4, size=(64, 64))
        self._seed_cache(_setup, path=clip)

        result = CliRunner().invoke(
            cli, ["prompt", str(clip), "face", "--last"],
        )
        assert result.exit_code == 2
        assert "mutually exclusive" in result.output

    def test_last_without_text_prompt_errors(self):
        result = CliRunner().invoke(cli, ["prompt", "--last"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output

    def test_no_input_no_last_errors(self):
        result = CliRunner().invoke(cli, ["prompt"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output


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
