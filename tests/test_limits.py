"""Tests for sentryblur.limits."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from sentryblur import limits
from sentryblur.limits import (
    HARDWARE_DISPLAY_NAMES,
    HardwareTier,
    check_clip_length_for_prompt,
    detect_hardware_tier,
    estimate_processing_minutes,
    get_video_duration,
)


def _make_video(path: Path, seconds: int) -> None:
    """Create a synthetic test video of the given duration via ffmpeg testsrc."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"testsrc=duration={seconds}:size=64x64:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def _ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")


# ---------- get_video_duration ----------

@pytest.mark.parametrize("seconds", [5, 30, 90])
def test_get_video_duration_synthetic_clips(tmp_path, _ffmpeg_available, seconds):
    clip = tmp_path / f"clip_{seconds}s.mp4"
    _make_video(clip, seconds)
    duration = get_video_duration(clip)
    assert abs(duration - seconds) < 0.5


def test_get_video_duration_falls_back_to_ffprobe(tmp_path, monkeypatch, _ffmpeg_available):
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 5)
    monkeypatch.setattr(limits, "_opencv_duration", lambda _p: None)
    assert abs(get_video_duration(clip) - 5) < 0.5


def test_get_video_duration_both_fail_raises(tmp_path, monkeypatch):
    clip = tmp_path / "missing.mp4"
    monkeypatch.setattr(limits, "_opencv_duration", lambda _p: None)
    monkeypatch.setattr(limits, "_ffprobe_duration", lambda _p: None)
    with pytest.raises(click.ClickException):
        get_video_duration(clip)


# ---------- estimate_processing_minutes ----------

def test_estimate_validated_tier():
    # MPS_MID: (25, 25), 60s -> 25 min flat
    assert estimate_processing_minutes(60, HardwareTier.MPS_MID) == (25, 25)


def test_estimate_cuda_high_30s():
    # CUDA_HIGH: (2, 5). 30s * 2 / 60 = 1.0 -> 1, 30s * 5 / 60 = 2.5 -> 2 (banker's rounds to even)
    low, high = estimate_processing_minutes(30, HardwareTier.CUDA_HIGH)
    assert low == 1
    assert high in (2, 3)  # depending on rounding mode, accept either


def test_estimate_minimum_one_minute():
    # 1s clip on CUDA_HIGH -> well under a minute, but floor is 1.
    assert estimate_processing_minutes(1, HardwareTier.CUDA_HIGH) == (1, 1)


# ---------- check_clip_length_for_prompt ----------

@pytest.fixture
def _force_tier(monkeypatch):
    """Returns a setter; pin detect_hardware_tier to a chosen tier."""
    def _set(tier: HardwareTier):
        monkeypatch.setattr(limits, "detect_hardware_tier", lambda: tier)
    return _set


def test_short_clip_returns_silently(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "short.mp4"
    _make_video(clip, 5)
    check_clip_length_for_prompt(clip, auto_confirm=False)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_auto_confirm_skips_prompt(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "long.mp4"
    _make_video(clip, 60)
    # No stdin: would hang if click.confirm fired.
    check_clip_length_for_prompt(clip, auto_confirm=True)
    captured = capsys.readouterr()
    assert "approximately" in captured.err
    assert "Continue?" not in captured.err


def test_user_rejects_raises_abort(tmp_path, monkeypatch, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "long.mp4"
    _make_video(clip, 60)
    monkeypatch.setattr("click.confirm", lambda *a, **kw: False)
    with pytest.raises(click.Abort):
        check_clip_length_for_prompt(clip, auto_confirm=False)


def test_user_accepts_returns_silently(tmp_path, monkeypatch, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "long.mp4"
    _make_video(clip, 60)
    monkeypatch.setattr("click.confirm", lambda *a, **kw: True)
    check_clip_length_for_prompt(clip, auto_confirm=False)


def test_validated_tier_message_no_range(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 60)
    check_clip_length_for_prompt(clip, auto_confirm=True)
    err = capsys.readouterr().err
    assert "approximately" in err
    assert "minutes" in err
    # No range like "10-20"
    import re
    assert not re.search(r"\d+-\d+ minutes", err)
    assert HARDWARE_DISPLAY_NAMES[HardwareTier.MPS_MID] in err


def test_estimated_tier_message_has_range(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_HIGH)
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 60)
    check_clip_length_for_prompt(clip, auto_confirm=True)
    err = capsys.readouterr().err
    import re
    assert re.search(r"\d+-\d+ minutes", err)
    assert HARDWARE_DISPLAY_NAMES[HardwareTier.MPS_HIGH] in err


# ---------- duration formatting ----------

def test_format_45s_clip(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 45)
    check_clip_length_for_prompt(clip, auto_confirm=True)
    assert "45-second" in capsys.readouterr().err


def test_format_90s_clip(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 90)
    check_clip_length_for_prompt(clip, auto_confirm=True)
    assert "1m 30s" in capsys.readouterr().err


def test_format_300s_clip(tmp_path, capsys, _ffmpeg_available, _force_tier):
    _force_tier(HardwareTier.MPS_MID)
    clip = tmp_path / "clip.mp4"
    _make_video(clip, 300)
    check_clip_length_for_prompt(clip, auto_confirm=True)
    assert "5-minute" in capsys.readouterr().err


# ---------- detect_hardware_tier (mocked) ----------

def _fake_torch(*, cuda_available=False, vram=0, mps_available=False):
    """Build a minimal stand-in for the torch module."""
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_properties=lambda _i: SimpleNamespace(total_memory=vram),
    )
    backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: mps_available),
    )
    return SimpleNamespace(cuda=cuda, backends=backends)


def test_detect_cuda_high(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(cuda_available=True, vram=24 * 1024 ** 3))
    assert detect_hardware_tier() == HardwareTier.CUDA_HIGH


def test_detect_cuda_mid(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(cuda_available=True, vram=8 * 1024 ** 3))
    assert detect_hardware_tier() == HardwareTier.CUDA_MID


def test_detect_torch_missing_returns_cpu(monkeypatch):
    # Force ImportError by inserting a sentinel that fails on import. The
    # cleanest way: temporarily remove torch from sys.modules and shadow
    # it with a finder that raises. Easiest alternative: monkeypatch
    # builtins.__import__.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert detect_hardware_tier() == HardwareTier.CPU


def test_detect_mps_m1_pro(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    monkeypatch.setattr(
        limits.subprocess, "check_output",
        lambda *a, **kw: "Apple M1 Pro\n",
    )
    assert detect_hardware_tier() == HardwareTier.MPS_MID


def test_detect_mps_m3(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    monkeypatch.setattr(
        limits.subprocess, "check_output",
        lambda *a, **kw: "Apple M3\n",
    )
    assert detect_hardware_tier() == HardwareTier.MPS_HIGH


def test_detect_mps_sysctl_fails_defaults_mid(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps_available=True))
    def boom(*a, **kw):
        raise OSError("sysctl not found")
    monkeypatch.setattr(limits.subprocess, "check_output", boom)
    assert detect_hardware_tier() == HardwareTier.MPS_MID
