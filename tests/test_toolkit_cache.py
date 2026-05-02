"""Tests for sentryblur._toolkit_cache."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentryblur import _toolkit_cache
from sentryblur._toolkit_cache import (
    LastClip,
    read_last_clip,
    write_last_clip,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the cache file into tmp_path so tests never touch ~/.cache."""
    cache_file = tmp_path / "sentry-toolkit" / "last_clip.json"
    monkeypatch.setattr(_toolkit_cache, "_cache_path", lambda: cache_file)
    return cache_file


def test_round_trip(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    write_last_clip(clip)

    got = read_last_clip()
    assert got is not None
    assert got.path == clip
    assert got.saved_by == "sentryblur"
    assert got.saved_at.tzinfo is not None
    assert abs(got.age_seconds) < 5


def test_custom_saved_by(tmp_path):
    clip = tmp_path / "clip.mp4"
    write_last_clip(clip, saved_by="sentrysearch")
    got = read_last_clip()
    assert got is not None
    assert got.saved_by == "sentrysearch"


def test_read_missing_returns_none():
    assert read_last_clip() is None


def test_read_corrupt_json_returns_none(_isolated_cache):
    _isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    _isolated_cache.write_text("{not valid json")
    assert read_last_clip() is None


def test_read_wrong_version_returns_none(_isolated_cache):
    _isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    _isolated_cache.write_text(json.dumps({
        "version": 999,
        "path": "/tmp/x.mp4",
        "saved_at": "2026-04-28T15:30:00Z",
        "saved_by": "sentrysearch",
    }))
    assert read_last_clip() is None


def test_read_missing_field_returns_none(_isolated_cache):
    _isolated_cache.parent.mkdir(parents=True, exist_ok=True)
    _isolated_cache.write_text(json.dumps({"version": 1, "path": "/tmp/x.mp4"}))
    assert read_last_clip() is None


def test_relative_path_rejected():
    with pytest.raises(ValueError):
        write_last_clip(Path("relative/clip.mp4"))


def test_atomic_write_preserves_previous_on_crash(tmp_path, monkeypatch, _isolated_cache):
    # First, write a valid cache.
    clip1 = tmp_path / "first.mp4"
    write_last_clip(clip1)
    assert read_last_clip().path == clip1

    # Simulate a crash mid-write: tempfile gets created, rename fails.
    def boom(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(_toolkit_cache.os, "replace", boom)

    clip2 = tmp_path / "second.mp4"
    with pytest.raises(OSError):
        write_last_clip(clip2)

    # Previous cache must still be intact and readable.
    got = read_last_clip()
    assert got is not None
    assert got.path == clip1

    # No leftover .tmp files in the cache dir.
    leftovers = [p for p in _isolated_cache.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_atomic_write_no_prior_cache_returns_none_on_crash(tmp_path, monkeypatch, _isolated_cache):
    def boom(src, dst):
        raise OSError("simulated crash")

    monkeypatch.setattr(_toolkit_cache.os, "replace", boom)

    clip = tmp_path / "clip.mp4"
    with pytest.raises(OSError):
        write_last_clip(clip)

    assert read_last_clip() is None


def test_age_seconds(tmp_path):
    clip = tmp_path / "clip.mp4"
    saved_at = datetime.now(timezone.utc) - timedelta(seconds=42)
    lc = LastClip(path=clip, saved_at=saved_at, saved_by="sentryblur")
    assert 41 <= lc.age_seconds <= 43


def test_file_exists(tmp_path):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x")
    missing = tmp_path / "missing.mp4"

    now = datetime.now(timezone.utc)
    assert LastClip(path=real, saved_at=now, saved_by="x").file_exists is True
    assert LastClip(path=missing, saved_at=now, saved_by="x").file_exists is False


def test_write_creates_parent_dir(tmp_path, _isolated_cache):
    assert not _isolated_cache.parent.exists()
    clip = tmp_path / "clip.mp4"
    write_last_clip(clip)
    assert _isolated_cache.is_file()


def test_payload_format_on_disk(tmp_path, _isolated_cache):
    clip = tmp_path / "clip.mp4"
    write_last_clip(clip)
    data = json.loads(_isolated_cache.read_text())
    assert data["version"] == 1
    assert data["path"] == str(clip)
    assert data["saved_by"] == "sentryblur"
    assert data["saved_at"].endswith("Z")
