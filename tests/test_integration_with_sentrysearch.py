"""End-to-end cross-tool flow: a clip cached as if by sentrysearch is
picked up by `sentryblur faces --last`.

We don't actually depend on sentrysearch — we just call our own
write_last_clip() with saved_by="sentrysearch" to simulate what the
sibling tool would write. This is safe because both tools mirror the
same _toolkit_cache module against a shared schema.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sentryblur import _toolkit_cache
from sentryblur._toolkit_cache import write_last_clip
from sentryblur.cli import cli


class _StubFaceDetector:
    name = "stub"

    def __init__(self, **_kwargs):
        pass

    def detect(self, _frame_bgr: np.ndarray) -> np.ndarray:
        return np.array([[10.0, 10.0, 30.0, 30.0, 0.91]], dtype=np.float32)


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


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the shared cache file into tmp_path so we never touch
    the user's real ~/.sentrysearch/."""
    cache_file = tmp_path / "sentry-toolkit" / "last_clip.json"
    monkeypatch.setattr(_toolkit_cache, "_cache_path", lambda: cache_file)
    return cache_file


def test_sentrysearch_writes_then_sentryblur_consumes(
    tmp_path, monkeypatch, _isolated_cache,
):
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector,
    )

    clip = tmp_path / "front_cam.mp4"
    _make_test_video(clip, frames=10, size=(64, 64))

    # Simulate sentrysearch caching the clip.
    write_last_clip(clip, saved_by="sentrysearch")

    result = CliRunner().invoke(
        cli, ["faces", "--last", "--yes", "--preview"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "front_cam_preview.jpg").exists()
    assert "Error" not in result.output


def test_stale_cache_over_seven_days_errors(
    tmp_path, monkeypatch, _isolated_cache,
):
    pytest.importorskip("cv2")
    monkeypatch.setattr(
        "sentryblur.detectors.SCRFDFaceDetector", _StubFaceDetector,
    )

    clip = tmp_path / "front_cam.mp4"
    _make_test_video(clip, frames=5, size=(64, 64))

    write_last_clip(clip, saved_by="sentrysearch")

    # Rewrite the on-disk saved_at to 8 days ago. We don't tunnel through
    # the API for this — the cache file is the integration contract.
    payload = json.loads(_isolated_cache.read_text())
    eight_days_ago = datetime.now(timezone.utc) - timedelta(days=8)
    payload["saved_at"] = eight_days_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    _isolated_cache.write_text(json.dumps(payload))

    result = CliRunner().invoke(
        cli, ["faces", "--last", "--yes", "--preview"],
    )

    assert result.exit_code != 0
    assert "more than 7 days" in result.output
