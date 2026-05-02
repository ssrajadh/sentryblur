# Mirror of sentrysearch/_toolkit_cache.py. Keep schema in sync.
# Schema changes require version bumps in both repos.
"""Shared "last clip" cache for cross-tool integration.

Self-contained: no other sentryblur imports. Designed to be a verbatim
mirror of the same module in sibling tools (e.g. sentrysearch) so they
share the cache file format and read/write semantics.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Shared with sentrysearch — both tools read/write ~/.sentrysearch/last_clip.json.
_CACHE_DIR_NAME = ".sentrysearch"
_CACHE_FILENAME = "last_clip.json"
_SCHEMA_VERSION = 1


def _cache_path() -> Path:
    return Path.home() / _CACHE_DIR_NAME / _CACHE_FILENAME


@dataclass(frozen=True)
class LastClip:
    path: Path
    saved_at: datetime
    saved_by: str

    @property
    def age_seconds(self) -> int:
        now = datetime.now(timezone.utc)
        return int((now - self.saved_at).total_seconds())

    @property
    def file_exists(self) -> bool:
        return self.path.is_file()


def write_last_clip(path: Path, saved_by: str = "sentryblur") -> None:
    """Atomically write the cache file."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {path}")

    cache_file = _cache_path()
    cache_dir = cache_file.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": _SCHEMA_VERSION,
        "path": str(path),
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "saved_by": saved_by,
    }

    fd, tmp_name = tempfile.mkstemp(
        prefix=_CACHE_FILENAME + ".", suffix=".tmp", dir=str(cache_dir),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, cache_file)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_last_clip() -> Optional[LastClip]:
    """Return the cached entry, or None if missing/corrupt/wrong-version."""
    cache_file = _cache_path()
    try:
        with open(cache_file) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict) or data.get("version") != _SCHEMA_VERSION:
        return None

    try:
        path = Path(data["path"])
        saved_at_str = data["saved_at"]
        saved_by = data["saved_by"]
    except (KeyError, TypeError):
        return None

    try:
        # Accept the "Z" suffix that we write, plus any ISO-8601 form
        # fromisoformat understands.
        if saved_at_str.endswith("Z"):
            saved_at = datetime.fromisoformat(saved_at_str[:-1]).replace(
                tzinfo=timezone.utc,
            )
        else:
            saved_at = datetime.fromisoformat(saved_at_str)
            if saved_at.tzinfo is None:
                saved_at = saved_at.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None

    if not isinstance(saved_by, str):
        return None

    return LastClip(path=path, saved_at=saved_at, saved_by=saved_by)
