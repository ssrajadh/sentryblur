"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_open_file(monkeypatch):
    """Stop the CLI's "open output in default viewer" hook from actually
    launching an image/video player during tests."""
    monkeypatch.setattr("sentryblur.cli._open_file", lambda _path: None)
