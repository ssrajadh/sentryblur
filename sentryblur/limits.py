"""Clip duration gating with hardware-aware time estimates.

Used by the `prompt` subcommand to warn users before kicking off a long
local-VLM run on a clip that is too long for their hardware.
"""

from __future__ import annotations

import enum
import math
import subprocess
import sys
from pathlib import Path

import click


class HardwareTier(enum.Enum):
    CUDA_HIGH = "cuda_high"
    CUDA_MID = "cuda_mid"
    MPS_HIGH = "mps_high"
    MPS_MID = "mps_mid"
    MPS_LOW = "mps_low"
    CPU = "cpu"


# (low_realtime, high_realtime). Equal values mean a validated measurement.
HARDWARE_MULTIPLIERS = {
    HardwareTier.CUDA_HIGH: (2, 5),
    HardwareTier.CUDA_MID: (5, 12),
    HardwareTier.MPS_HIGH: (10, 20),
    HardwareTier.MPS_MID: (25, 25),
    HardwareTier.MPS_LOW: (35, 50),
    HardwareTier.CPU: (150, 300),
}

HARDWARE_DISPLAY_NAMES = {
    HardwareTier.CUDA_HIGH: "NVIDIA GPU (16GB+)",
    HardwareTier.CUDA_MID: "NVIDIA GPU",
    HardwareTier.MPS_HIGH: "Apple Silicon (M2 Pro+)",
    HardwareTier.MPS_MID: "Apple M1 Pro / M1 Max",
    HardwareTier.MPS_LOW: "Apple M1",
    HardwareTier.CPU: "CPU",
}

_CUDA_HIGH_VRAM_THRESHOLD = 16 * 1024 ** 3  # 16 GiB
_GATE_THRESHOLD_S = 30


def detect_hardware_tier() -> HardwareTier:
    try:
        import torch
    except ImportError:
        return HardwareTier.CPU

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.total_memory >= _CUDA_HIGH_VRAM_THRESHOLD:
                return HardwareTier.CUDA_HIGH
            return HardwareTier.CUDA_MID

        if torch.backends.mps.is_available():
            return _detect_mps_tier()
    except Exception:
        return HardwareTier.CPU

    return HardwareTier.CPU


def _detect_mps_tier() -> HardwareTier:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return HardwareTier.MPS_MID

    chip = out.strip()
    # Order matters: M1 Pro/Max must be checked before bare M1.
    if "M1 Pro" in chip or "M1 Max" in chip:
        return HardwareTier.MPS_MID
    if "M1" in chip:
        return HardwareTier.MPS_LOW
    if "M2 Pro" in chip or "M2 Max" in chip or "M2 Ultra" in chip:
        return HardwareTier.MPS_HIGH
    if "M3" in chip or "M4" in chip:
        return HardwareTier.MPS_HIGH
    return HardwareTier.MPS_MID


def get_video_duration(video_path: Path) -> float:
    duration = _opencv_duration(video_path)
    if duration is not None:
        return duration

    duration = _ffprobe_duration(video_path)
    if duration is not None:
        return duration

    raise click.ClickException(
        f"Could not determine duration of {video_path}. "
        "Both opencv and ffprobe failed.",
    )


def _opencv_duration(video_path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not (frames and fps) or math.isnan(frames) or math.isnan(fps):
        return None
    if frames <= 0 or fps <= 0:
        return None
    return float(frames) / float(fps)


def _ffprobe_duration(video_path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(video_path),
            ],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        d = float(out.strip())
    except ValueError:
        return None
    if d <= 0 or math.isnan(d):
        return None
    return d


def estimate_processing_minutes(
    duration_seconds: float,
    tier: HardwareTier,
) -> tuple[int, int]:
    low_mult, high_mult = HARDWARE_MULTIPLIERS[tier]
    low = max(1, round(duration_seconds * low_mult / 60))
    high = max(1, round(duration_seconds * high_mult / 60))
    return low, high


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}-second"
    if s % 60 == 0:
        return f"{s // 60}-minute"
    return f"{s // 60}m {s % 60}s"


def check_clip_length_for_prompt(
    video_path: Path,
    auto_confirm: bool = False,
) -> None:
    duration = get_video_duration(video_path)
    if duration < _GATE_THRESHOLD_S:
        return

    tier = detect_hardware_tier()
    low, high = estimate_processing_minutes(duration, tier)
    hw_name = HARDWARE_DISPLAY_NAMES[tier]
    duration_str = _format_duration(duration)

    if low == high:
        message = (
            f"This {duration_str} clip will take approximately "
            f"{low} minutes on {hw_name}."
        )
    else:
        message = (
            f"This {duration_str} clip will likely take "
            f"{low}-{high} minutes on {hw_name}."
        )

    click.echo(message, err=True)

    if auto_confirm:
        return

    if not click.confirm("Continue?", default=False, err=True):
        raise click.Abort()
