"""Core blur pipeline: video -> detect -> dilate -> temporal union -> blur -> assemble."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from sentryblur.detectors import Detector


@dataclass
class BlurResult:
    n_frames: int
    fps: float
    covered_frames: int
    output_path: Path

    @property
    def coverage_pct(self) -> float:
        return 100 * self.covered_frames / self.n_frames if self.n_frames else 0.0


class FFmpegError(RuntimeError):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg failed")


def _video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Could not read fps from {path}")
    return fps


def _extract_frames(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(["-i", str(video), "-q:v", "2", f"{out_dir}/%05d.jpg"])
    return sorted(out_dir.glob("*.jpg"))


def _assemble(frame_dir: Path, fps: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "-framerate", str(fps),
        "-i", f"{frame_dir}/%05d.jpg",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output),
    ])


def _boxes_to_mask(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes.astype(int):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _temporal_union(masks: list[np.ndarray], window: int) -> list[np.ndarray]:
    if window <= 0:
        return masks
    n = len(masks)
    out = []
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        u = np.zeros_like(masks[i], dtype=bool)
        for j in range(lo, hi):
            u |= masks[j]
        out.append(u)
    return out


def _apply_blur(frame: np.ndarray, mask: np.ndarray, strength: int) -> np.ndarray:
    if not mask.any():
        return frame
    blurred = cv2.GaussianBlur(frame, (strength, strength), 0)
    out = frame.copy()
    out[mask] = blurred[mask]
    return out


def blur_video(
    input_path: Path,
    output_path: Path,
    detector: Detector,
    *,
    dilation_px: int = 15,
    temporal_window: int = 3,
    blur_strength: int = 51,
    progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> BlurResult:
    """Blur regions detected by `detector` in every frame of `input_path`,
    write to `output_path`. Atomic write via temp file."""

    if blur_strength % 2 == 0:
        blur_strength += 1  # cv2 GaussianBlur requires odd

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    t_start = time.perf_counter()
    fps = _video_fps(input_path)

    with tempfile.TemporaryDirectory(prefix="sentryblur_") as tmp:
        tmp_dir = Path(tmp)
        frames_in = tmp_dir / "in"
        frames_out = tmp_dir / "out"
        frames_out.mkdir()

        frames = _extract_frames(input_path, frames_in)
        n = len(frames)
        if n == 0:
            raise ValueError(f"No frames extracted from {input_path}")

        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]

        if verbose:
            print(
                f"sentryblur: {input_path}  {w}x{h} @ {fps:.2f} fps, {n} frames",
                file=sys.stderr,
            )

        if verbose:
            from tqdm.auto import tqdm
            frame_iter = tqdm(
                enumerate(frames), total=n, desc="Detecting",
                unit="frame", leave=False,
            )
        else:
            frame_iter = enumerate(frames)

        masks: list[np.ndarray] = []
        for i, fp in frame_iter:
            frame = cv2.imread(str(fp))
            boxes = detector.detect(frame)
            mask = _boxes_to_mask(boxes, (h, w)) if len(boxes) else np.zeros((h, w), dtype=bool)
            masks.append(mask)
            if progress:
                progress(i + 1, n)

        masks = [_dilate(m, dilation_px) for m in masks]
        masks = _temporal_union(masks, temporal_window)

        covered = 0
        for idx, (fp, m) in enumerate(zip(frames, masks)):
            frame = cv2.imread(str(fp))
            blurred = _apply_blur(frame, m, blur_strength)
            cv2.imwrite(str(frames_out / f"{idx:05d}.jpg"), blurred,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if m.any():
                covered += 1

        # Atomic write: assemble to tmp, then move to final.
        tmp_out = tmp_dir / "out.mp4"
        _assemble(frames_out, fps, tmp_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_out), str(output_path))

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(
            f"sentryblur: done in {elapsed:.1f}s -> {output_path}",
            file=sys.stderr,
        )

    return BlurResult(
        n_frames=n,
        fps=fps,
        covered_frames=covered,
        output_path=output_path,
    )
