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
    if boxes.ndim == 2 and boxes.shape[1] > 4:
        boxes = boxes[:, :4]
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


def _apply_pixelate(frame: np.ndarray, mask: np.ndarray, block: int) -> np.ndarray:
    if not mask.any():
        return frame
    h, w = frame.shape[:2]
    sw, sh = max(1, w // block), max(1, h // block)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[mask] = pixelated[mask]
    return out


def blur_video(
    input_path: Path,
    output_path: Path,
    detector: Detector,
    *,
    dilation_px: int = 15,
    temporal_window: int = 3,
    blur_strength: int = 51,
    blur_mode: str = "pixelate",
    pixel_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    verbose: bool = False,
) -> BlurResult:
    """Blur regions detected by `detector` in every frame of `input_path`,
    write to `output_path`. Atomic write via temp file."""

    if blur_mode not in ("pixelate", "gaussian"):
        raise ValueError(f"unknown blur_mode: {blur_mode!r}")
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
            if blur_mode == "pixelate":
                blurred = _apply_pixelate(frame, m, pixel_size)
            else:
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


def generate_preview(
    input_path: Path,
    output_path: Path,
    detector: Detector,
    *,
    max_width: int = 1920,
) -> Path:
    """Render a 3x3 contact sheet of detector output across 9 evenly-spaced
    keyframes. Lets users verify detection quality before committing to a
    full render. Atomic write via temp file."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(str(input_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise ValueError(f"Could not read frame count from {input_path}")

        cells: list[np.ndarray] = []
        for idx in _keyframe_indices(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cells.append(_annotate_boxes(frame, detector.detect(frame)))
    finally:
        cap.release()

    return _write_3x3_grid(cells, output_path, max_width)


def _annotate_boxes(
    frame: np.ndarray,
    detections: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw xyxy boxes (and optional confidence labels) onto a copy of `frame`."""
    if detections.size == 0:
        return frame.copy()
    boxes = detections[:, :4].astype(int)
    scores = (
        detections[:, 4]
        if detections.ndim == 2 and detections.shape[1] >= 5 else None
    )
    annotated = frame.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        label = f"{scores[i]:.2f}" if scores is not None else "det"
        ty = y0 - 5 if y0 > 15 else y0 + 15
        cv2.putText(annotated, label, (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def _keyframe_indices(n: int) -> list[int]:
    raw = [0, n // 8, 2 * n // 8, 3 * n // 8, 4 * n // 8,
           5 * n // 8, 6 * n // 8, 7 * n // 8, n - 1]
    seen: list[int] = []
    for i in raw:
        if i not in seen:
            seen.append(i)
    while len(seen) < 9:
        seen.append(seen[-1])
    return seen[:9]


def _write_3x3_grid(cells: list[np.ndarray], output_path: Path, max_width: int) -> Path:
    cell_w = max_width // 3
    h, w = cells[0].shape[:2]
    cell_h = max(1, int(round(cell_w * h / w)))
    resized = [cv2.resize(c, (cell_w, cell_h)) for c in cells]
    rows = [cv2.hconcat(resized[i:i + 3]) for i in range(0, 9, 3)]
    grid = cv2.vconcat(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=output_path.suffix or ".jpg", delete=False,
        dir=output_path.parent,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cv2.imwrite(str(tmp_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        shutil.move(str(tmp_path), str(output_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return output_path


def blur_video_nl(
    input_path: Path,
    output_path: Path,
    text_prompt: str,
    *,
    dilation_px: int = 15,
    temporal_window: int = 3,
    blur_mode: str = "pixelate",
    pixel_size: int = 16,
    blur_strength: int = 51,
    verbose: bool = False,
) -> BlurResult:
    """Mirror of blur_video() that uses NLDetector. SAM 3 already returns
    pixel-precise masks per frame, so this skips the box->mask conversion
    that blur_video does. Dilation, temporal smoothing, and blur application
    are identical."""
    if blur_mode not in ("pixelate", "gaussian"):
        raise ValueError(f"unknown blur_mode: {blur_mode!r}")
    if blur_strength % 2 == 0:
        blur_strength += 1

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    t_start = time.perf_counter()
    fps = _video_fps(input_path)

    with tempfile.TemporaryDirectory(prefix="sentryblur_nl_") as tmp:
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
                f"sentryblur: NL detection on {n} frames @ {w}x{h} "
                f"(prompt: {text_prompt!r})",
                file=sys.stderr,
            )

        from sentryblur.nl_detector import NLDetector
        detector = NLDetector(text_prompt)
        masks = detector.process_video(frames_in)
        if len(masks) != n:
            raise ValueError(
                f"NLDetector returned {len(masks)} masks for {n} frames"
            )

        masks = [_dilate(m, dilation_px) for m in masks]
        masks = _temporal_union(masks, temporal_window)

        covered = 0
        for idx, (fp, m) in enumerate(zip(frames, masks)):
            frame = cv2.imread(str(fp))
            if blur_mode == "pixelate":
                blurred = _apply_pixelate(frame, m, pixel_size)
            else:
                blurred = _apply_blur(frame, m, blur_strength)
            cv2.imwrite(str(frames_out / f"{idx:05d}.jpg"), blurred,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if m.any():
                covered += 1

        tmp_out = tmp_dir / "out.mp4"
        _assemble(frames_out, fps, tmp_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_out), str(output_path))

    if verbose:
        elapsed = time.perf_counter() - t_start
        print(
            f"sentryblur: NL done in {elapsed:.1f}s -> {output_path}",
            file=sys.stderr,
        )

    return BlurResult(
        n_frames=n, fps=fps, covered_frames=covered, output_path=output_path,
    )


def generate_preview_nl(
    input_path: Path,
    output_path: Path,
    text_prompt: str,
    *,
    max_width: int = 1920,
) -> Path:
    """3x3 contact sheet of SAM 3 detections per keyframe — runs the model
    on each keyframe as a 1-frame session, skipping the cross-frame
    tracking that the full propagation needs. Use to sanity-check the
    prompt before committing to a full video run."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    from sentryblur.nl_detector import NLDetector
    detector = NLDetector(text_prompt)

    cap = cv2.VideoCapture(str(input_path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            raise ValueError(f"Could not read frame count from {input_path}")

        cells: list[np.ndarray] = []
        for idx in _keyframe_indices(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cells.append(_annotate_boxes(frame, detector.detect_boxes(frame)))
    finally:
        cap.release()

    return _write_3x3_grid(cells, output_path, max_width)
