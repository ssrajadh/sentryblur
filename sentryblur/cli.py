"""Click-based CLI entry point."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click


def _open_file(path: str) -> None:
    """Open a file with the system's default application. Best-effort —
    swallow any failure (no GUI session, missing opener, etc.)."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_blurred{input_path.suffix}")


def _default_preview_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_preview.jpg")


_LAST_CLIP_MAX_AGE_S = 7 * 24 * 3600
_LAST_CLIP_WARN_AGE_S = 3600


def _format_age(seconds: int) -> str:
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _resolve_input(
    input_path: Path | None,
    use_last: bool,
    yes: bool,
) -> Path:
    """Resolve the command's input path, honoring --last.

    Exits the process on resolution failure. Returns an absolute Path
    on success.
    """
    if input_path is not None and use_last:
        raise click.UsageError(
            "INPUT_PATH and --last are mutually exclusive.",
        )
    if input_path is None and not use_last:
        raise click.UsageError("Missing argument 'INPUT_PATH'.")

    if not use_last:
        return input_path

    from sentryblur._toolkit_cache import read_last_clip

    cached = read_last_clip()
    if cached is None:
        click.secho(
            "Error: No cached clip found. "
            "Run sentrysearch first to save a clip.",
            fg="red", err=True,
        )
        raise SystemExit(1)

    if not cached.file_exists:
        click.secho(
            f"Error: Cached clip path no longer exists: {cached.path}\n"
            "The file may have been deleted or moved.",
            fg="red", err=True,
        )
        raise SystemExit(1)

    age = cached.age_seconds
    if age > _LAST_CLIP_MAX_AGE_S:
        when = cached.saved_at.strftime("%Y-%m-%d %H:%M UTC")
        click.secho(
            f"Error: Cached clip is more than 7 days old ({when}).\n"
            "Run sentrysearch again to refresh.",
            fg="red", err=True,
        )
        raise SystemExit(1)

    age_str = _format_age(age)
    if age > _LAST_CLIP_WARN_AGE_S:
        click.secho(
            f"Note: cached clip is from {age_str} ago at {cached.path}",
            fg="yellow", err=True,
        )

    if not yes:
        if not click.confirm(
            f"Last clip: {cached.path} "
            f"(saved {age_str} ago by {cached.saved_by})\n"
            "Process this clip?",
            default=True,
        ):
            raise SystemExit(1)

    return cached.path


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg",
            fg="red", err=True,
        )
        raise SystemExit(1)


def _handle_error(e: Exception) -> None:
    from sentryblur.detectors import MissingPlatesExtra
    if isinstance(e, MissingPlatesExtra):
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, FileNotFoundError):
        click.secho(f"Error: file not found: {e}", fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho(f"Error: {e}", fg="red", err=True)
        raise SystemExit(1)
    from sentryblur.pipeline import FFmpegError
    if isinstance(e, FFmpegError):
        click.secho(f"Error: ffmpeg failed: {e}", fg="red", err=True)
        raise SystemExit(1)
    raise e


def _run_blur_command(
    *,
    input_path: Path,
    output_path: Path | None,
    dilation: int,
    window: int,
    blur_strength: int,
    blur_mode: str,
    pixel_size: int,
    preview: bool,
    verbose: bool,
    detector_factory,
    target_label: str,
) -> None:
    """Shared body for `faces` and `plates` commands. Resolves the output
    path, instantiates the detector, then dispatches to either
    generate_preview() or blur_video()."""
    if not preview:
        _check_ffmpeg()

    if output_path is None:
        output_path = (
            _default_preview_output(input_path) if preview
            else _default_output(input_path)
        )
    if output_path.resolve() == input_path.resolve():
        click.secho("Error: output must differ from input.", fg="red", err=True)
        raise SystemExit(1)

    try:
        from sentryblur.pipeline import blur_video, generate_preview

        click.echo(f"Loading {target_label} detector...")
        detector = detector_factory()

        if preview:
            click.echo(f"Rendering preview {input_path.name} -> {output_path.name}")
            preview_path = generate_preview(input_path, output_path, detector)
            click.secho(
                f"Preview saved to {preview_path}. Review detections, then "
                "re-run without --preview to render the full video.",
                fg="green",
            )
            _open_file(str(preview_path))
            return

        click.echo(f"Blurring {input_path.name} -> {output_path.name}")
        t0 = time.time()
        if verbose:
            result = blur_video(
                input_path, output_path, detector,
                dilation_px=dilation, temporal_window=window,
                blur_strength=blur_strength, blur_mode=blur_mode,
                pixel_size=pixel_size, verbose=True,
            )
        else:
            with click.progressbar(length=100, label="Detecting") as bar:
                last = [0]
                def progress(done: int, total: int):
                    pct = int(100 * done / total)
                    if pct > last[0]:
                        bar.update(pct - last[0])
                        last[0] = pct
                result = blur_video(
                    input_path, output_path, detector,
                    dilation_px=dilation, temporal_window=window,
                    blur_strength=blur_strength, blur_mode=blur_mode,
                    pixel_size=pixel_size, progress=progress,
                )
        elapsed = time.time() - t0

        video_s = result.n_frames / result.fps
        speed = elapsed / video_s if video_s else 0.0
        click.secho(
            f"\nDone. {result.n_frames} frames, "
            f"coverage {result.coverage_pct:.1f}%, "
            f"{elapsed:.1f}s ({speed:.1f}x realtime)",
            fg="green",
        )
        click.echo(f"Output: {result.output_path}")
        _open_file(str(result.output_path))

    except Exception as e:
        _handle_error(e)


@click.group()
@click.version_option()
def cli():
    """Blur faces and license plates in dashcam footage."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                required=False, default=None)
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output path (default: <input>_blurred.<ext>).")
@click.option("--last", "use_last", is_flag=True,
              help="Use the most recent clip saved by sentrysearch (see "
                   "`sentrysearch search --save-top`). Cannot be combined "
                   "with INPUT_PATH.")
@click.option("-y", "--yes", is_flag=True,
              help="Skip the --last confirmation prompt.")
@click.option("--dilation", default=15, show_default=True, type=int,
              help="Pixels to dilate each detected box. Larger = safer margin.")
@click.option("--window", default=3, show_default=True, type=int,
              help="Temporal smoothing window (frames). Larger = catches dropouts.")
@click.option("--blur-mode", default="pixelate", show_default=True,
              type=click.Choice(["pixelate", "gaussian"]),
              help="Redaction style. Pixelate is harder to see through; "
                   "gaussian is softer but smaller targets may look weak.")
@click.option("--pixel-size", default=16, show_default=True, type=int,
              help="Mosaic block size in pixels (pixelate mode only). "
                   "Smaller = stronger redaction.")
@click.option("--blur-strength", default=51, show_default=True, type=int,
              help="Gaussian kernel size (gaussian mode only). Must be odd; "
                   "even values are bumped up.")
@click.option("--conf", default=0.25, show_default=True, type=float,
              help="Detector confidence threshold.")
@click.option("--gpu", is_flag=True, help="Use GPU for detection (CUDA only).")
@click.option("--preview", is_flag=True,
              help="Generate a 3x3 contact sheet of detections instead of "
                   "rendering the blurred video. Useful for sanity-checking "
                   "detection quality before a long render.")
@click.option("-v", "--verbose", is_flag=True,
              help="Print progress (tqdm) and timing info to stderr.")
def faces(input_path: Path | None, output_path: Path | None,
          use_last: bool, yes: bool,
          dilation: int, window: int,
          blur_mode: str, pixel_size: int, blur_strength: int,
          conf: float, gpu: bool, preview: bool, verbose: bool):
    """Blur faces in INPUT_PATH using SCRFD."""
    input_path = _resolve_input(input_path, use_last, yes)

    def factory():
        from sentryblur.detectors import SCRFDFaceDetector
        return SCRFDFaceDetector(conf=conf, use_gpu=gpu)

    _run_blur_command(
        input_path=input_path, output_path=output_path,
        dilation=dilation, window=window, blur_strength=blur_strength,
        blur_mode=blur_mode, pixel_size=pixel_size,
        preview=preview, verbose=verbose,
        detector_factory=factory, target_label="face",
    )


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
                required=False, default=None)
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output path (default: <input>_blurred.<ext>).")
@click.option("--last", "use_last", is_flag=True,
              help="Use the most recent clip saved by sentrysearch (see "
                   "`sentrysearch search --save-top`). Cannot be combined "
                   "with INPUT_PATH.")
@click.option("-y", "--yes", is_flag=True,
              help="Skip the --last confirmation prompt.")
@click.option("--dilation", default=15, show_default=True, type=int,
              help="Pixels to dilate each detected box. Larger = safer margin.")
@click.option("--window", default=3, show_default=True, type=int,
              help="Temporal smoothing window (frames). Larger = catches dropouts.")
@click.option("--blur-mode", default="pixelate", show_default=True,
              type=click.Choice(["pixelate", "gaussian"]),
              help="Redaction style. Pixelate is harder to see through; "
                   "gaussian is softer but smaller targets may look weak.")
@click.option("--pixel-size", default=16, show_default=True, type=int,
              help="Mosaic block size in pixels (pixelate mode only). "
                   "Smaller = stronger redaction.")
@click.option("--blur-strength", default=51, show_default=True, type=int,
              help="Gaussian kernel size (gaussian mode only). Must be odd; "
                   "even values are bumped up.")
@click.option("--conf", default=0.25, show_default=True, type=float,
              help="Detector confidence threshold.")
@click.option("--gpu", is_flag=True, help="Use GPU for detection (CUDA only).")
@click.option("--preview", is_flag=True,
              help="Generate a 3x3 contact sheet of detections instead of "
                   "rendering the blurred video.")
@click.option("-v", "--verbose", is_flag=True,
              help="Print progress (tqdm) and timing info to stderr.")
def plates(input_path: Path | None, output_path: Path | None,
           use_last: bool, yes: bool,
           dilation: int, window: int,
           blur_mode: str, pixel_size: int, blur_strength: int,
           conf: float, gpu: bool, preview: bool, verbose: bool):
    """Blur license plates in INPUT_PATH using open-image-models YOLOv9."""
    input_path = _resolve_input(input_path, use_last, yes)

    def factory():
        from sentryblur.detectors import OpenImagePlateDetector
        return OpenImagePlateDetector(conf=conf, use_gpu=gpu)

    _run_blur_command(
        input_path=input_path, output_path=output_path,
        dilation=dilation, window=window, blur_strength=blur_strength,
        blur_mode=blur_mode, pixel_size=pixel_size,
        preview=preview, verbose=verbose,
        detector_factory=factory, target_label="plate",
    )


_PROMPT_INSTALL_HINT = (
    "Install with:\n"
    "  uv tool install '.[prompt]'\n"
    "  pip install git+https://github.com/facebookresearch/sam2.git"
)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("text_prompt", metavar="TEXT_PROMPT")
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help="Output path (default: <input>_blurred.<ext>, "
                   "or <input>_preview.jpg with --preview).")
@click.option("--preview", is_flag=True,
              help="Render a 3x3 contact sheet of DINO detections instead "
                   "of running the full SAM 2 propagation. Fast — useful "
                   "for sanity-checking the prompt before committing.")
@click.option("-y", "--yes", is_flag=True,
              help="Skip the duration confirmation prompt.")
@click.option("--dilation", default=15, show_default=True, type=int,
              help="Pixels to dilate the SAM 2 mask. Larger = safer margin.")
@click.option("--window", default=3, show_default=True, type=int,
              help="Temporal smoothing window (frames).")
@click.option("--blur-mode", default="pixelate", show_default=True,
              type=click.Choice(["pixelate", "gaussian"]),
              help="Redaction style.")
@click.option("--pixel-size", default=16, show_default=True, type=int,
              help="Mosaic block size (pixelate mode only).")
@click.option("--blur-strength", default=51, show_default=True, type=int,
              help="Gaussian kernel size (gaussian mode only). Must be odd.")
@click.option("-v", "--verbose", is_flag=True,
              help="Print progress and timing info to stderr.")
def prompt(input_path: Path, text_prompt: str, output_path: Path | None,
           preview: bool, yes: bool, dilation: int, window: int,
           blur_mode: str, pixel_size: int, blur_strength: int, verbose: bool):
    """Redact arbitrary objects via natural language description.

    \b
    Uses Grounding DINO + SAM 2 for zero-shot promptable video segmentation.
    Slower than 'faces' or 'plates' (~25x realtime on Apple M1 Pro).
    Best for one-off tasks rather than batch processing.

    \b
    Examples:
      sentryblur prompt clip.mp4 "license plate"
      sentryblur prompt clip.mp4 "phone screen" --preview
      sentryblur prompt clip.mp4 "person in red shirt" --yes
    """
    from sentryblur.limits import (
        HardwareTier, check_clip_length_for_prompt, detect_hardware_tier,
    )

    tier = detect_hardware_tier()
    if tier == HardwareTier.CPU:
        click.secho(
            "Error: prompt requires a GPU or Apple Silicon. "
            "CPU is not supported.\n"
            "Detected: CPU.\n"
            "For face/plate redaction without a GPU, use 'sentryblur faces' "
            "or 'sentryblur plates' instead.",
            fg="red", err=True,
        )
        raise SystemExit(1)

    check_clip_length_for_prompt(input_path, auto_confirm=yes)

    if not preview:
        _check_ffmpeg()
    if output_path is None:
        output_path = (
            _default_preview_output(input_path) if preview
            else _default_output(input_path)
        )
    if output_path.resolve() == input_path.resolve():
        click.secho("Error: output must differ from input.", fg="red", err=True)
        raise SystemExit(1)

    try:
        from sentryblur.nl_detector import NLDetectionFailure
        if preview:
            from sentryblur.pipeline import generate_preview_nl
        else:
            from sentryblur.pipeline import blur_video_nl
    except ImportError as e:
        click.secho(
            f"Error: missing dependencies for `prompt`: {e}\n\n"
            f"{_PROMPT_INSTALL_HINT}",
            fg="red", err=True,
        )
        raise SystemExit(1)

    try:
        if preview:
            click.echo(f"Rendering preview {input_path.name} -> {output_path.name}")
            preview_path = generate_preview_nl(
                input_path, output_path, text_prompt,
            )
            click.secho(
                f"Preview saved to {preview_path}. Review detections, then "
                "re-run without --preview to render the full video.",
                fg="green",
            )
            _open_file(str(preview_path))
            return

        click.echo(f"Blurring {input_path.name} -> {output_path.name}")
        t0 = time.time()
        result = blur_video_nl(
            input_path, output_path, text_prompt,
            dilation_px=dilation, temporal_window=window,
            blur_mode=blur_mode, pixel_size=pixel_size,
            blur_strength=blur_strength, verbose=verbose,
        )
        elapsed = time.time() - t0
        video_s = result.n_frames / result.fps if result.fps else 0.0
        speed = elapsed / video_s if video_s else 0.0
        click.secho(
            f"\nDone. {result.n_frames} frames, "
            f"coverage {result.coverage_pct:.1f}%, "
            f"{elapsed:.1f}s ({speed:.1f}x realtime)",
            fg="green",
        )
        click.echo(f"Output: {result.output_path}")
        _open_file(str(result.output_path))
    except NLDetectionFailure:
        click.secho(
            f"Error: Could not find {text_prompt!r} in the first frame. "
            "Try a more specific or different prompt.",
            fg="red", err=True,
        )
        raise SystemExit(1)
    except ImportError as e:
        click.secho(
            f"Error: missing dependencies for `prompt`: {e}\n\n"
            f"{_PROMPT_INSTALL_HINT}",
            fg="red", err=True,
        )
        raise SystemExit(1)
    except Exception as e:
        _handle_error(e)


if __name__ == "__main__":
    cli()
