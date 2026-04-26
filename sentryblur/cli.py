"""Click-based CLI entry point."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import click


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_blurred{input_path.suffix}")


def _default_preview_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_preview.jpg")


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

    except Exception as e:
        _handle_error(e)


@click.group()
@click.version_option()
def cli():
    """Blur faces and license plates in dashcam footage."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output path (default: <input>_blurred.<ext>).")
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
def faces(input_path: Path, output_path: Path | None, dilation: int, window: int,
          blur_mode: str, pixel_size: int, blur_strength: int,
          conf: float, gpu: bool, preview: bool, verbose: bool):
    """Blur faces in INPUT_PATH using SCRFD."""
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
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output path (default: <input>_blurred.<ext>).")
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
def plates(input_path: Path, output_path: Path | None, dilation: int, window: int,
           blur_mode: str, pixel_size: int, blur_strength: int,
           conf: float, gpu: bool, preview: bool, verbose: bool):
    """Blur license plates in INPUT_PATH using open-image-models YOLOv9."""
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


if __name__ == "__main__":
    cli()
