# SentryBlur

[https://github.com/user-attachments/assets/87710fbb-8d63-44ea-b011-5fc9b46512c0](https://github.com/user-attachments/assets/6af8f234-63e2-42e5-b896-e8f24c8f3ee8)

Local, offline face and license plate redaction for video, with one command.

## Quickstart

```bash
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/ssrajadh/sentryblur.git
cd sentryblur
uv tool install .                         # base: faces only
uv tool install '.[plates]'               # add license plate detection

# 3. Run
sentryblur faces  dashcam.mp4             # → dashcam_blurred.mp4
sentryblur plates dashcam.mp4             # → dashcam_blurred.mp4
sentryblur faces  dashcam.mp4 --preview   # → dashcam_preview.jpg (3x3 contact sheet)
```

`ffmpeg` must be on PATH (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu).

## Why SentryBlur

- **Local, offline.** No API keys, no cloud upload. Footage never leaves your machine. Detection runs on CPU by default; CUDA optional.
- **Preview-first.** `--preview` renders a 3x3 contact sheet of detector output across keyframes so you can verify quality *before* committing to a full render of sensitive footage.
- **Composes with [SentrySearch](https://github.com/ssrajadh/sentrysearch).** Search dashcam footage with SentrySearch, then redact the matching clips with SentryBlur. (Direct pipe integration is on the roadmap; for now the two run as separate commands on the same files.)

## Install

From a local clone:

```bash
uv tool install .                         # faces only (~50 MB of deps)
uv tool install '.[plates]'               # adds open-image-models for plates
```

To upgrade after pulling new commits, re-run the same command.

A future `[prompt]` extra will add natural-language prompted redaction (`sentryblur prompt "laptop screen" video.mp4`) using Grounding DINO + SAM 2. Not shipped yet — see [Roadmap](#roadmap).

For development (editable install with test deps):

```bash
uv sync --group test
uv run pytest
```

System prerequisites:

- Python 3.11+
- `ffmpeg` on PATH
- [uv](https://docs.astral.sh/uv/) (the install snippet in [Quickstart](#quickstart) gets you this)

First run of each detector downloads weights:

| Detector | Size | Cache location |
|---|---|---|
| SCRFD (faces) | ~16 MB | `~/.insightface/` |
| YOLOv9-T (plates) | ~5 MB | `~/.cache/sentryblur/` |

## Usage

### Faces

```bash
$ sentryblur faces dashcam.mp4
Loading face detector...
Blurring dashcam.mp4 -> dashcam_blurred.mp4
Detecting  [####################################]  100%
Done. 720 frames, coverage 31.5%, 18.2s (0.6x realtime)
Output: /path/to/dashcam_blurred.mp4
```

### Plates

```bash
$ sentryblur plates dashcam.mp4
Loading plate detector...
sentryblur: loading yolo-v9-t-384-license-plates-end2end (downloads ~5 MB on first run)...
Blurring dashcam.mp4 -> dashcam_blurred.mp4
Detecting  [####################################]  100%
Done. 720 frames, coverage 84.2%, 22.7s (0.8x realtime)
```

### Flags (both commands)

| Flag | Default | Purpose |
|---|---|---|
| `INPUT_PATH` | — | Source video (positional, required) |
| `-o, --output PATH` | `<input>_blurred.<ext>` | Output path. With `--preview`, defaults to `<input>_preview.jpg` |
| `--dilation N` | 15 | Pixels to grow each detected box. Larger = safer margin around the target. |
| `--window N` | 3 | Temporal smoothing window in frames. The mask for frame *i* is the union across `[i-N, i+N]`, so a single-frame detection miss gets filled by its neighbors. |
| `--blur-mode MODE` | `pixelate` | Redaction style: `pixelate` (mosaic) or `gaussian`. Pixelate is the standard for redaction and harder to see through; gaussian can look weak on small targets. |
| `--pixel-size N` | 16 | Mosaic block size in pixels (pixelate mode only). Smaller = stronger redaction. |
| `--blur-strength N` | 51 | Gaussian kernel size (gaussian mode only). Must be odd; even values are bumped up. |
| `--conf F` | 0.25 | Detector confidence threshold. Lower = more recall, more false positives. |
| `--gpu` | off | Use CUDA for detection. Apple MPS not yet wired. |
| `--preview` | off | Render a 3x3 keyframe contact sheet with bounding boxes instead of blurring the full video. |
| `-v, --verbose` | off | Print progress (tqdm) and timing info to stderr. |

### Preview before render

Sensitive footage warrants a sanity check before committing to a long render:

```bash
$ sentryblur faces important.mp4 --preview
Loading face detector...
Rendering preview important.mp4 -> important_preview.jpg
Preview saved to important_preview.jpg. Review detections, then re-run without --preview to render the full video.
```

Open the JPG, verify the boxes land where you expect, then re-run without `--preview`.

## How it works

**1. Detect.** Each frame is fed to a per-target model — SCRFD ([insightface](https://github.com/deepinsight/insightface)) for faces, YOLOv9-T ([open-image-models](https://github.com/ankandrew/open-image-models)) for plates. The detector returns axis-aligned boxes plus confidences. No tracking, no Re-ID — just per-frame inference.

**2. Mask and dilate.** Each box becomes a binary mask the size of the frame. Masks are dilated by `--dilation` pixels with an elliptical kernel, so the blur region extends slightly past the detected box. This catches the case where the detector's bounding box clips the edge of a face or plate.

**3. Temporal smooth.** For frame *i*, the final mask is the union of dilated masks across `[i-window, i+window]`. A single-frame detection miss — which is the most common failure mode and the most damaging one for redaction — gets filled in by its neighbors as long as either side detected. `--window 3` covers a 7-frame radius (~230 ms at 30 fps), wide enough for transient miss but narrow enough not to over-blur during fast motion.

**4. Redact and reassemble.** The masked region of each frame is replaced — by default with a pixel mosaic (`--pixel-size`), or with a Gaussian blur (`--blur-strength`) if `--blur-mode gaussian`. Pixelation is the default because Gaussian blur on small targets (faces in dashcams are often 30–60 px) tends to collapse to a flat blob that looks weak; a mosaic preserves the visual signal "this region is redacted." The unmasked region is kept untouched, and ffmpeg reassembles to H.264 at CRF 18. The output is written atomically — to a tempfile, then `mv`'d into place — so a crash mid-render never overwrites your intended output with a half-finished file.

## Limitations

This section is honest, not aspirational. Read it before trusting SentryBlur with anything sensitive.

- **Detection misses small or distant targets, faces in profile, plates in low light, and partially occluded objects.** Temporal smoothing and dilation reduce miss rate but do not eliminate it. Always run `--preview` on sensitive footage and visually verify before committing to the full render.
- **Audio is not redacted.** Voices stating license plate numbers, addresses, or names will pass through untouched. SentryBlur is a video-frame redaction tool. Use a separate audio editor if your footage has identifying speech.
- **One missed frame defeats the purpose.** A leak of even a single unblurred frame in a published clip can be screenshotted and zoomed. SentryBlur reduces miss probability but cannot guarantee zero misses on arbitrary footage. For high-stakes redaction, watch the output at 0.25× speed before publishing.
- **Closed vocabulary.** Faces and plates only. Other identifying objects (visible monitor contents, name tags, building signage) require manual masking or the future `prompt` subcommand.

## Roadmap

- **`prompt` subcommand.** Natural-language prompted redaction via Grounding DINO + SAM 2. `sentryblur prompt "laptop screen" footage.mp4`. Tracking and segmenting general objects across frames; slower than the dedicated detectors but covers the long tail of identifying content.
- **SentrySearch pipe integration.** `sentrysearch search "..." | sentryblur faces -` so you can search-then-redact in one shell pipeline without intermediate files.
- **Apple MPS support.** GPU acceleration on Apple Silicon. Currently the `--gpu` flag is CUDA-only; MPS routing is straightforward but not yet wired.

## Acknowledgments

- **[insightface](https://github.com/deepinsight/insightface)** — SCRFD face detector.
- **[open-image-models](https://github.com/ankandrew/open-image-models)** — MIT-licensed YOLOv9-T license plate detector by ankandrew. Chosen specifically because the weights are MIT-licensed (most YOLOv8 plate weights on HuggingFace inherit Ultralytics's AGPL).
- **[ffmpeg](https://ffmpeg.org/)** — frame extraction and video reassembly.

## License

MIT.
