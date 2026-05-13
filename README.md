# SentryBlur

Local, offline object redaction for video, with one command.

[https://github.com/user-attachments/assets/87710fbb-8d63-44ea-b011-5fc9b46512c0](https://github.com/user-attachments/assets/6af8f234-63e2-42e5-b896-e8f24c8f3ee8)

> **Note**
> The demo above is sped up. Actual runtime depends on hardware, the clip was processed on an M1 Pro MBP (16 GB RAM); expect faster results on better hardware. See the [Performance](#performance) table for details.  


## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone and install
```bash
git clone https://github.com/ssrajadh/sentryblur.git
cd sentryblur

# Faces only (base)
uv tool install .

# Or, all detectors (faces + plates + natural-language prompt) in one command:
uv tool install '.[plates,prompt]'

# Optional: also include the legacy SAM 2 backend as a fallback (see below):
uv tool install '.[plates,prompt]' --with git+https://github.com/facebookresearch/sam2.git
```

`uv tool install` re-resolves on each invocation rather than merging, so multiple sequential installs would silently drop earlier extras. Use a single command with all the extras you want. To upgrade later, re-run the same command.

The `prompt` feature uses SAM 3.1 by default, which downloads weights from Hugging Face on first run. The weights are gated: visit [facebook/sam3](https://huggingface.co/facebook/sam3), accept Meta's SAM License, then run `huggingface-cli login` once with a token that has read access. If you can't get access (approval can take hours to days) or don't want to gate yourself on a HuggingFace account, install with the SAM 2 `--with` line above and pass `--backend sam2` to `sentryblur prompt`.

3. Run
```bash
sentryblur faces  video.mp4                          # → video_blurred.mp4
sentryblur plates video.mp4                          # → video_blurred.mp4
sentryblur prompt video.mp4 "road signs"          # → video_blurred.mp4
sentryblur faces  video.mp4 --preview                # → video_preview.jpg (3x3 contact sheet)
sentryblur prompt video.mp4 "phone screen" --preview
```

`ffmpeg` must be on PATH (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu).

## Why SentryBlur

- **Local, offline.** No API keys, no cloud upload. Footage never leaves your machine. Detection runs on CPU by default; CUDA optional.
- **Preview-first.** `--preview` renders a 3x3 contact sheet of detector output across keyframes so you can verify quality *before* committing to a full render of sensitive footage.
- **Composes with [SentrySearch](https://github.com/ssrajadh/sentrysearch).** Search footage with SentrySearch, then redact the matching clip with `sentryblur faces --last`, `sentryblur plates --last`, or `sentryblur prompt --last "..."` — `--last` picks up the most recent clip SentrySearch saved, no path-passing required.

## Install

From a local clone. Pick one — `uv tool install` re-resolves on each invocation, so running multiple installs sequentially will drop earlier extras. Always pass every extra you want in a single command.

```bash
# Faces only (~50 MB of deps)
uv tool install .

# Faces + plates
uv tool install '.[plates]'

# Faces + plates + prompt (SAM 3.1 backend, default).
uv tool install '.[plates,prompt]'

# Add the legacy SAM 2 + DINO backend as a fallback (sam2 is not on PyPI,
# so it has to come via --with so it lands in the same uv-managed venv
# as sentryblur itself):
uv tool install '.[plates,prompt]' --with git+https://github.com/facebookresearch/sam2.git
```

Hardware: `prompt` requires an NVIDIA GPU or Apple Silicon (16 GB+ unified memory). CPU is not supported.

The default SAM 3.1 backend requires accepting Meta's SAM License on the [SAM 3 model page](https://huggingface.co/facebook/sam3) and running `huggingface-cli login` once — the weights are gated. The faces and plates paths do not need this. If approval is taking too long or you'd rather skip the HF account, install with the SAM 2 `--with` line and use `sentryblur prompt --backend sam2`.

To upgrade after pulling new commits, re-run the same command.

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
| SAM 3.1 (prompt) | ~3.5 GB | `~/.cache/sentryblur/huggingface/` |

## Usage

### Faces

```bash
$ sentryblur faces video.mp4
Loading face detector...
Blurring video.mp4 -> video_blurred.mp4
Detecting  [####################################]  100%
Done. 720 frames, coverage 31.5%, 18.2s (0.6x realtime)
Output: /path/to/video_blurred.mp4
```

### Plates

```bash
$ sentryblur plates video.mp4
Loading plate detector...
sentryblur: loading yolo-v9-t-384-license-plates-end2end (downloads ~5 MB on first run)...
Blurring video.mp4 -> video_blurred.mp4
Detecting  [####################################]  100%
Done. 720 frames, coverage 84.2%, 22.7s (0.8x realtime)
```

### Prompt

Natural-language redaction for objects outside the closed faces/plates vocabulary — phone screens, monitors, name tags, specific people. SAM 3.1 (default) takes the text prompt and produces tracked pixel-precise masks across the full clip in a single pass; targets can first appear at any point in the video — there is no frame-0 dependency. Pass `--backend sam2` to fall back to the legacy Grounding DINO + SAM 2 path if you can't access the gated SAM 3 weights.

```bash
$ sentryblur prompt video.mp4 "license plate"
This 30-second clip will take approximately 12 minutes on Apple M1 Pro / M1 Max.
Continue? [y/N]: y
Blurring video.mp4 -> video_blurred.mp4
Done. 900 frames, coverage 18.6%, 758.4s (25.3x realtime)
Output: /home/user/video_blurred.mp4
```

`--preview` skips the cross-frame propagation and runs SAM 3 on 9 keyframes independently — gives you a contact sheet to verify the prompt before committing to the full video run:

```bash
$ sentryblur prompt video.mp4 "phone screen" --preview
Rendering preview video.mp4 -> video_preview.jpg
Preview saved to video_preview.jpg. Review detections, then re-run without --preview to render the full video.
```

`--last` picks up the most recent clip SentrySearch saved, so you can search-then-redact without retyping paths:

```bash
$ sentryblur prompt --last "license plate"
Last clip: /home/user/match_2026-04-15_14-30_02m15s-02m45s.mp4 (saved 2m ago by sentrysearch)
Process this clip? [Y/n]: y
This 30-second clip will take approximately 12 minutes on Apple M1 Pro / M1 Max.
Continue? [y/N]: y
Blurring match_2026-04-15_14-30_02m15s-02m45s.mp4 -> match_2026-04-15_14-30_02m15s-02m45s_blurred.mp4
Done. 900 frames, coverage 22.1%, 762.7s (25.4x realtime)
Output: /home/user/match_2026-04-15_14-30_02m15s-02m45s_blurred.mp4
```

Pass `-y/--yes` to skip both the `--last` and duration confirmation prompts (useful for scripts).

#### Performance

`prompt` is dominated by per-frame SAM 3.1 inference, so processing time scales with frame count, not just clip duration. A 60 fps clip takes roughly 2x longer than a 30 fps clip of the same duration.

On Apple M1 Pro / M1 Max (validated): a 10-second 30 fps clip (300 frames) processes in approximately 4 minutes.

| Hardware                  | Per-frame cost  | 10s @ 30fps          |
|---------------------------|-----------------|----------------------|
| NVIDIA GPU (16GB+)        | ~50-150 ms      | ~1-2 min             |
| NVIDIA GPU                | ~150-400 ms     | ~1-3 min             |
| Apple Silicon M2 Pro+     | ~400-800 ms     | ~2-4 min             |
| Apple M1 Pro / M1 Max     | ~0.8 s          | ~4 min (validated)   |
| Apple M1 (base)           | ~1.5-2.5 s      | ~8-13 min            |
| CPU                       | not supported   | —                    |

Only the M1 Pro / M1 Max row is from direct measurement; other rows are estimates and will be revised as users contribute real numbers.

### Flags

Shared by `faces`, `plates`, and `prompt` unless noted.

| Flag | Default | Purpose |
|---|---|---|
| `INPUT_PATH` | — | Source video (positional, required unless `--last`). `prompt` also takes a positional `TEXT_PROMPT` after the video. |
| `--last` | off | Use the most recent clip saved by SentrySearch (`sentrysearch search --save-top`). Mutually exclusive with `INPUT_PATH`. |
| `-y, --yes` | off | Skip the `--last` and (for `prompt`) duration-confirmation prompts. |
| `-o, --output PATH` | `<input>_blurred.<ext>` | Output path. With `--preview`, defaults to `<input>_preview.jpg`. |
| `--dilation N` | 15 | Pixels to grow each detected box/mask. Larger = safer margin around the target. |
| `--window N` | 3 | Temporal smoothing window in frames. The mask for frame *i* is the union across `[i-N, i+N]`, so a single-frame detection miss gets filled by its neighbors. |
| `--blur-mode MODE` | `pixelate` | Redaction style: `pixelate` (mosaic) or `gaussian`. Pixelate is the standard for redaction and harder to see through; gaussian can look weak on small targets. |
| `--pixel-size N` | 16 | Mosaic block size in pixels (pixelate mode only). Smaller = stronger redaction. |
| `--blur-strength N` | 51 | Gaussian kernel size (gaussian mode only). Must be odd; even values are bumped up. |
| `--conf F` | 0.25 | Detector confidence threshold. Lower = more recall, more false positives. *(faces, plates only)* |
| `--gpu` | off | Use CUDA for detection. Apple MPS not yet wired. *(faces, plates only — `prompt` always uses GPU/MPS via torch auto-detect)* |
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

**`prompt`.** SAM 3.1 takes the user's text prompt and the full clip, then runs its unified detector + tracker to produce pixel-precise masks across every frame in a single session. The detector and tracker share a backbone, and the tracker carries identity across frames so a target that first appears mid-clip still gets a stable mask. Frames are stored on CPU and streamed to GPU/MPS so 16 GB unified-memory Macs can handle multi-minute clips. Masks then go through the same dilation and temporal smoothing as `faces`/`plates` before blur and reassembly.

## Limitations

This section is honest, not aspirational. Read it before trusting SentryBlur with anything sensitive.

- **Detection misses small or distant targets, faces in profile, plates in low light, and partially occluded objects.** Temporal smoothing and dilation reduce miss rate but do not eliminate it. Always run `--preview` on sensitive footage and visually verify before committing to the full render.
- **Audio is not redacted.** Voices stating license plate numbers, addresses, or names will pass through untouched. SentryBlur is a video-frame redaction tool. Use a separate audio editor if your footage has identifying speech.
- **One missed frame defeats the purpose.** A leak of even a single unblurred frame in a published clip can be screenshotted and zoomed. SentryBlur reduces miss probability but cannot guarantee zero misses on arbitrary footage. For high-stakes redaction, watch the output at 0.25× speed before publishing.
- **Closed vocabulary.** The `faces` and `plates` detectors only know their respective targets. Other identifying objects (visible monitor contents, name tags, building signage) need the `prompt` subcommand or manual masking.

### `prompt`-specific

- **Gated SAM 3.1 weights.** The default `--backend sam3` path uses gated weights on Hugging Face. First use requires accepting Meta's SAM License on the [model page](https://huggingface.co/facebook/sam3) and running `huggingface-cli login`. Without this, the first run fails with a 401 and `sentryblur` will tell you to fix auth or pass `--backend sam2`. Approval can take from minutes to a day or more — the SAM 2 backend is the workaround in the meantime.
- **SAM 2 backend tradeoffs (`--backend sam2`).** No HF gating, but: (a) requires the `--with git+...sam2.git` install line, (b) only finds targets visible in *frame 0* of the clip — if the target enters mid-clip, SAM 2 misses it, and (c) less accurate on hard prompts than SAM 3.1 in our testing.
- **GPU/Apple Silicon only.** `prompt` requires CUDA or MPS. CPU is rejected up front. `faces` and `plates` still run on CPU.
- **Slow.** `prompt` is one to two orders of magnitude slower than `faces`/`plates` (see the [Performance](#performance) table). Suitable for one-off tasks, not batch workflows.
- **MPS quirks.** SAM 3 Video on Apple Silicon depends on a recent `transformers` (≥ 4.57) that has the `pin_memory()` fix for MPS. If you hit `RuntimeError: Attempted to set the storage of a tensor on device "cpu" to a storage on different device "mps:0"`, upgrade transformers.

## Roadmap

- **Faster `prompt` mode.** Sub-sampled SAM 3 detection plus a lightweight tracker between keyframes, instead of full per-frame propagation. Should bring `prompt` closer to `faces`/`plates` throughput on long clips.

## Acknowledgments

- **[insightface](https://github.com/deepinsight/insightface)** — SCRFD face detector.
- **[open-image-models](https://github.com/ankandrew/open-image-models)** — MIT-licensed YOLOv9-T license plate detector by ankandrew. Chosen specifically because the weights are MIT-licensed (most YOLOv8 plate weights on HuggingFace inherit Ultralytics's AGPL).
- **[ffmpeg](https://ffmpeg.org/)** — frame extraction and video reassembly.
- **[SAM 3](https://github.com/facebookresearch/sam3)** — Meta AI Research, accessed via HuggingFace `transformers` (`facebook/sam3`). Weights are released under Meta's custom [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE); this project's own code is Apache 2.0 (see `LICENSE`).
