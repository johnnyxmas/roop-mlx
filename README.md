# roop-mlx

Face swap images and video without training, tuned for Apple Silicon: a hybrid
inference engine (Apple Core AI + CoreML + MLX), hardware video decode/encode via
VideoToolbox, and a browser-based GUI. Apple Silicon (arm64 macOS) only.

[Installation](#installation) • [Usage](#usage) • [Engine](#engine) • [Changelog](CHANGELOG.md)

> On first run, models (~2 GB) are downloaded automatically to `models/`.
> The first launch also compiles the CoreML models (a few seconds each,
> cached on disk in `models/*/coreml_cache/` afterwards). On macOS 27+, the
> first use of each Core AI-routed model converts it (~5-10 s, cached in
> `models/*/coreai_cache/`).

---

## Features

- Browser-based GUI
- Multiple face swapping modes: first detected, by selection, by gender, random, all input faces
- Swap models: InsightFace inswapper, ReSwapper (128px & 256px), HyperSwap (256px)
- Batch processing of images and video
- Face occlusion masking via XSeg
- Face enhancers: GFPGAN, CodeFormer, GPEN (512/1024/2048px), RestoreFormer++
- Subsample upscaling for higher resolution swaps
- Frame processors: colorizer, upscaler, filters, background removal
- Live webcam face swap with virtual camera output
- Hardware video pipeline: AVFoundation/VideoToolbox decode and h264/hevc
  encode, with explicitly tagged BT.709 color end to end
- Multi-threaded processing overlapping CPU compositing with ANE/GPU inference
- Extras tab: video cutting, joining, repair, colorization, upscaling
- Settings persist between sessions via `config.yaml`
- Theme support

---

## Disclaimer and Conditions of Use

This software is provided for authorized research, academic study, and lawful use only. By using this software you agree to the following:

1. **Lawful use only.** You are solely responsible for compliance with all applicable laws governing synthetic media, biometric data, privacy, and impersonation.
2. **Consent required.** Where any identifiable individual's likeness is processed, you must obtain their documented, informed consent before processing, before distribution, and before any public display of outputs.
3. **Prohibited uses** include: non-consensual intimate imagery, fraud or deception, harassment or defamation, election interference, circumventing identity verification without authorization, or any other unlawful or harmful purpose.
4. **No warranty.** This software is provided "as is" without warranty of any kind.
5. **No liability.** The developers and contributors shall not be liable for any damages arising from use or misuse. You agree to indemnify and hold harmless the developers from any claims resulting from your use.

**Do not apply this software to illegal or unethical scenarios.**

---

## Installation

### Prerequisites

- **Apple Silicon Mac** (M1 / M2 / M3 / M4) running macOS 13.5 or newer
- **[uv](https://docs.astral.sh/uv/)** — `brew install uv`. It installs the
  right Python (3.13, from `.python-version`) itself; 3.11 – 3.14 are supported
- **ffmpeg** on your system PATH — `brew install ffmpeg` is the easy path

### Setup

```bash
uv sync
```

This creates `.venv/` with the exact dependency versions in `uv.lock`. No
provider flags, no CUDA wheels, no TensorRT to build. The `mlx` and
`onnxruntime` arm64 wheels install automatically.

Without uv, the same pinned versions install with plain pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

```bash
uv run python run.py
```

**CLI flags:**

- `--server_share` — expose via Gradio public URL (remote access; share with care)

The web UI opens automatically in your browser at `http://localhost:7860`.

---

## Settings Reference

All settings are stored in `config.yaml` and editable from the **⚙ Settings** tab.

### UI / Server

| Setting | Description |
| --- | --- |
| **Theme** | Visual style for the Gradio UI. Built-in options: `ocean` (default), `citrus`, `glass`, `monochrome`, `origin`, `soft`, `default`. Change requires a full server restart. |
| **Public Server** | Expose the UI via a public Gradio share URL. Anyone with the link can use your machine — share with care. |
| **Server Name** | Host/IP the Gradio server binds to. Leave blank for localhost-only (`127.0.0.1`). Set to `0.0.0.0` to allow LAN access. |
| **Server Port** | TCP port the Gradio server listens on. Leave at `0` to let Gradio pick an available port. |
| **Clear output folder before each run** | Delete everything in the output folder before each run starts. Useful for batch jobs where you don't want old results mixing in. |
| **Filename Output Template** | Template for saved result filenames. `{file}` = source filename, `{time}` = timestamp. File extension is added automatically. |

### Performance

| Setting | Description |
| --- | --- |
| **Max. Number of Threads** | Frames processed in parallel during video runs. Default: 2. Apple Silicon's unified memory means raising this also raises peak RAM. |
| **Max. Memory (GB)** | Hard cap on RAM for the process. `0` = no limit (OS decides). |

### Face Detection

| Setting | Description |
| --- | --- |
| **Use default Det-Size** | Use the 640×640 detector input. Uncheck for a larger 1024×1024 input — slower but finds smaller and more distant faces. |

### Output

| Setting | Description |
| --- | --- |
| **Image Output Format** | Format for still-image results. `png`: lossless, larger files. `jpg`: lossy, much smaller. `webp`: good compression with optional lossless mode. |
| **Video Codec** | FFmpeg codec for the final video. `auto`: VideoToolbox if available, else libx264. `libx264`/`libx265`: CPU H.264/H.265 — high quality, slower. `h264_videotoolbox`/`hevc_videotoolbox`: Apple hardware-encoded — faster, slightly lower quality per bitrate. `libvpx-vp9`: WebM/VP9, CPU-only. |
| **Video Output Format** | Container format. `mp4`: most compatible. `mkv`: best for unusual codec/audio combinations. `avi`: legacy. `webm`: pairs with libvpx-vp9. |
| **Video Quality** | Quality value. For `libx264` / `libx265` this is CRF (lower = higher quality, default 14, typical 18–28). For VideoToolbox this is `-q:v` (higher = better). |
| **Use OS temp folder** | Store extracted frames in the OS temp directory instead of the project folder. Helpful when your project drive is small. Temp files are cleared on reboot or via the Clean temp folder button. |
| **Show video in browser** | Play the finished video in the results panel. When off, a still frame is shown instead. |
| **Re-encode unplayable video for preview** | Off by default. Some codec/container choices (libx265, HEVC, mkv, avi, or an exotic audio track copied from the source) produce files browsers cannot play, and the results panel falls back to a still frame. Enable this to re-encode a temporary H.264 copy into the temp folder purely for preview — the saved output file is never modified. Costs a full extra encode pass, which is why it is opt-in. |

---

## Engine

Every model runs on its benchmark-fastest backend (see
[roop/inference.py](roop/inference.py)); each adoption was gated on
numerical parity against the ONNX reference implementation:

| Workload | Backend |
| --- | --- |
| Face detection (YOLOv8n-Face), recognition (ArcFace), landmarks, gender/age | CoreML (ANE/GPU) |
| Swap models: HyperSwap 1A/1B/1C | Core AI (macOS 27+), else CoreML |
| Swap models: inswapper, ReSwapper, AlphaFace | CoreML |
| Enhancers: GFPGAN, GPEN 512/1024/2048 | Core AI (macOS 27+), else MLX graph interpreter |
| Enhancers: CodeFormer, RestoreFormer++ | CoreML |
| Upscaler: SPAN ×4 | CoreML |
| XSeg, DeOldify, ISNet, Real-ESRGAN ×2 | MLX graph interpreter |
| Video decode / encode | VideoToolbox hardware |

Core AI is Apple's macOS 27 inference framework; it runs HyperSwap 1.33×,
GFPGAN 1.84× and GPEN 1.33-1.69× faster than the previous backend at equal
accuracy (M5 Pro). Each model is converted to `.aimodel` on first use with
[coreai-onnx](https://github.com/devin-lai/coreai-onnx), and called through a
small Swift bridge (`roop/coreai/`, shipped prebuilt). Anything that fails
falls back to the previous backend; `ROOP_DISABLE_COREAI=1` turns it off.

If `onnxruntime` is unavailable, everything falls back to the in-repo
ONNX-to-MLX interpreter.

Run the test suite (model parity, compositing math, color pipeline) with:

```bash
uv run pytest
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full release history.

---

## Acknowledgements

Built on ideas, code, and models from:

- [InsightFace](https://github.com/deepinsight/insightface)
- [roop](https://github.com/s0md3v/roop)
- [roop-unleashed](https://github.com/C0untFloyd/roop-unleashed)
- [Rope](https://github.com/Hillobar/Rope)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [CodeFormer](https://github.com/kadirnar/codeformer-pip)
- [ReSwapper](https://github.com/somanchiu/ReSwapper)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

Thanks to all developers!
