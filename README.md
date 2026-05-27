# roop-mlx

Face swap images and video without training, powered by Apple MLX and a browser-based GUI.
Apple Silicon (arm64 macOS) only.

[Installation](#installation) • [Usage](#usage) • [Changelog](CHANGELOG.md)

> On first run, models (~2 GB) are downloaded automatically to `models/`.

---

## Features

- Browser-based GUI
- Multiple face swapping modes: first detected, by selection, by gender, random, all input faces
- ReSwapper model option (128px & 256px) alongside InsightFace inswapper
- Batch processing of images and video
- Face occlusion masking via XSeg
- Face enhancers: GFPGAN, CodeFormer, GPEN (512/1024/2048px), RestoreFormer++
- Subsample upscaling for higher resolution swaps
- Frame processors: colorizer, upscaler, filters, background removal
- Live webcam face swap with virtual camera output
- VideoToolbox hardware encoding (h264/hevc) on Apple Silicon, libx264 fallback
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
- **Python 3.11** — [python.org](https://www.python.org/downloads/) or Homebrew
- **ffmpeg** on your system PATH — `brew install ffmpeg` is the easy path
- A virtual environment is strongly recommended

### Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No provider flags, no CUDA wheels, no TensorRT to build.
The `mlx` and `mlx-nn` packages pull arm64 wheels automatically.

---

## Usage

```bash
source venv/bin/activate
python run.py
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
| **Show video in browser** | Re-encode the final video for in-browser playback. Adds an extra encode pass — disable if you don't need browser preview. |

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
