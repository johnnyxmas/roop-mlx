# roop-floyd

Face swap images and video without training, powered by a browser-based GUI.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Changelog](CHANGELOG.md)

> On first run, models (~2 GB) are downloaded automatically to `models/`.

---

## Features

- Browser-based GUI — works on Windows, Linux, and macOS
- Multiple face swapping modes: first detected, by selection, by gender, random, all input faces
- ReSwapper model option (128px & 256px) alongside InsightFace inswapper
- Batch processing of images and video
- Face occlusion masking via text prompts (CLIP) or XSeg
- Face enhancers: GFPGAN, CodeFormer, DMDNet, GPEN (512/1024/2048px), RestoreFormer++
- Subsample upscaling for higher resolution swaps
- Frame processors: colorizer, upscaler, filters, background removal
- Live webcam face swap with virtual camera output
- Multi-GPU support — distribute frame workload across all available CUDA GPUs
- Per-GPU device selection in Settings (no restart required for single-GPU switch)
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

- **Python 3.11** — [python.org](https://www.python.org/downloads/)
- **ffmpeg** on your system PATH — [ffmpeg.org](https://ffmpeg.org/download.html) (required for video processing)
- A virtual environment is strongly recommended to isolate dependencies

---

### Windows — Quick Setup (Recommended)

Run the appropriate script for your GPU from the repo root. Each script creates a `venv/` folder and installs all dependencies automatically.

**NVIDIA RTX (any generation with CUDA 13 driver — recommended):**

```bat
setup_cuda13.bat
```

**NVIDIA RTX (CUDA 12.8 driver):**

```bat
setup_cuda12.bat
```

Then launch with:

```bat
roop-floyd.bat
```

---

### Manual Setup

**Step 1 — Create and activate a virtual environment (strongly recommended):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**Step 2 — Install PyTorch:**

| GPU | Command |
| --- | --- |
| NVIDIA / CUDA 13 (RTX 20-series and newer, recommended) | `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| NVIDIA / CUDA 12.8 | `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| AMD (Windows) | `pip install torch torchvision torchaudio` then use `onnxruntime-directml` below |
| macOS / CPU | `pip install torch torchvision torchaudio` |

**Step 3 — Install core dependencies:**

```bash
pip install uv
uv pip install numpy==1.26.4 opencv-python-headless onnx insightface albucore psutil tqdm ftfy regex pyvirtualcam
```

**Step 4 — Install ONNX Runtime:**

| Platform | Command |
| --- | --- |
| CUDA 13 | `pip install onnxruntime` then `pip install --no-cache-dir --no-deps onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/` |
| CUDA 12.8 | `pip install onnxruntime onnxruntime-gpu` |
| AMD (Windows) | `pip install onnxruntime onnxruntime-directml` |
| macOS Apple Silicon | `pip install onnxruntime onnxruntime-silicon` |
| CPU only | `pip install onnxruntime` |

**Step 5 — Final packages:**

```bash
pip install --force-reinstall pydantic==2.10.6
pip install gradio==5.13.0
```

---

### Linux

```bash
python -m venv venv && source venv/bin/activate
pip install uv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy==1.26.4 opencv-python-headless onnx insightface albucore psutil tqdm ftfy regex pyvirtualcam onnxruntime onnxruntime-gpu
pip install --force-reinstall pydantic==2.10.6
pip install gradio==5.13.0
python run.py
```

---

## Usage

**Windows:** `roop-floyd.bat` (or `python run.py` with venv active)

**Linux / macOS:**

```bash
source venv/bin/activate
python run.py
```

**CLI flags:**

- `--server_share` — expose via Gradio public URL (Hugging Face Spaces / remote access)
- `--cuda_device_id <N>` — select GPU by index (default: 0; overrides config)

The web UI opens automatically in your browser at `http://localhost:7860`.

---

## Settings Reference

All settings are stored in `config.yaml` and editable from the **⚙ Settings** tab. Changes take effect on the next run unless marked *requires restart*.

### UI / Server

| Setting | Description |
| --- | --- |
| **Theme** | Visual style for the Gradio UI. Built-in options: `ocean` (default), `citrus`, `glass`, `monochrome`, `origin`, `soft`, `default`. Change requires a full server restart. |
| **Public Server** | Expose the UI via a public Gradio share URL. Anyone with the link can use your local GPU — share with care. |
| **Server Name** | Host/IP the Gradio server binds to. Leave blank for localhost-only (`127.0.0.1`). Set to `0.0.0.0` to allow LAN access. |
| **Server Port** | TCP port the Gradio server listens on. Leave at `0` to let Gradio pick an available port. |
| **Clear output folder before each run** | Delete everything in the output folder before each run starts. Useful for batch jobs where you don't want old results mixing in. |
| **Filename Output Template** | Template for saved result filenames. `{file}` = source filename, `{time}` = timestamp. File extension is added automatically. |

### Hardware / Performance

| Setting | Description |
| --- | --- |
| **Provider** | ONNX Runtime execution provider. `cuda`: NVIDIA GPUs (fastest). `dml`: DirectML for AMD/Intel on Windows. `rocm`: AMD on Linux. `cpu`: fallback, very slow. *Requires restart.* |
| **CUDA Device** | Which NVIDIA GPU to use when multiple are installed. Only shown on multi-GPU systems. *Requires restart.* |
| **Use all GPUs (multi-GPU)** | Distribute frame workload across every detected CUDA GPU. Each GPU gets its own processor set — roughly linear scaling on batch jobs. Only shown when more than one CUDA GPU is detected. *Requires restart.* |
| **Force CPU for Face Analyser** | Run face detection and embedding on CPU even when a GPU provider is active. Swap and enhancer models still use the GPU. Useful when VRAM is tight — face analysis is cheap on CPU. |
| **Max. Number of Threads** | Frames processed in parallel during video runs. Default: 3. Higher values can saturate a fast GPU but increase VRAM and RAM usage. Don't exceed your physical CPU core count. |
| **Max. Memory (GB)** | Hard cap on system RAM for the process. `0` = no limit (OS decides). Useful on Windows to prevent runaway paging on long video encodes. |
| **GPU Video Decode (NVDEC)** | Use FFmpeg hardware-accelerated video decoding (NVDEC / DXVA2). Faster than CPU decode at 1080p+. Falls back to software decode if no hardware decoder is available. |

### Face Detection

| Setting | Description |
| --- | --- |
| **Face Detector** | Detection backend. **InsightFace**: Buffalo_l RetinaFace — accurate, moderate speed (default). **YOLOv8n-Face**: 3–5× faster, slightly less accurate on tiny or occluded faces. Both produce the same face data for the swap pipeline. |
| **Use default Det-Size** | Use InsightFace's default 640×640 detector input. Uncheck to use a larger 1024×1024 input — slower but finds smaller and more distant faces. |

### Performance Profile

Selected under **Performance Profile** in Settings.

| Profile | VRAM | Description |
| --- | --- | --- |
| `memory_saver` | ≤ 6 GB | Safest. Disables ORT memory arena, pattern, and buffer reuse. Face detection runs every frame. |
| `balanced` | ≥ 8 GB | Enables ORT allocators and preallocates CUDA input buffers. Typically 20–40% faster with ~200–500 MB extra VRAM. |
| `max_speed` | ≥ 12 GB | Adds TensorRT (FP16) and a larger detector stride. **First run is slow** — TensorRT builds an engine per model (2–10 min, app may appear frozen). Engines are cached under `models/trt_cache/`; subsequent runs are fast. Cache is invalidated by driver or ORT upgrades. |

**Detector Stride** (max_speed only): how often to run full face detection. `1` = every frame (safest). `3–5` = reuse previous boxes for intermediate frames. Optical-flow tracking smooths results at high strides; a scene-cut check forces redetection when the frame changes drastically.

### Output

| Setting | Description |
| --- | --- |
| **Image Output Format** | Format for still-image results. `png`: lossless, larger files. `jpg`: lossy, much smaller. `webp`: good compression with optional lossless mode. |
| **Video Codec** | FFmpeg codec for the final video. `auto`: NVENC if available, else libx264. `libx264`/`libx265`: CPU H.264/H.265 — high quality, slower. `h264_nvenc`/`hevc_nvenc`: GPU-encoded — faster, slightly lower quality per bitrate. `libvpx-vp9`: WebM/VP9, CPU-only. |
| **Video Output Format** | Container format. `mp4`: most compatible. `mkv`: best for unusual codec/audio combinations. `avi`: legacy. `webm`: pairs with libvpx-vp9. |
| **Video Quality (CRF)** | FFmpeg CRF value. Lower = higher quality, larger file. Default: 14 (near-lossless on x264). Typical range: 18–28; above 28 shows visible compression artifacts. Ignored by NVENC, which uses its own quality model. |
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
- [DMDNet](https://github.com/csxmli2016/DMDNet)
- [ReSwapper](https://github.com/somanchiu/ReSwapper)
- [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

Thanks to all developers!
