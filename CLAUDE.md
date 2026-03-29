# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
python run.py
```

Optional CLI flags:
- `--server_share` — expose via Gradio public URL
- `--cuda_device_id <N>` — select GPU index (default: 0)

On first run, models (~2 GB) are auto-downloaded to `models/`. ffmpeg must be installed and on PATH.

## Installation (CUDA 13 / 50-series GPU)

```bash
python -m venv venv && venv/scripts/activate
pip install uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install numpy opencv-python-headless onnx insightface albucore psutil onnxruntime tqdm ftfy regex pyvirtualcam
uv pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
pip install --force-reinstall pydantic==2.10.6
pip install --upgrade gradio==5.13.0
```

## Installation (CUDA 12.x / 50-series GPU)

```bash
python -m venv venv && venv/scripts/activate
pip install uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy opencv-python-headless onnx insightface albucore psutil onnxruntime onnxruntime-gpu tqdm ftfy regex pyvirtualcam
pip install --force-reinstall pydantic==2.10.6
pip install --upgrade gradio==5.13.0
```

For 30/40-series NVIDIA: `pip install -r requirements.txt` (uses CUDA 11.8+, onnxruntime-gpu).
For AMD: use `onnxruntime-directml` instead.

## Linting

```bash
flake8 .
```

Config in [.flake8](.flake8): checks E3, E4, F rules only. `roop/core.py` has E402 exempted.

## Architecture Overview

The app is a **Gradio-based web UI** for face-swapping images and video. Entry point: `run.py` → `roop/core.py:run()`.

### Startup Flow

1. `roop/core.py:run()` — parses args, runs `pre_check()` (downloads models), loads `config.yaml` into `Settings`, then calls `ui/main.py:run()`.
2. `ui/main.py` — sets up Gradio server, builds tab layout from `ui/tabs/`.
3. Execution provider (CUDA/CPU/DML/ROCM) is resolved at startup and stored in `roop.globals.execution_providers`.

### Core Processing Path

- **`roop/ProcessMgr.py`** (`ProcessMgr`) — orchestrates all processing. Initialized with `INPUT_FACESETS`, `TARGET_FACES`, and a `ProcessOptions` object. Two modes:
  - `run_batch()` — file-based (extract frames to disk, process, re-encode)
  - `run_batch_inmem()` — in-memory pipeline using a producer/consumer queue with `FFMPEG_VideoWriter` or `StreamWriter`
- **`roop/processors/`** — plugin-style processor classes. Each has `Initialize(plugin_options)`, `Run(source_face, target_face, frame)`, and `Release()`. Loaded dynamically via `str_to_class()`. Types:
  - `swap` — `FaceSwapInsightFace` (inswapper/reswapper ONNX models)
  - `mask` — `Mask_Clip2Seg`, `Mask_XSeg`
  - `enhance` — `Enhance_GFPGAN`, `Enhance_CodeFormer`, `Enhance_DMDNet`, `Enhance_GPEN`, `Enhance_RestoreFormerPPlus`
  - `frame` — `Frame_Colorizer`, `Frame_Upscale`, `Frame_Filter`, `Frame_Masking`

### Key Data Structures

- **`roop/globals.py`** — module-level mutable state shared across the app (execution providers, face data, settings object `CFG`, processing flags).
- **`roop/FaceSet.py`** — container for a set of source faces used in one swap slot.
- **`roop/ProcessOptions.py`** — immutable options bundle passed to `ProcessMgr.initialize()` (swap model, plugins, thresholds, blend ratio, mode, mask, etc.).
- **`roop/ProcessEntry.py`** — represents one input file with start/end frame range and output path.

### Configuration

`config.yaml` (auto-created) / `settings.py:Settings` — persists UI settings between sessions. Key fields: `provider`, `max_threads`, `memory_limit`, `output_video_codec`, `video_quality`, `use_os_temp_folder`.

### UI Structure

`ui/tabs/` contains one file per Gradio tab:
- `faceswap_tab.py` — main swap workflow
- `livecam_tab.py` — webcam real-time swap using `roop/virtualcam.py` and `roop/capturer.py`
- `facemgr_tab.py` — manage input/target face selections
- `extras_tab.py` — video cutting, colorization, upscaling
- `settings_tab.py` — UI for editing `config.yaml`

### FFmpeg Integration

`roop/util_ffmpeg.py` — wraps ffmpeg for frame extraction, video creation, audio restoration, GIF creation. `roop/ffmpeg_writer.py` — streams processed frames directly to ffmpeg stdin (in-memory path). NVENC is attempted first; falls back to CPU encoding on failure.

### Models Directory Layout

```
models/
  inswapper_128.onnx       # InsightFace swap model
  reswapper_128.onnx       # ReSwapper 128px
  reswapper_256.onnx       # ReSwapper 256px
  GFPGANv1.4.onnx
  CodeFormer/CodeFormerv0.1.onnx
  GPEN-BFR-512.onnx
  DMDNet.pth
  xseg.onnx
  CLIP/rd64-uni-refined.pth
  buffalo_l/               # InsightFace face analysis models
  Frame/                   # Upscale/colorize models (ESRGAN, DeOldify, etc.)
```
