# roop-floyd

Face swap images and video without training, powered by a browser-based GUI.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Changelog](#changelog)

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

**RTX 50-series or any GPU with CUDA 13 driver:**

```bat
setup_cuda13.bat
```

**RTX 30/40-series (CUDA 12):**

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
| RTX 50-series / CUDA 13 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` |
| RTX 30/40-series / CUDA 12 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
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
| CUDA 13 | `pip install onnxruntime onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/` |
| CUDA 12 | `pip install onnxruntime onnxruntime-gpu` |
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
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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

## Changelog

**2026-03-29** v4.5

- Multi-GPU support: frame workload distributed across all CUDA GPUs simultaneously
- GPU device selection dropdown in Settings UI (detected automatically at startup)
- GPEN enhancer now supports 512, 1024, and 2048px resolution variants with matched upscale pipeline
- Dynamic face crop size matches enhancer input resolution (no longer hardcoded to 512px)
- CUDA 13 support: PyTorch 2.11+cu130 and ONNX Runtime CUDA 13 build
- Upgraded ONNX Runtime to 1.24.4
- Fixed IO binding allocator crash in multi-GPU mode across all processors
- Fixed face analysis always running on GPU 0 regardless of selected device
- Improved FFMPEG error handling and NVENC failure recovery
- Fixed video output corruption and temp folder cleanup
- Fixed face swapping for subsequent runs after successful completion
- Added `setup_cuda13.bat`, `setup_cuda12.bat`, and `roop-floyd.bat` setup/launch scripts

**03.01.2025** v4.4.1

- Bugfix: Extras Tab
- Bugfix: Preview Mask
- Bugfix: DMDNet working on CUDA only
- Updated Colab Script to also install CUDNN

**31.12.2024** v4.4.0

- Added random face selection mode
- Added ReSwapper alternative swap model (128px & 256px output)
- Video repair added to Extras Tab
- Updated most packages to newer versions — CUDA >= 12.4 required
- Several minor bugfixes and QoL changes

**28.09.2024** v4.3.1

- Bugfix: Several possible memory leaks
- Added different output modes including virtual cam stream
- New swapping mode: All input faces
- Average total FPS displayed; setting for autorun

**05.09.2024** v4.2.0

- Added ability to reorder input & target faces
- New CLI arguments: `--server_share`, `--cuda_device_id`
- Added mask option and mouth area restoration
- Added webp image support

**14.07.2024** v4.1.0

- Added subsample upscaling to increase swap resolution

**12.05.2024** v4.0.0

- Added xseg masking to live-cam
- Added RealESRGAN x2 to frame processors
- Added colorize, upscale, and misc filters to Extras Tab

**22.04.2024** v3.9.0

- Added DFL XSeg support for partial face occlusion
- Massive rewrite of swapping architecture

**10.03.2024** v3.5.5

- Replaced RestoreFormer with RestoreFormer++
- Live Cam recoded to run independently from virtual cam

**08.01.2024** v3.5.0

- Auto rotation of horizontal faces
- Simple VR option for stereo images/video
- Added RestoreFormer enhancer

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
