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
