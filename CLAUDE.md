# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

roop-mlx: face swap for images/video, tuned for Apple Silicon only (arm64 macOS). Hybrid inference
engine mixing onnxruntime's CoreML execution provider (ANE/GPU) with an in-repo ONNX→MLX graph
interpreter, hardware video decode/encode via VideoToolbox, and a Gradio browser UI. There is no
CUDA/CPU-generic path — everything assumes Apple Silicon.

**Disclaimer/conditions of use are in README.md — this is a dual-use tool (legitimate research/VFX
vs. non-consensual deepfakes). Don't help circumvent the consent/lawful-use constraints described
there.**

## Commands

```bash
source venv/bin/activate
python run.py                    # launch the Gradio UI (http://localhost:7860)
python run.py --server_share     # expose via public Gradio share URL

pip install -r requirements-dev.txt
pytest                           # full suite (model parity, compositing math, color pipeline)
pytest tests/test_swap_parity.py -k some_test   # single test
flake8                           # E3/E4/F only, see .flake8 (excludes venv, clip)
```

Python 3.11–3.13 (`.python-version` pins 3.13.7; 3.14 works but is unverified on the face-swap path —
see memory). `pip install -r requirements.txt` pulls arm64 `mlx` and `onnxruntime` wheels; no
provider flags or manual wheel building needed. `ffmpeg` must be on PATH.

Models (~2 GB) auto-download to `models/` on first run via `roop/model_manager.py`; CoreML models
compile and cache to `models/*/coreml_cache/` on first use.

## Architecture

**Backend selection is the core design decision** (`roop/inference.py`): each ONNX model is
individually routed to onnxruntime+CoreML or the in-repo `OnnxMLXSession` interpreter
(`roop/mlx_runtime/onnx_loader.py`), based on benchmarked speed and numerical parity against the
ONNX reference — not a blanket "backend X for everything" choice. `_COREML_MODELS` lists which
models get CoreML; the routing rationale (and which models CoreML fails to compile, or partitions
poorly) is documented inline there — read it before changing which backend a model uses.
`create_session()` is the single entry point processors call to get a session; it falls back to the
MLX interpreter if onnxruntime is unavailable.

A related trap: CoreML's MLProgram runtime rejects unbounded/dynamic ONNX input dims (raises
"unbounded dimension" and silently falls back to CPU, forfeiting acceleration).
`_static_model_path()` in `inference.py` rewrites dynamic dims to fixed sizes before compilation —
`_STATIC_DIM_SIZES` maps per-model spatial dims (batch dim defaults to 1 if unmapped). If you add a
new CoreML-routed model that has any non-batch dynamic dim, it needs an entry here or it'll silently
degrade to CPU.

**Processing pipeline** (`roop/ProcessMgr.py`, ~1300 lines, the central orchestrator):
1. `ProcessMgr.initialize()` loads the enabled frame processors by name (`roop.globals.frame_processors`,
   e.g. `["face_swapper", "face_enhancer"]`) via `str_to_class` from `roop/processors/`.
2. `run_batch` / `run_batch_inmem` fan out frame processing across threads (reader thread → N worker
   threads calling `process_frame`/`process_videoframes` → writer thread), overlapping CPU-side
   decode/warp/composite with GPU/ANE inference. Thread count is deliberately small
   (`settings.py:_default_max_threads`) because MLX inference itself is serialized on the GPU — more
   workers only help the CPU-bound portions.
3. Per-frame: detect faces (`_detect_faces_with_tracker`, backed by `face_tracker.py`/`face_detector.py`),
   swap (`swap_faces` → processor's `Run`), optionally enhance/mask, then composite back
   (`paste_upscale`, `simple_blend_with_mask`, `blur_area`).

**Frame processors** (`roop/processors/*.py`) are independent plugins with a uniform
`Initialize(plugin_options)` / `Run(...)` / `Release()` shape, loaded dynamically by name — not
imported directly. Categories: face swap (`FaceSwapInsightFace`, `FaceSwapFaceFusion`), enhancers
(`Enhance_GFPGAN`, `Enhance_CodeFormer`, `Enhance_GPEN`, `Enhance_RestoreFormerPPlus`), masking
(`Mask_XSeg`, `Mask_Clip2Seg`), and frame-level effects (`Frame_Colorizer`, `Frame_Filter`,
`Frame_Masking`, `Frame_Upscale`). New processors follow this naming/lifecycle convention and get
registered wherever `ProcessMgr.initialize` resolves plugin names.

**State**: `roop/globals.py` holds mutable in-process run state (source/target paths, selected
enhancer, blend ratio, progress dict, etc.) — module-level globals, not a class. `settings.py`
(`Settings`, backed by `config.yaml`) holds persistent user-facing settings surfaced in the Settings
UI tab. `roop.globals.CFG` is the loaded `Settings` instance.

**UI** (`ui/`): Gradio app in `ui/main.py`, tabs under `ui/tabs/` (`faceswap_tab`, `livecam_tab`,
`extras_tab`, `facemgr_tab`, `settings_tab`) each building their own Gradio components and wiring
callbacks into `roop.core`/`ProcessMgr`.

**Video I/O**: `roop/ffmpeg_writer.py`, `roop/util_ffmpeg.py`, `roop/StreamWriter.py`,
`roop/virtualcam.py` handle hardware decode/encode (VideoToolbox) and BT.709 color tagging end to
end — color-space correctness across the encode path is a recurring source of subtle bugs here
(watch for frame-format mismatches feeding the raw bgr24 writer).

## Testing conventions

`tests/` mirrors the areas most prone to silent regressions on this stack: `test_swap_parity.py` and
`test_interpreter.py` check MLX-interpreter output against the ONNX reference numerically;
`test_inference_routing.py` checks the CoreML/MLX backend-selection logic itself;
`test_processmgr_math.py` covers compositing/blend math; `test_video_pipeline.py` and
`test_browser_preview.py` cover the ffmpeg/encode and preview-playback paths. When touching
`inference.py`, `onnx_loader.py`, or any processor's numerics, run the parity tests, not just import
checks.
