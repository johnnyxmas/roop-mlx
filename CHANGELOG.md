# Changelog

## v5.0 — 2026-11-09

roop-mlx is now an Apple Silicon–only fork: a hybrid inference engine (Apple Core AI, onnxruntime CoreML, and an in-repo ONNX→MLX interpreter) replaces the CUDA/TensorRT path. Note: there was no released 4.6 — those swap-model additions are already in v4.5 below — so this entry covers everything since v4.5.

### Inference engine
- **Apple Core AI backend (macOS 27+)**: HyperSwap, GFPGAN and GPEN run through a Swift C-ABI bridge (`libcoreai_bridge.dylib`), with models converted to `.aimodel` on first use and cached. Benchmarked 1.33x (HyperSwap), 1.84x (GFPGAN) and 1.33x (GPEN) over the previous path. `ROOP_DISABLE_COREAI=1` disables it
- Per-model backend routing (`create_session()`): Core AI, then onnxruntime CoreML, then the MLX interpreter, chosen by benchmarked speed and numerical parity; every failure falls through to the next backend
- ONNX→MLX graph interpreter replaces CUDA/ONNX Runtime GPU paths; all processors (InSwapper, HyperSwap, GFPGAN, CodeFormer, GPEN, RestoreFormer++, XSeg, Colorizer, Upscale, Masking) ported to MLX
- onnxruntime CoreML execution provider (ANE/GPU) for the detector stack, large enhancers and swap models; dynamic ONNX dims are rewritten to static sizes to avoid silent CPU fallback
- Interpreter speedups: separable bilinear Resize, Winograd off, true fp32 convs, attribute cache and weight pre-transpose (GFPGAN ~30% faster)
- Interpreter fixes: ONNX edge padding (DeOldify colour drift), opset-10 Resize, `Range` op
- Serialized CoreML compiles; poisoned/unloadable compile caches self-heal instead of wiping other models' caches; orphan cache growth fixed
- Verified on onnxruntime 1.30; dependencies modernised (OpenCV 5, onnxruntime 1.28+)

### Models
- Added SPAN x4 upscaler (512px tiles), replacing Real-ESRGAN x4 and LSDIR x4
- Added AlphaFace 256 swapper
- Added source identity weight (ported from FaceFusion's `face_swapper_weight`)
- Clip2Seg text masking ported to ONNX CLIPSeg (torch `clip` package dropped)
- Subsample-upscale dropdown gated to the swap model's native size
- Retired InSwapperNative / pixel-boost batch path, DMDNet, YOLO face detector (InsightFace buffalo_l restored on CoreML/ANE)
- `verify_all_models()` integrity sweep at startup catches truncated or corrupt downloads

### Detection and video
- Rotated/horizontal faces are now detected (rotation-retry detector) and 106-point landmarks no longer collapse; faceless frames cost ~16.5 ms
- Hardware video decode/encode via VideoToolbox; end-to-end BT.709 colour tagging
- Fixed diagonal output shear (stride desync) with a guard on frames fed to the raw bgr24 writer
- Fixed frame extraction on ffmpeg 9 (`-vsync` removed)
- Video downscale/re-encode buttons (480/720/1080/2160p + FPS) via `scale_video()`
- Opt-in re-encode for browser-unplayable output previews; preview now shows the converted target
- Stop button for target re-encodes; fixed stuck swap lock after Stop and FPS override truncating video

### UI and fixes
- Gradio 5 → 6 migration; dark theme rework; settings tooltips
- Fixed frame slider rejecting a newly selected video
- Fixed manual-mask toggle and mask sliders crashing (tuple offsets)
- Local-folder add now skips hidden files
- Fixed `rlim_t` overflow on macOS

### Project
- Dependencies managed with uv (`pyproject.toml` + `uv.lock`); `requirements*.txt` are generated exports
- Added regression test suite (swap/interpreter parity, routing, compositing, video pipeline), macOS ARM CI, Dependabot, and CLAUDE.md
- README rewritten as a standalone roop-mlx project

## v4.5 — 2026-04-12

- Major speed boost for video processing: CUDA OrtValue preallocation with per-thread io_binding eliminates repeated GPU buffer allocation across frames
- Fixed video processing hang: thread-local FaceTracker instances prevent cross-thread race conditions that caused deadlocks at high thread counts
- Fixed video processing deadlock: uncaught exceptions in worker threads now always unblock the writer thread
- Fixed scene-change artifacts: optical flow tracker now falls back to fresh face detection instead of returning stale keypoints from a prior scene
- Fixed rainbow/halftone artifacts on HyperSwap models: removed incompatible CUDA graph capture from HyperSwap inference session
- Fixed thread-safety crash in CodeFormer, RestoreFormer++, Frame_Colorizer, Frame_Upscale, and Frame_Masking: all shared `io_binding` objects replaced with per-call local bindings
- Fixed FP16 conversion failures for all face enhancement models (GFPGAN, GPEN, CodeFormer, RestoreFormer++): Conv nodes require matched types that `keep_io_types=True` breaks — enhancement models now run as FP32
- Added FP16 conversion for buffalo_l face analysis models stored in a separate directory so InsightFace's scanner is unaffected
- Added centralized model download registry and FP16 conversion pipeline (`model_manager.py`)
- Added YOLOv8n-Face as the face detector for `memory_saver` profile; InsightFace buffalo_l is used for `balanced` and `max_speed` (no separate setting)
- Added optical-flow face tracking between detection frames with automatic scene-cut redetection
- Added GPU-accelerated video decode (NVDEC) option in Settings
- Added performance profile selector: `memory_saver`, `balanced`, `max_speed` (TensorRT)
- Added configurable detector stride and TensorRT engine cache
- Startup now purges corrupt Gradio temp image files to prevent ImageEditor preprocessing crashes
- Fixed video frame seek failures (H.264 B-frame libavcodec assertion) with automatic VideoCapture reopen and retry
- Fixed InsightFace startup time: emap matrix cached as `.npy` sidecar to avoid parsing the full 500 MB ONNX protobuf on every launch
- Ocean dark theme set as default
- Multi-GPU support: frame workload distributed across all CUDA GPUs simultaneously
- GPU device selection dropdown in Settings UI (detected automatically at startup)
- GPEN now supports 512, 1024, and 2048px variants with matched upscale pipeline
- Dynamic face crop size matches enhancer input resolution (no longer hardcoded to 512px)
- CUDA 13 support: PyTorch 2.11+cu130 and ONNX Runtime CUDA 13 build
- CUDA 12.8 support via `setup_cuda12.bat`
- Upgraded ONNX Runtime to 1.24.4
- Added InSwapper 128 FP16 swap model (same quality, ~2x faster, half VRAM)
- Added HyperSwap 1A / 1B / 1C swap models (256px, best overall quality)
- Added Ghost 1 / 2 / 3 swap models (256px, better expression/lighting transfer)
- Added UniFace swap model (256px, improved off-angle face handling)
- New models auto-download on first use via dedicated `FaceSwapFaceFusion` processor
- Models now download during setup instead of first run
- InSwapper 128 fp32 labeled as "(CPU)" in model dropdown
- Added `setup_cuda13.bat`, `setup_cuda12.bat`, and `roop-floyd.bat` setup/launch scripts
- Added `download_models.py` for pre-downloading all models during setup
- Removed all Codeberg fallback URLs; all models now download from HuggingFace/GitHub
- Fixed IO binding allocator crash in multi-GPU mode across all processors
- Fixed face analysis always running on GPU 0 regardless of selected device
- Fixed null embeddings crash in FaceSet when averaging across multiple source faces
- Fixed download crash when a model URL returns 404
- Fixed GPEN 1024/2048 download (now sourced from facefusion/models-3.0.0)
- Fixed Ghost/HyperSwap input binding order (target=frame, source=embedding)
- Fixed UniFace preprocessing and input binding
- Fixed swap crash when source face data is None or empty
- Improved FFMPEG error handling and NVENC failure recovery
- Fixed video output corruption and temp folder cleanup
- Fixed face swapping for subsequent runs after successful completion
- Removed dead code: unused globals, imports, deprecated files, and upstream leftovers
- Pinned breakage-prone dependencies (pillow, markupsafe, scipy, scikit-image)

## v4.4.1 — 2025-01-03

- Bugfix: Extras Tab
- Bugfix: Preview Mask
- Bugfix: DMDNet working on CUDA only
- Updated Colab Script to also install CUDNN

## v4.4.0 — 2024-12-30

- Added random face selection mode
- Added ReSwapper alternative swap model (128px & 256px output)
- Video repair added to Extras Tab
- Updated most packages to newer versions — CUDA >= 12.4 required
- Several minor bugfixes and QoL changes

## v4.3.1 — 2024-09-28

- Bugfix: Several possible memory leaks
- Added different output modes including virtual cam stream
- New swapping mode: All input faces
- Average total FPS displayed; setting for autorun

## v4.2.0 — 2024-09-05

- Added ability to reorder input & target faces
- New CLI arguments: `--server_share`, `--cuda_device_id`
- Added mask option and mouth area restoration
- Added webp image support

## v4.1.0 — 2024-07-14

- Added subsample upscaling to increase swap resolution

## v4.0.0 — 2024-05-12

- Added xseg masking to live-cam
- Added RealESRGAN x2 to frame processors
- Added colorize, upscale, and misc filters to Extras Tab

## v3.9.0 — 2024-04-22

- Added DFL XSeg support for partial face occlusion
- Massive rewrite of swapping architecture

## v3.5.5 — 2024-03-10

- Replaced RestoreFormer with RestoreFormer++
- Live Cam recoded to run independently from virtual cam

## v3.5.0 — 2024-01-08

- Auto rotation of horizontal faces
- Simple VR option for stereo images/video
- Added RestoreFormer enhancer

## v2.7.0 — 2023-08-11

Initial Gradio version (TkInter version deprecated)
