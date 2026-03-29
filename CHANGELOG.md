# Changelog

## v4.5 — 2025-03-29

- Multi-GPU support: frame workload distributed across all CUDA GPUs simultaneously
- GPU device selection dropdown in Settings UI (detected automatically at startup)
- GPEN now supports 512, 1024, and 2048px variants with matched upscale pipeline
- Dynamic face crop size matches enhancer input resolution (no longer hardcoded to 512px)
- CUDA 13 support: PyTorch 2.11+cu130 and ONNX Runtime CUDA 13 build
- Upgraded ONNX Runtime to 1.24.4
- Fixed IO binding allocator crash in multi-GPU mode across all processors
- Fixed face analysis always running on GPU 0 regardless of selected device
- Fixed null embeddings crash in FaceSet when averaging across multiple source faces
- Improved FFMPEG error handling and NVENC failure recovery
- Fixed video output corruption and temp folder cleanup
- Fixed face swapping for subsequent runs after successful completion
- Added `setup_cuda13.bat`, `setup_cuda12.bat`, and `roop-floyd.bat` setup/launch scripts
- Added InSwapper 128 FP16 swap model (same quality, ~2x faster, half VRAM)
- Added HyperSwap 1A / 1B / 1C swap models (256px, best overall quality)
- Added Ghost 1 / 2 / 3 swap models (256px, better expression/lighting transfer)
- Added UniFace swap model (256px, improved off-angle face handling)
- New models auto-download on first use via dedicated `FaceSwapFaceFusion` processor

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
