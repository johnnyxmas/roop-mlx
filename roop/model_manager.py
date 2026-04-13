"""Model download registry and FP16 conversion.

Centralises every ONNX model the app can possibly use, downloads any that
are missing during ``pre_check()``, and optionally converts FP32 ONNX
models to FP16 variants on disk so the rest of the codebase can load them
for faster inference on tensor-core GPUs.
"""

import os
import numpy as np

from roop.utilities import resolve_relative_path, conditional_download


# ---------------------------------------------------------------------------
# Complete model registry
# ---------------------------------------------------------------------------
# Each entry: (relative_dir, filename, [url, ...fallback_url])
# relative_dir is relative to the project root ``models/`` folder.

ALL_MODELS = [
    # -- face swap (InsightFace) --
    ("", "inswapper_128.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"]),
    ("", "inswapper_128_fp16.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx"]),
    ("", "reswapper_128.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_128.onnx"]),
    ("", "reswapper_256.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_256.onnx"]),

    # -- face swap (FaceFusion / HyperSwap) --
    ("", "hyperswap_1a_256.onnx",
     ["https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx"]),
    ("", "hyperswap_1b_256.onnx",
     ["https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx"]),
    ("", "hyperswap_1c_256.onnx",
     ["https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx"]),

    # -- face enhancement --
    ("", "GFPGANv1.4.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx"]),
    ("", "GPEN-BFR-512.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx"]),
    ("", "GPEN-BFR-1024.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx"]),
    ("", "GPEN-BFR-2048.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_2048.onnx"]),
    ("", "restoreformer_plus_plus.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx"]),
    ("", "DMDNet.pth",
     ["https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth"]),
    ("CodeFormer", "CodeFormerv0.1.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx"]),

    # -- face masking --
    ("", "xseg.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx"]),
    ("CLIP", "rd64-uni-refined.pth",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth"]),

    # -- face detection / analysis (InsightFace buffalo_l) --
    ("buffalo_l", "1k3d68.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/1k3d68.onnx"]),
    ("buffalo_l", "2d106det.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx"]),
    ("buffalo_l", "det_10g.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/det_10g.onnx"]),
    ("buffalo_l", "genderage.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx"]),
    ("buffalo_l", "w600k_r50.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx"]),

    # -- YOLO face detection --
    ("", "yoloface_8n.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx"]),

    # -- frame processors --
    ("Frame", "deoldify_artistic.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx"]),
    ("Frame", "deoldify_stable.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx"]),
    ("Frame", "isnet-general-use.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx"]),
    ("Frame", "real_esrgan_x4.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx"]),
    ("Frame", "real_esrgan_x2.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx"]),
    ("Frame", "lsdir_x4.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx"]),
]


# Models eligible for FP16 conversion -- every ONNX model except those
# that are already FP16 (inswapper_128_fp16) or non-ONNX (.pth).
_FP16_ELIGIBLE = {
    # face swap
    "inswapper_128.onnx",
    # reswapper models have Gemm nodes that require matched types -- FP16
    # conversion creates float/float16 mismatches that ORT rejects.
    "hyperswap_1a_256.onnx",
    "hyperswap_1b_256.onnx",
    "hyperswap_1c_256.onnx",
    # face enhancement models all have Conv nodes that require matched types;
    # FP16 conversion with keep_io_types=True produces float/float16 mismatches
    # that ORT rejects.  Run these as FP32.
    # face masking
    "xseg.onnx",
    # face detection / analysis (buffalo_l)
    # FP16 variants are stored in buffalo_l_fp16/ so InsightFace never sees them
    "det_10g.onnx",
    "w600k_r50.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    # yolo detection
    "yoloface_8n.onnx",
    # frame processors
    "deoldify_artistic.onnx",
    "deoldify_stable.onnx",
    "isnet-general-use.onnx",
    # real_esrgan_x4/x2 and lsdir_x4 use a pytorch opset that onnx cannot
    # parse for modification -- they run fine as-is under onnxruntime.
}


def download_all_models():
    """Download every model in the registry that isn't already on disk."""
    models_root = resolve_relative_path("../models")

    # Group downloads by target directory for efficiency
    by_dir: dict[str, list] = {}
    for subdir, filename, urls in ALL_MODELS:
        target_dir = os.path.join(models_root, subdir) if subdir else models_root
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            continue
        by_dir.setdefault(target_dir, []).append((filename, urls))

    for target_dir, entries in by_dir.items():
        os.makedirs(target_dir, exist_ok=True)
        all_urls = []
        for filename, urls in entries:
            # Check if the file exists with a different case/name
            # (handles gpen_bfr_1024.onnx -> GPEN-BFR-1024.onnx rename)
            all_urls.extend(urls)
        if all_urls:
            conditional_download(target_dir, all_urls)

        # Handle filename renames (downloaded name != expected name)
        for filename, urls in entries:
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                continue
            # Check if file was downloaded with URL basename
            for url in urls:
                downloaded_name = os.path.basename(url)
                downloaded_path = os.path.join(target_dir, downloaded_name)
                if os.path.exists(downloaded_path) and downloaded_name != filename:
                    os.rename(downloaded_path, target_path)
                    print(f"  Renamed {downloaded_name} -> {filename}")
                    break


def _fp16_path(original_path: str) -> str:
    """Return the path for the FP16 variant of an ONNX model."""
    base, ext = os.path.splitext(original_path)
    return f"{base}_fp16{ext}"


def _buffalo_fp16_path(models_root: str, filename: str) -> str:
    """Return FP16 output path for a buffalo_l model.

    FP16 variants live in ``buffalo_l_fp16/`` (same filename, different dir)
    so InsightFace's scanner never sees them.
    """
    return os.path.join(models_root, "buffalo_l_fp16", filename)


def convert_models_to_fp16():
    """Convert eligible ONNX models to FP16, skipping those already done.

    This runs at startup and is idempotent -- if the FP16 file already
    exists, the model is skipped.

    buffalo_l models are stored in ``models/buffalo_l_fp16/`` (not inside
    ``buffalo_l/``) so InsightFace's directory scanner is unaffected.
    """
    models_root = resolve_relative_path("../models")

    for subdir, filename, _urls in ALL_MODELS:
        if filename not in _FP16_ELIGIBLE:
            continue

        target_dir = os.path.join(models_root, subdir) if subdir else models_root
        fp32_path = os.path.join(target_dir, filename)

        # buffalo_l FP16 variants go to a sibling directory
        if subdir == "buffalo_l":
            fp16_path = _buffalo_fp16_path(models_root, filename)
        else:
            fp16_path = _fp16_path(fp32_path)

        if not os.path.exists(fp32_path):
            continue
        if os.path.exists(fp16_path):
            continue

        os.makedirs(os.path.dirname(fp16_path), exist_ok=True)
        print(f"  Converting {filename} to FP16...")
        try:
            _convert_one_model(fp32_path, fp16_path)
            print(f"  -> {os.path.basename(fp16_path)}")
        except Exception as e:
            print(f"  FP16 conversion failed for {filename}: {e}")
            # Clean up partial file
            if os.path.exists(fp16_path):
                try:
                    os.remove(fp16_path)
                except OSError:
                    pass


def _load_onnx(path: str):
    """Load an ONNX model, trying external-data mode on parse failure."""
    import onnx
    try:
        return onnx.load(path)
    except Exception:
        # Some models use protobuf external data or exceed the 2 GB limit.
        # load_external_data=False reads only the graph structure.
        return onnx.load(path, load_external_data=False)


def _convert_one_model(src: str, dst: str):
    """Convert a single FP32 ONNX model to FP16.

    Tries ``onnxconverter_common`` first (best), then falls back to a
    manual weight-only conversion using just ``onnx`` + ``numpy``.
    """
    # Strategy 1: onnxconverter_common (handles graph types properly)
    try:
        from onnxconverter_common import float16 as oc_fp16

        model = _load_onnx(src)
        model_fp16 = oc_fp16.convert_float_to_float16(
            model,
            keep_io_types=True,       # keep I/O as float32 for compatibility
            min_positive_val=1e-7,
            max_finite_val=1e4,
        )
        import onnx
        onnx.save(model_fp16, dst)
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"    onnxconverter_common failed ({e}), trying manual conversion...")

    # Strategy 2: manual weight conversion (just converts initializers)
    from onnx import numpy_helper, TensorProto

    model = _load_onnx(src)

    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(initializer)
            # Clamp extreme values to avoid inf in FP16
            arr = np.clip(arr, -65504.0, 65504.0)
            arr_fp16 = arr.astype(np.float16)
            new_tensor = numpy_helper.from_array(arr_fp16, initializer.name)
            initializer.CopyFrom(new_tensor)
            initializer.data_type = TensorProto.FLOAT16

    import onnx
    onnx.save(model, dst)


def get_fp16_model_path(original_path: str) -> str:
    """Return the FP16 model path if it exists, otherwise the original.

    buffalo_l models are stored in ``buffalo_l_fp16/`` (same filename) to
    keep InsightFace's scanner clean.  All other models use ``_fp16.onnx``
    alongside the original.
    """
    # Check for buffalo_l special directory
    models_root = resolve_relative_path("../models")
    buffalo_dir = os.path.join(models_root, "buffalo_l") + os.sep
    if original_path.startswith(buffalo_dir):
        filename = os.path.basename(original_path)
        fp16 = _buffalo_fp16_path(models_root, filename)
        if os.path.exists(fp16):
            return fp16
        return original_path

    fp16 = _fp16_path(original_path)
    if os.path.exists(fp16):
        return fp16
    return original_path
