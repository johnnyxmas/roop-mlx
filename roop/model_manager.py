"""Model download registry for the MLX fork.

The list of remote model URLs is preserved so the existing pre_check()
download step still works.  The FP16 ONNX conversion that the upstream
repo ran at startup is gone -- MLX consumes its own weight format and
will need a per-model converter when each processor is ported.
"""

import os

from roop.utilities import resolve_relative_path, conditional_download


# Each entry: (relative_dir, filename, [url, ...fallback_url])
ALL_MODELS = [
    # -- face swap (InsightFace) --
    ("", "inswapper_128.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"]),
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

    ("CodeFormer", "CodeFormerv0.1.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx"]),

    # -- face masking --
    ("", "xseg.onnx",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx"]),
    ("CLIP", "rd64-uni-refined.pth",
     ["https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth"]),

    # -- face detection (YOLOv8n-Face) --
    ("", "yoloface_8n.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/yoloface_8n.onnx"]),

    # -- face analysis (InsightFace buffalo_l; detection itself is YOLO) --
    ("buffalo_l", "2d106det.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx"]),
    ("buffalo_l", "genderage.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx"]),
    ("buffalo_l", "w600k_r50.onnx",
     ["https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx"]),

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
    ("Frame", "span_kendata_x4.onnx",
     ["https://huggingface.co/facefusion/models-3.0.0/resolve/main/span_kendata_x4.onnx"]),
]


# Downloaded by older versions but never loaded by the MLX port.
_STALE_MODELS = [
    ("buffalo_l", "det_10g.onnx"),
    ("buffalo_l", "1k3d68.onnx"),
]


def _remove_stale_models(models_root: str):
    for subdir, fname in _STALE_MODELS:
        path = os.path.join(models_root, subdir, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"Removed unused model {subdir}/{fname}")
            except OSError:
                pass


_INTEGRITY_MARKER = ".integrity_ok"


def _file_is_corrupt(path: str, filename: str) -> bool:
    """Return True if *path* appears truncated or corrupt."""
    if os.path.getsize(path) < 1024:
        return True
    if filename.endswith(".onnx"):
        try:
            import onnx
            onnx.load(path)
        except Exception:
            return True
    # .pth files only get the size check: torch is not a dependency of
    # this fork, so a parse-based check would false-positive forever.
    return False


def _redownload_model(target_dir: str, filename: str, urls: list) -> bool:
    path = os.path.join(target_dir, filename)
    try:
        os.remove(path)
    except OSError:
        pass
    conditional_download(target_dir, urls)
    return os.path.exists(path) and os.path.getsize(path) > 1024


def verify_all_models(force: bool = False) -> None:
    """One-time integrity sweep: detect truncated/corrupt downloads and re-fetch.

    Guarded by a marker file so the full parse only runs once per install.
    Re-runs automatically if a previous sweep found a bad file (marker absent).
    """
    models_root = resolve_relative_path("../models")
    marker = os.path.join(models_root, _INTEGRITY_MARKER)
    if not force and os.path.exists(marker):
        return

    all_ok = True
    for subdir, filename, urls in ALL_MODELS:
        target_dir = os.path.join(models_root, subdir) if subdir else models_root
        path = os.path.join(target_dir, filename)
        if not os.path.exists(path):
            continue
        if _file_is_corrupt(path, filename):
            print(f"  {filename} failed integrity check — re-downloading.")
            if not _redownload_model(target_dir, filename, urls):
                print(f"  Could not repair {filename}; will retry next run.")
                all_ok = False

    if all_ok:
        try:
            os.makedirs(models_root, exist_ok=True)
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("ok\n")
        except OSError:
            pass


def download_all_models():
    """Download every model in the registry that isn't already on disk."""
    models_root = resolve_relative_path("../models")
    _remove_stale_models(models_root)

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
            all_urls.extend(urls)
        if all_urls:
            conditional_download(target_dir, all_urls)

        for filename, urls in entries:
            target_path = os.path.join(target_dir, filename)
            if os.path.exists(target_path):
                continue
            for url in urls:
                downloaded_name = os.path.basename(url)
                downloaded_path = os.path.join(target_dir, downloaded_name)
                if os.path.exists(downloaded_path) and downloaded_name != filename:
                    os.rename(downloaded_path, target_path)
                    print(f"  Renamed {downloaded_name} -> {filename}")
                    break
