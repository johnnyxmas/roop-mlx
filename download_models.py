"""Download all required models during setup."""
import os
import sys
import urllib.request
import urllib.error
from tqdm import tqdm
from urllib.parse import urlparse

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DOWNLOADS = {
    "models": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_128.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_256.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx",
        "https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
    ],
    "models/CLIP": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth",
    ],
    "models/buffalo_l": [
        "https://huggingface.co/halllooo/buffalo_l/resolve/main/1k3d68.onnx",
        "https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx",
        "https://huggingface.co/halllooo/buffalo_l/resolve/main/det_10g.onnx",
        "https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx",
        "https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx",
    ],
    "models/CodeFormer": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx",
    ],
    "models/Frame": [
        "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx",
        "https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx",
    ],
}


def download_file(url: str, dest_dir: str) -> bool:
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(dest_dir, filename)
    if os.path.exists(filepath):
        return True
    try:
        request = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        print(f"  FAILED: {filename} ({e.code} {e.reason})")
        return False
    total = int(request.headers.get("Content-Length", 0))
    with tqdm(total=total, desc=f"  {filename}", unit="B", unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(
            url, filepath,
            reporthook=lambda count, block_size, total_size: progress.update(block_size),
        )
    return True


def main():
    base = os.path.dirname(__file__)
    failed = []
    for subdir, urls in DOWNLOADS.items():
        dest = os.path.join(base, subdir)
        os.makedirs(dest, exist_ok=True)
        for url in urls:
            if not download_file(url, dest):
                failed.append(url)
    if failed:
        print(f"\n{len(failed)} download(s) failed:")
        for url in failed:
            print(f"  {url}")
        sys.exit(1)
    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
