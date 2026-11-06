import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib
import gradio
import tempfile
import cv2
import zipfile
import traceback
import random

from typing import Any

from pathlib import Path
from typing import List
from tqdm import tqdm
from scipy.spatial import distance
from urllib.parse import urlparse

import roop.template_parser as template_parser

import roop.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"


# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# https://github.com/facefusion/facefusion/blob/master/facefusion
def detect_fps(target_path: str) -> float:
    fps = 24.0
    cap = cv2.VideoCapture(target_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def sort_filenames_ignore_path(filenames):
    """Sorts a list of filenames containing a complete path by their filename,
    while retaining their original path.

    Args:
      filenames: A list of filenames containing a complete path.

    Returns:
      A sorted list of filenames containing a complete path.
    """
    filename_path_tuples = [
        (os.path.split(filename)[1], filename) for filename in filenames
    ]
    sorted_filename_path_tuples = sorted(filename_path_tuples, key=lambda x: x[0])
    return [
        filename_path_tuple[1] for filename_path_tuple in sorted_filename_path_tuples
    ]


def sort_rename_frames(path: str):
    filenames = os.listdir(path)
    filenames.sort()
    for i in range(len(filenames)):
        of = os.path.join(path, filenames[i])
        newidx = i + 1
        new_filename = os.path.join(
            path, f"{newidx:06d}." + roop.globals.CFG.output_image_format
        )
        os.rename(of, new_filename)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(
        (
            os.path.join(
                glob.escape(temp_directory_path),
                f"*.{roop.globals.CFG.output_image_format}",
            )
        )
    )


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def get_destfilename_from_path(
    srcfilepath: str, destfilepath: str, extension: str
) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if "." in extension:
        return os.path.join(destfilepath, f"{fn}{extension}")
    return os.path.join(destfilepath, f"{fn}{extension}{ext}")


def replace_template(file_path: str, index: int = 0) -> str:
    fn, ext = os.path.splitext(os.path.basename(file_path))

    # Remove the "__temp" placeholder that was used as a temporary filename
    fn = fn.replace("__temp", "")

    template = roop.globals.CFG.output_template
    replaced_filename = template_parser.parse(
        template, {"index": str(index), "file": fn}
    )

    return os.path.join(roop.globals.output_path, f"{replaced_filename}{ext}")


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def delete_temp_frames(filename: str) -> None:
    dir = os.path.dirname(os.path.dirname(filename))
    shutil.rmtree(dir)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg", "webp"))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        if image_path.endswith(".webp"):
            return True
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(
    download_directory_path: str, urls: List[str | List[str]]
) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        if isinstance(url, str):
            download_file_path = os.path.join(
                download_directory_path, os.path.basename(url)
            )
            if os.path.exists(download_file_path):
                continue
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc=f"Downloading {os.path.basename(urlparse(url).path)}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(
                    url,
                    download_file_path,
                    reporthook=lambda count, block_size, total_size: progress.update(
                        block_size
                    ),
                )  # type: ignore[attr-defined]
        elif isinstance(url, list):
            for _url in url:
                download_file_path = os.path.join(
                    download_directory_path, os.path.basename(_url)
                )
                if os.path.exists(download_file_path):
                    break
                try:
                    request = urllib.request.urlopen(_url)  # type: ignore[attr-defined]
                except urllib.error.HTTPError:
                    continue
                if not request.status == 200:
                    continue
                total = int(request.headers.get("Content-Length", 0))
                with tqdm(
                    total=total,
                    desc=f"Downloading {os.path.basename(urlparse(_url).path)}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    urllib.request.urlretrieve(
                        _url,
                        download_file_path,
                        reporthook=lambda count,
                        block_size,
                        total_size: progress.update(block_size),
                    )  # type: ignore[attr-defined]
                break


def get_local_files_from_folder(folder: str) -> List[str]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        # Hidden files (.DS_Store and friends) are never media, and Gradio
        # refuses to serve a dotfile under the working directory, which fails
        # the whole batch with InvalidPathError.
        if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))
    ]
    return files


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    """Return the active compute device name.

    On Apple Silicon (where MLX is supported) we always return ``"mlx"`` so
    callers can branch off it. PyTorch consumers should treat this as the
    MPS device.
    """
    return "mlx"


def str_to_class(module_name, class_name) -> Any:
    from importlib import import_module

    class_ = None
    try:
        module_ = import_module(module_name)
        try:
            class_ = getattr(module_, class_name)()
        except AttributeError:
            print(f"Class {class_name} does not exist")
    except ImportError:
        print(f"Module {module_name} does not exist")
    return class_


def is_installed(name: str) -> bool:
    return shutil.which(name)


# Taken from https://stackoverflow.com/a/68842705
def get_platform() -> str:
    if sys.platform == "linux":
        try:
            proc_version = open("/proc/version").read()
            if "Microsoft" in proc_version:
                return "wsl"
        except:
            pass
    return sys.platform


def open_with_default_app(filename: str):
    if filename == None:
        return
    platform = get_platform()
    if platform == "darwin":
        subprocess.call(("open", filename))
    elif platform in ["win64", "win32"]:
        os.startfile(filename.replace("/", "\\"))
    elif platform == "wsl":
        subprocess.call("cmd.exe /C start".split() + [filename])
    else:  # linux variants
        subprocess.call(["xdg-open", filename])


def prepare_for_batch(target_files) -> str:
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def zip(files, zipname):
    with zipfile.ZipFile(zipname, "w") as zip_file:
        for f in files:
            zip_file.write(f, os.path.basename(f))


def unzip(zipfilename: str, target_path: str):
    with zipfile.ZipFile(zipfilename, "r") as zip_file:
        zip_file.extractall(target_path)


def mkdir_with_umask(directory):
    oldmask = os.umask(0)
    # mode needs octal
    os.makedirs(directory, mode=0o775, exist_ok=True)
    os.umask(oldmask)


def open_folder(path: str):
    platform = get_platform()
    try:
        if platform == "darwin":
            subprocess.call(("open", path))
        elif platform in ["win64", "win32"]:
            open_with_default_app(path)
        elif platform == "wsl":
            subprocess.call("cmd.exe /C start".split() + [path])
        else:  # linux variants
            subprocess.Popen(["xdg-open", path])
    except Exception:
        traceback.print_exc()
        # import webbrowser
        # webbrowser.open(url)


def create_version_html() -> str:
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    try:
        import mlx.core as mx
        mlx_version = mx.__version__
    except Exception:
        mlx_version = "not installed"
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
mlx: {mlx_version}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2) -> float:
    return distance.cosine(emb1, emb2)


def clean_dir(path: str):
    contents = os.listdir(path)
    for item in contents:
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)


def shuffle_array(arr):
    """
    Shuffles the given array in place using the Fisher-Yates shuffle algorithm.

    Args:
      arr: The array to be shuffled.

    Returns:
      None. The array is shuffled in place.
    """
    for i in range(len(arr) - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
