#!/usr/bin/env python3

import os
import sys
import shutil

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith("--execution-provider") for arg in sys.argv):
    os.environ["OMP_NUM_THREADS"] = "1"

import warnings
from typing import List
import platform
import signal
import torch
import onnxruntime
import pathlib
import argparse

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.face_util import extract_face_images
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total, release_video


clip_text = None

call_display_ui = None

process_mgr = None


if "ROCMExecutionProvider" in roop.globals.execution_providers:
    del torch

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.globals.headless = False

    program = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100)
    )
    program.add_argument(
        "--server_share",
        help="Public server",
        dest="server_share",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--cuda_device_id",
        help="Index of the cuda gpu to use",
        dest="cuda_device_id",
        type=int,
        default=0,
    )
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ["face_swapper", "face_enhancer"]


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    list_providers = [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]

    try:
        for i in range(len(list_providers)):
            if list_providers[i] == "CUDAExecutionProvider":
                list_providers[i] = (
                    "CUDAExecutionProvider",
                    {"device_id": roop.globals.cuda_device_id},
                )
                torch.cuda.set_device(roop.globals.cuda_device_id)
                break
    except:
        pass

    return list_providers


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if "DmlExecutionProvider" in roop.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024**3
        if platform.system().lower() == "darwin":
            memory = roop.globals.max_memory * 1024**6
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(memory), ctypes.c_size_t(memory)
            )
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    import gc

    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    # Force garbage collection
    gc.collect()
    
    # Clear GPU memory if available
    if (
        "CUDAExecutionProvider" in roop.globals.execution_providers
        and torch.cuda.is_available()
    ):
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Additional GPU memory cleanup
            torch.cuda.synchronize()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(
            "Python version is not supported - please upgrade to 3.9 or higher."
        )
        return False

    download_directory_path = util.resolve_relative_path("../models")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/InSwapper/inswapper_128.onnx",
            ],
            [
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_128.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/ReSwapper/reswapper_128.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_256.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/ReSwapper/reswapper_256.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GFPGAN/GFPGANv1.4.onnx",
            ],
            [
                "https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DMDNet/DMDNet.pth",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GPEN/GPEN-BFR-512.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-1024.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GPEN/GPEN-BFR-1024.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-2048.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GPEN/GPEN-BFR-2048.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/RestoreFormer/restoreformer_plus_plus.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/xseg.onnx",
            ],
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/CLIP")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/rd64-uni-refined.pth",
            ]
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/buffalo_l")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/1k3d68.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/1k3d68.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/2d106det.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/det_10g.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/det_10g.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/genderage.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/w600k_r50.onnx",
            ],
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/CodeFormer")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/CodeFormer/CodeFormerv0.1.onnx",
            ]
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/Frame")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DeOldify/deoldify_artistic.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DeOldify/deoldify_stable.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/isnet-general-use.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/real_esrgan_x4.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/real_esrgan_x2.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/lsdir_x4.onnx",
            ],
        ],
    )

    if not shutil.which("ffmpeg"):
        update_status("ffmpeg is not installed.")
    return True


def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)


def start() -> None:
    if roop.globals.headless:
        print("Headless mode currently unsupported - starting UI!")
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'

    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine, swap_model=None):
    from roop.ProcessOptions import FF_SWAP_MODELS
    if swap_model is not None and swap_model in FF_SWAP_MODELS:
        processors = {"faceswap_ff": {}}
    else:
        processors = {"faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})

    if roop.globals.selected_enhancer == "GFPGAN":
        processors.update({"gfpgan": {}})
    elif roop.globals.selected_enhancer == "Codeformer":
        processors.update({"codeformer": {}})
    elif roop.globals.selected_enhancer == "DMDNet":
        processors.update({"dmdnet": {}})
    elif roop.globals.selected_enhancer in ("GPEN", "GPEN 512", "GPEN 1024", "GPEN 2048"):
        parts = roop.globals.selected_enhancer.split()
        res = int(parts[-1]) if len(parts) > 1 else 512
        processors.update({"gpen": {"resolution": res}})
    elif roop.globals.selected_enhancer == "Restoreformer++":
        processors.update({"restoreformer++": {}})
    return processors


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)

    #    if len(roop.globals.INPUT_FACESETS) <= selected_index:
    #        selected_index = 0
    process_mgr.initialize(
        roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options
    )
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def batch_process_regular(
    swap_model,
    output_method,
    files: list[ProcessEntry],
    masking_engine: str,
    new_clip_text: str,
    use_new_method,
    imagemask,
    restore_original_mouth,
    num_swap_steps,
    progress,
    selected_index=0,
) -> None:
    global clip_text, process_mgr

    # Only limit resources, don't release them before processing
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    mask = imagemask["layers"][0] if imagemask is not None else None
    if len(roop.globals.INPUT_FACESETS) <= selected_index:
        selected_index = 0
    options = ProcessOptions(
        swap_model,
        get_processing_plugins(masking_engine, swap_model),
        roop.globals.distance_threshold,
        roop.globals.blend_ratio,
        roop.globals.face_swap_mode,
        selected_index,
        new_clip_text,
        mask,
        num_swap_steps,
        roop.globals.subsample_size,
        False,
        restore_original_mouth,
    )
    process_mgr.initialize(
        roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options
    )
    batch_process(output_method, files, use_new_method)
    return


def batch_process_with_options(files: list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    # Only limit resources, don't release them before processing
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(
        roop.globals.INPUT_FACESETS, roop.globals.TARGET_FACES, options
    )
    roop.globals.keep_frames = False
    roop.globals.wait_after_extraction = False
    roop.globals.skip_audio = False
    batch_process("Files", files, True)


def batch_process(output_method, files: list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    roop.globals.processing = True

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        roop.globals.execution_threads = 1

    imagefiles: list[ProcessEntry] = []
    videofiles: list[ProcessEntry] = []

    update_status("Sorting videos/images")

    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(
                fullname,
                roop.globals.output_path,
                f".{roop.globals.CFG.output_image_format}",
            )
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(
                parents=True, exist_ok=True
            )
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ["gif"]):
            destination = util.get_destfilename_from_path(
                fullname,
                roop.globals.output_path,
                f"__temp.{roop.globals.CFG.output_video_format}",
            )
            f.finalname = destination
            videofiles.append(f)

    if len(imagefiles) > 0:
        update_status("Processing image(s)")
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, roop.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if len(videofiles) > 0:
        for index, v in enumerate(videofiles):
            if not roop.globals.processing:
                end_processing("Processing stopped!")
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(
                    f"Creating {os.path.basename(v.finalname)} with {fps} FPS..."
                )

            start_processing = time()
            if (
                is_streaming_only == False
                and roop.globals.keep_frames
                or not use_new_method
            ):
                util.create_temp(v.filename)
                update_status("Extracting frames...")
                ffmpeg.extract_frames(v.filename, v.startframe, v.endframe, fps)
                if not roop.globals.processing:
                    end_processing("Processing stopped!")
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(
                    temp_frame_paths, temp_frame_paths, roop.globals.execution_threads
                )
                if not roop.globals.processing:
                    end_processing("Processing stopped!")
                    return
                if roop.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)

                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not roop.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ["gif"]):
                    skip_audio = True
                else:
                    skip_audio = roop.globals.skip_audio
                process_mgr.run_batch_inmem(
                    output_method,
                    v.filename,
                    v.finalname,
                    v.startframe,
                    v.endframe,
                    fps,
                    roop.globals.execution_threads,
                )

            if not roop.globals.processing:
                end_processing("Processing stopped!")
                return

            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ""
                if util.has_extension(v.filename, ["gif"]):
                    gifname = util.get_destfilename_from_path(
                        v.filename, roop.globals.output_path, ".gif"
                    )
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(
                        parents=True, exist_ok=True
                    )

                    update_status("Creating final GIF")
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = roop.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(
                        parents=True, exist_ok=True
                    )

                    if not skip_audio:
                        ffmpeg.restore_audio(
                            video_file_name,
                            v.filename,
                            v.startframe,
                            v.endframe,
                            destination,
                        )
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)

            elif is_streaming_only == False:
                update_status(f"Failed processing {os.path.basename(v.finalname)}!")
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(
                f"\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s"
            )
    end_processing("Finished")


def end_processing(msg: str):
    update_status(msg)
    roop.globals.target_folder_path = None
    
    # No automatic temp folder cleanup - only do basic GPU cleanup
    # Temp folders will only be cleaned on explicit user action (Ctrl+C or cleanup button)
    basic_gpu_cleanup()


def basic_gpu_cleanup():
    """Basic GPU memory cleanup without releasing process manager"""
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU memory if available
    if (
        "CUDAExecutionProvider" in roop.globals.execution_providers
        and torch.cuda.is_available()
    ):
        try:
            with torch.cuda.device("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Additional GPU memory cleanup
                torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Warning: Could not perform GPU cleanup: {e}")


def cleanup_temp_folders_only():
    """Clean up only temporary folders without releasing face analyzer (for stopped jobs)"""
    import tempfile
    import shutil
    import time
    from roop import utilities as util
    
    try:
        # Clean up temp folder used for frame extraction
        if roop.globals.target_path:
            util.clean_temp(roop.globals.target_path)
        
        # Clean up batch processing temp folder
        tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
        if os.path.exists(tempfolder):
            try:
                shutil.rmtree(tempfolder)
                update_status("Cleaned batch processing temp folder")
            except Exception as e:
                print(f"Warning: Could not clean batch processing temp folder: {e}")
        
        # Clean up application temp folder if not using OS temp - but avoid Gradio files
        if not roop.globals.CFG.use_os_temp_folder:
            app_temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp"))
            if os.path.exists(app_temp_dir):
                cleaned_count = 0
                total_count = 0
                
                # Only clean contents, not the folder itself, and avoid Gradio temp files
                for item in os.listdir(app_temp_dir):
                    item_path = os.path.join(app_temp_dir, item)
                    total_count += 1
                    
                    # Skip Gradio temp files (they have long hex names and contain background.png, etc.)
                    if os.path.isdir(item_path):
                        # Check if this looks like a Gradio temp directory
                        try:
                            dir_contents = os.listdir(item_path)
                            # Skip directories that contain Gradio UI files
                            if any(fname in ['background.png', 'composite.png', 'layers.json'] for fname in dir_contents):
                                continue
                        except (OSError, PermissionError):
                            # If we can't read the directory, skip it to be safe
                            continue
                    
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        cleaned_count += 1
                    except PermissionError:
                        # File is in use, skip it to avoid corrupting active UI files
                        pass
                    except Exception as e:
                        # Other errors, skip silently
                        pass
                        
                if cleaned_count > 0:
                    update_status(f"Cleaned {cleaned_count}/{total_count} temp files (preserved UI files)")
                
    except Exception as e:
        print(f"Warning: Error during temp cleanup: {e}")


def cleanup_cache_folders():
    """Clean up temporary and cache folders after job completion"""
    import tempfile
    import shutil
    from roop import utilities as util
    
    try:
        # Clean up temp folder used for frame extraction
        if roop.globals.target_path:
            util.clean_temp(roop.globals.target_path)
        
        # Clean up batch processing temp folder
        tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
        if os.path.exists(tempfolder):
            shutil.rmtree(tempfolder)
            update_status("Cleaned batch processing temp folder")
        
        # Clean up application temp folder if not using OS temp - but avoid Gradio files
        if not roop.globals.CFG.use_os_temp_folder:
            app_temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp"))
            if os.path.exists(app_temp_dir):
                import time
                cleaned_count = 0
                total_count = 0
                
                # Only clean contents, not the folder itself, and avoid Gradio temp files
                for item in os.listdir(app_temp_dir):
                    item_path = os.path.join(app_temp_dir, item)
                    total_count += 1
                    
                    # Skip Gradio temp files (they have long hex names and contain background.png, etc.)
                    if os.path.isdir(item_path):
                        # Check if this looks like a Gradio temp directory
                        try:
                            dir_contents = os.listdir(item_path)
                            # Skip directories that contain Gradio UI files
                            if any(fname in ['background.png', 'composite.png', 'layers.json'] for fname in dir_contents):
                                continue
                        except (OSError, PermissionError):
                            # If we can't read the directory, skip it to be safe
                            continue
                    
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        cleaned_count += 1
                    except PermissionError:
                        # File is in use, skip it to avoid corrupting active UI files
                        pass
                    except Exception as e:
                        # Other errors, skip silently
                        pass
                        
                if cleaned_count > 0:
                    update_status(f"Cleaned {cleaned_count}/{total_count} temp files (preserved UI files)")
        
        # Clean up CLIP cache if it exists
        clip_cache_dir = os.path.expanduser("~/.cache/clip")
        if os.path.exists(clip_cache_dir):
            try:
                # Only clean if cache is larger than 1GB
                cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(clip_cache_dir)
                               for filename in filenames) / (1024**3)
                if cache_size > 1.0:
                    shutil.rmtree(clip_cache_dir)
                    update_status(f"Cleaned CLIP cache ({cache_size:.1f}GB)")
            except Exception as e:
                print(f"Warning: Could not clean CLIP cache: {e}")
        
        # Don't release face analyzer after successful completion to allow subsequent swaps
        # Only release it when explicitly shutting down the application
        update_status("Cleaned temporary files (keeping models loaded for next swap)")
                
    except Exception as e:
        print(f"Warning: Error during cache cleanup: {e}")


def destroy() -> None:
    """Called on Ctrl+C or script termination - do full cleanup"""
    update_status("Shutting down - cleaning up temp files...")
    
    # Do full temp folder cleanup on shutdown
    cleanup_cache_folders()
    
    # Release all resources
    release_resources()
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings("config.yaml")
    if roop.globals.startup_args.cuda_device_id != 0:
        roop.globals.CFG.cuda_device_id = roop.globals.startup_args.cuda_device_id
    roop.globals.cuda_device_id = roop.globals.CFG.cuda_device_id
    roop.globals.multi_gpu = roop.globals.CFG.multi_gpu
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = (
        roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    )
    if roop.globals.startup_args.server_share:
        roop.globals.CFG.server_share = True
    main.run()
