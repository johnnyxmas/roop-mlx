#!/usr/bin/env python3

import os
import sys
import shutil

import warnings
import signal
import pathlib
import argparse

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from roop.ProcessEntry import ProcessEntry
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame_total


call_display_ui = None

process_mgr = None


warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")


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
    roop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.globals.frame_processors = ["face_swapper", "face_enhancer"]


def release_resources() -> None:
    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(
            "Python version is not supported - please upgrade to 3.9 or higher."
        )
        return False

    from roop.model_manager import download_all_models, verify_all_models

    update_status("Checking / downloading models...")
    download_all_models()

    update_status("Verifying model integrity...")
    verify_all_models()

    # Compiled CoreML models are regenerable, and stale ones are unreachable
    # dead weight (they had reached 7 GB before the per-model cache layout).
    # Swept at launch and again in destroy().
    from roop.inference import prune_compile_caches

    prune_compile_caches()

    if not shutil.which("ffmpeg"):
        update_status("ffmpeg is not installed.")
    return True


def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    print(message)
    if call_display_ui is not None:
        call_display_ui(message)


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
    global process_mgr

    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    layers = (imagemask or {}).get("layers") or []
    mask = layers[0] if layers else None
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
    global process_mgr

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
    roop.globals.processing = True

    if not roop.globals.execution_threads or roop.globals.execution_threads < 1:
        roop.globals.execution_threads = roop.globals.CFG.max_threads

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

    import gc
    gc.collect()


def cleanup_temp_folders_only():
    """Clean up only temporary folders without releasing face analyzer (for stopped jobs)"""
    import tempfile
    import shutil

    try:
        if roop.globals.target_path:
            util.clean_temp(roop.globals.target_path)

        tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
        if os.path.exists(tempfolder):
            try:
                shutil.rmtree(tempfolder)
                update_status("Cleaned batch processing temp folder")
            except Exception as e:
                print(f"Warning: Could not clean batch processing temp folder: {e}")

        if not roop.globals.CFG.use_os_temp_folder:
            app_temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp"))
            if os.path.exists(app_temp_dir):
                cleaned_count = 0
                total_count = 0

                for item in os.listdir(app_temp_dir):
                    item_path = os.path.join(app_temp_dir, item)
                    total_count += 1

                    if os.path.isdir(item_path):
                        try:
                            dir_contents = os.listdir(item_path)
                            if any(fname in ['background.png', 'composite.png', 'layers.json'] for fname in dir_contents):
                                continue
                        except (OSError, PermissionError):
                            continue

                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        cleaned_count += 1
                    except PermissionError:
                        pass
                    except Exception:
                        pass

                if cleaned_count > 0:
                    update_status(f"Cleaned {cleaned_count}/{total_count} temp files (preserved UI files)")

    except Exception as e:
        print(f"Warning: Error during temp cleanup: {e}")


def cleanup_cache_folders():
    """Clean up temporary and cache folders after job completion"""
    import tempfile
    import shutil

    try:
        if roop.globals.target_path:
            util.clean_temp(roop.globals.target_path)

        tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
        if os.path.exists(tempfolder):
            shutil.rmtree(tempfolder)
            update_status("Cleaned batch processing temp folder")

        if not roop.globals.CFG.use_os_temp_folder:
            app_temp_dir = os.path.abspath(os.path.join(os.getcwd(), "temp"))
            if os.path.exists(app_temp_dir):
                cleaned_count = 0
                total_count = 0

                for item in os.listdir(app_temp_dir):
                    item_path = os.path.join(app_temp_dir, item)
                    total_count += 1

                    if os.path.isdir(item_path):
                        try:
                            dir_contents = os.listdir(item_path)
                            if any(fname in ['background.png', 'composite.png', 'layers.json'] for fname in dir_contents):
                                continue
                        except (OSError, PermissionError):
                            continue

                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                        cleaned_count += 1
                    except PermissionError:
                        pass
                    except Exception:
                        pass

                if cleaned_count > 0:
                    update_status(f"Cleaned {cleaned_count}/{total_count} temp files (preserved UI files)")

        clip_cache_dir = os.path.expanduser("~/.cache/clip")
        if os.path.exists(clip_cache_dir):
            try:
                cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(clip_cache_dir)
                               for filename in filenames) / (1024**3)
                if cache_size > 1.0:
                    shutil.rmtree(clip_cache_dir)
                    update_status(f"Cleaned CLIP cache ({cache_size:.1f}GB)")
            except Exception as e:
                print(f"Warning: Could not clean CLIP cache: {e}")

        update_status("Cleaned temporary files (keeping models loaded for next swap)")

    except Exception as e:
        print(f"Warning: Error during cache cleanup: {e}")


def destroy() -> None:
    """Called on Ctrl+C or script termination - do full cleanup"""
    update_status("Shutting down - cleaning up temp files...")

    cleanup_cache_folders()
    release_resources()

    # Sweep on the way out too, but only on a quiet exit: destroy() is also
    # the SIGINT handler (see the signal.signal call above), and a Ctrl+C
    # mid-job leaves worker threads holding sessions whose cache entries this
    # would delete.  The launch sweep in pre_check() covers that case safely —
    # nothing can create orphans while the app is down.
    if not roop.globals.processing:
        from roop.inference import prune_compile_caches

        prune_compile_caches()
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings("config.yaml")
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = (
        roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    )
    if roop.globals.startup_args.server_share:
        roop.globals.CFG.server_share = True

    from roop.inference import prewarm_sessions_async
    prewarm_sessions_async()

    main.run()
