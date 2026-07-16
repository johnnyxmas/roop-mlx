import os
import time
import cv2
import numpy as np
import psutil

from roop.ProcessOptions import ProcessOptions

from roop.face_util import (
    get_first_face,
    get_all_faces_fast,
    rotate_anticlockwise,
    rotate_clockwise,
    clamp_cut_values,
)
from roop.face_tracker import FaceTracker
from roop.perf_profile import (
    detector_stride as _perf_detector_stride,
    scene_cut_threshold as _perf_scene_cut_threshold,
)
from roop.utilities import (
    compute_cosine_distance,
    str_to_class,
    shuffle_array,
)

from typing import Any, List, Callable
from roop.typing import Frame, Face
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock, local as thread_local
from queue import Queue
from tqdm import tqdm
from roop.ffmpeg_writer import FFMPEG_VideoWriter
from roop.StreamWriter import StreamWriter
import roop.globals


# Poor man's enum to be able to compare to int
class eNoFaceAction:
    USE_ORIGINAL_FRAME = 0
    RETRY_ROTATED = 1
    SKIP_FRAME = 2
    SKIP_FRAME_IF_DISSIMILAR = (3,)
    USE_LAST_SWAPPED = 4


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


class ProcessMgr:
    imagemask = None

    options: ProcessOptions = None

    num_threads = 1
    current_index = 0
    processing_threads = 1
    buffer_wait_time = 0.1

    lock = Lock()

    frames_queue = None
    processed_queue = None

    videowriter = None
    streamwriter = None

    progress_gradio = None
    total_frames = 0

    num_frames_no_face = 0
    last_swapped_frame = None

    output_to_file = None
    output_to_cam = None

    plugins = {
        "faceswap": "FaceSwapInsightFace",
        "faceswap_ff": "FaceSwapFaceFusion",
        "mask_clip2seg": "Mask_Clip2Seg",
        "mask_xseg": "Mask_XSeg",
        "codeformer": "Enhance_CodeFormer",
        "gfpgan": "Enhance_GFPGAN",

        "gpen": "Enhance_GPEN",
        "restoreformer++": "Enhance_RestoreFormerPPlus",
        "colorizer": "Frame_Colorizer",
        "filter_generic": "Frame_Filter",
        "removebg": "Frame_Masking",
        "upscale": "Frame_Upscale",
    }

    def __init__(self, progress):
        if progress is not None:
            self.progress_gradio = progress
        self._thread_local = thread_local()
        self.processors = []
        self.input_face_datas = []
        self.target_face_datas = []

    def reuseOldProcessor(self, name: str):
        for p in self.processors:
            if p.processorname == name:
                return p

        return None

    def initialize(self, input_faces, target_faces, options):
        self.input_face_datas = input_faces
        self.target_face_datas = target_faces
        self.num_frames_no_face = 0
        self.last_swapped_frame = None
        self.options = options
        self._thread_local = thread_local()

        if options.swap_mode == "all_random":
            # don't modify original list
            self.input_face_datas = input_faces.copy()
            shuffle_array(self.input_face_datas)

        processors_to_remove = []
        for p in self.processors:
            newp = next(
                (x for x in options.processors.keys() if x == p.processorname), None
            )
            if newp is None:
                p.Release()
                processors_to_remove.append(p)
        for p in processors_to_remove:
            self.processors.remove(p)
            del p

        newprocessors = []
        for key, extoption in options.processors.items():
            p = self.reuseOldProcessor(key)
            if p is None:
                classname = self.plugins[key]
                module = "roop.processors." + classname
                p = str_to_class(module, classname)
            if p is not None:
                if p.type == "swap":
                    extoption.update({"modelname": self._resolve_modelname(self.options.swap_modelname)})
                p.Initialize(extoption)
                newprocessors.append(p)
            else:
                print(f"Not using {module}")
        self.processors = newprocessors

        if (
            isinstance(self.options.imagemask, dict)
            and self.options.imagemask.get("layers")
            and len(self.options.imagemask["layers"]) > 0
        ):
            self.options.imagemask = self.options.imagemask.get("layers")[0]
            self.options.imagemask = cv2.cvtColor(
                self.options.imagemask, cv2.COLOR_RGBA2GRAY
            )
            if np.any(self.options.imagemask):
                mo = self.input_face_datas[0].faces[0].mask_offsets
                self.options.imagemask = self.blur_area(
                    self.options.imagemask, mo[4], mo[5]
                )
                self.options.imagemask = self.options.imagemask.astype(np.float32) / 255
                self.options.imagemask = cv2.cvtColor(
                    self.options.imagemask, cv2.COLOR_GRAY2RGB
                )
            else:
                self.options.imagemask = None

        self.options.frame_processing = False
        for p in self.processors:
            if p.type.startswith("frame_"):
                self.options.frame_processing = True

    INSIGHTFACE_MODEL_FILES = {
        "InSwapper 128": "inswapper_128.onnx",
        "ReSwapper 128": "reswapper_128.onnx",
        "ReSwapper 256": "reswapper_256.onnx",
    }

    def _resolve_modelname(self, swap_modelname: str) -> str:
        return self.INSIGHTFACE_MODEL_FILES.get(swap_modelname, swap_modelname)

    def run_batch(self, source_files, target_files, threads: int = 1):
        progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        self.total_frames = len(source_files)
        self.num_threads = threads

        with tqdm(
            total=self.total_frames,
            desc="Processing",
            unit="frame",
            dynamic_ncols=True,
            bar_format=progress_bar_format,
        ) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                completed_count = 0
                while not queue.empty():
                    future = executor.submit(
                        self.process_frames,
                        source_files,
                        target_files,
                        pick_queue(queue, queue_per_future),
                        lambda: self.update_progress(progress),
                    )
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()
                    completed_count += 1

    def process_frames(
        self,
        source_files: List[str],
        target_files: List[str],
        current_files,
        update: Callable[[], None],
    ) -> None:
        _processors = self.processors
        file_to_idx = {f: i for i, f in enumerate(source_files)}
        for f in current_files:
            if not roop.globals.processing:
                return

            temp_frame = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if temp_frame is not None:
                try:
                    if self.options.frame_processing:
                        frame = temp_frame
                        for p in _processors:
                            frame = p.Run(frame)
                        resimg = frame
                    else:
                        resimg = self.process_frame(temp_frame)

                    if resimg is not None:
                        success, encoded = cv2.imencode(f".{roop.globals.CFG.output_image_format}", resimg)
                        if success:
                            encoded.tofile(target_files[file_to_idx[f]])
                        del resimg, encoded
                finally:
                    del temp_frame
            if update:
                update()

    def read_frames_thread(self, cap, frame_start, frame_end, num_threads):
        num_frame = 0
        total_num = frame_end - frame_start
        if frame_start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        while True and roop.globals.processing:
            ret, frame = cap.read()
            if not ret:
                break

            self.frames_queue[num_frame % num_threads].put(frame, block=True)
            num_frame += 1
            if num_frame == total_num:
                break

        for i in range(num_threads):
            self.frames_queue[i].put(None)

    def process_videoframes(self, threadindex, progress) -> None:
        _processors = self.processors
        while True:
            frame = self.frames_queue[threadindex].get()
            if frame is None:
                self.processing_threads -= 1
                self.processed_queue[threadindex].put((False, None))
                return
            else:
                resimg = None
                try:
                    if self.options.frame_processing:
                        processed_frame = frame
                        for p in _processors:
                            processed_frame = p.Run(processed_frame)
                        resimg = processed_frame
                    else:
                        resimg = self.process_frame(frame)
                except Exception as e:
                    print(f"[ProcessMgr] Exception in thread {threadindex}: {e}")
                    import traceback
                    traceback.print_exc()
                    resimg = frame  # fall back to unprocessed frame
                finally:
                    del frame
                self.processed_queue[threadindex].put((True, resimg))
                progress()

    def write_frames_thread(self):
        nextindex = 0
        num_producers = self.num_threads

        while True:
            process, frame = self.processed_queue[nextindex % self.num_threads].get()
            nextindex += 1
            if frame is not None:
                try:
                    if self.output_to_file:
                        self.videowriter.write_frame(frame)
                    if self.output_to_cam:
                        self.streamwriter.WriteToStream(frame)
                except (IOError, OSError, BrokenPipeError) as e:
                    print(f"Error writing frame: {e}")
                    import roop.globals
                    roop.globals.processing = False
                    return
                finally:
                    del frame
            elif process == False:
                num_producers -= 1
                if num_producers < 1:
                    return

    def run_batch_inmem(
        self,
        output_method,
        source_video,
        target_video,
        frame_start,
        frame_end,
        fps,
        threads: int = 1,
    ):
        if len(self.processors) < 1:
            print("No processor defined!")
            return

        from roop.capturer import open_video_capture
        cap = open_video_capture(source_video)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = (frame_end - frame_start) + 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, "getProcessedResolution"):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]

        self.total_frames = frame_count
        self.num_threads = threads

        self.processing_threads = self.num_threads
        self.frames_queue = []
        self.processed_queue = []
        for _ in range(threads):
            self.frames_queue.append(Queue(4))
            self.processed_queue.append(Queue(4))

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = (
            output_method == "Virtual Camera" or output_method == "Both"
        )

        if self.output_to_file:
            self.videowriter = FFMPEG_VideoWriter(
                target_video,
                (width, height),
                fps,
                codec=roop.globals.video_encoder,
                crf=roop.globals.video_quality,
                audiofile=None,
            )
        if self.output_to_cam:
            self.streamwriter = StreamWriter((width, height), int(fps))

        readthread = Thread(
            target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads)
        )
        readthread.start()

        writethread = Thread(target=self.write_frames_thread)
        writethread.start()

        progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        with tqdm(
            total=self.total_frames,
            desc="Processing",
            unit="frames",
            dynamic_ncols=True,
            bar_format=progress_bar_format,
        ) as progress:
            with ThreadPoolExecutor(
                thread_name_prefix="swap_proc", max_workers=self.num_threads
            ) as executor:
                futures = []

                for threadindex in range(threads):
                    future = executor.submit(
                        self.process_videoframes,
                        threadindex,
                        lambda: self.update_progress(progress),
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()
        readthread.join()
        writethread.join()
        cap.release()
        if self.output_to_file:
            try:
                self.videowriter.close()
            except Exception as e:
                print(f"Warning: Error closing video writer (video may still be valid): {e}")
        if self.output_to_cam:
            try:
                self.streamwriter.Close()
            except Exception as e:
                print(f"Warning: Error closing stream writer: {e}")

        self.frames_queue.clear()
        self.processed_queue.clear()

    _last_progress_time: float = 0.0
    _pending_frames: int = 0
    _psutil_process = None
    _PROGRESS_INTERVAL: float = 5.0   # seconds between progress updates

    def update_progress(self, progress: Any = None) -> None:
        # Called from every worker thread; guard the shared counters.
        with self.lock:
            self._pending_frames += 1

            now = time.time()
            if (now - self._last_progress_time) < self._PROGRESS_INTERVAL:
                return

            batch = self._pending_frames
            self._pending_frames = 0
            self._last_progress_time = now

        import roop.globals

        if self._psutil_process is None:
            self._psutil_process = psutil.Process(os.getpid())
        memory_usage = self._psutil_process.memory_info().rss / 1024 / 1024 / 1024

        # Read the live config value (updates when the user moves the slider)
        # rather than the startup snapshot in roop.globals.max_memory.
        cfg_limit = getattr(roop.globals.CFG, "memory_limit", 0) if roop.globals.CFG else 0
        if cfg_limit and cfg_limit > 0:
            mem_label = f"{memory_usage:.1f}/{cfg_limit:.0f}GB"
        else:
            total_gb = psutil.virtual_memory().total / 1024 ** 3
            mem_label = f"{memory_usage:.1f}/{total_gb:.0f}GB"

        progress.set_postfix(
            {
                "mem": f"{memory_usage:.1f}/{mem_label}",
                "threads": self.num_threads,
            }
        )
        progress.update(batch)

        current_frame = progress.n
        elapsed = now - roop.globals.progress_status["start_time"]
        fps = current_frame / elapsed if elapsed > 0 else 0
        remaining_frames = self.total_frames - current_frame
        eta_seconds = remaining_frames / fps if fps > 0 else 0
        percent = (current_frame / self.total_frames * 100) if self.total_frames > 0 else 0

        roop.globals.progress_status.update({
            "percent": percent,
            "fps": fps,
            "eta_seconds": eta_seconds,
            "current_frame": current_frame,
            "total_frames": self.total_frames,
            "active": True,
        })

        if self.progress_gradio is not None:
            self.progress_gradio(
                (current_frame, self.total_frames),
                desc="Processing",
                total=self.total_frames,
                unit="frames",
            )

    def process_frame(self, frame: Frame):
        if len(self.input_face_datas) < 1 and not self.options.show_face_masking:
            return frame
        temp_frame = frame.copy()
        num_swapped, temp_frame = self.swap_faces(frame, temp_frame)
        if num_swapped > 0:
            if roop.globals.no_face_action == eNoFaceAction.SKIP_FRAME_IF_DISSIMILAR:
                if len(self.input_face_datas) > num_swapped:
                    return None
            self.num_frames_no_face = 0
            self.last_swapped_frame = temp_frame.copy()
            return temp_frame
        if roop.globals.no_face_action == eNoFaceAction.USE_LAST_SWAPPED:
            if (
                self.last_swapped_frame is not None
                and self.num_frames_no_face < self.options.max_num_reuse_frame
            ):
                self.num_frames_no_face += 1
                return self.last_swapped_frame.copy()
            return frame

        elif roop.globals.no_face_action == eNoFaceAction.USE_ORIGINAL_FRAME:
            return frame
        if roop.globals.no_face_action == eNoFaceAction.SKIP_FRAME:
            # This only works with in-mem processing, as it simply skips the frame.
            # For 'extract frames' it simply leaves the unprocessed frame unprocessed and it gets used in the final output by ffmpeg.
            # If we could delete that frame here, that'd work but that might cause ffmpeg to fail unless the frames are renamed, and I don't think we have the info on what frame it actually is?????
            # alternatively, it could mark all the necessary frames for deletion, delete them at the end, then rename the remaining frames that might work?
            return None
        else:
            return self.retry_rotated(frame)

    def retry_rotated(self, frame):
        copyframe = frame.copy()
        copyframe = rotate_clockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_anticlockwise(temp_frame)

        copyframe = frame.copy()
        copyframe = rotate_anticlockwise(copyframe)
        temp_frame = copyframe.copy()
        num_swapped, temp_frame = self.swap_faces(copyframe, temp_frame)
        if num_swapped > 0:
            return rotate_clockwise(temp_frame)
        del copyframe
        return frame

    def _get_tracker(self):
        """Return the FaceTracker that belongs to the calling thread."""
        tl = self._thread_local
        if not hasattr(tl, "face_tracker"):
            tl.face_tracker = FaceTracker()
            tl.tracker_last_faces = None
            tl.tracker_frames_since_detect = 0
            tl.tracker_prev_small = None
        return tl

    def _needs_recognition(self):
        """Return True if the current swap mode requires face matching."""
        return self.options.swap_mode in ("selected",)

    def _needs_genderage(self):
        """Return True if the current swap mode filters faces by gender."""
        return self.options.swap_mode in ("all_female", "all_male")

    def _needs_landmarks_106(self):
        """Return True if any feature requires 106-point landmarks."""
        if self.options.restore_original_mouth:
            return True
        if roop.globals.autorotate_faces:
            return True

        return False

    def _detect_faces_with_tracker(self, frame):
        """Tracker-assisted face detection with optical flow.

        On stride=1 (memory_saver) this always runs full detection.  On
        stride>1 (balanced/max_speed) intermediate frames use Lucas-Kanade
        optical flow to warp cached bounding boxes and keypoints, giving
        smooth tracking even at high strides (10-15+).  A scene-cut check
        forces redetection when the frame changes drastically.
        """
        tl = self._get_tracker()
        stride = _perf_detector_stride()
        if stride <= 1:
            tl.tracker_last_faces = None
            tl.tracker_prev_small = None
            tl.face_tracker.reset()
            return get_all_faces_fast(
                frame,
                need_recognition=self._needs_recognition(),
                need_landmarks_106=self._needs_landmarks_106(),
                need_genderage=self._needs_genderage(),
            )

        force_redetect = False
        try:
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            if tl.tracker_prev_small is not None:
                diff = cv2.absdiff(small, tl.tracker_prev_small)
                if float(diff.mean()) > _perf_scene_cut_threshold():
                    force_redetect = True
            tl.tracker_prev_small = small
        except Exception:
            force_redetect = True

        need_detect = (
            force_redetect
            or tl.tracker_last_faces is None
            or not tl.tracker_last_faces
            or tl.tracker_frames_since_detect >= stride
        )
        if need_detect:
            faces = get_all_faces_fast(
                frame,
                need_recognition=self._needs_recognition(),
                need_landmarks_106=self._needs_landmarks_106(),
                need_genderage=self._needs_genderage(),
            )
            tl.tracker_last_faces = faces
            tl.tracker_frames_since_detect = 0
            tl.face_tracker.on_detection(frame, faces)
            return faces

        tl.tracker_frames_since_detect += 1

        tracked = tl.face_tracker.track(frame)
        if tracked is not None:
            return tracked

        # Optical flow lost all faces — don't return stale keypoints from a
        # prior scene; that produces bad alignment crops.  Fall back to a fresh
        # detection so scene-cut frames are handled correctly.
        faces = get_all_faces_fast(
            frame,
            need_recognition=self._needs_recognition(),
            need_landmarks_106=self._needs_landmarks_106(),
            need_genderage=self._needs_genderage(),
        )
        tl.tracker_last_faces = faces
        tl.tracker_frames_since_detect = 0
        tl.face_tracker.on_detection(frame, faces)
        return faces

    def swap_faces(self, frame, temp_frame):
        num_faces_found = 0

        if self.options.swap_mode == "first":
            faces = self._detect_faces_with_tracker(frame)
            if not faces:
                return num_faces_found, frame
            num_faces_found += 1
            temp_frame = self.process_face(
                self.options.selected_index, faces[0], temp_frame
            )

        else:
            faces = self._detect_faces_with_tracker(frame)
            if faces is None:
                return num_faces_found, frame

            try:
                if self.options.swap_mode == "all":
                    for face in faces:
                        num_faces_found += 1
                        temp_frame = self.process_face(
                            self.options.selected_index, face, temp_frame
                        )

                elif (
                    self.options.swap_mode == "all_input"
                    or self.options.swap_mode == "all_random"
                ):
                    for i, face in enumerate(faces):
                        if i >= len(self.input_face_datas):
                            break
                        num_faces_found += 1
                        temp_frame = self.process_face(i, face, temp_frame)

                elif self.options.swap_mode == "selected":
                    num_targetfaces = len(self.target_face_datas)
                    use_index = num_targetfaces == 1
                    
                    target_embeddings = [tf.embedding for tf in self.target_face_datas]
                    
                    for i, target_embedding in enumerate(target_embeddings):
                        if i >= len(self.input_face_datas):
                            continue
                            
                        for face in faces:
                            if (
                                compute_cosine_distance(target_embedding, face.embedding)
                                <= self.options.face_distance_threshold
                            ):
                                if use_index:
                                    temp_frame = self.process_face(
                                        self.options.selected_index, face, temp_frame
                                    )
                                else:
                                    temp_frame = self.process_face(i, face, temp_frame)
                                num_faces_found += 1
                                if (
                                    not roop.globals.vr_mode
                                    and num_faces_found == num_targetfaces
                                ):
                                    break
                        if (
                            not roop.globals.vr_mode
                            and num_faces_found == num_targetfaces
                        ):
                            break
                            
                elif (
                    self.options.swap_mode == "all_female"
                    or self.options.swap_mode == "all_male"
                ):
                    gender = "F" if self.options.swap_mode == "all_female" else "M"
                    for face in faces:
                        if face.sex == gender:
                            num_faces_found += 1
                            temp_frame = self.process_face(
                                self.options.selected_index, face, temp_frame
                            )

            finally:
                for face in faces:
                    del face
                faces.clear()

        if roop.globals.vr_mode and num_faces_found % 2 > 0:
            # stereo image, there has to be an even number of faces
            num_faces_found = 0
            return num_faces_found, frame
        if num_faces_found == 0:
            return num_faces_found, frame

        if (
            self.options.imagemask is not None
            and self.options.imagemask.shape == frame.shape
        ):
            temp_frame = self.simple_blend_with_mask(
                temp_frame, frame, self.options.imagemask
            )
        return num_faces_found, temp_frame

    def rotation_action(self, original_face: Face, frame: Frame):
        (height, width) = frame.shape[:2]

        bounding_box_width = original_face.bbox[2] - original_face.bbox[0]
        bounding_box_height = original_face.bbox[3] - original_face.bbox[1]
        horizontal_face = bounding_box_width > bounding_box_height

        center_x = width // 2.0
        start_x = original_face.bbox[0]
        bbox_center_x = start_x + (bounding_box_width // 2.0)

        # need to leverage the array of landmarks as decribed here:
        # https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg
        # basically, we should be able to check for the relative position of eyes and nose
        # then use that to determine which way the face is actually facing when in a horizontal position
        # and use that to determine the correct rotation_action

        forehead_x = original_face.landmark_2d_106[72][0]
        chin_x = original_face.landmark_2d_106[0][0]

        if horizontal_face:
            if chin_x < forehead_x:
                # this is someone lying down with their face like this (:
                return "rotate_anticlockwise"
            elif forehead_x < chin_x:
                # this is someone lying down with their face like this :)
                return "rotate_clockwise"
            if bbox_center_x >= center_x:
                # this is someone lying down with their face in the right hand side of the frame
                return "rotate_anticlockwise"
            if bbox_center_x < center_x:
                # this is someone lying down with their face in the left hand side of the frame
                return "rotate_clockwise"

        return None

    def auto_unrotate_frame(self, frame: Frame, rotation_action):
        if rotation_action == "rotate_anticlockwise":
            return rotate_clockwise(frame)
        elif rotation_action == "rotate_clockwise":
            return rotate_anticlockwise(frame)

        return frame

    def process_face(self, face_index, target_face: Face, frame: Frame):
        from roop.face_util import align_crop

        enhanced_frame = None
        if len(self.input_face_datas) > 0:
            inputface = self.input_face_datas[face_index].faces[0]
        else:
            inputface = None

        rotation_action = None
        if roop.globals.autorotate_faces:
            rotation_action = self.rotation_action(target_face, frame)
            if rotation_action is not None:
                (startX, startY, endX, endY) = target_face["bbox"].astype("int")
                width = endX - startX
                height = endY - startY
                offs = int(max(width, height) * 0.25)
                rotcutframe, startX, startY, endX, endY = self.cutout(
                    frame, startX - offs, startY - offs, endX + offs, endY + offs
                )
                if rotation_action == "rotate_anticlockwise":
                    rotcutframe = rotate_anticlockwise(rotcutframe)
                elif rotation_action == "rotate_clockwise":
                    rotcutframe = rotate_clockwise(rotcutframe)
                # rotate image and re-detect face to correct wonky landmarks
                rotface = get_first_face(rotcutframe)
                if rotface is None:
                    rotation_action = None
                else:
                    saved_frame = frame.copy()
                    frame = rotcutframe
                    target_face = rotface

        """ Code ported/adapted from Facefusion which borrowed the idea from Rope:
            Kind of subsampling the cutout and aligned face image and faceswapping slices of it up to
            the desired output resolution. This works around the current resolution limitations without using enhancers.
        """
        model_output_size = self.options.swap_output_size
        subsample_size = max(self.options.subsample_size, model_output_size)
        subsample_total = subsample_size // model_output_size
        aligned_img, M = align_crop(frame, target_face.kps, subsample_size)

        fake_frame = aligned_img
        target_face.matrix = M

        _processors = self.processors
        for p in _processors:
            if p.type == "swap":
                if p.processorname == "faceswap_ff":
                    swap_source = self.input_face_datas[face_index] if len(self.input_face_datas) > 0 else inputface
                    self.options._ff_swap_type = getattr(p, 'swap_type', 'ghost')
                else:
                    swap_source = inputface

                subsample_frames = self.implode_pixel_boost(
                    aligned_img, model_output_size, subsample_total
                )
                if getattr(p, "supports_batch", False):
                    # Swap every pixel-boost slice in a single batched
                    # dispatch instead of one GPU round-trip per slice.
                    frames = subsample_frames
                    for _ in range(0, self.options.num_swap_steps):
                        blob = self.prepare_crop_frames(frames)
                        blob = p.Run(swap_source, target_face, blob)
                        frames = self.normalize_swap_frames(blob)
                    swap_result_frames = list(frames)
                else:
                    swap_result_frames = []
                    for sliced_frame in subsample_frames:
                        for _ in range(0, self.options.num_swap_steps):
                            sliced_frame = self.prepare_crop_frame(sliced_frame)
                            sliced_frame = p.Run(swap_source, target_face, sliced_frame)
                            sliced_frame = self.normalize_swap_frame(sliced_frame)
                        swap_result_frames.append(sliced_frame)
                fake_frame = self.explode_pixel_boost(
                    swap_result_frames,
                    model_output_size,
                    subsample_total,
                    subsample_size,
                )
                fake_frame = fake_frame.astype(np.uint8)
                scale_factor = 0.0
            elif p.type == "mask":
                fake_frame = self.process_mask(p, aligned_img, fake_frame)
            else:
                enhanced_frame, scale_factor = p.Run(
                    self.input_face_datas[face_index], target_face, fake_frame
                )

        upscale = 512
        for p in _processors:
            if hasattr(p, 'input_resolution'):
                upscale = p.input_resolution
                break
        orig_width = fake_frame.shape[1]
        if orig_width != upscale:
            fake_frame = cv2.resize(fake_frame, (upscale, upscale), cv2.INTER_CUBIC)
        mask_offsets = (
            (0, 0, 0, 0, 1, 20) if inputface is None else inputface.mask_offsets
        )

        if enhanced_frame is None:
            scale_factor = int(upscale / orig_width)
            result = self.paste_upscale(
                fake_frame,
                fake_frame,
                target_face.matrix,
                frame,
                scale_factor,
                mask_offsets,
            )
        else:
            result = self.paste_upscale(
                fake_frame,
                enhanced_frame,
                target_face.matrix,
                frame,
                scale_factor,
                mask_offsets,
            )

        # Restore mouth before unrotating
        if self.options.restore_original_mouth:
            mouth_cutout, mouth_bb = self.create_mouth_mask(target_face, frame)
            result = self.apply_mouth_area(result, mouth_cutout, mouth_bb)

        if rotation_action is not None:
            fake_frame = self.auto_unrotate_frame(result, rotation_action)
            result = self.paste_simple(fake_frame, saved_frame, startX, startY)

        return result

    def cutout(self, frame: Frame, start_x, start_y, end_x, end_y):
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if end_x > frame.shape[1]:
            end_x = frame.shape[1]
        if end_y > frame.shape[0]:
            end_y = frame.shape[0]
        return frame[start_y:end_y, start_x:end_x], start_x, start_y, end_x, end_y

    def paste_simple(self, src: Frame, dest: Frame, start_x, start_y):
        end_x = start_x + src.shape[1]
        end_y = start_y + src.shape[0]

        start_x, end_x, start_y, end_y = clamp_cut_values(
            start_x, end_x, start_y, end_y, dest
        )
        dest[start_y:end_y, start_x:end_x] = src
        return dest

    def simple_blend_with_mask(self, image1, image2, mask):
        blended_image = (
            image1.astype(np.float32) * (1.0 - mask) + image2.astype(np.float32) * mask
        )
        return blended_image.astype(np.uint8)

    def paste_upscale(
        self, fake_face, upsk_face, M, target_img, scale_factor, mask_offsets
    ):
        M_scale = M * scale_factor
        IM = cv2.invertAffineTransform(M_scale)

        h = upsk_face.shape[0]
        w = upsk_face.shape[1]

        # Compositing only touches the warped face crop plus the mask blur.
        # Restrict all warping/blending to that ROI instead of running
        # full-frame float32 math for every face on every frame.
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        warped_corners = corners @ IM[:, :2].T + IM[:, 2]
        ext = max(
            warped_corners[:, 0].max() - warped_corners[:, 0].min(),
            warped_corners[:, 1].max() - warped_corners[:, 1].min(),
        )
        # Upper bound of blur_area's kernel reach (erosion only shrinks the
        # mask, so covering the Gaussian kernel size is sufficient).
        blur_amount = max(int(mask_offsets[5]), 1)
        margin = int(max(ext // blur_amount, blur_amount // 5)) + 8
        x0 = max(int(np.floor(warped_corners[:, 0].min())) - margin, 0)
        y0 = max(int(np.floor(warped_corners[:, 1].min())) - margin, 0)
        x1 = min(int(np.ceil(warped_corners[:, 0].max())) + margin, target_img.shape[1])
        y1 = min(int(np.ceil(warped_corners[:, 1].max())) + margin, target_img.shape[0])
        if x1 <= x0 or y1 <= y0:
            return target_img.copy()

        # Shift the inverse transform into ROI-local coordinates
        IM_roi = IM.copy()
        IM_roi[:, 2] -= (x0, y0)
        roi_size = (x1 - x0, y1 - y0)

        img_matte = np.zeros((h, w), dtype=np.uint8)
        top = int(mask_offsets[0] * h)
        bottom = int(h - (mask_offsets[1] * h))
        left = int(mask_offsets[2] * w)
        right = int(w - (mask_offsets[3] * w))
        img_matte[top:bottom, left:right] = 255

        img_matte = cv2.warpAffine(
            img_matte,
            IM_roi,
            roi_size,
            flags=cv2.INTER_NEAREST,
            borderValue=0.0,
        )
        # blacken the 1-pixel ROI border: at frame edges this matches the old
        # full-frame behavior; elsewhere the matte is already 0 at the border
        img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0
        if not img_matte.any():
            return target_img.copy()

        img_matte = self.blur_area(img_matte, mask_offsets[4], mask_offsets[5])
        img_matte = img_matte.astype(np.float32) / 255
        img_matte = np.reshape(img_matte, [img_matte.shape[0], img_matte.shape[1], 1])
        paste_face = cv2.warpAffine(
            upsk_face,
            IM_roi,
            roi_size,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if upsk_face is not fake_face:
            fake_face = cv2.warpAffine(
                fake_face,
                IM_roi,
                roi_size,
                borderMode=cv2.BORDER_REPLICATE,
            )
            paste_face = cv2.addWeighted(
                paste_face,
                self.options.blend_ratio,
                fake_face,
                1.0 - self.options.blend_ratio,
                0,
            )

        target_roi = target_img[y0:y1, x0:x1]
        paste_face = img_matte * paste_face
        paste_face = paste_face + (1 - img_matte) * target_roi.astype(np.float32)
        paste_face = paste_face.astype(np.uint8)
        if self.options.show_face_area_overlay:
            green_overlay = np.zeros_like(paste_face)
            green_overlay[:, :, 1] = np.where(img_matte[:, :, 0] > 0, 255, 0)
            paste_face = cv2.addWeighted(paste_face, 1 - 0.5, green_overlay, 0.5, 0)

        result = target_img.copy()
        result[y0:y1, x0:x1] = paste_face
        return result

    def blur_area(self, img_matte, num_erosion_iterations, blur_amount):
        mask_h_inds, mask_w_inds = np.where(img_matte == 255)
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h * mask_w))
        # Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
        # k = max(mask_size//12, 8)
        k = max(mask_size // (blur_amount // 2), blur_amount // 2)
        kernel = np.ones((k, k), np.uint8)
        img_matte = cv2.erode(img_matte, kernel, iterations=num_erosion_iterations)
        # Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
        # k = max(mask_size//24, 4)
        k = max(mask_size // blur_amount, blur_amount // 5)
        kernel_size = (k, k)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        return cv2.GaussianBlur(img_matte, blur_size, 0)

    def _get_swap_type(self):
        """Return the preprocessing type for the active swap model."""
        from roop.ProcessOptions import FF_SWAP_MODELS
        if hasattr(self, 'options') and self.options is not None:
            if self.options.swap_modelname in FF_SWAP_MODELS:
                return getattr(self.options, '_ff_swap_type', 'ghost')
        return "inswapper"

    def prepare_crop_frame(self, swap_frame):
        model_type = self._get_swap_type()

        if model_type in ("ghost", "hyperswap", "uniface"):
            swap_frame = swap_frame[:, :, ::-1] / 127.5 - 1
            swap_frame = swap_frame.transpose(2, 0, 1)
            swap_frame = np.expand_dims(swap_frame, axis=0).astype(np.float32)
        else:
            model_mean = [0.0, 0.0, 0.0]
            model_standard_deviation = [1.0, 1.0, 1.0]
            swap_frame = swap_frame[:, :, ::-1] / 255.0
            swap_frame = (swap_frame - model_mean) / model_standard_deviation
            swap_frame = swap_frame.transpose(2, 0, 1)
            swap_frame = np.expand_dims(swap_frame, axis=0).astype(np.float32)
        return swap_frame

    def prepare_crop_frames(self, frames):
        """Batched prepare_crop_frame: [N, H, W, 3] BGR uint8/float -> [N, 3, H, W] float32."""
        model_type = self._get_swap_type()
        frames = np.asarray(frames)
        if model_type in ("ghost", "hyperswap", "uniface"):
            blob = frames[:, :, :, ::-1] / 127.5 - 1
        else:
            blob = frames[:, :, :, ::-1] / 255.0
        return np.ascontiguousarray(blob.transpose(0, 3, 1, 2), dtype=np.float32)

    def normalize_swap_frames(self, blob):
        """Batched normalize_swap_frame: [N, 3, H, W] -> [N, H, W, 3] BGR."""
        model_type = self._get_swap_type()
        frames = np.asarray(blob).transpose(0, 2, 3, 1)
        if model_type in ("ghost", "hyperswap", "uniface"):
            frames = (frames * 127.5 + 127.5).round()
        else:
            frames = (frames * 255.0).round()
        return frames[:, :, :, ::-1]

    def normalize_swap_frame(self, swap_frame):
        model_type = self._get_swap_type()
        # Strip batch dimension if present (e.g. [1, C, H, W] -> [C, H, W])
        while swap_frame.ndim > 3:
            swap_frame = swap_frame[0]
        swap_frame = swap_frame.transpose(1, 2, 0)

        if model_type in ("ghost", "hyperswap", "uniface"):
            swap_frame = (swap_frame * 127.5 + 127.5).round()
        else:
            swap_frame = (swap_frame * 255.0).round()
        swap_frame = swap_frame[:, :, ::-1]
        return swap_frame

    def implode_pixel_boost(
        self, aligned_face_frame, model_size, pixel_boost_total: int
    ):
        # Use view instead of copy when possible to reduce memory usage
        try:
            subsample_frame = aligned_face_frame.reshape(
                model_size, pixel_boost_total, model_size, pixel_boost_total, 3
            )
            result = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(
                pixel_boost_total**2, model_size, model_size, 3
            )
            # Only copy if we need to preserve the original
            if result.base is aligned_face_frame:
                result = result.copy()
            return result
        except Exception:
            # Fallback to original method if reshape fails
            subsample_frame = aligned_face_frame.reshape(
                model_size, pixel_boost_total, model_size, pixel_boost_total, 3
            )
            subsample_frame = subsample_frame.transpose(1, 3, 0, 2, 4).reshape(
                pixel_boost_total**2, model_size, model_size, 3
            )
            result = subsample_frame.copy()
            del subsample_frame
            return result

    def explode_pixel_boost(
        self, subsample_frame, model_size, pixel_boost_total, pixel_boost_size
    ):
        final_frame = np.stack(subsample_frame, axis=0).reshape(
            pixel_boost_total, pixel_boost_total, model_size, model_size, 3
        )
        final_frame = final_frame.transpose(2, 0, 3, 1, 4).reshape(
            pixel_boost_size, pixel_boost_size, 3
        )
        result = final_frame.copy()
        del final_frame
        for frame in subsample_frame:
            del frame
        subsample_frame.clear()
        return result

    def process_mask(self, processor, frame: Frame, target: Frame):
        img_mask = processor.Run(frame, self.options.masking_text)
        img_mask = cv2.resize(img_mask, (target.shape[1], target.shape[0]))
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if self.options.show_face_masking:
            result = (1 - img_mask) * frame.astype(np.float32)
            return np.uint8(result)

        target = target.astype(np.float32)
        result = (1 - img_mask) * target
        result += img_mask * frame.astype(np.float32)
        return np.uint8(result)

    # Code for mouth restoration adapted from https://github.com/iVideoGameBoss/iRoopDeepFaceCam

    def create_mouth_mask(self, face: Face, frame: Frame):
        mouth_cutout = None

        landmarks = face.landmark_2d_106
        if landmarks is not None:
            mouth_points = landmarks[52:71].astype(np.int32)  # outer mouth
            min_x, min_y = np.min(mouth_points, axis=0)
            max_x, max_y = np.max(mouth_points, axis=0)
            min_x = max(0, min_x - (15 * 6))
            min_y = max(0, min_y - 22)
            max_x = min(frame.shape[1], max_x + (15 * 6))
            max_y = min(frame.shape[0], max_y + (90 * 6))
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        return mouth_cutout, (min_x, min_y, max_x, max_y)

    def create_feathered_mask(self, shape, feather_amount=30):
        mask = np.zeros(shape[:2], dtype=np.float32)
        center = (shape[1] // 2, shape[0] // 2)
        cv2.ellipse(
            mask,
            center,
            (shape[1] // 2 - feather_amount, shape[0] // 2 - feather_amount),
            0,
            0,
            360,
            1,
            -1,
        )
        mask = cv2.GaussianBlur(
            mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0
        )
        return mask / np.max(mask)

    def apply_mouth_area(
        self, frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple
    ) -> np.ndarray:
        min_x, min_y, max_x, max_y = mouth_box
        box_width = max_x - min_x
        box_height = max_y - min_y

        # Resize the mouth cutout to match the mouth box size
        if mouth_cutout is None or box_width is None or box_height is None:
            return frame
        try:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.shape != resized_mouth_cutout.shape:
                resized_mouth_cutout = cv2.resize(
                    resized_mouth_cutout, (roi.shape[1], roi.shape[0])
                )
            color_corrected_mouth = self.apply_color_transfer(resized_mouth_cutout, roi)
            feather_amount = min(30, box_width // 15, box_height // 15)
            mask = self.create_feathered_mask(resized_mouth_cutout.shape, feather_amount)
            mask = mask[:, :, np.newaxis]
            blended = (color_corrected_mouth * mask + roi * (1 - mask)).astype(np.uint8)
            frame[min_y:max_y, min_x:max_x] = blended
        except Exception as e:
            print(f"Error {e}")
            pass

        return frame

    def apply_color_transfer(self, source, target):
        """
        Apply color transfer from target to source image
        """
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        source_mean, source_std = cv2.meanStdDev(source)
        target_mean, target_std = cv2.meanStdDev(target)

        source_mean = source_mean.reshape(1, 1, 3)
        source_std = source_std.reshape(1, 1, 3)
        target_mean = target_mean.reshape(1, 1, 3)
        target_std = target_std.reshape(1, 1, 3)
        source = (source - source_mean) * (target_std / source_std) + target_mean
        return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    def release_resources(self):
        import tempfile
        import shutil

        for p in self.processors:
            p.Release()
        self.processors.clear()
        if self.videowriter is not None:
            self.videowriter.close()
            self.videowriter = None
        if self.streamwriter is not None:
            self.streamwriter.Close()
            self.streamwriter = None
        if hasattr(self, 'frames_queue') and self.frames_queue:
            self.frames_queue.clear()
            self.frames_queue = None
        if hasattr(self, 'processed_queue') and self.processed_queue:
            self.processed_queue.clear()
            self.processed_queue = None
        if hasattr(self, 'input_face_datas'):
            self.input_face_datas.clear()
        if hasattr(self, 'target_face_datas'):
            self.target_face_datas.clear()
        if hasattr(self, 'last_swapped_frame') and self.last_swapped_frame is not None:
            del self.last_swapped_frame
            self.last_swapped_frame = None
        try:
            tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
            if os.path.exists(tempfolder):
                shutil.rmtree(tempfolder)
        except Exception as e:
            print(f"Warning: Could not clean temp folder during release: {e}")

    def _cleanup_gpu_memory(self):
        """No-op on Apple Silicon (unified memory; MLX manages its own buffers)."""
        return None

    def _cleanup_gpu_memory_light(self):
        """No-op on Apple Silicon (unified memory)."""
        return None
