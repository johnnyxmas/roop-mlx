import os
import pathlib
import time
import threading
import cv2
import gradio as gr
import roop.utilities as util
import roop.globals
import ui.globals
from roop.face_util import extract_face_images, create_blank_image
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions, SWAP_MODEL_OUTPUT_SIZES
from roop.FaceSet import FaceSet
from roop.utilities import clean_dir

last_image = None


def _fpath(f) -> str:
    """Return filesystem path from a Gradio file object (FileData or NamedString)."""
    return f.path if hasattr(f, "path") else str(f)


def _selected_target_path(files) -> str:
    """Path of the selected target as it will be processed.

    A re-encode swaps the entry's file for the converted one while the Gradio
    file list keeps the original upload; frame numbers only line up with the
    converted file (a lower FPS has fewer frames).
    """
    if 0 <= selected_preview_index < len(list_files_process):
        return list_files_process[selected_preview_index].filename
    return _fpath(files[selected_preview_index])


IS_INPUT = True
SELECTED_FACE_INDEX = 0
SELECTION_FACES_DATA = []

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

input_faces = None
target_faces = None
face_selection = None
previewimage = None

selected_preview_index = 0

# Swap worker thread. "Running" is derived from its liveness rather than a
# flag: the Stop button cancels start_swap mid-yield, so any reset placed after
# the worker loop never executes.
_swap_thread = None


def _swap_running():
    return _swap_thread is not None and _swap_thread.is_alive()


list_files_process: list[ProcessEntry] = []
model_swap_choices = [
    "InSwapper 128",
    "ReSwapper 128",
    "ReSwapper 256",
    "HyperSwap 1A",
    "HyperSwap 1B",
    "HyperSwap 1C",
    "AlphaFace 256",
]

DEFAULT_SWAP_MODEL = "HyperSwap 1A"

# Subsample ("pixel boost") upscale targets. A value below the swap model's
# native output size is a no-op (it gets clamped up in ProcessMgr), so the
# dropdown only offers sizes >= the selected model's native output.
UPSCALE_CHOICES = ["128px", "256px", "512px"]


def upscale_choices_for_model(swap_model):
    native = SWAP_MODEL_OUTPUT_SIZES.get(swap_model, 128)
    return [c for c in UPSCALE_CHOICES if int(c[:3]) >= native]

no_face_choices = [
    "Use untouched original frame",
    "Retry rotated",
    "Skip Frame",
    "Skip Frame if no similar face",
    "Use last swapped",
]
swap_choices = [
    "First found",
    "All input faces",
    "All input faces (random)",
    "All female",
    "All male",
    "All faces",
    "Selected face",
]

current_video_fps = 50


_RES_HEIGHTS = {"480p": 480, "720p": 720, "1080p": 1080, "2160p": 2160}
_RES_TIERS = [480, 720, 1080, 2160]


def _current_target_filename():
    """Path of the currently-selected target (same source of truth as the swap
    pipeline). Returns None when no target is loaded."""
    if list_files_process and 0 <= selected_preview_index < len(list_files_process):
        return list_files_process[selected_preview_index].filename
    return None


def _current_target_height():
    """Pixel height of the current video target, or None for images/no target."""
    import cv2 as _cv2
    fn = _current_target_filename()
    if fn is None or not util.is_video(fn):
        return None
    cap = _cv2.VideoCapture(fn)
    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0
    cap.release()
    return h or None


def _conversion_button_states():
    """Enable/disable the five re-encode buttons based on the current target."""
    h = _current_target_height()
    is_video = h is not None
    size_states = [gr.update(interactive=(is_video and tier < h)) for tier in _RES_TIERS]
    return size_states + [gr.update(interactive=is_video)]


# The trailing entry is the Stop-conversion button, enabled only while converting.
def _lock_for_convert():
    return [gr.update(interactive=False)] * (3 + len(_RES_TIERS) + 1) + [gr.update(interactive=True)]


def _unlock_after_convert():
    return (
        [gr.update(interactive=True)] * 3
        + _conversion_button_states()
        + [gr.update(interactive=False)]
    )


def on_stop_convert():
    import roop.util_ffmpeg as ffmpeg

    if ffmpeg.cancel_scale_video():
        gr.Info("Stopping re-encode...")


def _on_convert_480(fps, progress=gr.Progress()):
    return on_convert_target_resolution("480p", fps, progress)


def _on_convert_720(fps, progress=gr.Progress()):
    return on_convert_target_resolution("720p", fps, progress)


def _on_convert_1080(fps, progress=gr.Progress()):
    return on_convert_target_resolution("1080p", fps, progress)


def _on_convert_2160(fps, progress=gr.Progress()):
    return on_convert_target_resolution("2160p", fps, progress)


def _on_convert_fps(fps, progress=gr.Progress()):
    return on_convert_target_resolution(None, fps, progress)


def on_convert_target_resolution(resolution, fps=0, progress=gr.Progress()):
    """Re-encode the selected target video (resize and/or FPS change)."""
    global current_video_fps
    import roop.util_ffmpeg as ffmpeg

    noop = (gr.update(), gr.update(), gr.update())
    try:
        if _swap_running():
            gr.Warning("Cannot re-encode while a swap is running (or still stopping).")
            return noop
        if not list_files_process or selected_preview_index >= len(list_files_process):
            gr.Warning("Load a target video first.")
            return noop
        entry = list_files_process[selected_preview_index]
        src = entry.filename
        if not util.is_video(src):
            gr.Warning("The selected target is not a video.")
            return noop

        try:
            fps_val = float(fps) if fps else 0.0
        except (TypeError, ValueError):
            fps_val = 0.0
        fps_val = fps_val if fps_val > 0 else None

        height = _RES_HEIGHTS.get(resolution) if resolution else None
        if height is None and fps_val is None:
            gr.Warning("Pick a smaller size and/or set an FPS to re-encode.")
            return noop
        if height is not None:
            src_height = _current_target_height()
            if src_height is not None and height >= src_height:
                gr.Warning(f"Source is {src_height}px tall; pick a smaller size than {resolution}.")
                return noop

        size_tag = resolution if resolution else "src"
        fps_tag = f"_{fps_val:g}fps" if fps_val else ""
        temp_dir = util.resolve_relative_path("../temp")
        os.makedirs(temp_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(temp_dir, f"{base}__{size_tag}{fps_tag}.mp4")

        desc = " + ".join(
            ([resolution] if height is not None else [])
            + ([f"{fps_val:g} fps"] if fps_val else [])
        )
        _desc = f"Re-encoding target to {desc}..."
        progress(0, desc=_desc)
        ok = ffmpeg.scale_video(
            src, out, height, fps_val,
            on_progress=lambda frac: progress(frac, desc=_desc),
        )
        if ffmpeg.scale_video_cancelled():
            gr.Info("Re-encode stopped — target unchanged.")
            return noop
        if not ok or not os.path.exists(out):
            gr.Warning("Video re-encode failed — target unchanged.")
            return noop

        entry.filename = out
        entry.startframe = 0
        total_frames = get_video_frame_total(out)
        entry.endframe = total_frames
        current_video_fps = util.detect_fps(out)
        entry.fps = current_video_fps
        info_html = get_target_file_info_html(out)
        progress(1.0)
        gr.Info(f"Re-encoded to {desc} and loaded as target.")
        return (
            gr.Slider(value=1, maximum=max(2, total_frames), info="0:00:00"),
            gen_processing_text(entry.startframe, entry.endframe),
            info_html,
        )
    except Exception as e:
        gr.Warning(f"Re-encode error: {e}")
        return noop

manual_masking = False


def faceswap_tab():
    global previewimage

    with gr.Tab("🎭 Face Swap"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                with gr.Row():
                    input_faces = gr.Gallery(
                        label="Input faces gallery",
                        allow_preview=False,
                        preview=False,
                        height=138,
                        columns=64,
                        object_fit="contain",
                        interactive=False,
                        elem_classes=["facestrip"],
                    )
                    target_faces = gr.Gallery(
                        label="Target faces gallery",
                        allow_preview=False,
                        preview=False,
                        height=138,
                        columns=64,
                        object_fit="contain",
                        interactive=False,
                        elem_classes=["facestrip"],
                    )
                with gr.Row():
                    bt_move_left_input = gr.Button("⬅ Move left", size="sm")
                    bt_move_right_input = gr.Button("➡ Move right", size="sm")
                    bt_move_left_target = gr.Button("⬅ Move left", size="sm")
                    bt_move_right_target = gr.Button("➡ Move right", size="sm")
                with gr.Row():
                    bt_remove_selected_input_face = gr.Button(
                        "❌ Remove selected", size="sm"
                    )
                    bt_clear_input_faces = gr.Button(
                        "💥 Clear all", variant="stop", size="sm"
                    )
                    bt_remove_selected_target_face = gr.Button(
                        "❌ Remove selected", size="sm"
                    )
                    bt_add_local = gr.Button("Add local files from", size="sm")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Accordion(label="Advanced Masking", open=False):
                            chk_showmaskoffsets = gr.Checkbox(
                                label="Show mask overlay in preview",
                                value=False,
                                interactive=True,
                            )
                            chk_restoreoriginalmouth = gr.Checkbox(
                                label="Restore original mouth area",
                                value=False,
                                interactive=True,
                            )
                            mask_top = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Top",
                                step=0.01,
                                interactive=True,
                            )
                            mask_bottom = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Bottom",
                                step=0.01,
                                interactive=True,
                            )
                            mask_left = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Left",
                                step=0.01,
                                interactive=True,
                            )
                            mask_right = gr.Slider(
                                0,
                                1.0,
                                value=0,
                                label="Offset Face Right",
                                step=0.01,
                                interactive=True,
                            )
                            mask_erosion = gr.Slider(
                                1.0,
                                3.0,
                                value=1.0,
                                label="Erosion Iterations",
                                step=1.00,
                                interactive=True,
                            )
                            mask_blur = gr.Slider(
                                10.0,
                                50.0,
                                value=20.0,
                                label="Blur size",
                                step=1.00,
                                interactive=True,
                            )
                            bt_toggle_masking = gr.Button(
                                "Toggle manual masking", variant="secondary", size="sm"
                            )
                            selected_mask_engine = gr.Dropdown(
                                ["None", "Clip2Seg", "DFL XSeg"],
                                value="None",
                                label="Face masking engine",
                            )
                            clip_text = gr.Textbox(
                                label="List of objects to mask and restore back on fake face",
                                value="cup,hands,hair,banana",
                                interactive=False,
                            )
                            bt_preview_mask = gr.Button(
                                "👥 Show Mask Preview", variant="secondary"
                            )
                    with gr.Column(scale=2):
                        local_folder = gr.Textbox(
                            show_label=False, placeholder="/content/", interactive=True
                        )
                with gr.Row(variant="panel"):
                    bt_srcfiles = gr.Files(
                        label="Source Images or Facesets",
                        file_count="multiple",
                        file_types=["image", ".fsz", ".webp"],
                        elem_id="filelist",
                        height=233,
                    )
                    bt_destfiles = gr.Files(
                        label="Target File(s)",
                        file_count="multiple",
                        file_types=["image", "video", ".webp"],
                        elem_id="filelist",
                        height=233,
                    )
                target_file_info = gr.HTML(value="", visible=True, elem_id="target_file_info")
                gr.Markdown(
                    "**Re-encode source video** — replaces the target with a "
                    "downscaled and/or FPS-changed copy. Size buttons re-encode "
                    "to that resolution (sizes at or above the source are "
                    "disabled). The FPS field can be applied on its own."
                )
                with gr.Row(variant="panel"):
                    bt_conv_480 = gr.Button("480p", variant="secondary", size="sm", interactive=False)
                    bt_conv_720 = gr.Button("720p", variant="secondary", size="sm", interactive=False)
                    bt_conv_1080 = gr.Button("1080p", variant="secondary", size="sm", interactive=False)
                    bt_conv_2160 = gr.Button("2160p", variant="secondary", size="sm", interactive=False)
                with gr.Row(variant="panel"):
                    target_fps = gr.Number(
                        value=0,
                        label="Re-encode FPS (0 = keep source)",
                        minimum=0,
                        maximum=240,
                        scale=2,
                    )
                    bt_conv_fps = gr.Button(
                        "Change FPS (keep size)", variant="secondary", size="sm",
                        interactive=False, scale=1,
                    )
                    bt_conv_stop = gr.Button(
                        "⏹ Stop", variant="stop", size="sm",
                        interactive=False, scale=1,
                    )
                with gr.Row(variant="panel"):
                    ui.globals.ui_selected_swap_model = gr.Dropdown(
                        model_swap_choices,
                        value=DEFAULT_SWAP_MODEL,
                        label="Specify Face Swap Model",
                    )
                    bt_test_all = gr.Button("Test All Swappers", variant="secondary")
                with gr.Row(variant="panel"):
                    forced_fps = gr.Slider(
                        minimum=0,
                        maximum=120,
                        value=0,
                        label="Video FPS",
                        info="Overrides detected fps if not 0",
                        step=1.0,
                        interactive=True,
                        container=True,
                    )

            with gr.Column(scale=2):
                previewimage = gr.Image(
                    label="Preview Image",
                    height=576,
                    type="numpy",
                    interactive=False,
                    visible=True,
                )
                maskimage = gr.ImageEditor(
                    label="Manual mask Image",
                    sources=["clipboard"],
                    transforms="",
                    type="numpy",
                    brush=gr.Brush(
                        color_mode="fixed", colors=["rgba(255, 255, 255, 1"]
                    ),
                    interactive=True,
                    visible=False,
                )
                with gr.Row(variant="panel"):
                    fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    bt_refresh_preview = gr.Button(
                        "🔄 Refresh", variant="secondary", size="sm"
                    )
                    bt_use_face_from_preview = gr.Button(
                        "Use Face from this Frame", variant="primary", size="sm"
                    )
                with gr.Row():
                    preview_frame_num = gr.Slider(
                        1,
                        2,
                        value=1,
                        label="Frame Number",
                        info="0:00:00",
                        step=1.0,
                        interactive=True,
                    )
                with gr.Row():
                    text_frame_clip = gr.Markdown("Processing frame range [0 - 0]")
                    set_frame_start = gr.Button("⬅ Set as Start", size="sm")
                    set_frame_end = gr.Button("➡ Set as End", size="sm")
        with gr.Row(visible=False) as dynamic_face_selection:
            with gr.Column(scale=2):
                face_selection = gr.Gallery(
                    label="Detected faces",
                    allow_preview=False,
                    preview=False,
                    height=138,
                    object_fit="cover",
                    columns=32,
                )
            with gr.Column():
                bt_faceselect = gr.Button("☑ Use selected face", size="sm")
                bt_cancelfaceselect = gr.Button("Done", size="sm")
            with gr.Column():
                gr.Markdown(" ")

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                selected_face_detection = gr.Dropdown(
                    swap_choices,
                    value="First found",
                    label="Specify face selection for swapping",
                )
            with gr.Column(scale=1):
                num_swap_steps = gr.Slider(
                    1,
                    5,
                    value=1,
                    step=1.0,
                    label="Number of swapping steps",
                    info="More steps may increase likeness",
                )
            with gr.Column(scale=2):
                ui.globals.ui_selected_enhancer = gr.Dropdown(
                    [
                        "None",
                        "Codeformer",

                        "GFPGAN",
                        "GPEN 512",
                        "GPEN 1024",
                        "GPEN 2048",
                        "Restoreformer++",
                    ],
                    value="None",
                    label="Select post-processing",
                )

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                max_face_distance = gr.Slider(
                    0.01,
                    1.0,
                    value=0.65,
                    label="Max Face Similarity Threshold",
                    info="0.0 = identical 1.0 = no similarity",
                )
            with gr.Column(scale=1):
                ui.globals.ui_upscale = gr.Dropdown(
                    upscale_choices_for_model(DEFAULT_SWAP_MODEL),
                    value=upscale_choices_for_model(DEFAULT_SWAP_MODEL)[0],
                    label="Subsample upscale to",
                    interactive=True,
                )
            with gr.Column(scale=2):
                ui.globals.ui_blend_ratio = gr.Slider(
                    0.0,
                    1.0,
                    value=0.65,
                    label="Original/Enhanced image blend ratio",
                    info="Only used with active post-processing",
                )
                ui.globals.ui_swapper_weight = gr.Slider(
                    0.0,
                    1.0,
                    value=0.5,
                    step=0.05,
                    label="Source identity weight",
                    info="0.5 leaves the identity untouched; lower keeps more "
                    "of the target's own face, higher pushes harder toward the "
                    "source",
                )

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                video_swapping_method = gr.Dropdown(
                    ["Extract Frames to media", "In-Memory processing"],
                    value="In-Memory processing",
                    label="Select video processing method",
                    interactive=True,
                )
                no_face_action = gr.Dropdown(
                    choices=no_face_choices,
                    value=no_face_choices[0],
                    label="Action on no face detected",
                    interactive=True,
                )
                vr_mode = gr.Checkbox(label="VR Mode", value=False)
            with gr.Column(scale=1):
                with gr.Group():
                    autorotate = gr.Checkbox(
                        label="Auto rotate horizontal Faces", value=True
                    )
                    roop.globals.skip_audio = gr.Checkbox(
                        label="Skip audio", value=False
                    )
                    roop.globals.keep_frames = gr.Checkbox(
                        label="Keep Frames (relevant only when extracting frames)",
                        value=False,
                    )
                    roop.globals.wait_after_extraction = gr.Checkbox(
                        label="Wait for user key press before creating video ",
                        value=False,
                    )

        with gr.Row(variant="panel"):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant="primary")
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant="secondary", interactive=False)
                gr.Button("👀 Open Output Folder", size="sm").click(
                    fn=lambda: util.open_folder(roop.globals.output_path)
                )
            with gr.Column(scale=2):
                output_method = gr.Dropdown(
                    ["File", "Virtual Camera", "Both"],
                    value="File",
                    label="Select Output Method",
                    interactive=True,
                )
        with gr.Row(variant="panel"):
            status_bar = gr.HTML(
                value='<div style="padding:8px 12px;background:#1a1a2e;border:1px solid #333;border-radius:6px;font-family:monospace;color:#888;font-size:14px;">Idle</div>',
                label="Processing Status",
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                resultfiles = gr.Files(label="Processed File(s)", interactive=False)
            with gr.Column():
                resultimage = gr.Image(
                    type="filepath", label="Final Image", interactive=False
                )
                resultvideo = gr.Video(
                    label="Final Video", interactive=False, visible=False
                )

    previewinputs = [
        ui.globals.ui_selected_swap_model,
        preview_frame_num,
        bt_destfiles,
        fake_preview,
        ui.globals.ui_selected_enhancer,
        selected_face_detection,
        max_face_distance,
        ui.globals.ui_blend_ratio,
        selected_mask_engine,
        clip_text,
        no_face_action,
        vr_mode,
        autorotate,
        maskimage,
        chk_showmaskoffsets,
        chk_restoreoriginalmouth,
        num_swap_steps,
        ui.globals.ui_upscale,
    ]
    previewoutputs = [previewimage, maskimage, preview_frame_num]
    input_faces.select(on_select_input_face, None, None).success(
        fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs
    )

    bt_move_left_input.click(
        fn=move_selected_input, inputs=[bt_move_left_input], outputs=[input_faces]
    )
    bt_move_right_input.click(
        fn=move_selected_input, inputs=[bt_move_right_input], outputs=[input_faces]
    )
    bt_move_left_target.click(
        fn=move_selected_target, inputs=[bt_move_left_target], outputs=[target_faces]
    )
    bt_move_right_target.click(
        fn=move_selected_target, inputs=[bt_move_right_target], outputs=[target_faces]
    )

    bt_remove_selected_input_face.click(
        fn=remove_selected_input_face, outputs=[input_faces]
    )
    bt_srcfiles.change(
        fn=on_srcfile_changed,
        show_progress="full",
        inputs=bt_srcfiles,
        outputs=[dynamic_face_selection, face_selection, input_faces, bt_srcfiles],
    )

    mask_top.release(fn=on_mask_top_changed, inputs=[mask_top], show_progress="hidden")
    mask_bottom.release(
        fn=on_mask_bottom_changed, inputs=[mask_bottom], show_progress="hidden"
    )
    mask_left.release(
        fn=on_mask_left_changed, inputs=[mask_left], show_progress="hidden"
    )
    mask_right.release(
        fn=on_mask_right_changed, inputs=[mask_right], show_progress="hidden"
    )
    mask_erosion.release(
        fn=on_mask_erosion_changed, inputs=[mask_erosion], show_progress="hidden"
    )
    mask_blur.release(
        fn=on_mask_blur_changed, inputs=[mask_blur], show_progress="hidden"
    )
    selected_mask_engine.change(
        fn=on_mask_engine_changed,
        inputs=[selected_mask_engine],
        outputs=[clip_text],
        show_progress="hidden",
    )
    ui.globals.ui_selected_swap_model.change(
        fn=on_swap_model_changed,
        inputs=[ui.globals.ui_selected_swap_model, ui.globals.ui_upscale],
        outputs=[ui.globals.ui_upscale],
        show_progress="hidden",
    )

    target_faces.select(on_select_target_face, None, None)
    bt_remove_selected_target_face.click(
        fn=remove_selected_target_face, outputs=[target_faces]
    )

    forced_fps.change(fn=on_fps_changed, inputs=[forced_fps], show_progress="hidden")
    ui.globals.ui_swapper_weight.change(
        fn=on_swapper_weight_changed,
        inputs=[ui.globals.ui_swapper_weight],
        show_progress="hidden",
    )
    _conversion_buttons = [bt_conv_480, bt_conv_720, bt_conv_1080, bt_conv_2160, bt_conv_fps]
    bt_destfiles.change(
        fn=on_destfiles_changed,
        inputs=[bt_destfiles],
        outputs=[preview_frame_num, text_frame_clip, target_file_info],
        show_progress="hidden",
    ).success(
        fn=on_preview_frame_changed,
        inputs=previewinputs,
        outputs=previewoutputs,
        show_progress="hidden",
    ).then(fn=_conversion_button_states, outputs=_conversion_buttons)
    bt_destfiles.select(
        fn=on_destfiles_selected,
        outputs=[preview_frame_num, text_frame_clip, forced_fps, target_file_info],
        show_progress="hidden",
    ).success(
        fn=on_preview_frame_changed,
        inputs=previewinputs,
        outputs=previewoutputs,
        show_progress="hidden",
    ).then(fn=_conversion_button_states, outputs=_conversion_buttons)
    bt_destfiles.clear(
        fn=on_clear_destfiles, outputs=[target_faces, selected_face_detection, target_file_info]
    ).then(fn=_conversion_button_states, outputs=_conversion_buttons)

    _convert_lockables = [bt_start, bt_srcfiles, bt_destfiles] + _conversion_buttons + [bt_conv_stop]
    # Unqueued so it runs while the conversion itself holds the queue.
    bt_conv_stop.click(fn=on_stop_convert, queue=False)
    for _btn, _conv_fn in (
        (bt_conv_480, _on_convert_480),
        (bt_conv_720, _on_convert_720),
        (bt_conv_1080, _on_convert_1080),
        (bt_conv_2160, _on_convert_2160),
        (bt_conv_fps, _on_convert_fps),
    ):
        _btn.click(
            fn=_lock_for_convert, outputs=_convert_lockables, queue=False,
        ).then(
            fn=_conv_fn, inputs=[target_fps],
            outputs=[preview_frame_num, text_frame_clip, target_file_info],
            show_progress="full",
        ).then(
            fn=on_preview_frame_changed, inputs=previewinputs,
            outputs=previewoutputs, show_progress="hidden",
        ).then(
            fn=_unlock_after_convert, outputs=_convert_lockables, queue=False,
        )
    resultfiles.select(
        fn=on_resultfiles_selected,
        inputs=[resultfiles],
        outputs=[resultimage, resultvideo],
    )

    face_selection.select(on_select_face, None, None)
    bt_faceselect.click(
        fn=on_selected_face,
        outputs=[input_faces, target_faces, selected_face_detection],
    )
    bt_cancelfaceselect.click(
        fn=on_end_face_selection, outputs=[dynamic_face_selection, face_selection]
    )

    bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

    bt_add_local.click(
        fn=on_add_local_folder, inputs=[local_folder], outputs=[bt_destfiles]
    )
    bt_preview_mask.click(
        fn=on_preview_mask,
        inputs=[
            ui.globals.ui_selected_swap_model,
            preview_frame_num,
            bt_destfiles,
            clip_text,
            selected_mask_engine,
        ],
        outputs=[previewimage],
    )

    start_event = bt_start.click(
        fn=start_swap,
        inputs=[
            ui.globals.ui_selected_swap_model,
            output_method,
            ui.globals.ui_selected_enhancer,
            selected_face_detection,
            roop.globals.keep_frames,
            roop.globals.wait_after_extraction,
            roop.globals.skip_audio,
            max_face_distance,
            ui.globals.ui_blend_ratio,
            selected_mask_engine,
            clip_text,
            video_swapping_method,
            no_face_action,
            vr_mode,
            autorotate,
            chk_restoreoriginalmouth,
            num_swap_steps,
            ui.globals.ui_upscale,
            maskimage,
        ],
        outputs=[bt_start, bt_stop, resultfiles, status_bar],
        show_progress="full",
    )
    after_swap_event = start_event.success(
        fn=on_resultfiles_finished,
        inputs=[resultfiles],
        outputs=[resultimage, resultvideo],
    )

    bt_stop.click(
        fn=stop_swap,
        cancels=[start_event, after_swap_event],
        outputs=[bt_start, bt_stop, status_bar],
        queue=False,
    )

    bt_refresh_preview.click(
        fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs
    )
    bt_toggle_masking.click(
        fn=on_toggle_masking,
        inputs=[previewimage, maskimage],
        outputs=[previewimage, maskimage],
    )
    fake_preview.change(
        fn=on_preview_frame_changed, inputs=previewinputs, outputs=previewoutputs
    )
    preview_frame_num.release(
        fn=on_preview_frame_changed,
        inputs=previewinputs,
        outputs=previewoutputs,
        show_progress="hidden",
    )
    bt_use_face_from_preview.click(
        fn=on_use_face_from_selected,
        show_progress="full",
        inputs=[bt_destfiles, preview_frame_num],
        outputs=[
            dynamic_face_selection,
            face_selection,
            target_faces,
            selected_face_detection,
        ],
    )
    bt_test_all.click(
        fn=test_all_swappers,
        inputs=[
            preview_frame_num,
            bt_destfiles,
            selected_face_detection,
            ui.globals.ui_upscale,
            selected_mask_engine,
            clip_text,
            no_face_action,
            vr_mode,
            autorotate,
            max_face_distance,
            ui.globals.ui_blend_ratio,
            num_swap_steps,
        ],
        outputs=[resultfiles],
        show_progress="full",
    )
    set_frame_start.click(
        fn=on_set_frame,
        inputs=[set_frame_start, preview_frame_num],
        outputs=[text_frame_clip],
    )
    set_frame_end.click(
        fn=on_set_frame,
        inputs=[set_frame_end, preview_frame_num],
        outputs=[text_frame_clip],
    )


def on_mask_top_changed(mask_offset):
    set_mask_offset(0, mask_offset)


def on_mask_bottom_changed(mask_offset):
    set_mask_offset(1, mask_offset)


def on_mask_left_changed(mask_offset):
    set_mask_offset(2, mask_offset)


def on_mask_right_changed(mask_offset):
    set_mask_offset(3, mask_offset)


def on_mask_erosion_changed(mask_offset):
    set_mask_offset(4, mask_offset)


def on_mask_blur_changed(mask_offset):
    set_mask_offset(5, mask_offset)


def set_mask_offset(index, mask_offset):
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        # Faces store their offsets as a tuple; edit a copy and store it back.
        offs = list(
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        )
        offs[index] = mask_offset
        if offs[0] + offs[1] > 0.99:
            offs[0] = 0.99
            offs[1] = 0.0
        if offs[2] + offs[3] > 0.99:
            offs[2] = 0.99
            offs[3] = 0.0
        roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[
            0
        ].mask_offsets = tuple(offs)


def on_swap_model_changed(swap_model, current_upscale):
    choices = upscale_choices_for_model(swap_model)
    value = current_upscale if current_upscale in choices else choices[0]
    return gr.Dropdown(choices=choices, value=value)


def on_mask_engine_changed(mask_engine):
    if mask_engine == "Clip2Seg":
        return gr.Textbox(interactive=True)
    return gr.Textbox(interactive=False)


def on_add_local_folder(folder):
    files = util.get_local_files_from_folder(folder)
    if files is None:
        gr.Warning("Empty folder or folder not found!")
    return files


def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global SELECTION_FACES_DATA, IS_INPUT, SELECTED_FACE_INDEX

    IS_INPUT = True
    SELECTED_FACE_INDEX = 0

    if srcfiles is None or len(srcfiles) < 1:
        return gr.Column(visible=False), None, ui.globals.ui_input_thumbs, None

    for f in srcfiles:
        source_path = _fpath(f)
        if source_path.lower().endswith("fsz"):
            progress(0, desc="Retrieving faces from Faceset File")
            import tempfile
            unzipfolder = os.path.join(os.environ.get("TEMP") or os.environ.get("TMPDIR") or tempfile.gettempdir(), "faceset")
            if os.path.isdir(unzipfolder):
                files = os.listdir(unzipfolder)
                for file in files:
                    os.remove(os.path.join(unzipfolder, file))
            else:
                os.makedirs(unzipfolder)
            util.mkdir_with_umask(unzipfolder)
            util.unzip(source_path, unzipfolder)
            is_first = True
            face_set = FaceSet()
            for file in os.listdir(unzipfolder):
                if file.endswith(".png"):
                    filename = os.path.join(unzipfolder, file)
                    progress(0, desc="Extracting faceset")
                    SELECTION_FACES_DATA = extract_face_images(filename, (False, 0))
                    for f in SELECTION_FACES_DATA:
                        face = f[0]
                        face.mask_offsets = (0, 0, 0, 0, 1, 20)
                        face_set.faces.append(face)
                        if is_first:
                            image = util.convert_to_gradio(f[1])
                            ui.globals.ui_input_thumbs.append(image)
                            is_first = False
                        face_set.ref_images.append(get_image_frame(filename))
            if len(face_set.faces) > 0:
                if len(face_set.faces) > 1:
                    face_set.AverageEmbeddings()
                roop.globals.INPUT_FACESETS.append(face_set)

        elif util.has_image_extension(source_path):
            progress(0, desc="Retrieving faces from image")
            roop.globals.source_path = source_path
            SELECTION_FACES_DATA = extract_face_images(
                roop.globals.source_path, (False, 0)
            )
            progress(0.5, desc="Retrieving faces from image")
            for f in SELECTION_FACES_DATA:
                face_set = FaceSet()
                face = f[0]
                face.mask_offsets = (0, 0, 0, 0, 1, 20)
                face_set.faces.append(face)
                face_set.ref_images.append(get_image_frame(roop.globals.source_path))
                image = util.convert_to_gradio(f[1])
                ui.globals.ui_input_thumbs.append(image)
                roop.globals.INPUT_FACESETS.append(face_set)

    progress(1.0)
    return gr.Column(visible=False), None, ui.globals.ui_input_thumbs, None


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index


def remove_selected_input_face():
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    if len(ui.globals.ui_input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)

    return ui.globals.ui_input_thumbs


def move_selected_input(button_text):
    if button_text == "⬅ Move left":
        if SELECTED_INPUT_FACE_INDEX <= 0:
            return ui.globals.ui_input_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_input_thumbs) <= SELECTED_INPUT_FACE_INDEX:
            return ui.globals.ui_input_thumbs
        offset = 1

    f = roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)
    roop.globals.INPUT_FACESETS.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    f = ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
    ui.globals.ui_input_thumbs.insert(SELECTED_INPUT_FACE_INDEX + offset, f)
    return ui.globals.ui_input_thumbs


def move_selected_target(button_text):
    if button_text == "⬅ Move left":
        if SELECTED_TARGET_FACE_INDEX <= 0:
            return ui.globals.ui_target_thumbs
        offset = -1
    else:
        if len(ui.globals.ui_target_thumbs) <= SELECTED_TARGET_FACE_INDEX:
            return ui.globals.ui_target_thumbs
        offset = 1

    f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    roop.globals.TARGET_FACES.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    f = ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    ui.globals.ui_target_thumbs.insert(SELECTED_TARGET_FACE_INDEX + offset, f)
    return ui.globals.ui_target_thumbs


def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index


def remove_selected_target_face():
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
    if len(ui.globals.ui_target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
    return ui.globals.ui_target_thumbs


def on_use_face_from_selected(files, frame_num):
    global IS_INPUT, SELECTION_FACES_DATA, SELECTED_FACE_INDEX

    IS_INPUT = False
    SELECTED_FACE_INDEX = 0
    thumbs = []

    roop.globals.target_path = _selected_target_path(files)
    if util.is_image(
        roop.globals.target_path
    ) and not roop.globals.target_path.lower().endswith(("gif")):
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (False, 0))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info("No faces detected!")
            roop.globals.target_path = None

    elif util.is_video(
        roop.globals.target_path
    ) or roop.globals.target_path.lower().endswith(("gif")):
        selected_frame = frame_num
        SELECTION_FACES_DATA = extract_face_images(
            roop.globals.target_path, (True, selected_frame)
        )
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = util.convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info("No faces detected!")
            roop.globals.target_path = None
    else:
        gr.Info("Unknown image/video type!")
        roop.globals.target_path = None

    if len(thumbs) == 1:
        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
        ui.globals.ui_target_thumbs.append(thumbs[0])
        return (
            gr.Row(visible=False),
            None,
            ui.globals.ui_target_thumbs,
            gr.Dropdown(value="Selected face"),
        )

    return (
        gr.Row(visible=True),
        thumbs,
        gr.Gallery(visible=True),
        gr.Dropdown(visible=True),
    )


def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index


def on_selected_face():
    if not SELECTION_FACES_DATA or SELECTED_FACE_INDEX >= len(SELECTION_FACES_DATA):
        return ui.globals.ui_input_thumbs, ui.globals.ui_target_thumbs, gr.Gallery(visible=True)
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = util.convert_to_gradio(fd[1])
    if IS_INPUT:
        face_set = FaceSet()
        fd[0].mask_offsets = (0, 0, 0, 0, 1, 20)
        face_set.faces.append(fd[0])
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(image)
        return (
            ui.globals.ui_input_thumbs,
            gr.Gallery(visible=True),
            gr.Dropdown(visible=True),
        )
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        ui.globals.ui_target_thumbs.append(image)
        return (
            gr.Gallery(visible=True),
            ui.globals.ui_target_thumbs,
            gr.Dropdown(value="Selected face"),
        )


#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])


def on_end_face_selection():
    return gr.Column(visible=False), None


def on_preview_frame_changed(
    swap_model,
    frame_num,
    files,
    fake_preview,
    enhancer,
    detection,
    face_distance,
    blend_ratio,
    selected_mask_engine,
    clip_text,
    no_face_action,
    vr_mode,
    auto_rotate,
    maskimage,
    show_face_area,
    restore_original_mouth,
    num_steps,
    upsample,
):
    global manual_masking, current_video_fps

    from roop.core import live_swap, get_processing_plugins

    manual_masking = False
    mask_offsets = (0, 0, 0, 0)
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0],
            "mask_offsets",
        ):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[
                0
            ].mask_offsets = mask_offsets
        mask_offsets = (
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        )

    timeinfo = "0:00:00"
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None, None, gr.Slider(info=timeinfo)

    filename = _selected_target_path(files)
    if util.is_video(filename) or filename.lower().endswith("gif"):
        current_frame = get_video_frame(filename, frame_num)
        if current_video_fps == 0:
            current_video_fps = 1
        secs = (frame_num - 1) / current_video_fps
        minutes = secs / 60
        secs = secs % 60
        hours = minutes / 60
        minutes = minutes % 60
        milliseconds = (secs - int(secs)) * 1000
        timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None, None, gr.Slider(info=timeinfo)

    if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
        return (
            util.convert_to_gradio(current_frame),
            gr.update(visible=False),
            gr.Slider(info=timeinfo),
        )

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = auto_rotate
    roop.globals.subsample_size = int(upsample[:3])

    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    face_index = SELECTED_INPUT_FACE_INDEX
    if len(roop.globals.INPUT_FACESETS) <= face_index:
        face_index = 0

    options = ProcessOptions(
        swap_model,
        get_processing_plugins(mask_engine, swap_model),
        roop.globals.distance_threshold,
        roop.globals.blend_ratio,
        roop.globals.face_swap_mode,
        face_index,
        clip_text,
        maskimage,
        num_steps,
        roop.globals.subsample_size,
        show_face_area,
        restore_original_mouth,
    )

    current_frame = live_swap(current_frame, options)
    if current_frame is None:
        return None, None, gr.Slider(info=timeinfo)
    return (
        gr.Image(value=util.convert_to_gradio(current_frame), visible=True),
        gr.ImageEditor(visible=False),
        gr.Slider(info=timeinfo),
    )


def map_mask_engine(selected_mask_engine, clip_text):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
            mask_engine = None
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


def on_toggle_masking(previewimage, mask):
    global manual_masking

    if not manual_masking and previewimage is None:
        gr.Warning("Show a preview frame first, then draw the mask on it.")
        return gr.Image(visible=True), gr.ImageEditor(visible=False)

    manual_masking = not manual_masking
    if manual_masking:
        # The editor passes None until it has held a value.
        layers = (mask or {}).get("layers") or []
        if len(layers) <= 1:
            layers = [create_blank_image(previewimage.shape[1], previewimage.shape[0])]
        return gr.Image(visible=False), gr.ImageEditor(
            value={"background": previewimage, "layers": layers, "composite": None},
            visible=True,
        )
    return gr.Image(visible=True), gr.ImageEditor(visible=False)


def gen_processing_text(start, end):
    return f"Processing frame range [{start} - {end}]"


def on_set_frame(sender: str, frame_num):
    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0, 0)

    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith("start"):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)

    return gen_processing_text(
        list_files_process[idx].startframe, list_files_process[idx].endframe
    )


def on_preview_mask(swap_model, frame_num, files, clip_text, mask_engine):
    from roop.core import live_swap, get_processing_plugins

    if (
        _swap_running()
        or files is None
        or selected_preview_index >= len(files)
        or clip_text is None
        or frame_num is None
    ):
        return None

    filename = _selected_target_path(files)
    if util.is_video(filename) or filename.lower().endswith("gif"):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None or mask_engine is None:
        return None
    if mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
        if clip_text is None or len(clip_text) < 1:
            mask_engine = None
    elif mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    options = ProcessOptions(
        swap_model,
        get_processing_plugins(mask_engine, swap_model),
        roop.globals.distance_threshold,
        roop.globals.blend_ratio,
        "all",
        0,
        clip_text,
        None,
        0,
        128,
        False,
        False,
        True,
    )

    current_frame = live_swap(current_frame, options)
    return util.convert_to_gradio(current_frame)


def test_all_swappers(
    frame_num,
    files,
    detection,
    upsample,
    selected_mask_engine,
    clip_text,
    no_face_action,
    vr_mode,
    auto_rotate,
    face_distance,
    blend_ratio,
    num_steps,
):
    from roop.core import live_swap, get_processing_plugins

    if files is None or len(files) < 1 or frame_num is None:
        return None

    filename = _selected_target_path(files)
    if util.is_video(filename) or filename.lower().endswith("gif"):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)

    if current_frame is None:
        return None

    output_dir = roop.globals.output_path

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    roop.globals.selected_enhancer = "None"
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = auto_rotate
    roop.globals.subsample_size = int(upsample[:3])

    frame_source = current_frame.copy()
    output_files = []
    for swap_model in model_swap_choices:
        print(f"[test_all_swappers] Testing {swap_model}...")
        options = ProcessOptions(
            swap_model,
            get_processing_plugins(None, swap_model),
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            SELECTED_INPUT_FACE_INDEX,
            clip_text,
            None,
            num_steps,
            roop.globals.subsample_size,
            False,
            False,
        )
        swapped_frame = live_swap(frame_source.copy(), options)
        if swapped_frame is None:
            print(f"[test_all_swappers] {swap_model} produced no output (no face?)")
            continue
        print(f"[test_all_swappers] {swap_model} done")
        safe_model = "".join(ch if ch.isalnum() else "_" for ch in swap_model).strip("_")
        target_basename = os.path.splitext(os.path.basename(filename))[0]
        output_filename = os.path.join(
            output_dir,
            f"{safe_model}-{target_basename}.{roop.globals.CFG.output_image_format}",
        )
        cv2.imwrite(output_filename, swapped_frame)
        output_files.append(output_filename)

    return output_files


def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    return ui.globals.ui_input_thumbs


def on_clear_destfiles():
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    return ui.globals.ui_target_thumbs, gr.Dropdown(value="First found"), ""


def index_of_no_face_action(dropdown_text):
    return no_face_choices.index(dropdown_text)


def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All input faces":
        return "all_input"
    elif dropdown_text == "All input faces (random)":
        return "all_random"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"

    return "all"


def _format_status_html(status):
    if not status.get("active"):
        return '<div style="padding:8px 12px;background:#1a1a2e;border:1px solid #333;border-radius:6px;font-family:monospace;color:#888;font-size:14px;">Idle</div>'
    percent = status.get("percent", 0)
    fps = status.get("fps", 0)
    eta = int(status.get("eta_seconds", 0))
    cur = status.get("current_frame", 0)
    total = status.get("total_frames", 0)
    h, rem = divmod(eta, 3600)
    m, s = divmod(rem, 60)
    eta_str = f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"
    bar_w = max(0, min(100, percent))
    return (
        '<div style="padding:8px 12px;background:#1a1a2e;border:1px solid #333;border-radius:6px;font-family:monospace;color:#e0e0e0;font-size:14px;">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'<span><b>{percent:5.1f}%</b> &nbsp; ({cur}/{total} frames)</span>'
        f'<span>FPS: <b>{fps:5.1f}</b> &nbsp; | &nbsp; ETA: <b>{eta_str}</b></span>'
        '</div>'
        '<div style="height:8px;background:#222;border-radius:4px;overflow:hidden;">'
        f'<div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,#4a9eff,#7b5cff);"></div>'
        '</div></div>'
    )


def start_swap(
    swap_model,
    output_method,
    enhancer,
    detection,
    keep_frames,
    wait_after_extraction,
    skip_audio,
    face_distance,
    blend_ratio,
    selected_mask_engine,
    clip_text,
    processing_method,
    no_face_action,
    vr_mode,
    autorotate,
    restore_original_mouth,
    num_swap_steps,
    upsample,
    imagemask,
    progress=gr.Progress(),
):
    from ui.main import prepare_environment
    from roop.core import batch_process_regular

    global _swap_thread

    idle_html = _format_status_html({"active": False})

    if list_files_process is None or len(list_files_process) <= 0:
        yield gr.Button(variant="primary"), None, None, idle_html
        return

    if roop.globals.CFG.clear_output:
        clean_dir(roop.globals.output_path)

    if not util.is_installed("ffmpeg"):
        msg = "ffmpeg is not installed! No video processing possible."
        gr.Warning(msg)

    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.subsample_size = int(upsample[:3])
    mask_engine = map_mask_engine(selected_mask_engine, clip_text)

    if roop.globals.face_swap_mode == "selected":
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error("No Target Face selected!")
            yield gr.Button(variant="primary"), None, None, idle_html
            return

    roop.globals.progress_status.update({
        "percent": 0, "fps": 0.0, "eta_seconds": 0,
        "current_frame": 0, "total_frames": 0,
        "start_time": time.time(), "active": True,
    })
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = (
        roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    )

    worker_exc = []

    def _worker():
        try:
            batch_process_regular(
                swap_model,
                output_method,
                list_files_process,
                mask_engine,
                clip_text,
                processing_method == "In-Memory processing",
                imagemask,
                restore_original_mouth,
                num_swap_steps,
                progress,
                SELECTED_INPUT_FACE_INDEX,
            )
        except Exception as e:
            worker_exc.append(e)

    t = threading.Thread(target=_worker, daemon=True)
    _swap_thread = t
    t.start()
    yield (
        gr.Button(variant="secondary", interactive=False),
        gr.Button(variant="primary", interactive=True),
        None,
        _format_status_html(roop.globals.progress_status),
    )

    while t.is_alive():
        t.join(timeout=0.5)
        yield (
            gr.Button(variant="secondary", interactive=False),
            gr.Button(variant="primary", interactive=True),
            None,
            _format_status_html(roop.globals.progress_status),
        )

    roop.globals.progress_status["active"] = False

    if worker_exc:
        raise worker_exc[0]

    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [str(item) for item in outdir.rglob("*") if item.is_file()]
    if len(outfiles) > 0:
        yield (
            gr.Button(variant="primary", interactive=True),
            gr.Button(variant="secondary", interactive=False),
            gr.Files(value=outfiles),
            idle_html,
        )
    else:
        yield (
            gr.Button(variant="primary", interactive=True),
            gr.Button(variant="secondary", interactive=False),
            None,
            idle_html,
        )


def stop_swap():
    roop.globals.processing = False
    roop.globals.progress_status["active"] = False
    gr.Info("Aborting processing - please wait for the remaining threads to be stopped")
    return (
        gr.Button(variant="primary", interactive=True),
        gr.Button(variant="secondary", interactive=False),
        _format_status_html({"active": False}),
    )


def on_swapper_weight_changed(weight):
    """Store the identity weight globally; ProcessOptions reads it from there."""
    roop.globals.swapper_weight = float(weight)


def on_fps_changed(fps):
    if (
        len(list_files_process) < 1
        or list_files_process[selected_preview_index].endframe < 1
    ):
        return
    list_files_process[selected_preview_index].fps = fps


def get_target_file_info_html(filename):
    if filename is None or not os.path.isfile(filename):
        return ""
    try:
        filesize = os.path.getsize(filename)
        if filesize > 1_000_000_000:
            size_str = f"{filesize / 1_000_000_000:.1f} GB"
        elif filesize > 1_000_000:
            size_str = f"{filesize / 1_000_000:.1f} MB"
        else:
            size_str = f"{filesize / 1_000:.1f} KB"

        if util.is_video(filename) or filename.lower().endswith("gif"):
            cap = cv2.VideoCapture(filename)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)])
                cap.release()
                if fps > 0 and total > 0:
                    duration_s = total / fps
                    mins, secs = divmod(int(duration_s), 60)
                    hrs, mins = divmod(mins, 60)
                    dur_str = f"{hrs}:{mins:02d}:{secs:02d}"
                else:
                    dur_str = "unknown"
                return (
                    f"<div style='padding:6px 10px;margin-top:2px;background:#1a1a2e;border:1px solid #333;border-radius:6px;"
                    f"font-size:13px;color:#ccc;line-height:1.6'>"
                    f"<b style='color:#7aa2f7'>Video</b> &nbsp; "
                    f"<b>{w}×{h}</b> &nbsp; "
                    f"<b>{fps:.2f}</b> fps &nbsp; "
                    f"{total} frames &nbsp; "
                    f"{dur_str} &nbsp; "
                    f"codec: {codec} &nbsp; "
                    f"{size_str}"
                    f"</div>"
                )
            cap.release()
        else:
            img = cv2.imread(filename)
            if img is not None:
                h, w = img.shape[:2]
                return (
                    f"<div style='padding:6px 10px;margin-top:2px;background:#1a1a2e;border:1px solid #333;border-radius:6px;"
                    f"font-size:13px;color:#ccc;line-height:1.6'>"
                    f"<b style='color:#7aa2f7'>Image</b> &nbsp; "
                    f"<b>{w}×{h}</b> &nbsp; "
                    f"{size_str}"
                    f"</div>"
                )
    except Exception:
        pass
    return ""


def on_destfiles_changed(destfiles):
    global selected_preview_index, current_video_fps

    list_files_process.clear()
    if destfiles is None or len(destfiles) < 1:
        return gr.Slider(value=1, maximum=2, info="0:00:00"), "", ""

    for f in destfiles:
        list_files_process.append(ProcessEntry(_fpath(f), 0, 0, 0))

    selected_preview_index = 0
    idx = selected_preview_index

    filename = list_files_process[idx].filename

    if util.is_video(filename) or filename.lower().endswith("gif"):
        total_frames = get_video_frame_total(filename)
        if total_frames is None or total_frames < 1:
            total_frames = 1
            gr.Warning(f"Corrupted video {filename}, can't detect number of frames!")
        else:
            current_video_fps = util.detect_fps(filename)
    else:
        total_frames = 1
    list_files_process[idx].endframe = total_frames
    info_html = get_target_file_info_html(filename)
    if total_frames > 1:
        return gr.Slider(
            value=1, maximum=max(2, total_frames), info="0:00:00"
        ), gen_processing_text(
            list_files_process[idx].startframe, list_files_process[idx].endframe
        ), info_html
    return gr.Slider(value=1, maximum=max(2, total_frames), info="0:00:00"), "", info_html


def on_destfiles_selected(evt: gr.SelectData):
    global selected_preview_index, current_video_fps

    if evt is not None:
        selected_preview_index = evt.index
    idx = selected_preview_index
    filename = list_files_process[idx].filename
    fps = list_files_process[idx].fps
    if util.is_video(filename) or filename.lower().endswith("gif"):
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename)
        if list_files_process[idx].endframe == 0:
            list_files_process[idx].endframe = total_frames
    else:
        total_frames = 1

    info_html = get_target_file_info_html(filename)
    if total_frames > 1:
        return (
            gr.Slider(
                # startframe 0 means "from the beginning"; the slider is
                # 1-based (minimum 1), and Gradio rejects an out-of-range
                # value on the next event that sends it.
                value=max(1, list_files_process[idx].startframe),
                maximum=max(2, total_frames),
                info="0:00:00",
            ),
            gen_processing_text(
                list_files_process[idx].startframe, list_files_process[idx].endframe
            ),
            fps,
            info_html,
        )
    return (
        gr.Slider(value=1, maximum=max(2, total_frames), info="0:00:00"),
        gen_processing_text(0, 0),
        fps,
        info_html,
    )


def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    filename = _fpath(files[selected_index])
    return display_output(filename)


def on_resultfiles_finished(files):
    selected_index = 0
    if files is None or len(files) < 1:
        return None, None

    filename = _fpath(files[selected_index])
    return display_output(filename)


def get_gradio_output_format():
    if roop.globals.CFG.output_image_format == "jpg":
        return "jpeg"
    return roop.globals.CFG.output_image_format


def display_output(filename):
    if util.is_video(filename) and roop.globals.CFG.output_show_video:
        playable = _browser_playable_path(filename)
        if playable is not None:
            return gr.Image(visible=False), gr.Video(visible=True, value=playable)

    if util.is_video(filename) or filename.lower().endswith("gif"):
        current_frame = get_video_frame(filename)
    else:
        current_frame = get_image_frame(filename)
    return gr.Image(
        visible=True, value=util.convert_to_gradio(current_frame)
    ), gr.Video(visible=False)


def _browser_playable_path(filename):
    """Path to play in the results panel, or None to show a still frame.

    Returns *filename* when the browser can decode it as-is.  Otherwise a
    re-encoded preview copy, but only when the user has opted in -- the
    extra encode pass never runs on its own.
    """
    import roop.util_ffmpeg as ffmpeg

    if ffmpeg.is_browser_compatible_video(filename):
        return filename

    if not roop.globals.CFG.output_reencode_preview:
        gr.Info(
            "This output uses a codec or container the browser cannot play, "
            "so a still frame is shown. Enable 'Re-encode unplayable video "
            "for preview' in Settings to preview it."
        )
        return None

    gr.Info("Output is not browser-playable — re-encoding a preview copy...")
    preview = ffmpeg.create_browser_preview(filename)
    if preview is None:
        gr.Warning("Preview re-encode failed — showing a still frame instead.")
        return None
    return preview
