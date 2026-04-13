import gradio as gr
import roop.globals
import ui.globals

available_themes = [
    "Default",
    "ocean",
    "citrus",
    "glass",
    "monochrome",
    "origin",
    "soft",
    "gradio/glass",
    "gradio/monochrome",
    "gradio/soft",
    "gstaff/xkcd",
    "freddyaboulton/dracula_revamped",
    "ysharma/steampunk",
]
image_formats = ["jpg", "png", "webp"]
video_formats = ["avi", "mkv", "mp4", "webm"]
video_codecs = [
    "auto", "libx264", "libx265", "libvpx-vp9", "h264_nvenc", "hevc_nvenc",
]

performance_profiles = ["memory_saver", "balanced", "max_speed"]
face_detectors = ["InsightFace", "YOLOv8n-Face"]

providerlist = None
settings_controls = []


TOOLTIPS = {
    "theme": [
        "Visual style for the Gradio UI.",
        "Change requires a full server restart to apply.",
    ],
    "server_share": [
        "Expose the UI via a public Gradio share URL.",
        "Anyone with the link can use your local GPU — share with care.",
    ],
    "clear_output": [
        "Delete everything in the output folder before each run starts.",
        "Useful for batch jobs where you don't want old results mixing in.",
    ],
    "output_template": [
        "Filename template for saved results.",
        "<code>{file}</code> = source filename, <code>{time}</code> = timestamp.",
        "File extension is added automatically based on output format.",
    ],
    "server_name": [
        "Host/IP the Gradio server binds to.",
        "Leave blank for localhost-only (127.0.0.1).",
        "Set to <code>0.0.0.0</code> to allow LAN access.",
    ],
    "server_port": [
        "TCP port the Gradio server listens on.",
        "Leave at 0 to let Gradio pick an available one.",
    ],
    "provider": [
        "ONNX Runtime execution provider for all models.",
        "<b>cuda</b>: NVIDIA GPUs (fastest on supported cards).",
        "<b>dml</b>: DirectML, for AMD/Intel GPUs on Windows.",
        "<b>rocm</b>: AMD GPUs on Linux.",
        "<b>cpu</b>: fallback, very slow.",
        "Requires restart.",
    ],
    "cuda_device_id": [
        "Which NVIDIA GPU to use when multiple are installed.",
        "Only visible on multi-GPU systems.",
        "Requires restart to take effect.",
    ],
    "multi_gpu": [
        "Split the frame workload across every detected CUDA GPU.",
        "Each GPU gets its own processor set — roughly linear scaling on "
        "batch jobs.",
        "Only shown when more than one CUDA GPU is detected.",
        "Requires restart.",
    ],
    "default_det_size": [
        "Use InsightFace's default detector input resolution (640×640).",
        "Uncheck to use a larger 1024×1024 detector input.",
        "Larger is slower but finds smaller/further faces.",
    ],
    "force_cpu": [
        "Run the face <i>analyser</i> (detection + embedding) on CPU even "
        "when a GPU provider is selected.",
        "Swap and enhancer models still use the GPU.",
        "Useful when GPU VRAM is tight — face analysis is relatively cheap "
        "on CPU.",
    ],
    "max_threads": [
        "Number of frames processed in parallel during video runs.",
        "Default: 3. Higher can saturate a fast GPU but also raises VRAM "
        "and system RAM use.",
        "Don't exceed the number of physical CPU cores.",
    ],
    "memory_limit": [
        "Hard cap on system RAM available to the process, in GB.",
        "0 = no limit (OS decides).",
        "Useful on Windows to prevent runaway swap when encoding long "
        "videos.",
    ],
    "output_image_format": [
        "File format for saved still-image results.",
        "<b>png</b>: lossless, larger files.",
        "<b>jpg</b>: lossy, much smaller.",
        "<b>webp</b>: good compression, supports lossless too.",
    ],
    "output_video_codec": [
        "Video codec used by FFmpeg when encoding final output.",
        "<b>auto</b>: NVENC if available, else libx264.",
        "<b>libx264/libx265</b>: CPU-encoded H.264/H.265. High quality, "
        "slower.",
        "<b>h264_nvenc/hevc_nvenc</b>: GPU-encoded. Much faster, slightly "
        "lower quality per bitrate.",
        "<b>libvpx-vp9</b>: WebM/VP9, CPU-only.",
    ],
    "output_video_format": [
        "Container for the final encoded video.",
        "<b>mp4</b>: most compatible.",
        "<b>mkv</b>: best for odd codec/audio combinations.",
        "<b>avi</b>: legacy.",
        "<b>webm</b>: pairs with libvpx-vp9.",
    ],
    "video_quality": [
        "FFmpeg CRF value. <b>Lower = higher quality</b>, larger files.",
        "Default: 14 (visually near-lossless on x264).",
        "Typical range: 18–28. Above 28 shows compression artifacts.",
        "Ignored by NVENC (which uses its own quality model).",
    ],
    "use_os_temp_folder": [
        "Store extracted frames in the OS temp directory instead of the "
        "project folder.",
        "Helps on small system drives if your project lives on a larger "
        "disk.",
        "Temp files are cleared on reboot or by the Clean temp folder "
        "button.",
    ],
    "output_show_video": [
        "Re-encode the final video so it can be previewed in the browser.",
        "Adds an extra encode pass — disable if you don't need in-browser "
        "playback.",
    ],
    "performance_profile": [
        "<b>memory_saver</b> — safest for ≤6 GB VRAM. Disables ORT memory "
        "arena/pattern/reuse. Face detection runs every frame. Current "
        "default.",
        "<b>balanced</b> — for ≥8 GB cards. Enables ORT allocators, "
        "preallocates CUDA input buffers, reuses the previous frame's "
        "face boxes every other frame. Typically 20–40% faster, ~200–"
        "500 MB extra VRAM.",
        "<b>max_speed</b> — for ≥12 GB cards. Adds TensorRT (FP16) and a "
        "larger detector stride.",
        "<b>⚠ First run on max_speed is slow:</b> TensorRT has to build "
        "an engine per model (2–10 minutes, app may appear frozen). "
        "Cached under <code>models/trt_cache/</code> — subsequent runs "
        "are fast.",
        "Cache is invalidated by driver/ORT upgrades or input resolution "
        "changes.",
    ],
    "detector_stride": [
        "How often to run full face detection in <b>max_speed</b> mode.",
        "1 = detect every frame (safest).",
        "3–5 = reuse previous boxes for intermediate frames (faster).",
        "Intermediate frames now use optical-flow tracking for smooth "
        "results even at high strides (10–15).",
        "A scene-cut check forces redetection when the frame changes "
        "drastically, so 3 is a safe default for talking-head video.",
        "Ignored unless profile is max_speed.",
    ],
    "face_detector": [
        "Face detection backend used to find faces in each frame.",
        "<b>InsightFace</b>: Buffalo_l RetinaFace — accurate, moderate "
        "speed. Default.",
        "<b>YOLOv8n-Face</b>: YOLO-based detector — 3–5× faster, "
        "slightly less accurate on tiny/occluded faces.",
        "Both backends produce the same face data for the swap pipeline.",
    ],
    "use_gpu_decode": [
        "Use FFmpeg hardware-accelerated video decoding (NVDEC / DXVA2).",
        "Faster than OpenCV CPU decode, especially for high-resolution "
        "video (1080p+).",
        "Falls back to software decode automatically if no hardware "
        "decoder is available.",
    ],
}



def _tip_html(bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    body = f"<ul>{items}</ul>"
    return (
        '<span class="roop-tip" tabindex="0">?'
        f'<span class="roop-tip-body">{body}</span>'
        "</span>"
    )


def tip(key):
    return gr.HTML(_tip_html(TOOLTIPS[key]), elem_classes=["roop-tip-wrap"])


def settings_tab():
    from roop.core import suggest_execution_providers
    from roop.utilities import get_cuda_device_list

    global providerlist

    providerlist = suggest_execution_providers()
    cuda_devices = get_cuda_device_list()
    with gr.Tab("⚙ Settings"):
        with gr.Row():
            with gr.Column():
                themes = gr.Dropdown(
                    available_themes,
                    label="Theme",
                    value=roop.globals.CFG.selected_theme,
                )
                tip("theme")
            with gr.Column():
                settings_controls.append(
                    gr.Checkbox(
                        label="Public Server",
                        value=roop.globals.CFG.server_share,
                        elem_id="server_share",
                        interactive=True,
                    )
                )
                tip("server_share")
                settings_controls.append(
                    gr.Checkbox(
                        label="Clear output folder before each run",
                        value=roop.globals.CFG.clear_output,
                        elem_id="clear_output",
                        interactive=True,
                    )
                )
                tip("clear_output")
                output_template = gr.Textbox(
                    label="Filename Output Template",
                    lines=1,
                    placeholder="{file}_{time}",
                    value=roop.globals.CFG.output_template,
                )
                tip("output_template")
            with gr.Column():
                input_server_name = gr.Textbox(
                    label="Server Name",
                    lines=1,
                    value=roop.globals.CFG.server_name,
                )
                tip("server_name")
            with gr.Column():
                input_server_port = gr.Number(
                    label="Server Port",
                    precision=0,
                    value=roop.globals.CFG.server_port,
                )
                tip("server_port")
        with gr.Row():
            with gr.Column():
                settings_controls.append(
                    gr.Dropdown(
                        providerlist,
                        label="Provider",
                        value=roop.globals.CFG.provider,
                        elem_id="provider",
                        interactive=True,
                    )
                )
                tip("provider")
                current_cuda = None
                if cuda_devices and roop.globals.CFG.cuda_device_id < len(
                    cuda_devices
                ):
                    raw = cuda_devices[roop.globals.CFG.cuda_device_id]
                    current_cuda = (
                        f"{roop.globals.CFG.cuda_device_id}: "
                        f"{raw.split(': ', 1)[1]}"
                    )
                cuda_device_id = gr.Dropdown(
                    choices=cuda_devices,
                    label="CUDA Device",
                    value=current_cuda,
                    interactive=bool(cuda_devices),
                    visible=bool(cuda_devices),
                )
                if cuda_devices:
                    tip("cuda_device_id")
                settings_controls.append(
                    gr.Checkbox(
                        label="Use all GPUs (multi-GPU)",
                        value=roop.globals.CFG.multi_gpu,
                        elem_id="multi_gpu",
                        interactive=True,
                        visible=len(cuda_devices) > 1,
                    )
                )
                if len(cuda_devices) > 1:
                    tip("multi_gpu")
                chk_det_size = gr.Checkbox(
                    label="Use default Det-Size",
                    value=True,
                    elem_id="default_det_size",
                    interactive=True,
                )
                tip("default_det_size")
                settings_controls.append(
                    gr.Dropdown(
                        face_detectors,
                        label="Face Detector",
                        value=roop.globals.CFG.face_detector,
                        elem_id="face_detector",
                        interactive=True,
                    )
                )
                tip("face_detector")
                settings_controls.append(
                    gr.Checkbox(
                        label="GPU Video Decode (NVDEC)",
                        value=roop.globals.CFG.use_gpu_decode,
                        elem_id="use_gpu_decode",
                        interactive=True,
                    )
                )
                tip("use_gpu_decode")
                settings_controls.append(
                    gr.Checkbox(
                        label="Force CPU for Face Analyser",
                        value=roop.globals.CFG.force_cpu,
                        elem_id="force_cpu",
                        interactive=True,
                    )
                )
                tip("force_cpu")
                max_threads = gr.Slider(
                    1,
                    32,
                    value=roop.globals.CFG.max_threads,
                    label="Max. Number of Threads",
                    step=1.0,
                    interactive=True,
                )
                tip("max_threads")
            with gr.Column():
                memory_limit = gr.Slider(
                    0,
                    128,
                    value=roop.globals.CFG.memory_limit,
                    label="Max. Memory to use (Gb)",
                    step=1.0,
                    interactive=True,
                )
                tip("memory_limit")
                settings_controls.append(
                    gr.Dropdown(
                        image_formats,
                        label="Image Output Format",
                        value=roop.globals.CFG.output_image_format,
                        elem_id="output_image_format",
                        interactive=True,
                    )
                )
                tip("output_image_format")
            with gr.Column():
                settings_controls.append(
                    gr.Dropdown(
                        video_codecs,
                        label="Video Codec",
                        value=roop.globals.CFG.output_video_codec,
                        elem_id="output_video_codec",
                        interactive=True,
                    )
                )
                tip("output_video_codec")
                settings_controls.append(
                    gr.Dropdown(
                        video_formats,
                        label="Video Output Format",
                        value=roop.globals.CFG.output_video_format,
                        elem_id="output_video_format",
                        interactive=True,
                    )
                )
                tip("output_video_format")
                video_quality = gr.Slider(
                    0,
                    100,
                    value=roop.globals.CFG.video_quality,
                    label="Video Quality (crf)",
                    step=1.0,
                    interactive=True,
                )
                tip("video_quality")
            with gr.Column():
                with gr.Group():
                    settings_controls.append(
                        gr.Checkbox(
                            label="Use OS temp folder",
                            value=roop.globals.CFG.use_os_temp_folder,
                            elem_id="use_os_temp_folder",
                            interactive=True,
                        )
                    )
                    tip("use_os_temp_folder")
                    settings_controls.append(
                        gr.Checkbox(
                            label="Show video in browser (re-encodes output)",
                            value=roop.globals.CFG.output_show_video,
                            elem_id="output_show_video",
                            interactive=True,
                        )
                    )
                    tip("output_show_video")
                settings_controls.append(
                    gr.Dropdown(
                        performance_profiles,
                        label="Performance Profile",
                        value=roop.globals.CFG.performance_profile,
                        elem_id="performance_profile",
                        interactive=True,
                    )
                )
                tip("performance_profile")
                detector_stride_slider = gr.Slider(
                    1,
                    10,
                    value=roop.globals.CFG.detector_stride,
                    label="Detector Stride (max_speed only)",
                    step=1.0,
                    interactive=True,
                )
                tip("detector_stride")
                button_apply_restart = gr.Button(
                    "Restart Server", variant="primary"
                )
                button_clean_temp = gr.Button("Clean temp folder")
                button_apply_settings = gr.Button("Apply Settings")

    cuda_device_id.input(
        fn=lambda a, b="cuda_device_id": on_settings_changed_misc(
            int(a.split(":")[0]), b
        ),
        inputs=[cuda_device_id],
    )
    chk_det_size.select(fn=on_option_changed)

    for s in settings_controls:
        s.select(fn=on_settings_changed)
    max_threads.input(
        fn=lambda a, b="max_threads": on_settings_changed_misc(a, b),
        inputs=[max_threads],
    )
    memory_limit.input(
        fn=lambda a, b="memory_limit": on_settings_changed_misc(a, b),
        inputs=[memory_limit],
    )
    video_quality.input(
        fn=lambda a, b="video_quality": on_settings_changed_misc(a, b),
        inputs=[video_quality],
    )
    detector_stride_slider.input(
        fn=lambda a, b="detector_stride": on_settings_changed_misc(
            int(a), b
        ),
        inputs=[detector_stride_slider],
    )

    button_clean_temp.click(fn=clean_temp)
    button_apply_settings.click(
        apply_settings,
        inputs=[themes, input_server_name, input_server_port, output_template],
    )
    button_apply_restart.click(restart)


def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f"Unhandled Setting for {evt.target}")


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return

    raise gr.Error(f"Unhandled Setting for {evt.target}")


def clean_temp():
    from ui.main import prepare_environment
    from roop.core import cleanup_cache_folders

    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs = []

    cleanup_cache_folders()

    prepare_environment()
    gr.Info("Temp Files removed")
    return None, None, None, None


def apply_settings(themes, input_server_name, input_server_port, output_template):
    from ui.main import show_msg

    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.output_template = output_template
    roop.globals.CFG.save()
    show_msg("Settings saved")


def restart():
    ui.globals.ui_restart_server = True
