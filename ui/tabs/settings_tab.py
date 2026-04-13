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

providerlist = None
settings_controls = []



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
            with gr.Column():
                settings_controls.append(
                    gr.Checkbox(
                        label="Public Server",
                        value=roop.globals.CFG.server_share,
                        elem_id="server_share",
                        interactive=True,
                    )
                )
                settings_controls.append(
                    gr.Checkbox(
                        label="Clear output folder before each run",
                        value=roop.globals.CFG.clear_output,
                        elem_id="clear_output",
                        interactive=True,
                    )
                )
                output_template = gr.Textbox(
                    label="Filename Output Template",
                    lines=1,
                    placeholder="{file}_{time}",
                    value=roop.globals.CFG.output_template,
                )
            with gr.Column():
                input_server_name = gr.Textbox(
                    label="Server Name",
                    lines=1,
                    value=roop.globals.CFG.server_name,
                )
            with gr.Column():
                input_server_port = gr.Number(
                    label="Server Port",
                    precision=0,
                    value=roop.globals.CFG.server_port,
                )
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
                settings_controls.append(
                    gr.Checkbox(
                        label="Use all GPUs (multi-GPU)",
                        value=roop.globals.CFG.multi_gpu,
                        elem_id="multi_gpu",
                        interactive=True,
                        visible=len(cuda_devices) > 1,
                    )
                )
                chk_det_size = gr.Checkbox(
                    label="Use default Det-Size",
                    value=True,
                    elem_id="default_det_size",
                    interactive=True,
                )
                settings_controls.append(
                    gr.Checkbox(
                        label="GPU Video Decode (NVDEC)",
                        value=roop.globals.CFG.use_gpu_decode,
                        elem_id="use_gpu_decode",
                        interactive=True,
                    )
                )
                settings_controls.append(
                    gr.Checkbox(
                        label="Force CPU for Face Analyser",
                        value=roop.globals.CFG.force_cpu,
                        elem_id="force_cpu",
                        interactive=True,
                    )
                )
                max_threads = gr.Slider(
                    1,
                    32,
                    value=roop.globals.CFG.max_threads,
                    label="Max. Number of Threads",
                    step=1.0,
                    interactive=True,
                )
            with gr.Column():
                memory_limit = gr.Slider(
                    0,
                    128,
                    value=roop.globals.CFG.memory_limit,
                    label="Max. Memory to use (Gb)",
                    step=1.0,
                    interactive=True,
                )
                settings_controls.append(
                    gr.Dropdown(
                        image_formats,
                        label="Image Output Format",
                        value=roop.globals.CFG.output_image_format,
                        elem_id="output_image_format",
                        interactive=True,
                    )
                )
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
                settings_controls.append(
                    gr.Dropdown(
                        video_formats,
                        label="Video Output Format",
                        value=roop.globals.CFG.output_video_format,
                        elem_id="output_video_format",
                        interactive=True,
                    )
                )
                video_quality = gr.Slider(
                    0,
                    100,
                    value=roop.globals.CFG.video_quality,
                    label="Video Quality (crf)",
                    step=1.0,
                    interactive=True,
                )
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
                    settings_controls.append(
                        gr.Checkbox(
                            label="Show video in browser (re-encodes output)",
                            value=roop.globals.CFG.output_show_video,
                            elem_id="output_show_video",
                            interactive=True,
                        )
                    )
                settings_controls.append(
                    gr.Dropdown(
                        performance_profiles,
                        label="Performance Profile",
                        value=roop.globals.CFG.performance_profile,
                        elem_id="performance_profile",
                        interactive=True,
                    )
                )
                detector_stride_slider = gr.Slider(
                    1,
                    10,
                    value=roop.globals.CFG.detector_stride,
                    label="Detector Stride (max_speed only)",
                    step=1.0,
                    interactive=True,
                )
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
