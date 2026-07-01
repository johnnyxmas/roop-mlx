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
    "auto", "libx264", "libx265", "libvpx-vp9",
    "h264_videotoolbox", "hevc_videotoolbox",
]

settings_controls = []


def settings_tab():
    with gr.Tab("⚙ Settings"):
        with gr.Row():
            with gr.Column():
                themes = gr.Dropdown(
                    available_themes,
                    label="Theme",
                    info="Color scheme for this interface. Cosmetic only; applies after restart.",
                    value=roop.globals.CFG.selected_theme,
                )
            with gr.Column():
                settings_controls.append(
                    gr.Checkbox(
                        label="Public Server",
                        info="Create a shareable public link so others can reach this app. Leave off for local-only use.",
                        value=roop.globals.CFG.server_share,
                        elem_id="server_share",
                        interactive=True,
                    )
                )
                settings_controls.append(
                    gr.Checkbox(
                        label="Clear output folder before each run",
                        info="Delete previous results before starting a new job so the output folder only holds the latest run.",
                        value=roop.globals.CFG.clear_output,
                        elem_id="clear_output",
                        interactive=True,
                    )
                )
                output_template = gr.Textbox(
                    label="Filename Output Template",
                    info="Naming pattern for saved files. Placeholders: {file} = original name, {time} = timestamp.",
                    lines=1,
                    placeholder="{file}_{time}",
                    value=roop.globals.CFG.output_template,
                )
            with gr.Column():
                input_server_name = gr.Textbox(
                    label="Server Name",
                    info="Network address to bind to. Blank = localhost (this Mac only). Use 0.0.0.0 to allow other devices on your network.",
                    lines=1,
                    value=roop.globals.CFG.server_name,
                )
            with gr.Column():
                input_server_port = gr.Number(
                    label="Server Port",
                    info="Network port for the web interface. Leave at 0 to pick one automatically.",
                    precision=0,
                    value=roop.globals.CFG.server_port,
                )
        with gr.Row():
            with gr.Column():
                chk_det_size = gr.Checkbox(
                    label="Use default Det-Size",
                    info="Use the standard face-detection resolution (640×640). Leave on unless you need a different size.",
                    value=True,
                    elem_id="default_det_size",
                    interactive=True,
                )
            with gr.Column():
                memory_limit = gr.Slider(
                    0,
                    128,
                    value=roop.globals.CFG.memory_limit,
                    label="Max. Memory to use (Gb)",
                    info="Upper limit on system RAM the app may use. 0 = no limit.",
                    step=1.0,
                    interactive=True,
                )
                settings_controls.append(
                    gr.Dropdown(
                        image_formats,
                        label="Image Output Format",
                        info="File type for saved images. jpg = smaller, png = lossless, webp = small with transparency.",
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
                        info="How output video is compressed. 'auto' picks h264_videotoolbox if VideoToolbox is available, otherwise libx264.",
                        value=roop.globals.CFG.output_video_codec,
                        elem_id="output_video_codec",
                        interactive=True,
                    )
                )
                settings_controls.append(
                    gr.Dropdown(
                        video_formats,
                        label="Video Output Format",
                        info="Container/file type for saved video. mp4 is the most widely compatible.",
                        value=roop.globals.CFG.output_video_format,
                        elem_id="output_video_format",
                        interactive=True,
                    )
                )
                video_quality = gr.Slider(
                    0,
                    100,
                    value=roop.globals.CFG.video_quality,
                    label="Video Quality (0=worst, 100=best)",
                    info="FFmpeg CRF/quality value: lower = better quality and larger files. Around 14-23 is typical.",
                    step=1.0,
                    interactive=True,
                )
            with gr.Column():
                with gr.Group():
                    settings_controls.append(
                        gr.Checkbox(
                            label="Use OS temp folder",
                            info="Store temporary working files in the system temp directory instead of the app folder. Useful if the app drive is low on space.",
                            value=roop.globals.CFG.use_os_temp_folder,
                            elem_id="use_os_temp_folder",
                            interactive=True,
                        )
                    )
                    settings_controls.append(
                        gr.Checkbox(
                            label="Show video in browser (re-encodes output)",
                            info="Preview the finished video in the browser. This re-encodes the file, which adds extra processing time.",
                            value=roop.globals.CFG.output_show_video,
                            elem_id="output_show_video",
                            interactive=True,
                        )
                    )
                button_apply_restart = gr.Button(
                    "Restart Server", variant="primary"
                )
                button_clean_temp = gr.Button("Clean temp folder")
                button_apply_settings = gr.Button("Apply Settings")

    chk_det_size.select(fn=on_option_changed)

    for s in settings_controls:
        s.select(fn=on_settings_changed)
    memory_limit.input(
        fn=lambda a, b="memory_limit": on_settings_changed_misc(a, b),
        inputs=[memory_limit],
    )
    video_quality.input(
        fn=lambda a, b="video_quality": on_settings_changed_misc(a, b),
        inputs=[video_quality],
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
