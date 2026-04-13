import os
import time
import gradio as gr
import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None


def _purge_corrupt_temp_images(temp_dir: str):
    """Remove 0-byte or unreadable image files left by previous sessions.

    Gradio caches ImageEditor layers (background.png etc.) to disk.  If a
    prior session wrote a corrupt file (e.g. after a codec assertion crash),
    PIL raises UnidentifiedImageError during input preprocessing and crashes
    the event handler before our code even runs.  Cleaning them up at startup
    prevents the stale cache from causing crashes in the new session.
    """
    import glob
    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    removed = 0
    for pattern in image_exts:
        for path in glob.glob(os.path.join(temp_dir, "**", pattern), recursive=True):
            try:
                if os.path.getsize(path) == 0:
                    os.remove(path)
                    removed += 1
                    continue
                from PIL import Image
                with Image.open(path):
                    pass  # valid file
            except Exception:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
    if removed:
        print(f"[startup] Removed {removed} corrupt image file(s) from temp dir")


def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(
            os.path.join(os.getcwd(), "temp")
        )
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
    _purge_corrupt_temp_images(os.environ["TEMP"])


def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)

    run_server = True
    uii.ui_restart_server = False
    mycss = """
        span {color: var(--block-info-text-color)}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
        .image-container.svelte-1l6wqyv {height: 100%}

    """

    while run_server:
        if roop.globals.CFG.provider == "cuda" and util.has_cuda_device() == False:
            roop.globals.CFG.provider = "cpu"
        roop.globals.cuda_device_id = roop.globals.CFG.cuda_device_id
        roop.globals.multi_gpu = roop.globals.CFG.multi_gpu
        roop.globals.execution_providers = decode_execution_providers(
            [roop.globals.CFG.provider]
        )
        gputype = util.get_device()
        if gputype == "cuda":
            util.print_cuda_info()
        print(f"Using provider {roop.globals.execution_providers} - Device:{gputype}")

        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == "0.0.0.0" else True
        _BUILTIN_THEMES = {
            "ocean": gr.themes.Ocean(),
            "base": gr.themes.Base(),
            "citrus": gr.themes.Citrus(),
            "default": gr.themes.Default(),
            "glass": gr.themes.Glass(),
            "monochrome": gr.themes.Monochrome(),
            "origin": gr.themes.Origin(),
            "soft": gr.themes.Soft(),
        }
        _theme_val = roop.globals.CFG.selected_theme or "Default"
        theme_obj = _BUILTIN_THEMES.get(_theme_val.lower(), _theme_val)
        with gr.Blocks(
            title=f"{roop.metadata.name} {roop.metadata.version}",
            theme=theme_obj,
            css=mycss,
            delete_cache=(60, 86400),
        ) as ui:
            with gr.Row(variant="compact"):
                gr.Markdown(
                    f"### [{roop.metadata.name} {roop.metadata.version}](https://gitlab.com/roop-unleashed/roop-unleashed)"
                )
                gr.HTML(util.create_version_html(), elem_id="versions")
            faceswap_tab()
            livecam_tab()
            facemgr_tab()
            extras_tab()
            settings_tab()
        launch_browser = roop.globals.CFG.launch_browser

        uii.ui_restart_server = False
        try:
            ui.queue().launch(
                inbrowser=launch_browser,
                server_name=server_name,
                server_port=server_port,
                share=roop.globals.CFG.server_share,
                ssl_verify=ssl_verify,
                prevent_thread_lock=True,
                show_error=True,
            )
        except Exception as e:
            print(f"Exception {e} when launching Gradio Server!")
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)
