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
        app_temp = os.path.abspath(os.path.join(os.getcwd(), "temp"))
        os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = app_temp
    import tempfile
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = temp_dir
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"
    _purge_corrupt_temp_images(temp_dir)


def run():
    from roop.core import set_display_ui

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

        html, body { background: #0f0f0f !important; color: #e0e0e0 !important; }

        /* Override Gradio 5 design tokens — these beat Svelte-scoped selectors */
        :root, .gradio-container {
            /* Primary fill tokens (used by block wrappers, headers, rows) */
            --background-fill-primary: #1a1a1a;
            --background-fill-secondary: #111;

            --body-background-fill: #0f0f0f;
            --body-text-color: #e0e0e0;
            --body-text-color-subdued: #aaa;

            --block-background-fill: #1a1a1a;
            --block-border-color: #333;
            --block-label-background-fill: #1a1a1a;
            --block-label-text-color: #aaa;
            --block-title-text-color: #ccc;
            --block-info-text-color: #aaa;

            --panel-background-fill: #111;
            --panel-border-color: #333;

            --input-background-fill: #222;
            --input-border-color: #444;
            --input-placeholder-color: #666;
            --input-text-color: #e0e0e0;

            --button-secondary-background-fill: #2a2a2a;
            --button-secondary-background-fill-hover: #383838;
            --button-secondary-border-color: #555;
            --button-secondary-text-color: #e0e0e0;

            --button-primary-background-fill: #1a7a7a;
            --button-primary-background-fill-hover: #1a9090;
            --button-primary-border-color: #1a9090;
            --button-primary-text-color: #fff;

            --button-cancel-background-fill: #5a1a1a;
            --button-cancel-text-color: #ffaaaa;

            --checkbox-background-color: #222;
            --checkbox-border-color: #555;
            --checkbox-label-background-fill: #1a1a1a;
            --checkbox-label-text-color: #ccc;

            --slider-color: #1a9090;
            --table-even-background-fill: #1a1a1a;
            --table-odd-background-fill: #222;
            --table-row-focus: #2a2a2a;

            --color-accent: #1a9090;
            --color-accent-soft: #1a404040;
            --link-text-color: #4fc3c3;
            --link-text-color-hover: #6fd5d5;

            /* Neutral scale used by Base theme for backgrounds */
            --neutral-50: #111;
            --neutral-100: #1a1a1a;
            --neutral-200: #222;
            --neutral-300: #333;
            --neutral-400: #555;
            --neutral-500: #888;
            --neutral-600: #aaa;
            --neutral-700: #ccc;
            --neutral-800: #e0e0e0;
            --neutral-900: #f0f0f0;
            --neutral-950: #f8f8f8;
        }

        /* Hard overrides for elements that ignore the tokens */
        html, body, .gradio-container { background: #0f0f0f !important; color: #e0e0e0 !important; }
        .block, fieldset, .form { background: #1a1a1a !important; border-color: #333 !important; }
        .gap, .gap-hide { background: #111 !important; }
        .compact { background: #0f0f0f !important; }

        /* Tab nav bar */
        .tab-nav { background: #0f0f0f !important; border-bottom-color: #333 !important; }
        .tab-nav button { background: transparent !important; color: #aaa !important; border-color: transparent !important; }
        .tab-nav button.selected { color: #fff !important; border-color: #1a9090 !important; }

        /* Accordion / summary */
        .label-wrap, .label-wrap span { background: #1a1a1a !important; color: #ccc !important; }
        details > summary { background: #222 !important; border-color: #444 !important; color: #ccc !important; }

        /* Dropdown menus */
        .wrap-inner, ul.options { background: #1a1a1a !important; border-color: #444 !important; }
        ul.options li { color: #e0e0e0 !important; }
        ul.options li:hover, ul.options li.selected { background: #2a2a2a !important; }

        /* Inputs */
        input[type=number], input[type=text], textarea, select {
            background: #222 !important; color: #e0e0e0 !important; border-color: #444 !important;
        }
    """

    while run_server:
        gputype = util.get_device()
        print(f"Using device: {gputype}")

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
        _theme_val = roop.globals.CFG.selected_theme or "base"
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
