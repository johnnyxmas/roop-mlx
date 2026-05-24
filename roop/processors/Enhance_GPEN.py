"""Enhance_GPEN -- MLX port pending.

Upstream supported three GPEN-BFR variants (512/1024/2048) selected by
the ``resolution`` plugin option.  Loaded via ONNX Runtime.
"""


class Enhance_GPEN:
    plugin_options: dict = None
    name = None
    devicename = None

    processorname = "gpen"
    type = "enhance"

    GPEN_DOWNLOAD_URLS = {
        512: "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
        1024: "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx",
        2048: "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_2048.onnx",
    }

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Enhance_GPEN has not been ported to MLX yet."
        )

    def Run(self, source_faceset, target_face, temp_frame):
        raise NotImplementedError(
            "Enhance_GPEN has not been ported to MLX yet."
        )

    def Release(self):
        return None
