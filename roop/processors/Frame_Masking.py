"""Frame_Masking -- MLX port pending.

Upstream loaded ``isnet-general-use.onnx`` via ONNX Runtime to generate
a background-removal mask.
"""


class Frame_Masking:
    plugin_options: dict = None
    devicename = None
    name = None

    processorname = "removebg"
    type = "frame_masking"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Frame_Masking has not been ported to MLX yet."
        )

    def Run(self, temp_frame):
        raise NotImplementedError(
            "Frame_Masking has not been ported to MLX yet."
        )

    def Release(self):
        return None
