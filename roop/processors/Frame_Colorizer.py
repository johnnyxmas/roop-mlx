"""Frame_Colorizer -- MLX port pending.

Upstream supported DeOldify artistic + stable (``deoldify_artistic.onnx``,
``deoldify_stable.onnx``) via ONNX Runtime.
"""


class Frame_Colorizer:
    plugin_options: dict = None
    devicename = None
    prev_type = None

    processorname = "deoldify"
    type = "frame_colorizer"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Frame_Colorizer has not been ported to MLX yet."
        )

    def Run(self, temp_frame):
        raise NotImplementedError(
            "Frame_Colorizer has not been ported to MLX yet."
        )

    def Release(self):
        return None

    def getProcessedResolution(self, width, height):
        return None
