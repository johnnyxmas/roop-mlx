"""Frame_Upscale -- MLX port pending.

Upstream supported ESRGAN x2/x4 and LSDIR x4 via ONNX Runtime.
"""


class Frame_Upscale:
    plugin_options: dict = None
    devicename = None
    prev_type = None

    processorname = "upscale"
    type = "frame_enhancer"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Frame_Upscale has not been ported to MLX yet."
        )

    def Run(self, temp_frame):
        raise NotImplementedError(
            "Frame_Upscale has not been ported to MLX yet."
        )

    def Release(self):
        return None

    def getProcessedResolution(self, width, height):
        return None
