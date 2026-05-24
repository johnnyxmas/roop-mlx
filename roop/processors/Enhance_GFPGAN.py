"""Enhance_GFPGAN -- MLX port pending.

Upstream loaded ``GFPGANv1.4.onnx`` via ONNX Runtime and ran a single
512x512 face restoration pass.  The pre/post processing
(BGR->RGB, normalize to [-1,1], CHW transpose) is pure NumPy/OpenCV
and will be reused once the model is wired to MLX.
"""


class Enhance_GFPGAN:
    plugin_options: dict = None
    name = None
    devicename = None

    processorname = "gfpgan"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Enhance_GFPGAN has not been ported to MLX yet."
        )

    def Run(self, source_faceset, target_face, temp_frame):
        raise NotImplementedError(
            "Enhance_GFPGAN has not been ported to MLX yet."
        )

    def Release(self):
        return None
