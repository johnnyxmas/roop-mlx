"""Mask_XSeg -- MLX port pending.

Upstream loaded ``xseg.onnx`` (DeepFaceLab XSeg) via ONNX Runtime.
"""


class Mask_XSeg:
    plugin_options: dict = None

    processorname = "mask_xseg"
    type = "mask"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Mask_XSeg has not been ported to MLX yet."
        )

    def Run(self, frame, masking_text=None):
        raise NotImplementedError(
            "Mask_XSeg has not been ported to MLX yet."
        )

    def Release(self):
        return None
