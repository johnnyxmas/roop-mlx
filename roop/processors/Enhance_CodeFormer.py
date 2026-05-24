"""Enhance_CodeFormer -- MLX port pending.

Upstream loaded ``CodeFormer/CodeFormerv0.1.onnx`` via ONNX Runtime
with the CodeFormer-specific weight (``w``) input.
"""


class Enhance_CodeFormer:
    plugin_options: dict = None
    name = None
    devicename = None

    processorname = "codeformer"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Enhance_CodeFormer has not been ported to MLX yet."
        )

    def Run(self, source_faceset, target_face, temp_frame):
        raise NotImplementedError(
            "Enhance_CodeFormer has not been ported to MLX yet."
        )

    def Release(self):
        return None
