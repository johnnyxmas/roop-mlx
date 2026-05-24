"""Enhance_RestoreFormerPPlus -- MLX port pending.

Upstream loaded ``restoreformer_plus_plus.onnx`` via ONNX Runtime.
"""


class Enhance_RestoreFormerPPlus:
    plugin_options: dict = None
    devicename = None
    name = None

    processorname = "restoreformer++"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "Enhance_RestoreFormerPPlus has not been ported to MLX yet."
        )

    def Run(self, source_faceset, target_face, temp_frame):
        raise NotImplementedError(
            "Enhance_RestoreFormerPPlus has not been ported to MLX yet."
        )

    def Release(self):
        return None
