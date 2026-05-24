"""FaceSwapFaceFusion -- MLX port pending.

Handles the FaceFusion / HyperSwap family (``hyperswap_1a/1b/1c_256``)
in the upstream repo via an ONNX Runtime path with custom pre/post-
processing (different mean/std than InsightFace inswapper).

The model catalogue (``FACE_SWAPPER_MODELS``) is preserved so the
UI dropdown that lists swap models still works -- only Initialize/Run
raise.
"""


class FaceSwapFaceFusion:
    plugin_options: dict = None
    devicename: str = None
    device_id: int = 0
    swap_type: str = None

    processorname = "faceswap_ff"
    type = "swap"

    FACE_SWAPPER_MODELS = {
        "HyperSwap 1A": {
            "file": "hyperswap_1a_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
            "crossface": None,
            "crossface_url": None,
            "swap_type": "hyperswap",
            "size": 256,
        },
        "HyperSwap 1B": {
            "file": "hyperswap_1b_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx",
            "crossface": None,
            "crossface_url": None,
            "swap_type": "hyperswap",
            "size": 256,
        },
        "HyperSwap 1C": {
            "file": "hyperswap_1c_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx",
            "crossface": None,
            "crossface_url": None,
            "swap_type": "hyperswap",
            "size": 256,
        },
    }

    def Initialize(self, plugin_options: dict):
        raise NotImplementedError(
            "FaceSwapFaceFusion has not been ported to MLX yet."
        )

    def Run(self, source_face, target_face, temp_frame):
        raise NotImplementedError(
            "FaceSwapFaceFusion has not been ported to MLX yet."
        )

    def Release(self):
        return None
