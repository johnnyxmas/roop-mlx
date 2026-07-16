"""FaceSwapFaceFusion -- MLX backend.

Handles the FaceFusion / HyperSwap family (hyperswap_1a/1b/1c_256).
Unlike InsightFace inswapper, HyperSwap takes the raw normed_embedding
directly (no emap projection) and returns both the swapped face and a
blend mask (mask is not used by the caller).
"""

import os

import numpy as np

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path


class FaceSwapFaceFusion:
    plugin_options: dict = None
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
        if self.plugin_options is not None:
            if self.plugin_options.get("modelname") != plugin_options.get("modelname"):
                self.Release()

        self.plugin_options = plugin_options

        if getattr(self, "_session", None) is not None:
            return

        from roop.inference import create_session

        modelname = plugin_options.get("modelname", "HyperSwap 1B")
        config = self.FACE_SWAPPER_MODELS[modelname]
        self.swap_type = config["swap_type"]

        model_path = resolve_relative_path("../models/" + config["file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HyperSwap model not found: {model_path}")

        self._session = create_session(model_path)
        self._output_name = self._session.get_outputs()[0].name

    def Run(self, source_faceset, target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_faceset, "faces") and source_faceset.faces:
            source_face = source_faceset.faces[0]
        else:
            source_face = source_faceset

        source = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

        result = self._session.run(
            [self._output_name],
            {"source": source, "target": np.ascontiguousarray(temp_frame)},
        )
        return result[0]

    def Release(self):
        self._session = None
