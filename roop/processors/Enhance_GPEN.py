"""Enhance_GPEN -- MLX backend.

GPEN-BFR-{512,1024,2048}: GAN prior face restoration.  Resolution is
selected via the ``resolution`` plugin option (default 512).  Input is
an N×N BGR face crop; output is an N×N BGR restored face.
"""

import cv2
import numpy as np

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path

_MODEL_NAMES = {
    512:  "GPEN-BFR-512.onnx",
    1024: "GPEN-BFR-1024.onnx",
    2048: "GPEN-BFR-2048.onnx",
}


class Enhance_GPEN:
    plugin_options: dict = None

    processorname = "gpen"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        resolution = int(plugin_options.get("resolution", 512))
        if self.plugin_options is not None:
            if int(self.plugin_options.get("resolution", 512)) != resolution:
                self.Release()

        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        model_name = _MODEL_NAMES.get(resolution, "GPEN-BFR-512.onnx")
        model_path = resolve_relative_path(f"../models/{model_name}")
        self._resolution = resolution

        # Routed like every other model (Core AI, else the MLX interpreter).
        from roop.inference import create_session
        self._session = create_session(model_path)

    def Run(self, source_faceset: FaceSet, target_face: Face,
            temp_frame: Frame):
        input_size = temp_frame.shape[1]
        res = self._resolution

        frame = cv2.resize(temp_frame, (res, res), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = (frame / 255.0 - 0.5) / 0.5
        frame = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)  # [1,3,res,res]

        result = self._session.run(["output"], {"input": frame})[0][0]  # [3,res,res]

        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        self._session = None
