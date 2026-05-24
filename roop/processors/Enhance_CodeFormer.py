"""Enhance_CodeFormer -- MLX backend.

CodeFormerv0.1: VQ-VAE + Transformer face restoration.  Input is a
512x512 BGR face crop plus a scalar fidelity weight w ∈ [0, 1];
output is a 512x512 BGR restored face.
"""

import cv2
import numpy as np

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


class Enhance_CodeFormer:
    plugin_options: dict = None

    processorname = "codeformer"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        from roop.mlx_runtime.onnx_loader import OnnxMLXSession
        model_path = resolve_relative_path("../models/CodeFormer/CodeFormerv0.1.onnx")
        self._session = OnnxMLXSession(model_path)

    def Run(self, source_faceset: FaceSet, target_face: Face,
            temp_frame: Frame):
        input_size = temp_frame.shape[1]
        fidelity = float(self.plugin_options.get("fidelity_weight", 0.5))

        frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = (frame / 255.0 - 0.5) / 0.5
        frame = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)  # [1,3,512,512]

        w = np.array(fidelity, dtype=np.float32)
        result = self._session.run(["y"], {"x": frame, "w": w})[0][0]  # [3,512,512]

        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        self._session = None
