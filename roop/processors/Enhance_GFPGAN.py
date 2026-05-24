"""Enhance_GFPGAN -- MLX backend.

GFPGANv1.4: StyleGAN2-based face restoration.  Input is a 512x512 BGR
face crop; output is a 512x512 BGR restored face.
"""

import cv2
import numpy as np

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


class Enhance_GFPGAN:
    plugin_options: dict = None

    processorname = "gfpgan"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        from roop.mlx_runtime.onnx_loader import OnnxMLXSession
        model_path = resolve_relative_path("../models/GFPGANv1.4.onnx")
        self._session = OnnxMLXSession(model_path)

    def Run(self, source_faceset: FaceSet, target_face: Face,
            temp_frame: Frame):
        input_size = temp_frame.shape[1]

        # pre-process: BGR→RGB, resize to 512, normalise to [-1, 1], CHW
        frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = (frame / 255.0 - 0.5) / 0.5
        frame = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)  # [1,3,512,512]

        result = self._session.run(["1288"], {"input": frame})[0][0]  # [3,512,512]

        # post-process: clamp, denorm, CHW→HWC, RGB→BGR
        result = np.clip(result, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        self._session = None
