"""Mask_XSeg -- MLX backend.

XSeg is a DeepFaceLab segmentation model exported from TF.  The ONNX
graph begins with a Transpose(NHWC→NCHW) and ends with a
Reshape/Sigmoid that returns NHWC [N,H,W,1].  Input is fed as NHWC.
"""

import cv2
import numpy as np

from roop.typing import Frame
from roop.utilities import resolve_relative_path


class Mask_XSeg:
    plugin_options: dict = None

    processorname = "mask_xseg"
    type = "mask"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        from roop.mlx_runtime.onnx_loader import OnnxMLXSession
        model_path = resolve_relative_path("../models/xseg.onnx")
        self._session = OnnxMLXSession(model_path)
        self._inp = self._session.get_inputs()[0].name
        self._out = self._session.get_outputs()[0].name

    def Run(self, img: Frame, masking_text: str = None) -> Frame:
        # resize to model input size, normalise to [0, 1]
        frame = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        frame = frame.astype(np.float32) / 255.0
        frame = frame[np.newaxis]           # [1, 256, 256, 3] NHWC

        result = self._session.run([self._out], {self._inp: frame})[0]
        result = result[0]                  # [256, 256, 1]
        result = np.clip(result, 0.0, 1.0)
        result[result < 0.1] = 0.0
        result = 1.0 - result               # invert: 1 = keep, 0 = mask
        return result

    def Release(self):
        self._session = None
