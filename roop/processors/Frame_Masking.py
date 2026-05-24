"""Frame_Masking -- MLX backend.

ISNet general-use background removal.  Input is any BGR frame; output is
the frame multiplied by the predicted foreground mask (background → black).
"""

import cv2
import numpy as np

from roop.typing import Frame
from roop.utilities import resolve_relative_path

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class Frame_Masking:
    plugin_options: dict = None

    processorname = "removebg"
    type = "frame_masking"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        from roop.mlx_runtime.onnx_loader import OnnxMLXSession
        model_path = resolve_relative_path("../models/Frame/isnet-general-use.onnx")
        self._session = OnnxMLXSession(model_path)

    def Run(self, temp_frame: Frame) -> Frame:
        orig_h, orig_w = temp_frame.shape[:2]

        frame = cv2.resize(temp_frame, (1024, 1024))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = frame / 255.0 - _MEAN
        frame = np.expand_dims(frame.transpose(2, 0, 1), 0)  # [1,3,1024,1024]

        result = self._session.run(["output_image"], {"input_image": frame})[0][0, 0]

        # min-max normalise → soft mask in [0, 1]
        lo, hi = result.min(), result.max()
        if hi > lo:
            mask = (result - lo) / (hi - lo)
        else:
            mask = result - lo

        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = mask[:, :, np.newaxis]
        return (mask * temp_frame.astype(np.float32)).astype(np.uint8)

    def Release(self):
        self._session = None
