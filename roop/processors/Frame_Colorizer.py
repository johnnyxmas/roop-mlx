"""Frame_Colorizer -- MLX backend.

DeOldify artistic/stable colorization.  Takes any BGR frame (typically
from a grayscale source) and returns a colorized BGR frame.

Pre/post-processing matches the upstream ORT pipeline exactly:
  - input:  grayscale → RGB stack, 256×256, raw uint8 as float32, NCHW
  - output: CHW → HWC, resize to original, merge L from original + AB from
            model output in LAB space.
"""

import cv2
import numpy as np

from roop.typing import Frame
from roop.utilities import resolve_relative_path


class Frame_Colorizer:
    plugin_options: dict = None
    prev_type: str = None

    processorname = "deoldify"
    type = "frame_colorizer"

    def Initialize(self, plugin_options: dict):
        subtype = plugin_options.get("subtype", "deoldify_artistic")
        if self.prev_type is not None and self.prev_type != subtype:
            self.Release()

        self.plugin_options = plugin_options
        self.prev_type = subtype

        if getattr(self, "_session", None) is not None:
            return

        fname = f"{subtype}.onnx"
        model_path = resolve_relative_path(f"../models/Frame/{fname}")

        from roop.mlx_runtime.onnx_loader import OnnxMLXSession
        self._session = OnnxMLXSession(model_path)
        self._output_name = self._session.get_outputs()[0].name

    def getProcessedResolution(self, width: int, height: int):
        return (width, height)

    def Run(self, temp_frame: Frame) -> Frame:
        orig_h, orig_w = temp_frame.shape[:2]

        # Pre-process: convert to grayscale then back to 3-channel RGB
        gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        frame = cv2.resize(frame, (256, 256))
        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, 0).astype(np.float32)  # [1,3,256,256], raw [0-255]

        result = self._session.run([self._output_name], {"input": frame})[0][0]

        # Post-process: CHW → HWC, resize to original, merge L + AB in LAB
        colorized = result.transpose(1, 2, 0)
        colorized = cv2.resize(colorized, (orig_w, orig_h))

        # Extract L from original frame, AB from model output
        orig_lab = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2LAB)
        orig_l = orig_lab[:, :, 0:1]

        colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        colorized_lab = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2LAB)
        color_ab = colorized_lab[:, :, 1:]

        merged = np.concatenate([orig_l, color_ab], axis=2)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def Release(self):
        self._session = None
