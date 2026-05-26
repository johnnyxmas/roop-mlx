"""Enhance_DMDNet -- MLX backend.

DMDNet blind face restoration.  Loads DMDNet.pth (PyTorch checkpoint) via
the torch-free loader in roop.mlx_runtime.dmdnet and runs inference on MLX.

Only the generic (dictionary-free) path is implemented here.  The full
"specific" path that uses reference face memory banks is left for a future
sprint.
"""

import cv2
import numpy as np

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

# 106-point → 68-point landmark index map (standard conversion)
_LM106_TO_68 = [
    1, 10, 12, 14, 16, 3, 5, 7, 0, 23, 21, 19, 32, 30, 28, 26, 17,
    43, 48, 49, 51, 50, 102, 103, 104, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 82, 81, 80, 79, 78, 77, 74,
    90, 91, 92, 93, 94, 95, 88, 89, 86, 87, 84, 85, 98, 97, 96, 100,
]


def _landmarks106_to_68(pt106: np.ndarray) -> np.ndarray:
    """Map 106-point landmarks to 68-point landmarks."""
    return pt106[_LM106_TO_68]


def _get_component_location(lm68: np.ndarray):
    """Compute [LE, RE, NO, MO] bounding boxes from 68-point landmarks.

    Returns list of (x1, y1, x2, y2) tuples, one per facial component.
    Indices follow the standard 68-point layout:
      LE=36-41, RE=42-47, NO=27-35, MO=48-67
    """
    def _bbox(pts):
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw = max(x2 - x1, y2 - y1) * 0.75
        return (int(cx - hw), int(cy - hw), int(cx + hw), int(cy + hw))

    le = _bbox(lm68[36:42])
    re = _bbox(lm68[42:48])
    no = _bbox(lm68[27:36])
    mo = _bbox(lm68[48:68])
    return [le, re, no, mo]


class Enhance_DMDNet:
    plugin_options: dict = None

    processorname = "dmdnet"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_model", None) is not None:
            return

        from roop.mlx_runtime.dmdnet import DMDNetMLX
        model_path = resolve_relative_path("../models/DMDNet.pth")
        self._model = DMDNetMLX(model_path)

    def Run(self, source_faceset, target_face: Face, temp_frame: Frame):
        input_size = 512
        orig_h, orig_w = temp_frame.shape[:2]
        scale_factor = int(input_size / orig_w) if orig_w > 0 else 1

        # Resize to 512×512 if needed
        if orig_h != input_size or orig_w != input_size:
            frame = cv2.resize(temp_frame, (input_size, input_size),
                               interpolation=cv2.INTER_LINEAR)
        else:
            frame = temp_frame.copy()

        # BGR→RGB, [0,255] → float32 NCHW [-1, 1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame_rgb.astype(np.float32) / 127.5 - 1.0      # [0,255] → [-1,1]
        img_nchw = img.transpose(2, 0, 1)[np.newaxis]          # [1,3,512,512]

        # Build component locations from target face landmarks (if available)
        locs = None
        if target_face is not None and hasattr(target_face, "landmark") and \
                target_face.landmark is not None:
            lm106 = np.array(target_face.landmark)
            if lm106.shape[0] == 106:
                lm68 = _landmarks106_to_68(lm106)
                # Scale landmarks to 512×512 space
                scale_x = input_size / orig_w
                scale_y = input_size / orig_h
                lm68 = lm68.astype(np.float32)
                lm68[:, 0] *= scale_x
                lm68[:, 1] *= scale_y
                locs = _get_component_location(lm68)  # list of 4 (x1,y1,x2,y2) tuples

        if locs is None:
            # Default component locations for a standard 512×512 aligned face
            locs = [
                (140, 155, 220, 225),   # LE
                (290, 155, 370, 225),   # RE
                (215, 215, 295, 315),   # NO
                (175, 325, 335, 395),   # MO
            ]

        # dmdnet.forward expects locs shape [1, 4, 4] (batch × component × coords)
        locs_arr = np.array([locs], dtype=np.float32)           # [1, 4, 4]
        out_nchw = self._model.forward(img_nchw, locs_arr)     # [1,3,512,512]

        # Post-process: [-1,1] → [0,1] → [0,255] uint8, RGB→BGR
        out = out_nchw[0] * 0.5 + 0.5                          # [0,1]
        out = out.transpose(1, 2, 0)                            # HWC RGB
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)           # HWC BGR

        # Remove the ~12px periodic grid from the dilated encoder via median filter.
        # Median filter kills regular grid patterns while preserving edges/detail.
        result = cv2.medianBlur(result, 5)

        return result, scale_factor

    def Release(self):
        self._model = None
