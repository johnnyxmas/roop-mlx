"""Mask_Clip2Seg -- text-prompted face masking via CLIPSeg (ONNX).

Runs the CLIPSeg rd64-refined model through onnxruntime (CoreML EP with
CPU fallback), replacing the original torch implementation.  Tokenization
uses the pure-python CLIP BPE tokenizer in clip/simple_tokenizer.py; the
pre/post-processing pipeline (border mask, threshold, dilate, blur)
replicates the torch version, with the model's native 352px resolution
instead of 256.
"""

import threading

import cv2
import numpy as np

from roop.typing import Frame
from roop.utilities import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()

_CONTEXT_LENGTH = 77  # CLIP's fixed text context
_INPUT_SIZE = 352     # CLIPSeg rd64-refined native resolution
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Mask_Clip2Seg:
    plugin_options: dict = None

    processorname = "clip2seg"
    type = "mask"

    def Initialize(self, plugin_options: dict):
        self.plugin_options = plugin_options
        if getattr(self, "_session", None) is not None:
            return

        model_path = resolve_relative_path(
            "../models/CLIP/clipseg_rd64_refined.onnx"
        )
        from roop.inference import try_coreml_session

        self._session = try_coreml_session(model_path)
        if self._session is None:
            # The 520MB transformer graph is not interpreter territory;
            # plain ORT CPU is the fallback.
            import onnxruntime as ort

            ort.set_default_logger_severity(3)
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )

        from clip.simple_tokenizer import SimpleTokenizer

        self._tokenizer = SimpleTokenizer()
        self._sot = self._tokenizer.encoder["<|startoftext|>"]
        self._eot = self._tokenizer.encoder["<|endoftext|>"]

    def _tokenize(self, prompt: str):
        tokens = [self._sot] + self._tokenizer.encode(prompt) + [self._eot]
        tokens = tokens[:_CONTEXT_LENGTH]
        ids = np.zeros((1, _CONTEXT_LENGTH), dtype=np.int64)
        mask = np.zeros((1, _CONTEXT_LENGTH), dtype=np.int64)
        ids[0, : len(tokens)] = tokens
        mask[0, : len(tokens)] = 1
        return ids, mask

    def _predict(self, pixel_values: np.ndarray, prompt: str) -> np.ndarray:
        ids, mask = self._tokenize(prompt.strip())
        logits = self._session.run(
            None,
            {
                "input_ids": ids,
                "pixel_values": pixel_values,
                "attention_mask": mask,
            },
        )[0]
        return logits.reshape(_INPUT_SIZE, _INPUT_SIZE)

    def Run(self, img1, keywords: str) -> Frame:
        if keywords is None or len(keywords) < 1 or img1 is None:
            return img1

        small = cv2.resize(img1, (_INPUT_SIZE, _INPUT_SIZE))

        # Feathered full-frame border mask, as in the torch version.
        img_mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
        mask_border = 1
        mask_blur = 5
        clip_blur = 5
        img_mask = cv2.rectangle(
            img_mask,
            (mask_border, mask_border),
            (_INPUT_SIZE - mask_border - 1, _INPUT_SIZE - mask_border - 1),
            (255, 255, 255),
            -1,
        )
        img_mask = cv2.GaussianBlur(img_mask, (mask_blur * 2 + 1, mask_blur * 2 + 1), 0)
        img_mask /= 255

        # BGR uint8 -> normalized RGB [1, 3, H, W]
        rgb = small[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        pixel_values = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[np.newaxis]
        )

        thresh = 0.5
        prompts = [p for p in keywords.split(",") if p.strip()]
        with THREAD_LOCK_CLIP:
            clip_mask = np.zeros((_INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
            for prompt in prompts:
                logits = self._predict(pixel_values, prompt)
                clip_mask += 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        clip_mask = np.clip(clip_mask, 0, 1)
        clip_mask[clip_mask > thresh] = 1.0
        clip_mask[clip_mask <= thresh] = 0.0
        kernel = np.ones((5, 5), np.float32)
        clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
        clip_mask = cv2.GaussianBlur(
            clip_mask, (clip_blur * 2 + 1, clip_blur * 2 + 1), 0
        )

        img_mask *= clip_mask
        img_mask[img_mask < 0.0] = 0.0
        return img_mask

    def Release(self):
        self._session = None
        self._tokenizer = None
