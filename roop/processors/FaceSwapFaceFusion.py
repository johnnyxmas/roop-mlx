import os
from typing import Optional, Union

import numpy as np
import onnxruntime
import cv2
from skimage import transform as trans
import roop.globals

from roop.FaceSet import FaceSet
from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download


# FFHQ 512x512 5-point landmark template (eyes, nose, mouth corners)
FFHQ_LANDMARKS_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)


class FaceSwapFaceFusion:
    plugin_options: Optional[dict] = None
    model_swap: Optional[onnxruntime.InferenceSession] = None
    model_crossface: Optional[onnxruntime.InferenceSession] = None
    devicename: Optional[str] = None
    device_id: int = 0
    swap_type: Optional[str] = None
    preprocessing: str = "ghost"

    processorname = "faceswap_ff"
    type = "swap"

    FACE_SWAPPER_MODELS = {
        "Ghost 1": {
            "file": "ghost_1_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx",
            "crossface": "crossface_ghost.onnx",
            "crossface_url": "https://huggingface.co/facefusion/models-3.4.0/resolve/main/crossface_ghost.onnx",
            "swap_type": "ghost",
            "size": 256,
        },
        "Ghost 2": {
            "file": "ghost_2_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_2_256.onnx",
            "crossface": "crossface_ghost.onnx",
            "crossface_url": "https://huggingface.co/facefusion/models-3.4.0/resolve/main/crossface_ghost.onnx",
            "swap_type": "ghost",
            "size": 256,
        },
        "Ghost 3": {
            "file": "ghost_3_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_3_256.onnx",
            "crossface": "crossface_ghost.onnx",
            "crossface_url": "https://huggingface.co/facefusion/models-3.4.0/resolve/main/crossface_ghost.onnx",
            "swap_type": "ghost",
            "size": 256,
        },
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
        "UniFace": {
            "file": "uniface_256.onnx",
            "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/uniface_256.onnx",
            "crossface": None,
            "crossface_url": None,
            "swap_type": "uniface",
            "size": 256,
        },
    }

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if (
                self.plugin_options["devicename"] != plugin_options["devicename"]
                or self.plugin_options["modelname"] != plugin_options["modelname"]
            ):
                self.Release()

        self.plugin_options = plugin_options
        modelname = plugin_options["modelname"]
        config = self.FACE_SWAPPER_MODELS[modelname]
        self.swap_type = config["swap_type"]
        self.preprocessing = config["swap_type"]

        if self.model_swap is None:
            self.devicename = plugin_options["devicename"].replace("mps", "cpu")
            self.device_id = int(plugin_options.get("device_id", 0))

            models_dir = resolve_relative_path("../models")
            model_path = os.path.join(models_dir, config["file"])
            if not os.path.exists(model_path):
                conditional_download(models_dir, [[config["url"]]])

            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            self.model_swap = onnxruntime.InferenceSession(
                model_path, sess_options, providers=roop.globals.execution_providers
            )

            if config["crossface"] is not None:
                crossface_path = os.path.join(models_dir, config["crossface"])
                if not os.path.exists(crossface_path):
                    conditional_download(models_dir, [[config["crossface_url"]]])
                self.model_crossface = onnxruntime.InferenceSession(
                    crossface_path, None, providers=roop.globals.execution_providers
                )

    def _convert_embedding(self, normed_embedding: np.ndarray) -> np.ndarray:
        embedding = normed_embedding.reshape(1, -1).astype(np.float32)
        if self.model_crossface is None:
            return embedding
        io_binding = self.model_crossface.io_binding()
        io_binding.bind_cpu_input(self.model_crossface.get_inputs()[0].name, embedding)
        io_binding.bind_output(
            self.model_crossface.get_outputs()[0].name, self.devicename, self.device_id
        )
        self.model_crossface.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]

    def _warp_face_ffhq(self, kps: np.ndarray, source_image: np.ndarray, size: int = 256) -> np.ndarray:
        scale = size / 512.0
        dst = FFHQ_LANDMARKS_512 * scale
        tform = trans.SimilarityTransform.from_estimate(kps, dst)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(source_image, M, (size, size), borderValue=0.0)
        warped = warped[:, :, ::-1] / 127.5 - 1
        warped = warped.transpose(2, 0, 1)
        return np.expand_dims(warped, axis=0).astype(np.float32)

    def Run(self, source_faceset: Union[FaceSet, Face], target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_faceset, 'faces') and source_faceset.faces:
            source_face = source_faceset.faces[0]
            ref_images = getattr(source_faceset, 'ref_images', [])
        else:
            source_face = source_faceset
            ref_images = []

        io_binding = self.model_swap.io_binding()

        if self.swap_type in ("ghost", "hyperswap"):
            embedding = self._convert_embedding(source_face.normed_embedding)
            io_binding.bind_cpu_input("target", temp_frame)
            io_binding.bind_cpu_input("source", embedding)
        elif self.swap_type == "uniface":
            if not ref_images:
                return temp_frame
            source_crop = self._warp_face_ffhq(source_face.kps, ref_images[0])
            io_binding.bind_cpu_input("target", temp_frame)
            io_binding.bind_cpu_input("source", source_crop)

        io_binding.bind_output(
            self.model_swap.get_outputs()[0].name, self.devicename, self.device_id
        )
        self.model_swap.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0][0]

    def Release(self):
        if self.model_swap is not None:
            del self.model_swap
            self.model_swap = None
        if self.model_crossface is not None:
            del self.model_crossface
            self.model_crossface = None
