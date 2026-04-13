import os
from typing import Optional, Union

import numpy as np
import onnxruntime
import roop.globals

from roop.FaceSet import FaceSet
from roop.perf_profile import (
    augment_providers,
    get_session_options,
)
from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path, conditional_download



class FaceSwapFaceFusion:
    plugin_options: Optional[dict] = None
    model_swap: Optional[onnxruntime.InferenceSession] = None
    model_crossface: Optional[onnxruntime.InferenceSession] = None
    devicename: Optional[str] = None
    device_id: int = 0
    swap_type: Optional[str] = None

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
            if (
                self.plugin_options["devicename"] != plugin_options["devicename"]
                or self.plugin_options["modelname"] != plugin_options["modelname"]
            ):
                self.Release()

        self.plugin_options = plugin_options
        modelname = plugin_options["modelname"]
        config = self.FACE_SWAPPER_MODELS[modelname]
        self.swap_type = config["swap_type"]

        if self.model_swap is None:
            self.devicename = plugin_options["devicename"].replace("mps", "cpu")
            self.device_id = int(plugin_options.get("device_id", 0))

            models_dir = resolve_relative_path("../models")
            model_path = os.path.join(models_dir, config["file"])
            if not os.path.exists(model_path):
                conditional_download(models_dir, [[config["url"]]])

            # HyperSwap output reads don't survive CUDA graph capture correctly;
            # use standard session options (no CUDA graph).
            sess_options = get_session_options()
            providers = augment_providers(roop.globals.execution_providers)
            self.model_swap = onnxruntime.InferenceSession(
                model_path, sess_options, providers=providers
            )

            if config["crossface"] is not None:
                crossface_path = os.path.join(models_dir, config["crossface"])
                if not os.path.exists(crossface_path):
                    conditional_download(models_dir, [[config["crossface_url"]]])
                self.model_crossface = onnxruntime.InferenceSession(
                    crossface_path, get_session_options(), providers=providers
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

    def Run(self, source_faceset: Union[FaceSet, Face], target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_faceset, 'faces') and source_faceset.faces:
            source_face = source_faceset.faces[0]
            ref_images = getattr(source_faceset, 'ref_images', [])
        else:
            source_face = source_faceset
            ref_images = []

        if self.swap_type in ("ghost", "hyperswap"):
            source_data = self._convert_embedding(source_face.normed_embedding)
        else:
            return temp_frame

        io_binding = self.model_swap.io_binding()
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", source_data)
        # Bind ALL model outputs; some ORT versions require every output to be
        # bound when using run_with_iobinding.  We only consume the first
        # output (the swapped face), but the mask output must still be bound.
        for out in self.model_swap.get_outputs():
            io_binding.bind_output(out.name, self.devicename, self.device_id)
        self.model_swap.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0][0]

    def Release(self):
        if self.model_swap is not None:
            del self.model_swap
            self.model_swap = None
        if self.model_crossface is not None:
            del self.model_crossface
            self.model_crossface = None
