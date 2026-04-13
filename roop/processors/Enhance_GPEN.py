import os
import cv2
import numpy as np
import onnxruntime
import roop.globals

from roop.perf_profile import (
    augment_providers,
    get_session_options,
    get_session_options_with_cuda_graphs,
    should_use_cuda_graphs,
    should_use_fp16_models,
)
from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path, conditional_download


class Enhance_GPEN:
    plugin_options: dict = None

    model_gpen = None
    name = None
    devicename = None

    processorname = "gpen"
    type = "enhance"

    GPEN_DOWNLOAD_URLS = {
        512: "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
        1024: "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_1024.onnx",
        2048: "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_2048.onnx",
    }

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if (self.plugin_options["devicename"] != plugin_options["devicename"]
                    or self.plugin_options.get("resolution", 512) != plugin_options.get("resolution", 512)):
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gpen is None:
            self.input_resolution = self.plugin_options.get("resolution", 512)
            model_path = resolve_relative_path(f"../models/GPEN-BFR-{self.input_resolution}.onnx")
            if not os.path.exists(model_path):
                url = self.GPEN_DOWNLOAD_URLS.get(self.input_resolution)
                if url:
                    models_dir = resolve_relative_path("../models")
                    conditional_download(models_dir, [url])
                    downloaded = os.path.join(models_dir, os.path.basename(url))
                    if downloaded != model_path and os.path.exists(downloaded):
                        os.rename(downloaded, model_path)
            if should_use_fp16_models():
                from roop.model_manager import get_fp16_model_path
                model_path = get_fp16_model_path(model_path)
            if should_use_cuda_graphs():
                _sess_opts = get_session_options_with_cuda_graphs()
            else:
                _sess_opts = get_session_options()
            self.model_gpen = onnxruntime.InferenceSession(
                model_path,
                _sess_opts,
                providers=augment_providers(roop.globals.execution_providers),
            )
            self.devicename = self.plugin_options["devicename"].replace("mps", "cpu")
            self.device_id = int(self.plugin_options.get("device_id", 0))

        self.name = self.model_gpen.get_inputs()[0].name

    def Run(
        self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame
    ) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (self.input_resolution, self.input_resolution), cv2.INTER_CUBIC)

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype("float32") / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_gpen.io_binding()
        io_binding.bind_cpu_input("input", temp_frame)
        io_binding.bind_output("output", self.devicename, self.device_id)
        self.model_gpen.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        if self.model_gpen is not None:
            del self.model_gpen
            self.model_gpen = None
