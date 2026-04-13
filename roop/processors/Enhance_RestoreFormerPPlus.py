import cv2
import numpy as np
import onnxruntime
import roop.globals

from roop.perf_profile import (
    augment_providers,
    get_enhancer_session_options,
)
from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path


class Enhance_RestoreFormerPPlus:
    plugin_options: dict = None
    model_restoreformerpplus = None
    devicename = None
    name = None

    processorname = "restoreformer++"
    type = "enhance"

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_restoreformerpplus is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace("mps", "cpu")
            self.device_id = int(self.plugin_options.get("device_id", 0))
            model_path = resolve_relative_path("../models/restoreformer_plus_plus.onnx")
            _sess_opts = get_enhancer_session_options()
            self.model_restoreformerpplus = onnxruntime.InferenceSession(
                model_path,
                _sess_opts,
                providers=augment_providers(roop.globals.execution_providers),
            )
            self.model_inputs = self.model_restoreformerpplus.get_inputs()
            self._output_name = self.model_restoreformerpplus.get_outputs()[0].name

    def Run(
        self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame
    ) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), cv2.INTER_CUBIC)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype("float32") / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_restoreformerpplus.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self._output_name, self.devicename, self.device_id)
        self.model_restoreformerpplus.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs

        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        scale_factor = int(result.shape[1] / input_size)
        return result.astype(np.uint8), scale_factor

    def Release(self):
        del self.model_restoreformerpplus
        self.model_restoreformerpplus = None
