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
from roop.utilities import resolve_relative_path
from roop.typing import Frame


class Frame_Colorizer:
    plugin_options: dict = None
    model_colorizer = None
    devicename = None
    prev_type = None

    processorname = "deoldify"
    type = "frame_colorizer"

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if (
            self.prev_type is not None
            and self.prev_type != self.plugin_options["subtype"]
        ):
            self.Release()
        self.prev_type = self.plugin_options["subtype"]
        if self.model_colorizer is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace("mps", "cpu")
            self.device_id = int(self.plugin_options.get("device_id", 0))
            if self.prev_type == "deoldify_artistic":
                model_path = resolve_relative_path(
                    "../models/Frame/deoldify_artistic.onnx"
                )
            elif self.prev_type == "deoldify_stable":
                model_path = resolve_relative_path(
                    "../models/Frame/deoldify_stable.onnx"
                )

            if should_use_fp16_models():
                from roop.model_manager import get_fp16_model_path
                model_path = get_fp16_model_path(model_path)
            if should_use_cuda_graphs():
                _sess_opts = get_session_options_with_cuda_graphs()
            else:
                _sess_opts = get_session_options()
            onnxruntime.set_default_logger_severity(3)
            self.model_colorizer = onnxruntime.InferenceSession(
                model_path,
                _sess_opts,
                providers=augment_providers(roop.globals.execution_providers),
            )
            self.model_inputs = self.model_colorizer.get_inputs()
            self._output_name = self.model_colorizer.get_outputs()[0].name

    def Run(self, input_frame: Frame) -> Frame:
        temp_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_GRAY2RGB)
        temp_frame = cv2.resize(temp_frame, (256, 256))
        temp_frame = temp_frame.transpose((2, 0, 1))
        temp_frame = np.expand_dims(temp_frame, axis=0).astype(np.float32)
        io_binding = self.model_colorizer.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self._output_name, self.devicename, self.device_id)
        self.model_colorizer.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        del ort_outs
        colorized_frame = result.transpose(1, 2, 0)
        colorized_frame = cv2.resize(
            colorized_frame, (input_frame.shape[1], input_frame.shape[0])
        )
        temp_blue_channel, _, _ = cv2.split(input_frame)
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2RGB).astype(
            np.uint8
        )
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_BGR2LAB)
        _, color_green_channel, color_red_channel = cv2.split(colorized_frame)
        colorized_frame = cv2.merge(
            (temp_blue_channel, color_green_channel, color_red_channel)
        )
        colorized_frame = cv2.cvtColor(colorized_frame, cv2.COLOR_LAB2BGR)
        return colorized_frame.astype(np.uint8)

    def Release(self):
        del self.model_colorizer
        self.model_colorizer = None
