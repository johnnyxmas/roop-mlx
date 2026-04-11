from threading import local as thread_local

import numpy as np
import onnx
import onnxruntime

import roop.globals
from roop.perf_profile import (
    augment_providers,
    get_session_options,
    should_preallocate_io,
)
from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path


class FaceSwapInsightFace:
    plugin_options: dict = None
    model_swap_insightface = None

    processorname = "faceswap"
    type = "swap"

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            prev = self.plugin_options
            device_changed = (
                prev["devicename"] != plugin_options["devicename"]
            )
            model_changed = (
                prev["modelname"] != plugin_options["modelname"]
            )
            if device_changed or model_changed:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_swap_insightface is None:
            model_path = resolve_relative_path(
                "../models/" + self.plugin_options["modelname"]
            )
            graph = onnx.load(model_path).graph
            self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
            raw_devicename = self.plugin_options["devicename"]
            self.devicename = raw_devicename.replace("mps", "cpu")
            self.device_id = int(self.plugin_options.get("device_id", 0))
            self.input_mean = 0.0
            self.input_std = 255.0

            sess_options = get_session_options()
            providers = augment_providers(roop.globals.execution_providers)

            self.model_swap_insightface = onnxruntime.InferenceSession(
                model_path, sess_options, providers=providers
            )
            outputs = self.model_swap_insightface.get_outputs()
            self._output_name = outputs[0].name
            self._tls = thread_local()

    def _get_tls(self):
        tls = self._tls
        if not hasattr(tls, "io_binding"):
            tls.io_binding = None
            tls.target_ortvalue = None
            tls.latent_ortvalue = None
            tls.target_shape = None
            tls.latent_shape = None
        return tls

    def _run_with_preallocated(self, target, latent):
        """Fast path: reuse a persistent IoBinding + CUDA OrtValues.

        The binding and OrtValues are stored in ``threading.local()`` so each
        worker thread owns its own buffers. ORT sessions are thread-safe for
        ``run_with_iobinding``, but io_binding/OrtValue instances are NOT —
        sharing them between threads corrupts device memory. Per-thread
        caches give us the alloc-amortization benefit without the race.
        """
        tls = self._get_tls()

        target_missing = tls.target_ortvalue is None
        target_shape_changed = tls.target_shape != target.shape
        if target_missing or target_shape_changed:
            tls.target_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                target, self.devicename, self.device_id
            )
            tls.target_shape = target.shape
        else:
            tls.target_ortvalue.update_inplace(target)

        latent_missing = tls.latent_ortvalue is None
        latent_shape_changed = tls.latent_shape != latent.shape
        if latent_missing or latent_shape_changed:
            tls.latent_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                latent, self.devicename, self.device_id
            )
            tls.latent_shape = latent.shape
        else:
            tls.latent_ortvalue.update_inplace(latent)

        if tls.io_binding is None:
            tls.io_binding = self.model_swap_insightface.io_binding()
        io_binding = tls.io_binding
        io_binding.bind_ortvalue_input("target", tls.target_ortvalue)
        io_binding.bind_ortvalue_input("source", tls.latent_ortvalue)
        io_binding.bind_output(
            self._output_name, self.devicename, self.device_id
        )
        self.model_swap_insightface.run_with_iobinding(io_binding)
        return io_binding.copy_outputs_to_cpu()[0]

    def Run(self, source_face, target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_face, "faces") and source_face.faces:
            source_face = source_face.faces[0]
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        latent = np.ascontiguousarray(latent.astype(np.float32))
        target = np.ascontiguousarray(temp_frame)

        if should_preallocate_io() and self.devicename == "cuda":
            try:
                ort_outs = self._run_with_preallocated(target, latent)
                return ort_outs[0]
            except Exception:
                tls = self._get_tls()
                tls.io_binding = None
                tls.target_ortvalue = None
                tls.latent_ortvalue = None

        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input("target", target)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output(
            self._output_name, self.devicename, self.device_id
        )
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self._tls = thread_local()
