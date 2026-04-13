import os
from threading import local as thread_local

import numpy as np
import onnx
import onnxruntime

import roop.globals
from roop.perf_profile import (
    augment_providers,
    get_session_options,
    get_session_options_with_cuda_graphs,
    should_preallocate_io,
    should_use_cuda_graphs,
    should_use_fp16_models,
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

            # Load FP16 model if available and profile allows
            if should_use_fp16_models():
                from roop.model_manager import get_fp16_model_path
                model_path = get_fp16_model_path(model_path)

            # Cache emap as a tiny .npy sidecar to avoid re-parsing
            # the full 500 MB+ ONNX protobuf on every startup.
            emap_cache = model_path + ".emap.npy"
            if os.path.exists(emap_cache):
                self.emap = np.load(emap_cache).astype(np.float32)
            else:
                graph = onnx.load(model_path).graph
                self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                if self.emap.dtype != np.float32:
                    self.emap = self.emap.astype(np.float32)
                np.save(emap_cache, self.emap)
            raw_devicename = self.plugin_options["devicename"]
            self.devicename = raw_devicename.replace("mps", "cpu")
            self.device_id = int(self.plugin_options.get("device_id", 0))
            self.input_mean = 0.0
            self.input_std = 255.0

            # Use CUDA graphs for fixed-size swap models
            if should_use_cuda_graphs() and self.devicename == "cuda":
                sess_options = get_session_options_with_cuda_graphs()
            else:
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

        Bindings are set up ONCE and reused every frame.  Only update_inplace
        touches GPU memory between calls so the CUDA graph (if enabled) can
        replay without re-capture.  Re-binding an io_binding after CUDA graph
        capture forces a new capture (~30 s) every frame — avoid it.

        OrtValue/IoBinding objects are stored in threading.local() so each
        worker thread owns independent buffers (ORT sessions are thread-safe
        but io_binding/OrtValue are not).
        """
        tls = self._get_tls()

        target_shape_changed = tls.target_shape != target.shape
        latent_shape_changed = tls.latent_shape != latent.shape
        need_rebind = (
            tls.io_binding is None
            or target_shape_changed
            or latent_shape_changed
        )

        # Allocate or update target OrtValue
        if tls.target_ortvalue is None or target_shape_changed:
            tls.target_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                target, self.devicename, self.device_id
            )
            tls.target_shape = target.shape
        else:
            tls.target_ortvalue.update_inplace(target)

        # Allocate or update latent OrtValue
        if tls.latent_ortvalue is None or latent_shape_changed:
            tls.latent_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
                latent, self.devicename, self.device_id
            )
            tls.latent_shape = latent.shape
        else:
            tls.latent_ortvalue.update_inplace(latent)

        # Bind inputs/output only on first call or if shapes changed
        if need_rebind:
            tls.io_binding = self.model_swap_insightface.io_binding()
            tls.io_binding.bind_ortvalue_input("target", tls.target_ortvalue)
            tls.io_binding.bind_ortvalue_input("source", tls.latent_ortvalue)
            tls.io_binding.bind_output(
                self._output_name, self.devicename, self.device_id
            )

        self.model_swap_insightface.run_with_iobinding(tls.io_binding)
        return tls.io_binding.copy_outputs_to_cpu()[0]

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

    def RunBatch(self, source_faces, target_faces, temp_frames):
        """Process multiple face crops in a single batched inference call.

        Parameters
        ----------
        source_faces : list
            Source face objects (one per crop).
        target_faces : list[Face]
            Not used directly -- kept for API symmetry.
        temp_frames : list[ndarray]
            Pre-processed face crops (already in model input format).

        Returns
        -------
        list[ndarray]
            Swap results, one per input crop.
        """
        if not temp_frames:
            return []

        # Build batched inputs
        latents = []
        for sf in source_faces:
            if hasattr(sf, "faces") and sf.faces:
                sf = sf.faces[0]
            lat = sf.normed_embedding.reshape((1, -1))
            lat = np.dot(lat, self.emap)
            lat /= np.linalg.norm(lat)
            latents.append(lat.astype(np.float32))

        # Stack targets along batch dim
        targets = np.concatenate(
            [np.ascontiguousarray(t) for t in temp_frames], axis=0
        )
        latent_batch = np.concatenate(latents, axis=0)

        # Try batched inference (model may or may not support batch > 1)
        try:
            io_binding = self.model_swap_insightface.io_binding()
            io_binding.bind_cpu_input("target", targets)
            io_binding.bind_cpu_input("source", latent_batch)
            io_binding.bind_output(
                self._output_name, self.devicename, self.device_id
            )
            self.model_swap_insightface.run_with_iobinding(io_binding)
            ort_outs = io_binding.copy_outputs_to_cpu()[0]
            return [ort_outs[i] for i in range(len(temp_frames))]
        except Exception:
            # Fall back to sequential if batched fails
            results = []
            for sf, tf, frame in zip(source_faces, target_faces, temp_frames):
                results.append(self.Run(sf, tf, frame))
            return results

    def Release(self):
        del self.model_swap_insightface
        self.model_swap_insightface = None
        self._tls = thread_local()
