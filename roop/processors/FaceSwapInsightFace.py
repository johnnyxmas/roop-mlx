"""FaceSwapInsightFace -- MLX backend.

Loads inswapper_128 via native MLX model (fast path) or falls back to
the OnnxMLXSession interpreter for reswapper models.
"""

import os

import numpy as np

from roop.typing import Face, Frame
from roop.utilities import resolve_relative_path

class FaceSwapInsightFace:
    plugin_options: dict = None

    processorname = "faceswap"
    type = "swap"
    # True when the loaded model accepts [N, C, H, W] batches, letting
    # ProcessMgr swap all pixel-boost slices in one dispatch.
    supports_batch = False

    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("modelname") != plugin_options.get("modelname"):
                self.Release()

        self.plugin_options = plugin_options

        if getattr(self, "_session", None) is not None:
            return

        model_path = resolve_relative_path(
            "../models/" + self.plugin_options["modelname"]
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Swap model not found: {model_path}")

        # Load emap from .npy sidecar or extract from ONNX initializers.
        emap_cache = model_path + ".emap.npy"
        if os.path.exists(emap_cache):
            self.emap = np.load(emap_cache).astype(np.float32)
        else:
            import onnx
            from onnx import numpy_helper
            self.emap = numpy_helper.to_array(
                onnx.load(model_path).graph.initializer[-1]
            ).astype(np.float32)
            np.save(emap_cache, self.emap)

        if os.path.basename(model_path) == "inswapper_128.onnx":
            # CoreML EP: 33 ms/swap, bit-exact vs the ONNX reference.  Its
            # graph declares batch=1, so pixel-boost slices run per-slice.
            from roop.inference import try_coreml_session
            self._session = try_coreml_session(model_path)
            if self._session is not None:
                self.supports_batch = False
            else:
                # CoreML unavailable — use the hand-ported compiled mlx.nn
                # model (fp16, batch-capable) instead of the interpreter.
                import mlx.core as mx
                from roop.mlx_runtime.inswapper_native import InSwapperNative
                self._session = InSwapperNative.from_onnx(model_path, dtype=mx.float16)
                self.supports_batch = True
            self._output_name = "output"
        else:
            # reswapper models: CoreML EP (with MLX fp16 fallback)
            from roop.inference import create_session
            self._session = create_session(model_path)
            self._output_name = self._session.get_outputs()[0].name
            self.supports_batch = False

    def Run(self, source_face, target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_face, "faces") and source_face.faces:
            source_face = source_face.faces[0]

        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent) + 1e-6
        latent = np.ascontiguousarray(latent.astype(np.float32))

        result = self._session.run(
            [self._output_name],
            {"target": np.ascontiguousarray(temp_frame), "source": latent},
        )
        return result[0]

    def RunBatch(self, source_faces, target_faces, temp_frames):
        return [self.Run(sf, tf, fr)
                for sf, tf, fr in zip(source_faces, target_faces, temp_frames)]

    def Release(self):
        self._session = None
        self.emap = None
        self.supports_batch = False
