"""FaceSwapInsightFace -- hybrid backend.

Runs inswapper/reswapper models through roop.inference.create_session
(CoreML EP with MLX-interpreter fallback).  The hand-ported native MLX
inswapper and its pixel-boost batch path were retired 2026-07 once the
CoreML default proved stable (33 ms/swap, bit-exact).
"""

import os

import numpy as np

from roop.typing import Face, Frame
from roop.face_util import balance_source_embedding
from roop.utilities import resolve_relative_path


class FaceSwapInsightFace:
    plugin_options: dict = None

    processorname = "faceswap"
    type = "swap"

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

        # CoreML EP with MLX-interpreter fallback for all swap models.
        from roop.inference import create_session
        self._session = create_session(model_path)
        self._output_name = self._session.get_outputs()[0].name

    def Run(self, source_face, target_face: Face, temp_frame: Frame) -> Frame:
        if hasattr(source_face, "faces") and source_face.faces:
            source_face = source_face.faces[0]

        latent = source_face.normed_embedding.reshape((1, -1))
        # Identity balancing happens on the ArcFace vector, before the emap
        # projection — the same order FaceFusion uses.
        latent = balance_source_embedding(
            latent,
            getattr(target_face, "embedding", None),
            self.plugin_options.get("swapper_weight", 0.5),
        )
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent) + 1e-6
        latent = np.ascontiguousarray(latent.astype(np.float32))

        result = self._session.run(
            [self._output_name],
            {"target": np.ascontiguousarray(temp_frame), "source": latent},
        )
        return result[0]

    def Release(self):
        self._session = None
        self.emap = None
