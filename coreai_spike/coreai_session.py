"""ONNX-Runtime-shaped session over Core AI, via the libcoreai_bridge dylib.

    sess = CoreAISession("model.aimodel", "model.onnx")   # .onnx supplies I/O metadata
    out = sess.run(None, {"target": x, "source": e})
"""
import ctypes
import os

import numpy as np
import onnx

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "libcoreai_bridge.dylib")
_UNITS = {"default": 0, "cpu": 1, "gpu": 2, "ane": 3}
_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_LIB_PATH)
        lib.cai_load.restype = ctypes.c_void_p
        lib.cai_load.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        lib.cai_free.argtypes = [ctypes.c_void_p]
        lib.cai_run.restype = ctypes.c_int32
        _lib = lib
    return _lib


class _Arg:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class CoreAISession:
    def __init__(self, aimodel_path, onnx_path, unit="default"):
        m = onnx.load(onnx_path, load_external_data=False)
        init = {i.name for i in m.graph.initializer}

        def dims(vi):
            return [d.dim_value if d.dim_value > 0 else None for d in vi.type.tensor_type.shape.dim]
        self._inputs = [_Arg(i.name, dims(i)) for i in m.graph.input if i.name not in init]
        self._outputs = [_Arg(o.name, dims(o)) for o in m.graph.output]
        self._lib = _load_lib()
        self._h = self._lib.cai_load(os.path.abspath(aimodel_path).encode(), _UNITS[unit])
        if not self._h:
            raise RuntimeError(f"Core AI could not load {aimodel_path}")

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        out_args = [o for o in self._outputs if not output_names or o.name in output_names]
        arrays = [np.ascontiguousarray(feed[i.name], dtype=np.float32) for i in self._inputs]
        n_in = len(arrays)
        in_names = (ctypes.c_char_p * n_in)(*[i.name.encode() for i in self._inputs])
        in_data = (ctypes.POINTER(ctypes.c_float) * n_in)(
            *[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrays])
        shapes = [d for a in arrays for d in a.shape]
        in_shapes = (ctypes.c_int64 * len(shapes))(*shapes)
        in_ranks = (ctypes.c_int32 * n_in)(*[a.ndim for a in arrays])

        outs = [np.empty([d or 1 for d in o.shape], dtype=np.float32) for o in out_args]
        n_out = len(outs)
        out_names = (ctypes.c_char_p * n_out)(*[o.name.encode() for o in out_args])
        out_data = (ctypes.POINTER(ctypes.c_float) * n_out)(
            *[o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for o in outs])
        out_counts = (ctypes.c_int64 * n_out)(*[o.size for o in outs])

        rc = self._lib.cai_run(ctypes.c_void_p(self._h), n_in, in_names, in_data, in_shapes, in_ranks,
                               n_out, out_names, out_data, out_counts)
        if rc != 0:
            raise RuntimeError(f"Core AI run failed (code {rc})")
        return outs

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.cai_free(ctypes.c_void_p(self._h))
            self._h = None
