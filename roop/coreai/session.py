"""Apple Core AI backend (macOS 27+).

Core AI runs ``.aimodel`` bundles through a Swift-only runtime.  We reach it
through ``libcoreai_bridge.dylib`` (built from CoreAIBridge.swift, shipped
prebuilt) via ctypes, and convert each ONNX model on first use with the
coreai-onnx converter.  The converted bundle is cached next to the model in
``coreai_cache/``; the converter's own precision check against ONNX Runtime
must pass or nothing is cached.

``CoreAISession`` mirrors the slice of ``onnxruntime.InferenceSession`` the
processors use: ``get_inputs()``, ``get_outputs()``, ``run(names, feed)``.

Set ROOP_DISABLE_COREAI=1 to turn the backend off.
"""
from __future__ import annotations

import ctypes
import glob
import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libcoreai_bridge.dylib")
CACHE_DIR_NAME = "coreai_cache"
# Core AI's default placement.  Preferring .neuralEngine or .gpu measured
# identical (the preference is a hint), and preferring .cpu returned WRONG
# output (hyperswap 25 dB, max error 1.2) -- never pass 1.
_UNIT_DEFAULT = 0

_lib = None
_available = None


def available() -> bool:
    """Core AI usable here: macOS 27+, Apple Silicon, bridge + converter present."""
    global _available
    if _available is None:
        _available = _check_available()
    return _available


def _check_available() -> bool:
    if os.environ.get("ROOP_DISABLE_COREAI"):
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        if int(platform.mac_ver()[0].split(".")[0]) < 27:
            return False
    except ValueError:
        return False
    if importlib.util.find_spec("coreai_onnx") is None:
        return False
    try:
        _load_lib()
    except OSError as e:
        print(f"Core AI bridge unavailable ({e})")
        return False
    return True


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


def _cache_tag(model_path: str) -> str:
    """Invalidates on a converter upgrade or a replaced model file."""
    ver = importlib.metadata.version("coreai-onnx")
    return f"{ver}-{os.path.getsize(model_path)}"


def aimodel_path(model_path: str) -> str:
    cache = os.path.join(os.path.dirname(model_path), CACHE_DIR_NAME)
    name = os.path.basename(model_path)
    return os.path.join(cache, f"{name}.{_cache_tag(model_path)}.aimodel")


def ensure_aimodel(model_path: str) -> str | None:
    """Return the cached .aimodel for *model_path*, converting it if needed.

    Conversion runs in a child process (a crash in the beta runtime cannot
    take the app down) and only succeeds if the converter's precision check
    against ONNX Runtime passes.  Returns None on any failure.
    """
    out = aimodel_path(model_path)
    if os.path.isdir(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    name = os.path.basename(model_path)
    # Drop bundles from older converters / model files.
    for stale in glob.glob(os.path.join(os.path.dirname(out), f"{name}.*.aimodel")):
        subprocess.run(["rm", "-rf", stale], check=False)
    print(f"Converting {name} for Core AI (first use only)...")
    proc = subprocess.run(
        [sys.executable, "-m", "coreai_onnx._cli", "convert", model_path,
         "-o", out, "--compute-unit", "ane"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.isdir(out):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        print(f"Core AI conversion failed for {name}: {' | '.join(tail)}")
        subprocess.run(["rm", "-rf", out], check=False)
        return None
    return out


class _Arg:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class CoreAISession:
    """ORT-shaped session over a converted .aimodel."""

    def __init__(self, model_path: str):
        import onnx

        aimodel = ensure_aimodel(model_path)
        if aimodel is None:
            raise RuntimeError("conversion failed")
        m = onnx.load(model_path, load_external_data=False)
        init = {i.name for i in m.graph.initializer}

        def dims(vi):
            return [d.dim_value if d.dim_value > 0 else None
                    for d in vi.type.tensor_type.shape.dim]
        self._inputs = [_Arg(i.name, dims(i)) for i in m.graph.input if i.name not in init]
        self._outputs = [_Arg(o.name, dims(o)) for o in m.graph.output]
        if any(d is None for o in self._outputs for d in o.shape):
            # Output buffers are preallocated from the ONNX shapes.
            raise RuntimeError("dynamic output shape")
        # One call at a time per session.  Concurrent runs on one function
        # corrupted output intermittently: GPEN-2048 under 4 threads gave 1 in
        # 40 results off by up to 1.82 (of a 2.74 range); serial runs are
        # bit-repeatable.  Costs ~13% of the overlap gain (hyperswap 14.9 ms
        # per run concurrent vs 17.2 serial).
        import threading
        self._run_lock = threading.Lock()
        self._lib = _load_lib()
        self._h = self._lib.cai_load(aimodel.encode(), _UNIT_DEFAULT)
        if not self._h:
            raise RuntimeError(f"Core AI could not load {aimodel}")
        self.model_path = model_path

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CoreAI"]

    def run(self, output_names, feed):
        wanted = [o for o in self._outputs if not output_names or o.name in output_names]
        if output_names:  # keep the caller's order, like ORT
            order = {n: i for i, n in enumerate(output_names)}
            wanted.sort(key=lambda o: order[o.name])
        arrays = [np.ascontiguousarray(feed[i.name], dtype=np.float32) for i in self._inputs]
        n_in = len(arrays)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        in_names = (ctypes.c_char_p * n_in)(*[i.name.encode() for i in self._inputs])
        in_data = (c_float_p * n_in)(*[a.ctypes.data_as(c_float_p) for a in arrays])
        shapes = [d for a in arrays for d in a.shape]
        in_shapes = (ctypes.c_int64 * len(shapes))(*shapes)
        in_ranks = (ctypes.c_int32 * n_in)(*[a.ndim for a in arrays])

        outs = [np.empty(o.shape, dtype=np.float32) for o in wanted]
        n_out = len(outs)
        out_names = (ctypes.c_char_p * n_out)(*[o.name.encode() for o in wanted])
        out_data = (c_float_p * n_out)(*[o.ctypes.data_as(c_float_p) for o in outs])
        out_counts = (ctypes.c_int64 * n_out)(*[o.size for o in outs])

        with self._run_lock:
            rc = self._lib.cai_run(ctypes.c_void_p(self._h), n_in, in_names, in_data,
                                   in_shapes, in_ranks, n_out, out_names, out_data, out_counts)
        if rc != 0:
            raise RuntimeError(f"Core AI run failed (code {rc})")
        return outs

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.cai_free(ctypes.c_void_p(h))
            self._h = None
