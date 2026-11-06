"""ONNX-to-MLX graph executor.

Loads an ONNX model and runs it node-by-node using MLX operations.
Provides the same ``run(output_names, input_dict)`` interface as
``onnxruntime.InferenceSession`` so callers can swap backends.

Spatial tensors (4-D) are kept in ONNX NCHW order throughout and
converted to MLX's native NHWC only for Conv / MaxPool / Resize ops,
then immediately converted back.  This avoids axis-mismatch bugs in
Reshape / Split / Concat and makes the mapping to ONNX semantics
explicit.

Supported ops (sufficient for YOLO, InSwapper, w600k_r50, 2d106det):
  Conv, Gemm, BatchNormalization, PRelu, LeakyRelu, Relu, Sigmoid,
  Tanh, Add, Sub, Mul, Div, Sqrt, Floor, ReduceMean, Reshape, Concat,
  Split, Slice, Unsqueeze, Flatten, Shape, Cast, Constant, MaxPool,
  Resize, Pad.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Winograd conv2d off, unless the user set it.  On M5 (GPU neural
# accelerators) MLX's winograd path runs fp32 3x3 convs at TF32 precision,
# and the transforms amplify the rounding (MLX PR #4242, closed unmerged).
# Measured 2026-11-05 on M5 Pro, mlx 0.32.4: every interpreter-routed model
# goes from ~70 dB to ~130 dB vs the ORT CPU reference (i.e. true fp32) at
# neutral speed (GPEN-2048 551 -> 532 ms, GPEN-512 142 -> 146, xseg 11.2 ->
# 11.4).  MLX reads the switch lazily, so this works even if mlx was
# imported first; set before the import anyway.
os.environ.setdefault("MLX_CONV_WINOGRAD", "0")

import mlx.core as mx  # noqa: E402
import mlx.nn as mlx_nn  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, numpy_helper  # noqa: E402


# Serializes all MLX inference across worker threads.  The GPU executes
# one graph at a time anyway; the lock buys thread-safety so the CPU-side
# work (decode, warps, compositing, encode) can overlap GPU inference.
MLX_RUN_LOCK = threading.Lock()


_ONNX_TO_MLX_DTYPE = {
    TensorProto.FLOAT:   mx.float32,
    TensorProto.FLOAT16: mx.float16,
    TensorProto.DOUBLE:  mx.float32,  # downcast
    TensorProto.INT32:   mx.int32,
    TensorProto.INT64:   mx.int32,    # MLX has no int64
    TensorProto.UINT8:   mx.uint8,
    TensorProto.BOOL:    mx.bool_,
}

_ONNX_TO_NP_DTYPE = {
    TensorProto.FLOAT:   np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE:  np.float64,
    TensorProto.INT32:   np.int32,
    TensorProto.INT64:   np.int64,
    TensorProto.UINT8:   np.uint8,
    TensorProto.BOOL:    np.bool_,
}


def _np_to_mx(arr: np.ndarray) -> mx.array:
    """Convert numpy array to MLX, downcasting int64 → int32.

    Large int64 sentinel values (INT64_MIN/MAX) used by ONNX Slice are
    clipped into int32 range so they remain detectable as sentinels
    (abs value >= 2**30) rather than wrapping to small values like 0 or 1.
    """
    if arr.dtype == np.int64:
        arr = arr.clip(-(2**31), 2**31 - 1).astype(np.int32)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return mx.array(arr)


def _attr_value(a):
    """Parse one AttributeProto into a Python value."""
    t = a.type
    if t == onnx.AttributeProto.FLOAT:
        return a.f
    if t == onnx.AttributeProto.INT:
        return a.i
    if t == onnx.AttributeProto.STRING:
        return a.s
    if t == onnx.AttributeProto.TENSOR:
        return numpy_helper.to_array(a.t)
    if t == onnx.AttributeProto.FLOATS:
        return list(a.floats)
    if t == onnx.AttributeProto.INTS:
        return list(a.ints)
    return None


# mirrors onnxruntime.NodeArg
class _NodeArg:
    def __init__(self, name: str, shape: list):
        self._name = name
        self._shape = shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> list:
        return self._shape


class OnnxMLXSession:
    """Execute an ONNX model with MLX.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` file.
    dtype:
        Float dtype for inference (``mx.float32`` or ``mx.float16``).
        Defaults to float32 for accuracy.
    """

    # Ops heavy enough that we force mx.eval() on their output to keep the
    # lazy computation graph small and avoid OOM on large models.
    # Disabled for small models (< 500 nodes) where the full lazy graph
    # fits comfortably and single-shot eval is faster.
    _EVAL_AFTER_OPS = frozenset({"Gemm", "MatMul"})

    # Metal memory limits are set once per process, not per session.
    _metal_configured: bool = False

    @staticmethod
    def _configure_metal() -> None:
        """Set Metal memory and cache limits once at first session creation."""
        try:
            import psutil
            total = psutil.virtual_memory().total
            try:
                import roop.globals
                limit_gb = getattr(getattr(roop.globals, "CFG", None), "memory_limit", 0) or 0
            except Exception:
                limit_gb = 0
            if limit_gb > 0:
                metal_limit = int(limit_gb * 1024 ** 3)
            else:
                # Reserve 2 GB for OS + non-MLX app memory.
                metal_limit = max(total - 2 * 1024 ** 3, total // 2)
            mx.set_memory_limit(metal_limit)
            # Keep a 1 GB buffer pool for recently freed Metal buffers.
            mx.set_cache_limit(1 * 1024 ** 3)
        except Exception:
            pass

    def __init__(self, model_path: str, dtype: Any = mx.float32):
        if not OnnxMLXSession._metal_configured:
            OnnxMLXSession._configure_metal()
            OnnxMLXSession._metal_configured = True

        self._dtype = dtype
        # Per-node parsed attributes, filled lazily on first access so
        # run() never re-scans protobuf attribute lists per frame.
        self._attr_cache: dict[int, dict] = {}
        model = onnx.load(model_path)
        graph = model.graph

        # Pin one Python wrapper per node.  protobuf's C backend creates
        # transient wrappers on every `graph.node` iteration, which would
        # make the id()-keyed attribute cache collide — and re-creating
        # wrappers per frame costs time in run() anyway.
        self._nodes = list(graph.node)

        # Extract weights as MLX arrays, then immediately free the serialized
        # bytes from the protobuf so the ONNX file data is not held in Python
        # memory alongside the MLX copies (would double the model footprint).
        self._weights: dict[str, mx.array] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init).copy()
            if arr.dtype in (np.float32, np.float64, np.float16):
                arr = arr.astype(np.float32 if dtype == mx.float32 else np.float16)
            self._weights[init.name] = _np_to_mx(arr)
            # Free serialized bytes immediately — they're now in self._weights.
            init.ClearField("raw_data")
            init.ClearField("float_data")
            init.ClearField("double_data")
            init.ClearField("int32_data")
            init.ClearField("int64_data")

        # Pre-compute all Constant nodes and store them alongside weights.
        # GFPGAN has 262 Constant nodes; without this they are recreated as
        # mx.array() on every single inference call (one per frame per face).
        # Storing them in self._weights means run() just copies the reference.
        _float_dtype = dtype if dtype in (mx.float16, mx.float32) else mx.float32
        for node in self._nodes:
            if node.op_type != "Constant" or not node.output or not node.output[0]:
                continue
            out_name = node.output[0]
            val_attr = self._attr(node, "value", None)
            if val_attr is not None:
                self._weights[out_name] = _np_to_mx(val_attr)
            else:
                fval = self._attr(node, "value_float", None)
                ival = self._attr(node, "value_int", None)
                if fval is not None:
                    self._weights[out_name] = mx.array(fval, dtype=_float_dtype)
                elif ival is not None:
                    self._weights[out_name] = mx.array(ival, dtype=mx.int32)

        # Pre-evaluate all weights + constants so Metal GPU-upload happens at
        # load time rather than spiking on the first inference frame.
        if self._weights:
            mx.eval(*self._weights.values())

        # Keep only the graph (nodes + topology); model can be GC-ed.
        self._graph = graph

        self._input_names = [i.name for i in graph.input
                             if i.name not in self._weights]
        self._output_names = [o.name for o in graph.output]

        self._inputs_meta = [
            _NodeArg(i.name,
                     [d.dim_value for d in i.type.tensor_type.shape.dim])
            for i in graph.input if i.name not in self._weights
        ]
        self._outputs_meta = [
            _NodeArg(o.name,
                     [d.dim_value for d in o.type.tensor_type.shape.dim])
            for o in graph.output
        ]

        # Precompute last-use index for each non-weight value so run() can
        # delete intermediate activations the moment they're no longer needed.
        self._last_use: dict[str, int] = {}
        for i, node in enumerate(self._nodes):
            for inp in node.input:
                if inp:
                    self._last_use[inp] = i

        self._dispatch = {
            "Conv":               self._op_conv,
            "Gemm":               self._op_gemm,
            "BatchNormalization": self._op_batchnorm,
            "PRelu":              self._op_prelu,
            "LeakyRelu":          self._op_leakyrelu,
            "Relu":               self._op_relu,
            "Sigmoid":            self._op_sigmoid,
            "Tanh":               self._op_tanh,
            "Add":                self._op_add,
            "Sub":                self._op_sub,
            "Mul":                self._op_mul,
            "Div":                self._op_div,
            "Sqrt":               self._op_sqrt,
            "Floor":              self._op_floor,
            "ReduceMean":         self._op_reducemean,
            "Reshape":            self._op_reshape,
            "Concat":             self._op_concat,
            "Split":              self._op_split,
            "Slice":              self._op_slice,
            "Unsqueeze":          self._op_unsqueeze,
            "Squeeze":            self._op_squeeze,
            "Flatten":            self._op_flatten,
            "Shape":              self._op_shape,
            "Cast":               self._op_cast,
            "Constant":           self._op_constant,
            "MaxPool":            self._op_maxpool,
            "AveragePool":        self._op_avgpool,
            "GlobalAveragePool":  self._op_global_avgpool,
            "Resize":             self._op_resize,
            "Pad":                self._op_pad,
            "Transpose":          self._op_transpose,
            "MatMul":             self._op_matmul,
            "Clip":               self._op_clip,
            "Expand":             self._op_expand,
            "Gather":             self._op_gather,
            "Softmax":            self._op_softmax,
            "Reciprocal":         self._op_reciprocal,
            "Max":                self._op_max,
            "Min":                self._op_min,
            "ConvTranspose":      self._op_conv_transpose,
            "Pow":                self._op_pow,
            "ReduceSum":          self._op_reducesum,
            "ReduceMax":          self._op_reducemax,
            "ConstantOfShape":    self._op_constant_of_shape,
            "Tile":               self._op_tile,
            "Identity":           self._op_identity,
            "Abs":                self._op_abs,
            "Sign":               self._op_sign,
            "Round":              self._op_round,
            "Less":               self._op_less,
            "Where":              self._op_where,
            "Neg":                self._op_neg,
            "Equal":              self._op_equal,
            "Erf":                self._op_erf,
            "Exp":                self._op_exp,
            "Log":                self._op_log,
            "InstanceNormalization": self._op_instance_norm,
            "ReduceProd":         self._op_reduceprod,
            "ScatterElements":    self._op_scatter_elements,
            "TopK":               self._op_topk,
            "ArgMin":             self._op_argmin,
            "ArgMax":             self._op_argmax,
            "CumSum":             self._op_cumsum,
            "NonZero":            self._op_nonzero,
            "ScatterND":          self._op_scatter_nd,
            "GatherND":           self._op_gather_nd,
            "GatherElements":     self._op_gather_elements,
            "Einsum":             self._op_einsum,
            "Not":                self._op_not,
            "And":                self._op_and,
            "Or":                 self._op_or,
            "GreaterOrEqual":     self._op_greater_equal,
            "LessOrEqual":        self._op_less_equal,
            "Greater":            self._op_greater,
            "HardSigmoid":        self._op_hard_sigmoid,
            "DepthToSpace":       self._op_depth_to_space,
            "ReduceL2":           self._op_reduce_l2,
        }

        n_nodes = sum(1 for n in self._nodes if n.op_type != "Constant")
        self._eval_after = self._EVAL_AFTER_OPS if n_nodes >= 800 else frozenset()

        # Pre-transpose 2-D conv weights to MLX's OHWI layout at load time
        # instead of once per node per frame.  Only weights whose every use
        # is a Conv weight input can be converted in place.
        weight_uses: dict[str, set] = {}
        for node in self._nodes:
            for pos, inp in enumerate(node.input):
                if inp in self._weights:
                    weight_uses.setdefault(inp, set()).add((node.op_type, pos))
        self._conv_w_ohwi: set[str] = set()
        for wname, uses in weight_uses.items():
            if uses == {("Conv", 1)} and self._weights[wname].ndim == 4:
                self._weights[wname] = mx.transpose(self._weights[wname], [0, 2, 3, 1])
                self._conv_w_ohwi.add(wname)
        if self._conv_w_ohwi:
            mx.eval(*[self._weights[n] for n in self._conv_w_ohwi])

    def _attr(self, node, name, default=None):
        """Cached ONNX attribute lookup (parsed once per node, not per run)."""
        cached = self._attr_cache.get(id(node))
        if cached is None:
            cached = {a.name: _attr_value(a) for a in node.attribute}
            self._attr_cache[id(node)] = cached
        val = cached.get(name)
        return default if val is None else val

    def get_inputs(self) -> list[_NodeArg]:
        return self._inputs_meta

    def get_outputs(self) -> list[_NodeArg]:
        return self._outputs_meta

    def run(self, output_names: list[str] | None,
            input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference.  Returns a list of numpy arrays.

        Memory strategy
        ---------------
        MLX uses lazy evaluation: operations build a computation graph and only
        execute when mx.eval() is called.  For large models (>500 MB) running
        the entire graph as one lazy expression causes peak memory proportional
        to all intermediate activations simultaneously.

        We break this in two ways:
        1. After Conv / Gemm / MatMul, call mx.eval() on the output to
           materialise it as a concrete buffer and sever the dependency chain.
        2. Delete activations from the values dict as soon as the last node
           that reads them has run (precomputed _last_use map).
        """
        if output_names is None:
            output_names = self._output_names
        output_set = frozenset(n for n in output_names if n)

        with MLX_RUN_LOCK:
            values: dict[str, mx.array] = dict(self._weights)
            for k, v in input_feed.items():
                values[k] = _np_to_mx(np.asarray(v))

            for i, node in enumerate(self._nodes):
                op = node.op_type
                if op == "Constant":
                    continue
                handler = self._dispatch.get(op)
                if handler is None:
                    raise NotImplementedError(
                        f"ONNX op '{op}' is not implemented in OnnxMLXSession"
                    )
                handler(node, values)

                if op in self._eval_after:
                    for out_name in node.output:
                        if out_name and out_name in values:
                            mx.eval(values[out_name])

                for inp in node.input:
                    if (inp
                            and self._last_use.get(inp) == i
                            and inp not in self._weights
                            and inp not in output_set
                            and inp in values):
                        del values[inp]

            mx.eval(*[values[n] for n in output_names if n in values])
            outputs = [np.array(values[n]) for n in output_names]
            # Drop all remaining activations (weights stay in self._weights).
            for k in [k for k in values if k not in self._weights]:
                del values[k]
            return outputs

    def _v(self, values: dict, name: str) -> mx.array | None:
        """Fetch a named value; return None if name is empty."""
        if not name:
            return None
        return values.get(name)

    def _set(self, values: dict, node, idx: int, val: mx.array):
        if idx < len(node.output) and node.output[idx]:
            values[node.output[idx]] = val

    @staticmethod
    def _to_nhwc(x: mx.array) -> mx.array:
        return mx.transpose(x, [0, 2, 3, 1])

    @staticmethod
    def _to_nchw(x: mx.array) -> mx.array:
        return mx.transpose(x, [0, 3, 1, 2])

    def _op_conv(self, node, values):
        x = self._v(values, node.input[0])
        w = self._v(values, node.input[1])
        b = self._v(values, node.input[2]) if len(node.input) > 2 and node.input[2] else None

        strides   = list(self._attr(node, "strides",   [1, 1]))
        pads      = list(self._attr(node, "pads",      [0, 0, 0, 0]))
        dilations = list(self._attr(node, "dilations", [1, 1]))
        groups    = self._attr(node, "group",     1)
        auto_pad  = self._attr(node, "auto_pad",  b"NOTSET")

        if x.ndim == 3:
            # 1-D convolution: input [N, C, L] → MLX needs [N, L, C]
            pads_1d = pads[:2] if len(pads) >= 2 else [0, 0]
            if pads_1d[0] != (pads_1d[1] if len(pads_1d) > 1 else 0):
                x = mx.pad(x, [(0,0),(0,0),(pads_1d[0], pads_1d[1] if len(pads_1d) > 1 else 0)])
                pad1d = 0
            elif auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
                pad1d = "same"
            else:
                pad1d = pads_1d[0]
            x_nlc = mx.transpose(x, [0, 2, 1])
            w_okc = mx.transpose(w, [0, 2, 1])
            out = mx.conv1d(x_nlc, w_okc, stride=strides[0], padding=pad1d,
                            dilation=dilations[0], groups=groups)
            if b is not None:
                out = out + b.reshape(1, 1, -1)
            values[node.output[0]] = mx.transpose(out, [0, 2, 1])
            return

        # ONNX pads: [top, left, bottom, right]
        while len(pads) < 4:
            pads.append(0)
        # mx.conv2d requires symmetric padding; use explicit mx.pad for asymmetric cases
        if pads[0] != pads[2] or pads[1] != pads[3]:
            x = mx.pad(x, [(0,0),(0,0),(pads[0],pads[2]),(pads[1],pads[3])])
            padding = 0
        elif auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
            padding = "same"
        else:
            padding = pads[0]  # symmetric; ONNX guarantees pads[0]==pads[2]

        # NCHW → NHWC for input; OIHW → OHWI for weight (pre-transposed at
        # load time for static weights, converted here only for dynamic ones)
        x_nhwc = self._to_nhwc(x)
        if node.input[1] in self._conv_w_ohwi:
            w_ohwi = w
        else:
            w_ohwi = mx.transpose(w, [0, 2, 3, 1])

        out = mx.conv2d(
            x_nhwc, w_ohwi,
            stride=strides,
            padding=padding,
            dilation=dilations,
            groups=groups,
        )
        if b is not None:
            out = out + b.reshape(1, 1, 1, -1)
        values[node.output[0]] = self._to_nchw(out)

    def _op_gemm(self, node, values):
        A = self._v(values, node.input[0])
        B = self._v(values, node.input[1])
        C = self._v(values, node.input[2]) if len(node.input) > 2 and node.input[2] else None

        alpha  = self._attr(node, "alpha",  1.0)
        beta   = self._attr(node, "beta",   1.0)
        transA = self._attr(node, "transA", 0)
        transB = self._attr(node, "transB", 0)

        if transA:
            A = mx.transpose(A)
        if transB:
            B = mx.transpose(B)

        out = alpha * (A @ B)
        if C is not None:
            out = out + beta * C
        values[node.output[0]] = out

    def _op_matmul(self, node, values):
        A = self._v(values, node.input[0])
        B = self._v(values, node.input[1])
        values[node.output[0]] = A @ B

    def _op_batchnorm(self, node, values):
        x     = self._v(values, node.input[0])
        scale = self._v(values, node.input[1])
        bias  = self._v(values, node.input[2])
        mean  = self._v(values, node.input[3])
        var   = self._v(values, node.input[4])
        eps   = self._attr(node, "epsilon", 1e-5)

        if x.ndim == 4:
            # NCHW: broadcast over N,H,W
            scale = scale.reshape(1, -1, 1, 1)
            bias  = bias.reshape(1, -1, 1, 1)
            mean  = mean.reshape(1, -1, 1, 1)
            var   = var.reshape(1, -1, 1, 1)
        # else 2D [N, C]: params already shape (C,) — broadcast works as-is

        out = (x - mean) / mx.sqrt(var + eps) * scale + bias
        values[node.output[0]] = out

    def _op_prelu(self, node, values):
        x     = self._v(values, node.input[0])
        slope = self._v(values, node.input[1])
        # slope shape: (C,) or (1,C,1,1)
        if slope.ndim == 1:
            slope = slope.reshape(1, -1, 1, 1) if x.ndim == 4 else slope
        values[node.output[0]] = mx.where(x >= 0, x, slope * x)

    def _op_leakyrelu(self, node, values):
        x     = self._v(values, node.input[0])
        alpha = self._attr(node, "alpha", 0.01)
        values[node.output[0]] = mx.where(x >= 0, x, alpha * x)

    def _op_relu(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.maximum(x, 0)

    def _op_sigmoid(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.sigmoid(x)

    def _op_tanh(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.tanh(x)

    def _op_add(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = a + b

    def _op_sub(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = a - b

    def _op_mul(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = a * b

    def _op_div(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = a / b

    def _op_sqrt(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.sqrt(x)

    def _op_floor(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.floor(x)

    def _op_clip(self, node, values):
        x   = self._v(values, node.input[0])
        mn  = self._v(values, node.input[1]) if len(node.input) > 1 and node.input[1] else None
        mx_ = self._v(values, node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if mn is None:
            mn = self._attr(node, "min", None)
        if mx_ is None:
            mx_ = self._attr(node, "max", None)
        if mn is not None:
            mn = float(np.array(mn))
            x = mx.maximum(x, mn)
        if mx_ is not None:
            mx_ = float(np.array(mx_))
            x = mx.minimum(x, mx_)
        values[node.output[0]] = x

    def _op_reducemean(self, node, values):
        x        = self._v(values, node.input[0])
        axes     = self._attr(node, "axes", None)
        keepdims = bool(self._attr(node, "keepdims", 1))
        if axes is None and len(node.input) > 1 and node.input[1]:
            axes_arr = self._v(values, node.input[1])
            axes = [int(a) for a in np.array(axes_arr)]
        values[node.output[0]] = mx.mean(x, axis=axes, keepdims=keepdims)

    def _op_reshape(self, node, values):
        x     = self._v(values, node.input[0])
        shape = self._v(values, node.input[1])
        shape_list = [int(s) for s in np.array(shape)]
        # Replace 0 with the original dim size
        orig = list(x.shape)
        shape_list = [orig[i] if shape_list[i] == 0 else shape_list[i]
                      for i in range(min(len(shape_list), len(orig)))] + shape_list[len(orig):]
        values[node.output[0]] = mx.reshape(x, shape_list)

    def _op_concat(self, node, values):
        inputs = [self._v(values, n) for n in node.input]
        axis   = self._attr(node, "axis", 0)
        values[node.output[0]] = mx.concatenate(inputs, axis=axis)

    def _op_split(self, node, values):
        x    = self._v(values, node.input[0])
        axis = self._attr(node, "axis", 0)
        if len(node.input) > 1 and node.input[1]:
            split_arr = self._v(values, node.input[1])
            split = [int(s) for s in np.array(split_arr)]
        else:
            split = self._attr(node, "split", None)

        if split is None:
            n_out = len(node.output)
            dim   = x.shape[axis]
            chunk = dim // n_out
            split = [chunk] * n_out

        # convert sizes to indices for mx.split
        indices = []
        acc = 0
        for s in split[:-1]:
            acc += s
            indices.append(acc)

        parts = mx.split(x, indices, axis=axis)
        for i, name in enumerate(node.output):
            if name:
                values[name] = parts[i]

    def _op_slice(self, node, values):
        data = self._v(values, node.input[0])
        starts_t = self._v(values, node.input[1])
        ends_t   = self._v(values, node.input[2])
        axes_t   = self._v(values, node.input[3]) if len(node.input) > 3 and node.input[3] else None
        steps_t  = self._v(values, node.input[4]) if len(node.input) > 4 and node.input[4] else None

        starts = [int(s) for s in np.array(starts_t)]
        ends   = [int(e) for e in np.array(ends_t)]
        axes   = [int(a) for a in np.array(axes_t)] if axes_t is not None else list(range(len(starts)))
        steps  = [int(s) for s in np.array(steps_t)] if steps_t is not None else [1] * len(starts)

        # Build a full slice tuple
        ndim = data.ndim
        slices = [slice(None)] * ndim
        for ax, st, en, step in zip(axes, starts, ends, steps):
            # Clamp to INT_MAX sentinel → None for open-ended slice
            en_val = None if abs(en) >= 2**30 else en
            slices[ax] = slice(st, en_val, step)
        values[node.output[0]] = data[tuple(slices)]

    def _op_unsqueeze(self, node, values):
        x = self._v(values, node.input[0])
        if len(node.input) > 1 and node.input[1]:
            axes_t = self._v(values, node.input[1])
            axes = sorted([int(a) for a in np.array(axes_t)])
        else:
            axes = sorted(self._attr(node, "axes", [0]))
        for ax in axes:
            x = mx.expand_dims(x, ax)
        values[node.output[0]] = x

    def _op_squeeze(self, node, values):
        x = self._v(values, node.input[0])
        if len(node.input) > 1 and node.input[1]:
            axes = sorted([int(a) for a in np.array(self._v(values, node.input[1]))],
                          reverse=True)
        else:
            axes = sorted(self._attr(node, "axes", []), reverse=True)
        for ax in axes:
            x = mx.squeeze(x, ax)
        values[node.output[0]] = x

    def _op_flatten(self, node, values):
        x    = self._v(values, node.input[0])
        axis = self._attr(node, "axis", 1)
        if axis < 0:
            axis = x.ndim + axis
        outer = 1
        for d in x.shape[:axis]:
            outer *= d
        inner = 1
        for d in x.shape[axis:]:
            inner *= d
        values[node.output[0]] = mx.reshape(x, [outer, inner])

    def _op_shape(self, node, values):
        x = self._v(values, node.input[0])
        start = self._attr(node, "start", 0)
        end   = self._attr(node, "end", None)
        shape = list(x.shape)
        shape_slice = shape[start:end]
        values[node.output[0]] = mx.array(shape_slice, dtype=mx.int32)

    def _op_cast(self, node, values):
        x   = self._v(values, node.input[0])
        to  = self._attr(node, "to", TensorProto.FLOAT)
        dst = _ONNX_TO_MLX_DTYPE.get(to, mx.float32)
        values[node.output[0]] = x.astype(dst)

    def _op_constant(self, node, values):
        val = self._attr(node, "value", None)
        if val is not None:
            values[node.output[0]] = _np_to_mx(val)
        else:
            # float/int scalar attributes — respect model dtype so fp16 models
            # don't upcast every broadcast op that uses this constant.
            _fdtype = self._dtype if self._dtype in (mx.float16, mx.float32) else mx.float32
            fval = self._attr(node, "value_float", None)
            ival = self._attr(node, "value_int", None)
            if fval is not None:
                values[node.output[0]] = mx.array(fval, dtype=_fdtype)
            elif ival is not None:
                values[node.output[0]] = mx.array(ival, dtype=mx.int32)

    def _op_maxpool(self, node, values):
        x          = self._v(values, node.input[0])
        ks      = self._attr(node, "kernel_shape", [1, 1])
        strides = self._attr(node, "strides", [1, 1])
        pads    = self._attr(node, "pads", [0, 0, 0, 0])

        pad_h = pads[0]  # assume symmetric
        pad_w = pads[1]

        x_nhwc = self._to_nhwc(x)
        pool   = mlx_nn.MaxPool2d(
            kernel_size=ks,
            stride=strides,
            padding=[pad_h, pad_w],
        )
        out = pool(x_nhwc)
        values[node.output[0]] = self._to_nchw(out)

    def _op_avgpool(self, node, values):
        x       = self._v(values, node.input[0])
        ks      = self._attr(node, "kernel_shape", [1, 1])
        strides = self._attr(node, "strides", [1, 1])
        pads    = self._attr(node, "pads", [0, 0, 0, 0])

        x_nhwc = self._to_nhwc(x)
        pool   = mlx_nn.AvgPool2d(
            kernel_size=ks,
            stride=strides,
            padding=[pads[0], pads[1]],
        )
        out = pool(x_nhwc)
        values[node.output[0]] = self._to_nchw(out)

    def _op_global_avgpool(self, node, values):
        x = self._v(values, node.input[0])
        # x is NCHW; mean over H and W
        out = mx.mean(x, axis=(2, 3), keepdims=True)
        values[node.output[0]] = out

    def _op_resize(self, node, values):
        x = self._v(values, node.input[0])

        scales = None
        sizes  = None
        if len(node.input) == 2:
            # Opset 10: inputs are [X, scales]
            scales_arr = self._v(values, node.input[1])
            if scales_arr is not None:
                scales = [float(s) for s in np.array(scales_arr)]
        else:
            # Opset 11+: inputs are [X, roi, scales, sizes]
            if len(node.input) > 2 and node.input[2]:
                scales_arr = self._v(values, node.input[2])
                if scales_arr is not None:
                    scales = [float(s) for s in np.array(scales_arr)]
            if len(node.input) > 3 and node.input[3]:
                sizes_arr = self._v(values, node.input[3])
                if sizes_arr is not None:
                    sizes = [int(s) for s in np.array(sizes_arr)]
        if scales is None and sizes is None:
            raise NotImplementedError(
                f"Resize node without scales/sizes (inputs: {list(node.input)})"
            )

        mode = self._attr(node, "mode", b"nearest")
        if isinstance(mode, bytes):
            mode = mode.decode()

        n, c, h, w = x.shape

        if sizes is not None:
            out_h, out_w = sizes[2], sizes[3]
        elif scales is not None:
            out_h = int(h * scales[2])
            out_w = int(w * scales[3])
        else:
            values[node.output[0]] = x
            return

        x_nhwc = self._to_nhwc(x)  # [N,H,W,C]

        if mode == "nearest":
            # nearest-neighbour via repeat
            scale_h = out_h // h
            scale_w = out_w // w
            if scale_h * h == out_h and scale_w * w == out_w:
                # integer scale: use repeat
                out = mx.repeat(mx.repeat(x_nhwc, scale_h, axis=1), scale_w, axis=2)
            else:
                # non-integer: use mx.image.resize
                out = self._mlx_resize(x_nhwc, out_h, out_w, "nearest")
        else:
            out = self._mlx_resize(x_nhwc, out_h, out_w, "linear")

        values[node.output[0]] = self._to_nchw(out)

    @staticmethod
    def _mlx_resize(x_nhwc: mx.array, out_h: int, out_w: int, mode: str) -> mx.array:
        """Resize [N,H,W,C] to [N,out_h,out_w,C] using pure MLX ops."""
        _, h, w, _ = x_nhwc.shape

        if mode != "linear":
            y_idx = (mx.arange(out_h) * (h / out_h)).astype(mx.int32)
            x_idx = (mx.arange(out_w) * (w / out_w)).astype(mx.int32)
            return mx.take(mx.take(x_nhwc, mx.clip(y_idx, 0, h - 1), axis=1),
                           mx.clip(x_idx, 0, w - 1), axis=2)

        # Bilinear, half-pixel coordinate transform.  Separable: interpolate
        # rows first, then columns of that result.  Same maths as blending the
        # four corners at once, but 4 gathers and 2 blends instead of 8 and 4,
        # and the column pass runs on the height-resized (not full) tensor —
        # GFPGAN 212 -> 148 ms end to end (42 Resize nodes), parity unchanged.
        def _axis(out_n: int, n: int):
            src = (mx.arange(out_n, dtype=mx.float32) + 0.5) * (n / out_n) - 0.5
            src = mx.clip(src, 0.0, n - 1.0)
            i0 = mx.floor(src).astype(mx.int32)
            return i0, mx.minimum(i0 + 1, n - 1), src - i0.astype(mx.float32)

        y0, y1, fy = _axis(out_h, h)
        x0, x1, fx = _axis(out_w, w)

        top = mx.take(x_nhwc, y0, axis=1)
        rows = top + (mx.take(x_nhwc, y1, axis=1) - top) * fy.reshape(1, -1, 1, 1)
        left = mx.take(rows, x0, axis=2)
        return left + (mx.take(rows, x1, axis=2) - left) * fx.reshape(1, 1, -1, 1)

    def _op_pad(self, node, values):
        x = self._v(values, node.input[0])
        if len(node.input) > 1 and node.input[1]:
            pads_t = self._v(values, node.input[1])
            pads   = [int(p) for p in np.array(pads_t)]
        else:
            pads = self._attr(node, "pads", [0] * (2 * x.ndim))

        const_val = 0.0
        if len(node.input) > 2 and node.input[2]:
            cv = self._v(values, node.input[2])
            if cv is not None:
                const_val = float(np.array(cv))

        mode = self._attr(node, "mode", b"constant")
        if isinstance(mode, bytes):
            mode = mode.decode()

        # ONNX pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        ndim = x.ndim
        pad_width = [(pads[i], pads[i + ndim]) for i in range(ndim)]
        if mode == "constant":
            values[node.output[0]] = mx.pad(x, pad_width, constant_values=const_val)
        elif mode == "reflect":
            values[node.output[0]] = self._reflect_pad(x, pad_width)
        elif mode == "edge":
            # Replicate the border value.  Used to fall through to reflect,
            # which mirrors the pixel one step in instead — DeOldify's
            # ReplicationPad+AvgPool blur then drifted up to 17/255 per
            # channel from the ONNX reference.
            values[node.output[0]] = mx.pad(x, pad_width, mode="edge")
        else:
            raise NotImplementedError(f"Pad mode '{mode}' is not implemented")

    def _op_transpose(self, node, values):
        x    = self._v(values, node.input[0])
        perm = self._attr(node, "perm", None)
        if perm is None:
            perm = list(range(x.ndim - 1, -1, -1))
        values[node.output[0]] = mx.transpose(x, perm)

    def _op_expand(self, node, values):
        x          = self._v(values, node.input[0])
        shape_hint = [int(s) for s in np.array(self._v(values, node.input[1]))]
        # ONNX Expand uses numpy broadcast rules: compute the true output shape
        # from both the input shape and the hint, then broadcast to it.
        out_shape = list(np.broadcast_shapes(x.shape, shape_hint))
        values[node.output[0]] = mx.broadcast_to(x, out_shape)

    def _op_gather(self, node, values):
        x       = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        axis    = self._attr(node, "axis", 0)
        values[node.output[0]] = mx.take(x, indices.astype(mx.int32), axis=axis)

    @staticmethod
    def _reflect_pad(x: mx.array, pad_width: list[tuple[int, int]]) -> mx.array:
        """Reflect padding via index reversal (MLX only supports constant pad)."""
        for axis, (before, after) in enumerate(pad_width):
            if before == 0 and after == 0:
                continue
            n = x.shape[axis]
            parts = []
            if before > 0:
                # reflect: indices [before, before-1, ..., 1]
                idx = mx.array(list(range(before, 0, -1)), dtype=mx.int32)
                parts.append(mx.take(x, idx, axis=axis))
            parts.append(x)
            if after > 0:
                idx = mx.array(list(range(n - 2, n - 2 - after, -1)), dtype=mx.int32)
                parts.append(mx.take(x, idx, axis=axis))
            x = mx.concatenate(parts, axis=axis)
        return x

    def _op_softmax(self, node, values):
        x    = self._v(values, node.input[0])
        axis = self._attr(node, "axis", -1)
        values[node.output[0]] = mx.softmax(x, axis=axis)

    def _op_reciprocal(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.reciprocal(x)

    def _op_max(self, node, values):
        result = self._v(values, node.input[0])
        for name in node.input[1:]:
            result = mx.maximum(result, self._v(values, name))
        values[node.output[0]] = result

    def _op_min(self, node, values):
        result = self._v(values, node.input[0])
        for name in node.input[1:]:
            result = mx.minimum(result, self._v(values, name))
        values[node.output[0]] = result

    def _op_conv_transpose(self, node, values):
        x = self._v(values, node.input[0])
        w = self._v(values, node.input[1])
        b = self._v(values, node.input[2]) if len(node.input) > 2 and node.input[2] else None

        strides   = self._attr(node, "strides",          [1, 1])
        pads      = self._attr(node, "pads",              [0, 0, 0, 0])
        groups    = self._attr(node, "group",             1)
        out_pads  = self._attr(node, "output_padding",    [0, 0])

        # ONNX ConvTranspose pads = output crop amounts: [h_beg, w_beg, h_end, w_end]
        # Run with no padding then crop manually to support asymmetric pads.
        # ONNX ConvTranspose weight: (C_in, C_out/g, kH, kW) → MLX (C_out, kH, kW, C_in/g)
        w_mlx = mx.transpose(w, [1, 2, 3, 0])

        x_nhwc = self._to_nhwc(x)
        out = mx.conv_transpose2d(
            x_nhwc, w_mlx,
            stride=strides,
            padding=0,          # we handle cropping ourselves
            output_padding=out_pads,
            groups=groups,
        )
        if b is not None:
            out = out + b.reshape(1, 1, 1, -1)

        out = self._to_nchw(out)  # [N, C, H, W]

        # Crop output per ONNX pads [h_beg, w_beg, h_end, w_end]
        if any(pads):
            h_b, w_b, h_e, w_e = pads
            H, W = out.shape[2], out.shape[3]
            h_end = H - h_e if h_e else H
            w_end = W - w_e if w_e else W
            out = out[:, :, h_b:h_end, w_b:w_end]

        values[node.output[0]] = out

    def _op_pow(self, node, values):
        x = self._v(values, node.input[0])
        y = self._v(values, node.input[1])
        values[node.output[0]] = mx.power(x, y)

    def _op_reducesum(self, node, values):
        x        = self._v(values, node.input[0])
        keepdims = bool(self._attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes_arr = self._v(values, node.input[1])
            axes = [int(a) for a in np.array(axes_arr)]
        else:
            axes = self._attr(node, "axes", None)
        if axes is not None and len(axes) == 0:
            axes = None
        values[node.output[0]] = mx.sum(x, axis=axes, keepdims=keepdims)

    def _op_reducemax(self, node, values):
        x        = self._v(values, node.input[0])
        keepdims = bool(self._attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes_arr = self._v(values, node.input[1])
            axes = [int(a) for a in np.array(axes_arr)]
        else:
            axes = self._attr(node, "axes", None)
        values[node.output[0]] = mx.max(x, axis=axes, keepdims=keepdims)

    def _op_constant_of_shape(self, node, values):
        shape_arr = self._v(values, node.input[0])
        shape = [int(s) for s in np.array(shape_arr)]
        val_attr = self._attr(node, "value", None)
        if val_attr is not None:
            fill_val = float(val_attr.flat[0])
            dtype = _ONNX_TO_MLX_DTYPE.get(
                TensorProto.FLOAT, mx.float32
            )
        else:
            fill_val = 0.0
            dtype = mx.float32
        values[node.output[0]] = mx.full(shape, fill_val, dtype=dtype)

    def _op_range(self, node, values):
        start = float(self._v(values, node.input[0]))
        limit = float(self._v(values, node.input[1]))
        delta = float(self._v(values, node.input[2]))
        values[node.output[0]] = mx.arange(start, limit, delta, dtype=mx.float32)

    def _op_tile(self, node, values):
        x       = self._v(values, node.input[0])
        repeats = [int(r) for r in np.array(self._v(values, node.input[1]))]
        result  = x
        for axis, rep in enumerate(repeats):
            if rep != 1:
                result = mx.concatenate([result] * rep, axis=axis)
        values[node.output[0]] = result

    def _op_identity(self, node, values):
        values[node.output[0]] = self._v(values, node.input[0])

    def _op_abs(self, node, values):
        values[node.output[0]] = mx.abs(self._v(values, node.input[0]))

    def _op_sign(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = mx.sign(x)

    def _op_round(self, node, values):
        values[node.output[0]] = mx.round(self._v(values, node.input[0]))

    def _op_neg(self, node, values):
        values[node.output[0]] = -self._v(values, node.input[0])

    def _op_less(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = (a < b)

    def _op_equal(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = mx.equal(a, b)

    def _op_where(self, node, values):
        cond = self._v(values, node.input[0])
        x    = self._v(values, node.input[1])
        y    = self._v(values, node.input[2])
        values[node.output[0]] = mx.where(cond, x, y)

    def _op_erf(self, node, values):
        values[node.output[0]] = mx.erf(self._v(values, node.input[0]))

    def _op_exp(self, node, values):
        values[node.output[0]] = mx.exp(self._v(values, node.input[0]))

    def _op_log(self, node, values):
        values[node.output[0]] = mx.log(self._v(values, node.input[0]))

    def _op_instance_norm(self, node, values):
        x     = self._v(values, node.input[0])   # [N, C, d1, ..., dn]
        scale = self._v(values, node.input[1])   # [C]
        bias  = self._v(values, node.input[2])   # [C]
        eps   = self._attr(node, "epsilon", 1e-5)

        # Normalise over all spatial axes (everything beyond [N, C])
        spatial_axes = tuple(range(2, x.ndim))
        mean = mx.mean(x, axis=spatial_axes, keepdims=True)
        var  = mx.mean((x - mean) ** 2, axis=spatial_axes, keepdims=True)
        out  = (x - mean) / mx.sqrt(var + eps)

        # Broadcast scale/bias to match input rank
        param_shape = (1, -1) + (1,) * (x.ndim - 2)
        out = out * scale.reshape(param_shape) + bias.reshape(param_shape)
        values[node.output[0]] = out

    def _op_reduceprod(self, node, values):
        x        = self._v(values, node.input[0])
        keepdims = bool(self._attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes = [int(a) for a in np.array(self._v(values, node.input[1]))]
        else:
            axes = self._attr(node, "axes", None)
        values[node.output[0]] = mx.prod(x, axis=axes, keepdims=keepdims)

    def _op_scatter_elements(self, node, values):
        data    = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        updates = self._v(values, node.input[2])
        axis = self._attr(node, "axis", 0)
        arr  = np.array(data)
        idx = np.array(indices).astype(np.intp)
        upd = np.array(updates)
        np.put_along_axis(arr, idx, upd, axis=axis)
        values[node.output[0]] = _np_to_mx(arr)

    def _op_topk(self, node, values):
        x    = self._v(values, node.input[0])
        k_t  = self._v(values, node.input[1])
        k    = int(np.array(k_t))
        axis = self._attr(node, "axis", -1)
        largest = bool(self._attr(node, "largest", 1))
        sorted_ = bool(self._attr(node, "sorted", 1))

        arr = np.array(x)
        if largest:
            idx = np.argsort(arr, axis=axis)[..., -k:]
            if sorted_:
                # reverse so largest comes first
                idx = np.flip(idx, axis=axis)
        else:
            idx = np.argsort(arr, axis=axis)[..., :k]
            if sorted_:
                idx = np.flip(idx, axis=axis)
        vals = np.take_along_axis(arr, idx, axis=axis)
        values[node.output[0]] = _np_to_mx(vals.copy())
        if len(node.output) > 1 and node.output[1]:
            values[node.output[1]] = _np_to_mx(idx.astype(np.int32).copy())

    def _op_argmin(self, node, values):
        x        = self._v(values, node.input[0])
        axis     = self._attr(node, "axis", 0)
        keepdims = bool(self._attr(node, "keepdims", 1))
        values[node.output[0]] = mx.argmin(x, axis=axis, keepdims=keepdims)

    def _op_argmax(self, node, values):
        x        = self._v(values, node.input[0])
        axis     = self._attr(node, "axis", 0)
        keepdims = bool(self._attr(node, "keepdims", 1))
        values[node.output[0]] = mx.argmax(x, axis=axis, keepdims=keepdims)

    def _op_cumsum(self, node, values):
        x    = self._v(values, node.input[0])
        axis = int(np.array(self._v(values, node.input[1])))
        exclusive = bool(self._attr(node, "exclusive", 0))
        reverse   = bool(self._attr(node, "reverse", 0))
        arr = np.array(x)
        if reverse:
            arr = np.flip(arr, axis=axis)
        out = np.cumsum(arr, axis=axis)
        if exclusive:
            out = np.roll(out, 1, axis=axis)
            slices = [slice(None)] * arr.ndim
            slices[axis] = slice(0, 1)
            out[tuple(slices)] = 0
        if reverse:
            out = np.flip(out, axis=axis)
        values[node.output[0]] = _np_to_mx(np.ascontiguousarray(out))

    def _op_nonzero(self, node, values):
        x   = self._v(values, node.input[0])
        arr = np.array(x)
        idx = np.nonzero(arr)
        values[node.output[0]] = _np_to_mx(np.array(idx, dtype=np.int32))

    def _op_scatter_nd(self, node, values):
        data    = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        updates = self._v(values, node.input[2])
        arr = np.array(data)
        idx = tuple(np.array(indices)[..., i] for i in range(indices.shape[-1]))
        arr[idx] = np.array(updates)
        values[node.output[0]] = _np_to_mx(arr)

    def _op_gather_nd(self, node, values):
        x       = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        arr = np.array(x)
        idx = np.array(indices)
        idx_t = tuple(idx[..., i] for i in range(idx.shape[-1]))
        values[node.output[0]] = _np_to_mx(arr[idx_t])

    def _op_gather_elements(self, node, values):
        x       = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        axis    = self._attr(node, "axis", 0)
        arr = np.array(x)
        idx = np.array(indices).astype(np.intp)
        values[node.output[0]] = _np_to_mx(np.take_along_axis(arr, idx, axis=axis))

    def _op_einsum(self, node, values):
        equation = self._attr(node, "equation", b"").decode()
        inputs   = [np.array(self._v(values, n)) for n in node.input]
        values[node.output[0]] = _np_to_mx(np.einsum(equation, *inputs))

    def _op_not(self, node, values):
        x = self._v(values, node.input[0])
        values[node.output[0]] = ~x.astype(mx.bool_)

    def _op_and(self, node, values):
        a = self._v(values, node.input[0]).astype(mx.bool_)
        b = self._v(values, node.input[1]).astype(mx.bool_)
        values[node.output[0]] = a & b

    def _op_or(self, node, values):
        a = self._v(values, node.input[0]).astype(mx.bool_)
        b = self._v(values, node.input[1]).astype(mx.bool_)
        values[node.output[0]] = a | b

    def _op_greater_equal(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = (a >= b)

    def _op_less_equal(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = (a <= b)

    def _op_greater(self, node, values):
        a = self._v(values, node.input[0])
        b = self._v(values, node.input[1])
        values[node.output[0]] = (a > b)

    def _op_hard_sigmoid(self, node, values):
        x     = self._v(values, node.input[0])
        alpha = self._attr(node, "alpha", 0.2)
        beta  = self._attr(node, "beta",  0.5)
        values[node.output[0]] = mx.clip(alpha * x + beta, 0.0, 1.0)

    def _op_depth_to_space(self, node, values):
        x         = self._v(values, node.input[0])   # [N, C, H, W]
        blocksize = self._attr(node, "blocksize", 2)
        mode      = self._attr(node, "mode", b"DCR")
        if isinstance(mode, bytes):
            mode = mode.decode()
        N, C, H, W = x.shape
        c_out = C // (blocksize * blocksize)
        if mode == "CRD":
            x = mx.reshape(x, (N, c_out, blocksize, blocksize, H, W))
            x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        else:  # DCR
            x = mx.reshape(x, (N, blocksize, blocksize, c_out, H, W))
            x = mx.transpose(x, (0, 3, 4, 1, 5, 2))
        values[node.output[0]] = mx.reshape(x, (N, c_out, H * blocksize, W * blocksize))

    def _op_reduce_l2(self, node, values):
        x        = self._v(values, node.input[0])
        keepdims = bool(self._attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes = [int(a) for a in np.array(self._v(values, node.input[1]))]
        else:
            axes = self._attr(node, "axes", None)
        values[node.output[0]] = mx.sqrt(mx.sum(x * x, axis=axes, keepdims=keepdims))
