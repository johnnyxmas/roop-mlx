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

from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------
def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
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
    return default


# ---------------------------------------------------------------------------
# NodeArg stub (mirrors onnxruntime.NodeArg)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------
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

    def __init__(self, model_path: str, dtype: Any = mx.float32):
        self._dtype = dtype
        model = onnx.load(model_path)
        graph = model.graph
        self._graph = graph

        # Pre-load all initializer weights as MLX arrays
        self._weights: dict[str, mx.array] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init).copy()
            if arr.dtype in (np.float32, np.float64, np.float16):
                arr = arr.astype(np.float32 if dtype == mx.float32 else np.float16)
            self._weights[init.name] = _np_to_mx(arr)

        # Collect graph input / output metadata
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

        # Dispatch table
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
        }

    # -- public interface (mirrors onnxruntime.InferenceSession) -----------

    def get_inputs(self) -> list[_NodeArg]:
        return self._inputs_meta

    def get_outputs(self) -> list[_NodeArg]:
        return self._outputs_meta

    def run(self, output_names: list[str] | None,
            input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference.  Returns a list of numpy arrays."""
        if output_names is None:
            output_names = self._output_names

        values: dict[str, mx.array] = dict(self._weights)
        for k, v in input_feed.items():
            values[k] = _np_to_mx(np.asarray(v))

        for node in self._graph.node:
            op = node.op_type
            handler = self._dispatch.get(op)
            if handler is None:
                raise NotImplementedError(
                    f"ONNX op '{op}' is not implemented in OnnxMLXSession"
                )
            handler(node, values)

        mx.eval(*[values[n] for n in output_names if n in values])
        return [np.array(values[n]) for n in output_names]

    # -- helpers -----------------------------------------------------------

    def _v(self, values: dict, name: str) -> mx.array | None:
        """Fetch a named value; return None if name is empty."""
        if not name:
            return None
        return values.get(name)

    def _set(self, values: dict, node, idx: int, val: mx.array):
        if idx < len(node.output) and node.output[idx]:
            values[node.output[idx]] = val

    # -- spatial layout helpers (NCHW ↔ NHWC) -----------------------------

    @staticmethod
    def _to_nhwc(x: mx.array) -> mx.array:
        return mx.transpose(x, [0, 2, 3, 1])

    @staticmethod
    def _to_nchw(x: mx.array) -> mx.array:
        return mx.transpose(x, [0, 3, 1, 2])

    # -- op implementations ------------------------------------------------

    def _op_conv(self, node, values):
        x = self._v(values, node.input[0])
        w = self._v(values, node.input[1])
        b = self._v(values, node.input[2]) if len(node.input) > 2 and node.input[2] else None

        strides   = _attr(node, "strides",   [1, 1])
        pads      = _attr(node, "pads",      [0, 0, 0, 0])
        dilations = _attr(node, "dilations", [1, 1])
        groups    = _attr(node, "group",     1)
        auto_pad  = _attr(node, "auto_pad",  b"NOTSET")

        # asymmetric padding not directly supported in mx.conv2d;
        # use explicit mx.pad if needed
        if pads[0] != pads[2] or pads[1] != pads[3]:
            # pad H first then W: pads = [h_start, w_start, h_end, w_end]
            x = mx.pad(x, [(0,0),(0,0),(pads[0],pads[2]),(pads[1],pads[3])])
            padding = 0
        elif auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
            padding = "same"
        else:
            padding = pads[0]  # symmetric; ONNX guarantees pads[0]==pads[2]

        # NCHW → NHWC for input; OIHW → OHWI for weight
        x_nhwc = self._to_nhwc(x)
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

        alpha  = _attr(node, "alpha",  1.0)
        beta   = _attr(node, "beta",   1.0)
        transA = _attr(node, "transA", 0)
        transB = _attr(node, "transB", 0)

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
        eps   = _attr(node, "epsilon", 1e-5)

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
        alpha = _attr(node, "alpha", 0.01)
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
            mn = _attr(node, "min", None)
        if mx_ is None:
            mx_ = _attr(node, "max", None)
        if mn is not None:
            mn = float(np.array(mn))
            x = mx.maximum(x, mn)
        if mx_ is not None:
            mx_ = float(np.array(mx_))
            x = mx.minimum(x, mx_)
        values[node.output[0]] = x

    def _op_reducemean(self, node, values):
        x        = self._v(values, node.input[0])
        axes     = _attr(node, "axes", None)
        keepdims = bool(_attr(node, "keepdims", 1))
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
        axis   = _attr(node, "axis", 0)
        values[node.output[0]] = mx.concatenate(inputs, axis=axis)

    def _op_split(self, node, values):
        x    = self._v(values, node.input[0])
        axis = _attr(node, "axis", 0)
        if len(node.input) > 1 and node.input[1]:
            split_arr = self._v(values, node.input[1])
            split = [int(s) for s in np.array(split_arr)]
        else:
            split = _attr(node, "split", None)

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
            axes = sorted(_attr(node, "axes", [0]))
        for ax in axes:
            x = mx.expand_dims(x, ax)
        values[node.output[0]] = x

    def _op_squeeze(self, node, values):
        x = self._v(values, node.input[0])
        if len(node.input) > 1 and node.input[1]:
            axes = sorted([int(a) for a in np.array(self._v(values, node.input[1]))],
                          reverse=True)
        else:
            axes = sorted(_attr(node, "axes", []), reverse=True)
        for ax in axes:
            x = mx.squeeze(x, ax)
        values[node.output[0]] = x

    def _op_flatten(self, node, values):
        x    = self._v(values, node.input[0])
        axis = _attr(node, "axis", 1)
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
        start = _attr(node, "start", 0)
        end   = _attr(node, "end", None)
        shape = list(x.shape)
        shape_slice = shape[start:end]
        values[node.output[0]] = mx.array(shape_slice, dtype=mx.int32)

    def _op_cast(self, node, values):
        x   = self._v(values, node.input[0])
        to  = _attr(node, "to", TensorProto.FLOAT)
        dst = _ONNX_TO_MLX_DTYPE.get(to, mx.float32)
        values[node.output[0]] = x.astype(dst)

    def _op_constant(self, node, values):
        val = _attr(node, "value", None)
        if val is not None:
            values[node.output[0]] = _np_to_mx(val)
        else:
            # float/int scalar attributes
            fval = _attr(node, "value_float", None)
            ival = _attr(node, "value_int", None)
            if fval is not None:
                values[node.output[0]] = mx.array(fval, dtype=mx.float32)
            elif ival is not None:
                values[node.output[0]] = mx.array(ival, dtype=mx.int32)

    def _op_maxpool(self, node, values):
        x          = self._v(values, node.input[0])
        ks      = _attr(node, "kernel_shape", [1, 1])
        strides = _attr(node, "strides", [1, 1])
        pads    = _attr(node, "pads", [0, 0, 0, 0])

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
        ks      = _attr(node, "kernel_shape", [1, 1])
        strides = _attr(node, "strides", [1, 1])
        pads    = _attr(node, "pads", [0, 0, 0, 0])

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

        # Opset 11+: inputs are [X, roi, scales, sizes]
        scales = None
        sizes  = None
        if len(node.input) > 2 and node.input[2]:
            scales_arr = self._v(values, node.input[2])
            if scales_arr is not None:
                scales = [float(s) for s in np.array(scales_arr)]
        if len(node.input) > 3 and node.input[3]:
            sizes_arr = self._v(values, node.input[3])
            if sizes_arr is not None:
                sizes = [int(s) for s in np.array(sizes_arr)]

        mode = _attr(node, "mode", b"nearest")
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
        """Resize [N,H,W,C] to [N,out_h,out_w,C].

        Uses scipy.ndimage.zoom which handles arbitrary channel counts,
        unlike cv2 which is limited to ≤4 channels.
        """
        from scipy.ndimage import zoom

        arr = np.array(x_nhwc)  # [N, H, W, C]
        n, h, w, c = arr.shape
        zh = out_h / h
        zw = out_w / w

        order = 1 if mode == "linear" else 0  # bilinear vs nearest
        out = zoom(arr, (1, zh, zw, 1), order=order, prefilter=False)
        return mx.array(out.astype(arr.dtype))

    def _op_pad(self, node, values):
        x = self._v(values, node.input[0])
        if len(node.input) > 1 and node.input[1]:
            pads_t = self._v(values, node.input[1])
            pads   = [int(p) for p in np.array(pads_t)]
        else:
            pads = _attr(node, "pads", [0] * (2 * x.ndim))

        const_val = 0.0
        if len(node.input) > 2 and node.input[2]:
            cv = self._v(values, node.input[2])
            if cv is not None:
                const_val = float(np.array(cv))

        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes):
            mode = mode.decode()

        # ONNX pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        ndim = x.ndim
        pad_width = [(pads[i], pads[i + ndim]) for i in range(ndim)]
        if mode == "constant":
            values[node.output[0]] = mx.pad(x, pad_width, constant_values=const_val)
        else:
            values[node.output[0]] = self._reflect_pad(x, pad_width)

    def _op_transpose(self, node, values):
        x    = self._v(values, node.input[0])
        perm = _attr(node, "perm", None)
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
        axis    = _attr(node, "axis", 0)
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
        axis = _attr(node, "axis", -1)
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

        strides   = _attr(node, "strides",          [1, 1])
        pads      = _attr(node, "pads",              [0, 0, 0, 0])
        groups    = _attr(node, "group",             1)
        out_pads  = _attr(node, "output_padding",    [0, 0])

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
        keepdims = bool(_attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes_arr = self._v(values, node.input[1])
            axes = [int(a) for a in np.array(axes_arr)]
        else:
            axes = _attr(node, "axes", None)
        if axes is not None and len(axes) == 0:
            axes = None
        values[node.output[0]] = mx.sum(x, axis=axes, keepdims=keepdims)

    def _op_reducemax(self, node, values):
        x        = self._v(values, node.input[0])
        keepdims = bool(_attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes_arr = self._v(values, node.input[1])
            axes = [int(a) for a in np.array(axes_arr)]
        else:
            axes = _attr(node, "axes", None)
        values[node.output[0]] = mx.max(x, axis=axes, keepdims=keepdims)

    def _op_constant_of_shape(self, node, values):
        shape_arr = self._v(values, node.input[0])
        shape = [int(s) for s in np.array(shape_arr)]
        val_attr = _attr(node, "value", None)
        if val_attr is not None:
            fill_val = float(val_attr.flat[0])
            dtype = _ONNX_TO_MLX_DTYPE.get(
                TensorProto.FLOAT, mx.float32
            )
        else:
            fill_val = 0.0
            dtype = mx.float32
        values[node.output[0]] = mx.full(shape, fill_val, dtype=dtype)

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
        eps   = _attr(node, "epsilon", 1e-5)

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
        keepdims = bool(_attr(node, "keepdims", 1))
        if len(node.input) > 1 and node.input[1]:
            axes = [int(a) for a in np.array(self._v(values, node.input[1]))]
        else:
            axes = _attr(node, "axes", None)
        values[node.output[0]] = mx.prod(x, axis=axes, keepdims=keepdims)

    def _op_scatter_elements(self, node, values):
        data    = self._v(values, node.input[0])
        indices = self._v(values, node.input[1])
        updates = self._v(values, node.input[2])
        axis = _attr(node, "axis", 0)
        arr  = np.array(data)
        idx = np.array(indices).astype(np.intp)
        upd = np.array(updates)
        np.put_along_axis(arr, idx, upd, axis=axis)
        values[node.output[0]] = _np_to_mx(arr)

    def _op_topk(self, node, values):
        x    = self._v(values, node.input[0])
        k_t  = self._v(values, node.input[1])
        k    = int(np.array(k_t))
        axis = _attr(node, "axis", -1)
        largest = bool(_attr(node, "largest", 1))
        sorted_ = bool(_attr(node, "sorted", 1))

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
        axis     = _attr(node, "axis", 0)
        keepdims = bool(_attr(node, "keepdims", 1))
        values[node.output[0]] = mx.argmin(x, axis=axis, keepdims=keepdims)

    def _op_argmax(self, node, values):
        x        = self._v(values, node.input[0])
        axis     = _attr(node, "axis", 0)
        keepdims = bool(_attr(node, "keepdims", 1))
        values[node.output[0]] = mx.argmax(x, axis=axis, keepdims=keepdims)

    def _op_cumsum(self, node, values):
        x    = self._v(values, node.input[0])
        axis = int(np.array(self._v(values, node.input[1])))
        exclusive = bool(_attr(node, "exclusive", 0))
        reverse   = bool(_attr(node, "reverse", 0))
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
        axis    = _attr(node, "axis", 0)
        arr = np.array(x)
        idx = np.array(indices).astype(np.intp)
        values[node.output[0]] = _np_to_mx(np.take_along_axis(arr, idx, axis=axis))

    def _op_einsum(self, node, values):
        equation = _attr(node, "equation", b"").decode()
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
        alpha = _attr(node, "alpha", 0.2)
        beta  = _attr(node, "beta",  0.5)
        values[node.output[0]] = mx.clip(alpha * x + beta, 0.0, 1.0)
