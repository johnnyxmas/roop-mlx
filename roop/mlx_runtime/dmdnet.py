"""DMDNet face restoration -- pure MLX inference, no torch dependency.

Architecture reference:
  "Blind Face Restoration via Deep Multi-scale Component Dictionaries"
  (Li et al., ECCV 2020)  https://github.com/csxmli2016/DMDNet

Weight source: ``models/DMDNet.pth`` (PyTorch checkpoint with SpectralNorm).
We read ``weight_orig`` directly for every Conv2d (spectral norm only scales
weights at training time; skipping it at inference is a standard approximation
that preserves the model's qualitative behaviour).

Public API
----------
>>> model = DMDNetMLX("models/DMDNet.pth")
>>> out_nchw = model.forward(img_nchw, locs)   # np.ndarray [1,3,512,512] [-1,1]
"""

from __future__ import annotations

import io
import math
import pickle
import re
import zipfile
from collections import OrderedDict
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Weight loading (torch-free ZIP + pickle reader)
# ---------------------------------------------------------------------------

# Map storage class names → numpy dtype
_STORAGE_DTYPE: Dict[str, np.dtype] = {
    "FloatStorage":  np.float32,
    "HalfStorage":   np.float16,
    "DoubleStorage": np.float64,
    "LongStorage":   np.int64,
    "IntStorage":    np.int32,
    "ShortStorage":  np.int16,
    "ByteStorage":   np.uint8,
    "BoolStorage":   np.uint8,
    "CharStorage":   np.int8,
}


def _rebuild_tensor(storage: np.ndarray, offset: int, shape: tuple,
                    stride: tuple, *args) -> np.ndarray:
    """Reconstruct a numpy array from a flat storage blob."""
    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        return np.zeros(shape, dtype=storage.dtype)
    chunk = storage[offset: offset + numel]
    # stride-aware copy: use numpy stride tricks if non-contiguous
    if len(shape) > 0:
        arr = np.lib.stride_tricks.as_strided(
            chunk,
            shape=shape,
            strides=tuple(s * storage.dtype.itemsize for s in stride),
        )
        return arr.copy().astype(np.float32)
    return chunk.reshape(shape).astype(np.float32)


def load_dmdnet_weights(path: str) -> Dict[str, np.ndarray]:
    """Load ``DMDNet.pth`` and return a dict mapping key → float32 ndarray.

    Completely torch-free.  The .pth is a ZIP archive whose ``data.pkl``
    encodes a state_dict.  Storage blobs live in ``archive/data/<N>`` members.
    """
    with zipfile.ZipFile(path) as zf:
        # ---- collect raw data blobs by integer index -----------------------
        blobs: Dict[int, bytes] = {}
        for name in zf.namelist():
            if name.startswith("archive/data/"):
                tail = name[len("archive/data/"):]
                if tail.isdigit():
                    blobs[int(tail)] = zf.read(name)

        with zf.open("archive/data.pkl") as f:
            pkl_bytes = f.read()

    # ---- pre-scan: map blob index → numpy dtype ----------------------------
    # The PERSID tuple is ('storage', <StorageCls>, '<idx>', 'cuda:0', numel).
    # We locate each GLOBAL opcode for a Storage type, then find the string
    # key that follows it in the PERSID sequence.
    _blob_dtype: Dict[int, np.dtype] = {}
    # Pattern: GLOBAL opcode bytes → 'torch\x00FloatStorage'
    for m in re.finditer(rb'torch\x00([A-Za-z]+Storage)', pkl_bytes):
        sname = m.group(1).decode()
        dtype = _STORAGE_DTYPE.get(sname, np.float32)
        # The key string follows shortly after the GLOBAL opcode.
        # It appears as a BINUNICODE-encoded digit string.
        chunk = pkl_bytes[m.end(): m.end() + 40]
        km = re.search(rb'(\d+)', chunk)
        if km:
            try:
                _blob_dtype[int(km.group(1))] = dtype
            except ValueError:
                pass

    # ---- unpickle with custom handlers ------------------------------------
    class _Loader(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            # Tensor rebuild functions
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return _rebuild_tensor
            # _rebuild_from_type_v2(cls, func, args, state): forward to func
            if name == "_rebuild_from_type_v2":
                return lambda cls, func, args, state: func(*args)
            # _rebuild_parameter wraps a tensor; just return the data
            if name == "_rebuild_parameter":
                return lambda data, req_grad, bw_hooks: data
            # Standard collections
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            # Everything else: return a no-op stub
            return lambda *a, **kw: None

        def persistent_load(self, pid):
            # pid = ('storage', StorageCls, str_key, device, numel)
            if not isinstance(pid, (list, tuple)) or len(pid) < 3:
                return np.array([], dtype=np.float32)
            str_key = pid[2]
            int_key = int(str_key) if str(str_key).isdigit() else -1
            dtype = _blob_dtype.get(int_key, np.float32)
            if int_key in blobs:
                return np.frombuffer(blobs[int_key], dtype=dtype).copy()
            return np.array([], dtype=dtype)

    loader = _Loader(io.BytesIO(pkl_bytes))
    state_dict = loader.load()

    # ---- extract numpy arrays from the result ------------------------------
    result: Dict[str, np.ndarray] = {}
    if isinstance(state_dict, dict):
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                result[str(k)] = v.astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Low-level MLX helpers
# ---------------------------------------------------------------------------

def _to_nhwc(x: mx.array) -> mx.array:
    """NCHW → NHWC"""
    return mx.transpose(x, [0, 2, 3, 1])


def _to_nchw(x: mx.array) -> mx.array:
    """NHWC → NCHW"""
    return mx.transpose(x, [0, 3, 1, 2])


def _leaky_relu(x: mx.array, neg_slope: float = 0.2) -> mx.array:
    return mx.where(x >= 0, x, neg_slope * x)


def _bilinear_resize(x_nchw: mx.array, h: int, w: int) -> mx.array:
    """Bilinear resize NCHW tensor to (h, w) using scipy.ndimage.zoom."""
    arr = np.array(x_nchw)  # [N, C, H, W]
    n, c, in_h, in_w = arr.shape
    # zoom over H and W axes; leave N and C untouched
    zh = h / in_h
    zw = w / in_w
    out = zoom(arr, (1, 1, zh, zw), order=1, prefilter=False)
    return mx.array(out.astype(np.float32))


def _roi_crop_resize(
    feat_nchw: mx.array,
    box_xy: np.ndarray,  # [x1, y1, x2, y2]  (integer coords at feat scale)
    th: int,
    tw: int,
) -> mx.array:
    """Crop a region from feat_nchw and bilinear-resize to (th, tw).

    Coordinates are clamped to the feature map boundary.
    """
    arr = np.array(feat_nchw)  # [1, C, H, W]
    _, c, fh, fw = arr.shape
    x1, y1, x2, y2 = int(box_xy[0]), int(box_xy[1]), int(box_xy[2]), int(box_xy[3])
    # clamp
    x1 = max(0, min(x1, fw - 1))
    y1 = max(0, min(y1, fh - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y2 = max(y1 + 1, min(y2, fh))
    crop = arr[:, :, y1:y2, x1:x2]  # [1, C, ph, pw]
    _, _, ph, pw = crop.shape
    if ph == th and pw == tw:
        return mx.array(crop.astype(np.float32))
    zh = th / ph
    zw = tw / pw
    resized = zoom(crop, (1, 1, zh, zw), order=1, prefilter=False)
    return mx.array(resized.astype(np.float32))


def _instance_norm_4d(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Instance normalisation (no affine) over H and W for NCHW input."""
    mean = mx.mean(x, axis=(2, 3), keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=(2, 3), keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def _adaptive_instance_norm(content: mx.array, style: mx.array) -> mx.array:
    """AdaIN: normalise content, then apply style mean/std (NCHW)."""
    norm_content = _instance_norm_4d(content)
    style_mean = mx.mean(style, axis=(2, 3), keepdims=True)
    style_std = mx.sqrt(mx.mean((style - style_mean) ** 2, axis=(2, 3), keepdims=True) + 1e-5)
    return norm_content * style_std + style_mean


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax; degenerate all-NaN rows become uniform."""
    shifted = x - np.nanmax(x, axis=axis, keepdims=True)
    # Clamp exponent to avoid overflow; underflow → 0 is fine
    e = np.exp(np.clip(shifted, -500.0, 0.0))
    s = e.sum(axis=axis, keepdims=True)
    # Guard against zero-sum (all-NaN or all-zero queries)
    s = np.where(s == 0, 1.0, s)
    result = e / s
    # Replace any remaining NaN rows with uniform distribution
    S = x.shape[axis]
    nan_rows = ~np.isfinite(result).all(axis=axis, keepdims=True)
    result = np.where(nan_rows, 1.0 / S, result)
    return result


def _read_mem(
    k,   # mx.array or np.ndarray [S, K, kH, kW]
    v,   # mx.array or np.ndarray [S, V, vH, vW]
    q,   # mx.array or np.ndarray [1, K, kH, kW]
) -> Tuple[mx.array, np.ndarray]:
    """Memory-attention read.

    Returns (retrieved_value [1,V,vH,vW], argmax_indices [1]).
    """
    k_np = np.array(k, dtype=np.float32)   # [S, K, kH, kW]
    v_np = np.array(v, dtype=np.float32)   # [S, V, vH, vW]
    q_np = np.array(q, dtype=np.float32)   # [1, K, kH, kW]

    S, K, kH, kW = k_np.shape
    scale = math.sqrt(float(K * kH * kW))

    # Dot-product attention: sim[b, s] = sum(q[b] * k[s])
    # Use float64 for the einsum to avoid overflow with large activations
    sim = np.einsum('bkhw,skhw->bs', q_np.astype(np.float64),
                    k_np.astype(np.float64)) / scale  # [1, S]
    score = _softmax_np(sim.astype(np.float32), axis=1)   # [1, S]

    S, V, vH, vW = v_np.shape
    # Weighted sum of values
    mem_out = (score.astype(np.float32) @ v_np.reshape(S, -1)).reshape(1, V, vH, vW)
    # Fallback: if still NaN (e.g. zero query), return mean of values
    if np.isnan(mem_out).any():
        mem_out = v_np.mean(axis=0, keepdims=True)  # [1, V, vH, vW]
    max_inds = np.argmax(score, axis=1)  # [1]
    return mx.array(mem_out.astype(np.float32)), max_inds


# ---------------------------------------------------------------------------
# SN Conv2d helper
# ---------------------------------------------------------------------------

def _sn_conv2d(
    weights: Dict[str, np.ndarray],
    prefix: str,
    x: mx.array,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
) -> mx.array:
    """Apply a spectral-normalised Conv2d to NCHW input *x*.

    Reads ``{prefix}.weight_orig`` (already divided by sigma in __init__)
    and ``{prefix}.bias`` from the weight dict.

    MLX conv2d uses NHWC input and OHWI weights, so we transpose accordingly.
    """
    w_oihw = weights[f"{prefix}.weight_orig"]   # [O, I, kH, kW] np.ndarray
    b_np   = weights.get(f"{prefix}.bias")       # [O] or None

    # OIHW → OHWI for MLX conv2d
    w_mx   = mx.array(w_oihw)
    w_ohwi = mx.transpose(w_mx, [0, 2, 3, 1])

    x_nhwc = _to_nhwc(x)

    out = mx.conv2d(
        x_nhwc,
        w_ohwi,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    if b_np is not None:
        out = out + mx.array(b_np).reshape(1, 1, 1, -1)

    return _to_nchw(out)


# ---------------------------------------------------------------------------
# Building-block forward functions
# ---------------------------------------------------------------------------

def _dilate_res_block(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "E_lq.res1"
    x: mx.array,
    dilation0: int,
    dilation1: int,
) -> mx.array:
    """DilateResBlock: x + Res(x).

    Res = SNConv(d0) → LReLU → SNConv(d1)
    Keys: {prefix}.Res.0.weight_orig / bias, {prefix}.Res.2.weight_orig / bias
    """
    pad0 = dilation0
    pad1 = dilation1
    h = _sn_conv2d(weights, f"{prefix}.Res.0", x, stride=1, padding=pad0, dilation=dilation0)
    h = _leaky_relu(h)
    h = _sn_conv2d(weights, f"{prefix}.Res.2", h, stride=1, padding=pad1, dilation=dilation1)
    return x + h


def _query_module(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "E_lq.LE_256_Q"
    x: mx.array,
) -> mx.array:
    """Query: SNConv → LReLU → SNConv.

    Keys: {prefix}.Query.0.*, {prefix}.Query.2.*
    """
    h = _sn_conv2d(weights, f"{prefix}.Query.0", x)
    h = _leaky_relu(h)
    h = _sn_conv2d(weights, f"{prefix}.Query.2", h)
    return h


def _key_value(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "LE_256_KV"
    x: mx.array,
) -> Tuple[mx.array, mx.array]:
    """KeyValue: returns (Key(x), Value(x)).

    Keys: {prefix}.Key.0/2, {prefix}.Value.0/2
    """
    k = _sn_conv2d(weights, f"{prefix}.Key.0", x)
    k = _leaky_relu(k)
    k = _sn_conv2d(weights, f"{prefix}.Key.2", k)

    v = _sn_conv2d(weights, f"{prefix}.Value.0", x)
    v = _leaky_relu(v)
    v = _sn_conv2d(weights, f"{prefix}.Value.2", v)
    return k, v


def _attention_block(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "LE_256_Attention"
    x: mx.array,
) -> mx.array:
    """AttentionBlock: Sequential(SNConv, LReLU, SNConv).

    Keys: {prefix}.0.*, {prefix}.2.*
    """
    h = _sn_conv2d(weights, f"{prefix}.0", x)
    h = _leaky_relu(h)
    h = _sn_conv2d(weights, f"{prefix}.2", h)
    return h


def _mask_attention(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "LE_256_Mask"
    x: mx.array,
    y: mx.array,
    z: mx.array,
) -> mx.array:
    """MaskAttention: three parallel SNConv branches concatenated then merged.

    Keys: {prefix}.conv1.0/2, conv2.0/2, conv3.0/2, convCat.0/2
    """
    c1 = _sn_conv2d(weights, f"{prefix}.conv1.0", x)
    c1 = _leaky_relu(c1)
    c1 = _sn_conv2d(weights, f"{prefix}.conv1.2", c1)

    c2 = _sn_conv2d(weights, f"{prefix}.conv2.0", y)
    c2 = _leaky_relu(c2)
    c2 = _sn_conv2d(weights, f"{prefix}.conv2.2", c2)

    c3 = _sn_conv2d(weights, f"{prefix}.conv3.0", z)
    c3 = _leaky_relu(c3)
    c3 = _sn_conv2d(weights, f"{prefix}.conv3.2", c3)

    cat = mx.concatenate([c1, c2, c3], axis=1)

    out = _sn_conv2d(weights, f"{prefix}.convCat.0", cat)
    out = _leaky_relu(out)
    out = _sn_conv2d(weights, f"{prefix}.convCat.2", out)
    return out


def _conv_u_block(
    weights: Dict[str, np.ndarray],
    prefix: str,
    x: mx.array,
    dilation: int = 1,
) -> mx.array:
    """convU: Sequential(SNConv2d, LReLU, SNConv2d) with uniform dilation.

    Keys: {prefix}.0.*, {prefix}.2.*
    """
    pad = dilation  # for 3x3 conv: (3-1)//2 * dilation = dilation
    h = _sn_conv2d(weights, f"{prefix}.0", x, dilation=dilation, padding=pad)
    h = _leaky_relu(h)
    h = _sn_conv2d(weights, f"{prefix}.2", h, dilation=dilation, padding=pad)
    return h


def _msdilate_block(
    weights: Dict[str, np.ndarray],
    x: mx.array,
) -> mx.array:
    """MSDilateBlock with dilations [4,3,2,1].

    Keys: MSDilate.conv1-4.0/2, MSDilate.convi.*
    """
    branches = []
    for i, d in enumerate([4, 3, 2, 1], start=1):
        b = _conv_u_block(weights, f"MSDilate.conv{i}", x, dilation=d)
        branches.append(b)
    cat = mx.concatenate(branches, axis=1)  # [1, 256*4, H, W]
    out = _sn_conv2d(weights, "MSDilate.convi", cat)
    return x + out


def _styled_up_block(
    weights: Dict[str, np.ndarray],
    prefix: str,     # "up1", "up2", "up3"
    inp: mx.array,   # NCHW input
    style: mx.array, # NCHW style (attended feature)
) -> mx.array:
    """StyledUpBlock forward.

    conv1: SNConv(in→out, 3,1,1) [no upsample at 0 based on actual keys]
    convup: Upsample→SNConv(out→out)→LReLU→SNConv(out→out)
    ScaleModel1: SNConv(in→out)→LReLU→SNConv(out→out)
    ShiftModel1: same
    forward:
      out = lrelu(conv1(inp))
      scale = ScaleModel1(style), shift = ShiftModel1(style)
      out = out * scale + shift
      outup = convup applied with 2x upsample then two convs
    """
    # conv1: {prefix}.conv1.0 and {prefix}.conv1.2
    out = _sn_conv2d(weights, f"{prefix}.conv1.0", inp)
    out = _leaky_relu(out)
    out = _sn_conv2d(weights, f"{prefix}.conv1.2", out)
    out = _leaky_relu(out)

    # AdaIN-style: scale & shift from style
    scale = _sn_conv2d(weights, f"{prefix}.ScaleModel1.0", style)
    scale = _leaky_relu(scale)
    scale = _sn_conv2d(weights, f"{prefix}.ScaleModel1.2", scale)

    shift = _sn_conv2d(weights, f"{prefix}.ShiftModel1.0", style)
    shift = _leaky_relu(shift)
    shift = _sn_conv2d(weights, f"{prefix}.ShiftModel1.2", shift)

    out = out * scale + shift

    # convup: Upsample(2x) → SNConv(up1.convup.1) → LReLU → SNConv(up1.convup.3)
    _, _, h, w = out.shape
    out_up = _bilinear_resize(out, h * 2, w * 2)
    out_up = _sn_conv2d(weights, f"{prefix}.convup.1", out_up)
    out_up = _leaky_relu(out_up)
    out_up = _sn_conv2d(weights, f"{prefix}.convup.3", out_up)

    return out_up


def _up_res_block(
    weights: Dict[str, np.ndarray],
    prefix: str,   # e.g. "up4.2" or "up4.3"
    x: mx.array,
) -> mx.array:
    """UpResBlock: x + Model(x).

    Keys: {prefix}.Model.0.*, {prefix}.Model.2.*
    """
    h = _sn_conv2d(weights, f"{prefix}.Model.0", x)
    h = _leaky_relu(h)
    h = _sn_conv2d(weights, f"{prefix}.Model.2", h)
    return x + h


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

# Part sizes (original pixel coords, 512x512 input):
#   LE/RE: 80x80, MO: 110x110 (height x width)
# At 256-scale (stride 2): divide by 2 → LE/RE 40, MO 55
# At 128-scale (stride 4): divide by 4 → LE/RE 20, MO 27
# At  64-scale (stride 8): divide by 8 → LE/RE 10, MO 13

def _feature_extractor(
    weights: Dict[str, np.ndarray],
    enc: str,         # "E_lq" or "E_hq"
    img_nchw: mx.array,
    locs: np.ndarray,  # [1, 4, 4]  boxes [[x1,y1,x2,y2] * 4]  (LE,RE,NO,MO)
) -> Dict[str, mx.array]:
    """Run the FeatureExtractor (encoder) and extract per-part ROI patches."""

    # -- Encoder backbone --
    # conv1: stride 2 → 256x256
    f1 = _sn_conv2d(weights, f"{enc}.conv1.0", img_nchw, stride=2, padding=1)
    f1 = _leaky_relu(f1)
    f1 = _sn_conv2d(weights, f"{enc}.conv1.2", f1, stride=1, padding=1)
    f1 = _dilate_res_block(weights, f"{enc}.res1", f1, dilation0=5, dilation1=3)
    f2 = _sn_conv2d(weights, f"{enc}.conv2.0", f1, stride=1, padding=1)
    f2 = _leaky_relu(f2)
    f2 = _sn_conv2d(weights, f"{enc}.conv2.2", f2, stride=1, padding=1)
    f2 = _dilate_res_block(weights, f"{enc}.res2", f2, dilation0=5, dilation1=3)
    # f2: [1, 64, 256, 256]

    # conv3: stride 2 → 128x128
    f3 = _sn_conv2d(weights, f"{enc}.conv3.0", f2, stride=2, padding=1)
    f3 = _leaky_relu(f3)
    f3 = _sn_conv2d(weights, f"{enc}.conv3.2", f3, stride=1, padding=1)
    f3 = _dilate_res_block(weights, f"{enc}.res3", f3, dilation0=3, dilation1=1)
    f4 = _sn_conv2d(weights, f"{enc}.conv4.0", f3, stride=1, padding=1)
    f4 = _leaky_relu(f4)
    f4 = _sn_conv2d(weights, f"{enc}.conv4.2", f4, stride=1, padding=1)
    f4 = _dilate_res_block(weights, f"{enc}.res4", f4, dilation0=3, dilation1=1)
    # f4: [1, 128, 128, 128]

    # conv5: stride 2 → 64x64
    f5 = _sn_conv2d(weights, f"{enc}.conv5.0", f4, stride=2, padding=1)
    f5 = _leaky_relu(f5)
    f5 = _sn_conv2d(weights, f"{enc}.conv5.2", f5, stride=1, padding=1)
    f5 = _dilate_res_block(weights, f"{enc}.res5", f5, dilation0=1, dilation1=1)
    f6 = _sn_conv2d(weights, f"{enc}.conv6.0", f5, stride=1, padding=1)
    f6 = _leaky_relu(f6)
    f6 = _sn_conv2d(weights, f"{enc}.conv6.2", f6, stride=1, padding=1)
    f6 = _dilate_res_block(weights, f"{enc}.res6", f6, dilation0=1, dilation1=1)
    # f6: [1, 256, 64, 64]

    # -- ROI crop for LE (left eye), RE (right eye), MO (mouth) --
    # locs[0, 0] = LE box, locs[0, 1] = RE box, locs[0, 2] = NO (nose, unused), locs[0, 3] = MO
    le_box = locs[0, 0]  # [x1, y1, x2, y2]
    re_box = locs[0, 1]
    mo_box = locs[0, 3]

    # Scale 256 (divide coords by 2)
    le_part_256 = _roi_crop_resize(f2, le_box // 2, 40, 40)
    re_part_256 = _roi_crop_resize(f2, re_box // 2, 40, 40)
    mo_part_256 = _roi_crop_resize(f2, mo_box // 2, 55, 55)

    # Scale 128 (divide by 4)
    le_part_128 = _roi_crop_resize(f4, le_box // 4, 20, 20)
    re_part_128 = _roi_crop_resize(f4, re_box // 4, 20, 20)
    mo_part_128 = _roi_crop_resize(f4, mo_box // 4, 27, 27)

    # Scale 64 (divide by 8)
    le_part_64 = _roi_crop_resize(f6, le_box // 8, 10, 10)
    re_part_64 = _roi_crop_resize(f6, re_box // 8, 10, 10)
    mo_part_64 = _roi_crop_resize(f6, mo_box // 8, 13, 13)

    # -- Apply queries --
    le_256_q = _query_module(weights, f"{enc}.LE_256_Q", le_part_256)
    re_256_q = _query_module(weights, f"{enc}.RE_256_Q", re_part_256)
    mo_256_q = _query_module(weights, f"{enc}.MO_256_Q", mo_part_256)

    le_128_q = _query_module(weights, f"{enc}.LE_128_Q", le_part_128)
    re_128_q = _query_module(weights, f"{enc}.RE_128_Q", re_part_128)
    mo_128_q = _query_module(weights, f"{enc}.MO_128_Q", mo_part_128)

    le_64_q = _query_module(weights, f"{enc}.LE_64_Q", le_part_64)
    re_64_q = _query_module(weights, f"{enc}.RE_64_Q", re_part_64)
    mo_64_q = _query_module(weights, f"{enc}.MO_64_Q", mo_part_64)

    return dict(
        f256=f2, f128=f4, f64=f6,
        le_part_256=le_part_256, re_part_256=re_part_256, mo_part_256=mo_part_256,
        le_part_128=le_part_128, re_part_128=re_part_128, mo_part_128=mo_part_128,
        le_part_64=le_part_64, re_part_64=re_part_64, mo_part_64=mo_part_64,
        le_256_q=le_256_q, re_256_q=re_256_q, mo_256_q=mo_256_q,
        le_128_q=le_128_q, re_128_q=re_128_q, mo_128_q=mo_128_q,
        le_64_q=le_64_q, re_64_q=re_64_q, mo_64_q=mo_64_q,
        le_box=le_box, re_box=re_box, mo_box=mo_box,
    )


# ---------------------------------------------------------------------------
# Enhancer (memory-augmented attention per feature scale)
# ---------------------------------------------------------------------------

def _to_mx(v) -> mx.array:
    """Ensure value is mx.array (converts np.ndarray if needed)."""
    if isinstance(v, mx.array):
        return v
    return mx.array(np.asarray(v).astype(np.float32))


def _part_enhance(
    weights: Dict[str, np.ndarray],
    scale: str,            # "256", "128", "64"
    part: str,             # "LE", "RE", "MO"
    part_feat: mx.array,   # [1, C, pH, pW]
    query: mx.array,       # [1, K, qH, qW]
) -> mx.array:
    """Memory-read + MaskAttention + AttentionBlock for one part at one scale.

    Returns the style feature [1, C, pH, pW].
    """
    mem_key = f"{part.lower()}_{scale}_mem_key"    # e.g. le_256_mem_key
    mem_val = f"{part.lower()}_{scale}_mem_value"
    mask_prefix = f"{part}_{scale}_Mask"
    att_prefix  = f"{part}_{scale}_Attention"

    retrieved, _ = _read_mem(
        weights[mem_key],
        weights[mem_val],
        query,
    )
    retrieved = _to_mx(retrieved)

    # MaskAttention(part_feat, retrieved, part_feat)
    att = _mask_attention(weights, mask_prefix, part_feat, retrieved, part_feat)

    # AttentionBlock for style
    style = _attention_block(weights, att_prefix, att)
    return style


def _enhancer_generic(
    weights: Dict[str, np.ndarray],
    fs: Dict[str, mx.array],
) -> Tuple[mx.array, mx.array, mx.array]:
    """Run the generic-path enhancer (lq path only).

    Returns three style tensors:
      EnMem64  – list of [le_style, re_style, mo_style] at 64-scale (256ch)
      EnMem128 – list at 128-scale (128ch)
      EnMem256 – list at 256-scale (64ch)

    Each list contains the three per-part styles; the caller aggregates them
    by resizing to the decoder feature map size and averaging.
    """
    # ---- 256-scale (feature map 256x256, parts 40/55 px) ------------------
    le_style256 = _part_enhance(weights, "256", "LE", fs["le_part_256"], fs["le_256_q"])
    re_style256 = _part_enhance(weights, "256", "RE", fs["re_part_256"], fs["re_256_q"])
    mo_style256 = _part_enhance(weights, "256", "MO", fs["mo_part_256"], fs["mo_256_q"])

    # ---- 128-scale (parts 20/27 px) ----------------------------------------
    le_style128 = _part_enhance(weights, "128", "LE", fs["le_part_128"], fs["le_128_q"])
    re_style128 = _part_enhance(weights, "128", "RE", fs["re_part_128"], fs["re_128_q"])
    mo_style128 = _part_enhance(weights, "128", "MO", fs["mo_part_128"], fs["mo_128_q"])

    # ---- 64-scale (parts 10/13 px) -----------------------------------------
    le_style64 = _part_enhance(weights, "64", "LE", fs["le_part_64"], fs["le_64_q"])
    re_style64 = _part_enhance(weights, "64", "RE", fs["re_part_64"], fs["re_64_q"])
    mo_style64 = _part_enhance(weights, "64", "MO", fs["mo_part_64"], fs["mo_64_q"])

    # Return lists; spatial aggregation happens in _reconstruct where we know
    # the target decoder feature-map size.
    EnMem256 = [le_style256, re_style256, mo_style256]
    EnMem128 = [le_style128, re_style128, mo_style128]
    EnMem64  = [le_style64,  re_style64,  mo_style64]

    return EnMem256, EnMem128, EnMem64


def _aggregate_styles(styles: list, target_h: int, target_w: int) -> mx.array:
    """Resize each style to (target_h, target_w) and average them."""
    resized = [_bilinear_resize(s, target_h, target_w) for s in styles]
    out = resized[0]
    for r in resized[1:]:
        out = out + r
    return out / len(resized)


# ---------------------------------------------------------------------------
# Decoder / reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(
    weights: Dict[str, np.ndarray],
    fs: Dict[str, mx.array],
    EnMem256: list,
    EnMem128: list,
    EnMem64: list,
) -> mx.array:
    """Decoder path: MSDilate + 3 StyledUpBlocks + up4.

    EnMem64/128/256 are lists of per-part style tensors.  They are resized
    to the decoder feature-map size and averaged before being passed as style.

    Network decoder dimensions:
      f64  → [1, 256, 64, 64]   (MSDilate input/output)
      up1 (256→128): style from EnMem64  (256ch parts at ~10px)
        → output [1, 128, 128, 128]
      up2 (128→64): style from EnMem128  (128ch parts at ~20px)
        → output [1, 64, 256, 256]
      up3 (64→64): style from EnMem256   (64ch parts at ~40px)
        → output [1, 64, 512, 512]
      up4: SNConv(64→64) + LReLU + 2×UpResBlock + SNConv(64→3) + Tanh
    """
    x = fs["f64"]  # [1, 256, 64, 64]
    x = _msdilate_block(weights, x)

    # up1: style needs to match x channel/spatial size for ScaleModel1.
    # ScaleModel1.0 has shape (128, 256, 3, 3) → takes 256ch input (same as x).
    # We resize the per-part EnMem64 styles [1,256,10,10] to [1,256,64,64].
    _, _, h64, w64 = x.shape
    style64 = _aggregate_styles(EnMem64, h64, w64)   # [1, 256, 64, 64]
    x = _styled_up_block(weights, "up1", x, style64)
    # x: [1, 128, 128, 128]

    _, _, h128, w128 = x.shape
    style128 = _aggregate_styles(EnMem128, h128, w128)  # [1, 128, 128, 128]
    x = _styled_up_block(weights, "up2", x, style128)
    # x: [1, 64, 256, 256]

    _, _, h256, w256 = x.shape
    style256 = _aggregate_styles(EnMem256, h256, w256)  # [1, 64, 256, 256]
    x = _styled_up_block(weights, "up3", x, style256)
    # x: [1, 64, 512, 512]

    # up4: Sequential(SNConv(64→64), LReLU, UpResBlock, UpResBlock, SNConv(64→3), Tanh)
    x = _sn_conv2d(weights, "up4.0", x)
    x = _leaky_relu(x)
    x = _up_res_block(weights, "up4.2", x)
    x = _up_res_block(weights, "up4.3", x)
    x = _sn_conv2d(weights, "up4.4", x)
    x = mx.tanh(x)

    return x  # [1, 3, 512, 512] in [-1, 1]


# ---------------------------------------------------------------------------
# Memory tensor accessor
# ---------------------------------------------------------------------------

def _get_mem(weights: Dict[str, np.ndarray], key: str) -> mx.array:
    """Return a memory buffer as mx.array, handling np.ndarray or mx.array."""
    v = weights[key]
    if isinstance(v, np.ndarray):
        return mx.array(v.astype(np.float32))
    return v


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DMDNetMLX:
    """DMDNet face restoration model running on Apple MLX.

    Parameters
    ----------
    path:
        Path to ``DMDNet.pth``.

    Usage
    -----
    >>> model = DMDNetMLX("models/DMDNet.pth")
    >>> # img_nchw: np.ndarray or mx.array [1, 3, 512, 512], values in [-1, 1]
    >>> # locs: np.ndarray [1, 4, 4], face component boxes (LE, RE, NO, MO)
    >>> out = model.forward(img_nchw, locs)  # np.ndarray [1, 3, 512, 512] in [-1, 1]
    """

    def __init__(self, path: str):
        raw = load_dmdnet_weights(path)
        self._weights = self._preprocess_weights(raw)

    @staticmethod
    def _preprocess_weights(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Pre-apply spectral norm to all conv weights and convert memory to mx.

        This avoids recomputing sigma on every forward call and eliminates
        the need to keep weight_u / weight_v around during inference.
        """
        w = dict(raw)

        # ---- Spectral-normalise every weight_orig --------------------------
        orig_keys = [k for k in w if k.endswith(".weight_orig")]
        for wk in orig_keys:
            prefix = wk[: -len(".weight_orig")]
            u_key = f"{prefix}.weight_u"
            v_key = f"{prefix}.weight_v"
            u_np = w.get(u_key)
            v_np = w.get(v_key)
            wo = w[wk]                               # [O, I, kH, kW] float32

            if u_np is not None and v_np is not None:
                W_flat = wo.reshape(wo.shape[0], -1).astype(np.float64)
                try:
                    Wv = W_flat @ v_np.astype(np.float64)
                    sigma = float(u_np.astype(np.float64) @ Wv)
                except (FloatingPointError, OverflowError):
                    sigma = float("nan")
                if not math.isfinite(sigma) or abs(sigma) < 1e-8:
                    sigma = float(np.linalg.svd(W_flat, compute_uv=False)[0])
            else:
                # No u/v available (should not happen for this model)
                W_flat = wo.reshape(wo.shape[0], -1).astype(np.float64)
                sigma = float(np.linalg.svd(W_flat, compute_uv=False)[0])

            if abs(sigma) > 1e-8:
                w[wk] = (wo / float(sigma)).astype(np.float32)

        # ---- Convert memory tensors to mx.array ----------------------------
        mem_keys = [
            "le_256_mem_key", "le_256_mem_value",
            "re_256_mem_key", "re_256_mem_value",
            "mo_256_mem_key", "mo_256_mem_value",
            "le_128_mem_key", "le_128_mem_value",
            "re_128_mem_key", "re_128_mem_value",
            "mo_128_mem_key", "mo_128_mem_value",
            "le_64_mem_key",  "le_64_mem_value",
            "re_64_mem_key",  "re_64_mem_value",
            "mo_64_mem_key",  "mo_64_mem_value",
        ]
        for mk in mem_keys:
            if mk in w:
                v = w[mk]
                if isinstance(v, np.ndarray):
                    w[mk] = mx.array(v.astype(np.float32))

        return w

    @property
    def weights(self) -> Dict[str, np.ndarray]:
        return self._weights

    def forward(
        self,
        img_nchw: "np.ndarray | mx.array",
        locs: np.ndarray,
    ) -> np.ndarray:
        """Run face restoration.

        Parameters
        ----------
        img_nchw:
            Input image tensor, shape [1, 3, 512, 512], values in [-1, 1].
            Can be numpy or mlx array.
        locs:
            Face component bounding boxes, shape [1, 4, 4].
            Axis-1 order: [LE, RE, NO, MO]; each box is [x1, y1, x2, y2]
            in original image pixel coordinates (512x512 space).

        Returns
        -------
        np.ndarray [1, 3, 512, 512] in [-1, 1].
        """
        if isinstance(img_nchw, np.ndarray):
            img_mx = mx.array(img_nchw.astype(np.float32))
        else:
            img_mx = img_nchw

        locs_np = np.array(locs).astype(np.float32)

        # Encode LQ image
        fs_lq = _feature_extractor(self._weights, "E_lq", img_mx, locs_np)

        # Memory-augmented enhancement (generic path)
        EnMem256, EnMem128, EnMem64 = _enhancer_generic(self._weights, fs_lq)

        # Decode
        out = _reconstruct(self._weights, fs_lq, EnMem256, EnMem128, EnMem64)

        # Materialise
        mx.eval(out)
        return np.array(out)
