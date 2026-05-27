"""Native MLX implementation of InSwapper 128.

Replaces the ONNX node-by-node interpreter with direct mlx.nn layers,
eliminating ~40ms of per-frame Python dispatch overhead.

Architecture (from ONNX graph analysis):
  Encoder:  3→128(7×7)→256(3×3)→512(3×3,s2)→1024(3×3,s2)  [128→32 spatial]
  6× AdaIN style blocks (1024ch, source-conditioned via Linear→scale/shift)
  Decoder:  upsample+1024→512→upsample+512→256→128→3(7×7)+tanh  [32→128 spatial]
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _reflect_pad(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    if pad_h > 0:
        top = mx.take(x, mx.arange(pad_h, 0, -1), axis=1)
        bot_start = x.shape[1] - 2
        bot = mx.take(x, mx.arange(bot_start, bot_start - pad_h, -1), axis=1)
        x = mx.concatenate([top, x, bot], axis=1)
    if pad_w > 0:
        left = mx.take(x, mx.arange(pad_w, 0, -1), axis=2)
        right_start = x.shape[2] - 2
        right = mx.take(x, mx.arange(right_start, right_start - pad_w, -1), axis=2)
        x = mx.concatenate([left, x, right], axis=2)
    return x


def _upsample_bilinear_2x(x: mx.array) -> mx.array:
    _, h, w, _ = x.shape
    out_h, out_w = h * 2, w * 2
    y_src = (mx.arange(out_h, dtype=mx.float32) + 0.5) * 0.5 - 0.5
    x_src = (mx.arange(out_w, dtype=mx.float32) + 0.5) * 0.5 - 0.5
    y_src = mx.clip(y_src, 0.0, h - 1.0)
    x_src = mx.clip(x_src, 0.0, w - 1.0)
    y0 = mx.floor(y_src).astype(mx.int32)
    x0 = mx.floor(x_src).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, h - 1)
    x1 = mx.minimum(x0 + 1, w - 1)
    fy = (y_src - y0.astype(mx.float32)).reshape(1, -1, 1, 1)
    fx = (x_src - x0.astype(mx.float32)).reshape(1, 1, -1, 1)
    tl = mx.take(mx.take(x, y0, axis=1), x0, axis=2)
    tr = mx.take(mx.take(x, y0, axis=1), x1, axis=2)
    bl = mx.take(mx.take(x, y1, axis=1), x0, axis=2)
    br = mx.take(mx.take(x, y1, axis=1), x1, axis=2)
    return tl * (1 - fy) * (1 - fx) + tr * (1 - fy) * fx + bl * fy * (1 - fx) + br * fy * fx


class AdaINBlock(nn.Module):
    def __init__(self, ch: int = 1024, src_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 3)
        self.style1 = nn.Linear(src_dim, ch * 2)
        self.style2 = nn.Linear(src_dim, ch * 2)

    @staticmethod
    def _instance_norm(x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        return (x - mean) / mx.sqrt(var + 1e-8)

    @staticmethod
    def _style_mod(x: mx.array, style: mx.array) -> mx.array:
        scale, shift = mx.split(style, 2, axis=-1)
        return x * scale.reshape(1, 1, 1, -1) + shift.reshape(1, 1, 1, -1)

    def __call__(self, x: mx.array, source: mx.array) -> mx.array:
        residual = x
        h = self.conv1(_reflect_pad(x, 1, 1))
        h = self._style_mod(self._instance_norm(h), self.style1(source))
        h = nn.relu(h)
        h = self.conv2(_reflect_pad(h, 1, 1))
        h = self._style_mod(self._instance_norm(h), self.style2(source))
        return h + residual


class InSwapperNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = nn.Conv2d(3, 128, 7)
        self.enc1 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.blocks = [AdaINBlock(1024, 512) for _ in range(6)]
        self.dec0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dec1 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec3 = nn.Conv2d(128, 3, 7)

    def __call__(self, target: mx.array, source: mx.array) -> mx.array:
        x = _reflect_pad(target, 3, 3)
        x = nn.leaky_relu(self.enc0(x), negative_slope=0.2)
        x = nn.leaky_relu(self.enc1(x), negative_slope=0.2)
        x = nn.leaky_relu(self.enc2(x), negative_slope=0.2)
        x = nn.leaky_relu(self.enc3(x), negative_slope=0.2)
        for blk in self.blocks:
            x = blk(x, source)
        x = nn.leaky_relu(self.dec0(_upsample_bilinear_2x(x)), negative_slope=0.2)
        x = nn.leaky_relu(self.dec1(_upsample_bilinear_2x(x)), negative_slope=0.2)
        x = nn.leaky_relu(self.dec2(x), negative_slope=0.2)
        x = mx.tanh(self.dec3(_reflect_pad(x, 3, 3)))
        return (x + 1.0) / 2.0

    @staticmethod
    def from_onnx(model_path: str, dtype=mx.float16) -> "InSwapperNative":
        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load(model_path)
        w = {i.name: numpy_helper.to_array(i) for i in onnx_model.graph.initializer}

        def conv_w(name: str) -> mx.array:
            return mx.array(w[name].transpose(0, 2, 3, 1))

        model = InSwapperNative()

        enc_names = [
            ("onnx::Conv_833", "onnx::Conv_834"),
            ("onnx::Conv_836", "onnx::Conv_837"),
            ("onnx::Conv_839", "onnx::Conv_840"),
            ("onnx::Conv_842", "onnx::Conv_843"),
        ]
        for (wn, bn), layer in zip(enc_names, [model.enc0, model.enc1, model.enc2, model.enc3]):
            layer.weight = conv_w(wn)
            layer.bias = mx.array(w[bn])

        for i in range(6):
            blk = model.blocks[i]
            p = f"styles.{i}"
            blk.conv1.weight = conv_w(f"{p}.conv1.1.weight")
            blk.conv1.bias = mx.array(w[f"{p}.conv1.1.bias"])
            blk.conv2.weight = conv_w(f"{p}.conv2.1.weight")
            blk.conv2.bias = mx.array(w[f"{p}.conv2.1.bias"])
            blk.style1.weight = mx.array(w[f"{p}.style1.linear.weight"])
            blk.style1.bias = mx.array(w[f"{p}.style1.linear.bias"])
            blk.style2.weight = mx.array(w[f"{p}.style2.linear.weight"])
            blk.style2.bias = mx.array(w[f"{p}.style2.linear.bias"])

        dec_names = [
            ("onnx::Conv_845", "onnx::Conv_846"),
            ("onnx::Conv_848", "onnx::Conv_849"),
            ("onnx::Conv_851", "onnx::Conv_852"),
            ("up0.1.weight", "up0.1.bias"),
        ]
        for (wn, bn), layer in zip(dec_names, [model.dec0, model.dec1, model.dec2, model.dec3]):
            layer.weight = conv_w(wn)
            layer.bias = mx.array(w[bn])

        if dtype == mx.float16:
            params = model.parameters()
            def to_fp16(p):
                if isinstance(p, mx.array) and p.dtype == mx.float32:
                    return p.astype(mx.float16)
                elif isinstance(p, dict):
                    return {k: to_fp16(v) for k, v in p.items()}
                elif isinstance(p, list):
                    return [to_fp16(v) for v in p]
                return p
            model.update(to_fp16(model.parameters()))

        mx.eval(model.parameters())
        return model

    def run(self, output_names, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        target_nchw = input_feed["target"]
        source = input_feed["source"]

        target_nhwc = mx.array(target_nchw.transpose(0, 2, 3, 1))
        source_mx = mx.array(source)

        out_nhwc = self(target_nhwc, source_mx)
        mx.eval(out_nhwc)

        out_nchw = np.array(out_nhwc).transpose(0, 3, 1, 2)
        return [out_nchw]
