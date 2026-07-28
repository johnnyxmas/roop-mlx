"""AutoencoderKL in MLX -- the VAE used by SD 1.5, SDXL and FLUX.

Everything runs NHWC (MLX's native conv layout); callers hand in and get
back NHWC arrays.  Parameter names match the diffusers checkpoint.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Linear(in_ch, out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        residual = self.conv_shortcut(x) if "conv_shortcut" in self else x
        h = self.conv1(_silu(self.norm1(x)))
        h = self.conv2(_silu(self.norm2(h)))
        return h + residual


class _Attention(nn.Module):
    """Single-head spatial self-attention (diffusers `Attention`, vae flavour)."""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.group_norm(x).reshape(b, h * w, c)
        q, k, v = self.to_q(y), self.to_k(y), self.to_v(y)
        scale = 1.0 / math.sqrt(c)
        out = mx.fast.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None], scale=scale
        )[:, 0]
        return x + self.to_out[0](out).reshape(b, h, w, c)


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # diffusers pads (0,1,0,1) -- bottom/right only.
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        return self.conv(x.reshape(b, h * 2, w * 2, c))


class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, layers, add_downsample, groups):
        super().__init__()
        self.resnets = [
            _ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups)
            for i in range(layers)
        ]
        if add_downsample:
            self.downsamplers = [_Downsample(out_ch)]

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
        return x


class _UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, layers, add_upsample, groups):
        super().__init__()
        self.resnets = [
            _ResnetBlock(in_ch if i == 0 else out_ch, out_ch, groups)
            for i in range(layers)
        ]
        if add_upsample:
            self.upsamplers = [_Upsample(out_ch)]

    def __call__(self, x):
        for r in self.resnets:
            x = r(x)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x


class _MidBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.resnets = [_ResnetBlock(channels, channels, groups),
                        _ResnetBlock(channels, channels, groups)]
        self.attentions = [_Attention(channels, groups)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class _Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chans: List[int] = cfg["block_out_channels"]
        groups = cfg.get("norm_num_groups", 32)
        layers = cfg.get("layers_per_block", 2)
        latent = cfg.get("latent_channels", 4)
        double_z = cfg.get("double_z", True)

        self.conv_in = nn.Conv2d(cfg.get("in_channels", 3), chans[0], 3,
                                 padding=1)
        blocks = []
        for i, ch in enumerate(chans):
            blocks.append(_DownBlock(chans[max(i - 1, 0)], ch, layers,
                                     i < len(chans) - 1, groups))
        self.down_blocks = blocks
        self.mid_block = _MidBlock(chans[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, chans[-1],
                                          pytorch_compatible=True)
        self.conv_out = nn.Conv2d(chans[-1], 2 * latent if double_z else latent,
                                  3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for b in self.down_blocks:
            x = b(x)
        x = self.mid_block(x)
        return self.conv_out(_silu(self.conv_norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        chans: List[int] = cfg["block_out_channels"]
        groups = cfg.get("norm_num_groups", 32)
        layers = cfg.get("layers_per_block", 2) + 1
        latent = cfg.get("latent_channels", 4)
        rchans = list(reversed(chans))

        self.conv_in = nn.Conv2d(latent, rchans[0], 3, padding=1)
        self.mid_block = _MidBlock(rchans[0], groups)
        blocks = []
        for i, ch in enumerate(rchans):
            blocks.append(_UpBlock(rchans[max(i - 1, 0)], ch, layers,
                                   i < len(rchans) - 1, groups))
        self.up_blocks = blocks
        self.conv_norm_out = nn.GroupNorm(groups, rchans[-1],
                                          pytorch_compatible=True)
        self.conv_out = nn.Conv2d(rchans[-1], cfg.get("out_channels", 3), 3,
                                  padding=1)

    def __call__(self, x):
        x = self.mid_block(self.conv_in(x))
        for b in self.up_blocks:
            x = b(x)
        return self.conv_out(_silu(self.conv_norm_out(x)))


class AutoencoderKL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        latent = cfg.get("latent_channels", 4)
        self._scaling_factor = cfg.get("scaling_factor", 0.18215)
        self._shift_factor = cfg.get("shift_factor", None)
        self.encoder = _Encoder(cfg)
        self.decoder = _Decoder(cfg)
        # SD keeps 1x1 quant convs around the latent; FLUX's VAE drops them.
        if cfg.get("use_quant_conv", True):
            self.quant_conv = nn.Linear(2 * latent, 2 * latent)
        if cfg.get("use_post_quant_conv", True):
            self.post_quant_conv = nn.Linear(latent, latent)

    def encode(self, x: mx.array, key: Optional[mx.array] = None) -> mx.array:
        """NHWC image in [-1, 1] -> scaled latent."""
        h = self.encoder(x)
        if "quant_conv" in self:
            h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=-1)
        if key is None:
            z = mean
        else:
            std = mx.exp(0.5 * mx.clip(logvar, -30.0, 20.0))
            z = mean + std * mx.random.normal(mean.shape, key=key)
        if self._shift_factor is not None:
            z = z - self._shift_factor
        return z * self._scaling_factor

    def decode(self, z: mx.array) -> mx.array:
        """Scaled latent -> NHWC image in [-1, 1]."""
        z = z / self._scaling_factor
        if self._shift_factor is not None:
            z = z + self._shift_factor
        if "post_quant_conv" in self:
            z = self.post_quant_conv(z)
        return self.decoder(z)
