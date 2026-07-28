"""UNet2DConditionModel in MLX -- the denoiser for SD 1.5 and SDXL.

Both variants come out of the same class; the differences (dual text
encoder conditioning, the added time-id embedding, linear vs conv
projections in the transformer blocks, per-level transformer depth) are all
driven by the diffusers unet/config.json.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0,
                       flip_sin_to_cos: bool = True,
                       downscale_freq_shift: float = 0.0) -> mx.array:
    """diffusers get_timestep_embedding, MLX port."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    exponent = exponent / (half - downscale_freq_shift)
    emb = mx.exp(exponent)[None] * t.astype(mx.float32)[:, None]
    if flip_sin_to_cos:
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x):
        return self.linear_2(_silu(self.linear_1(x)))


class _ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Linear(in_ch, out_ch)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        residual = self.conv_shortcut(x) if "conv_shortcut" in self else x
        h = self.conv1(_silu(self.norm1(x)))
        h = h + self.time_emb_proj(_silu(temb))[:, None, None, :]
        h = self.conv2(_silu(self.norm2(h)))
        return h + residual


class _CrossAttention(nn.Module):
    def __init__(self, query_dim: int, heads: int, context_dim: int):
        super().__init__()
        self.heads = heads
        inner = query_dim
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = [nn.Linear(inner, query_dim)]

    def __call__(self, x: mx.array, context: Optional[mx.array] = None):
        ctx = x if context is None else context
        b, l, _ = x.shape
        h = self.heads
        q = self.to_q(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        k = self.to_k(ctx).reshape(b, ctx.shape[1], h, -1).transpose(0, 2, 1, 3)
        v = self.to_v(ctx).reshape(b, ctx.shape[1], h, -1).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(q.shape[-1])
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out[0](out)


class _GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x):
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        # `net` is a Sequential in diffusers: [GEGLU, Dropout, Linear].
        self.net = [_GEGLU(dim, dim * mult), nn.Identity(),
                    nn.Linear(dim * mult, dim)]

    def __call__(self, x):
        return self.net[2](self.net[0](x))


class _BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, context_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _CrossAttention(dim, heads, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _CrossAttention(dim, heads, context_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)

    def __call__(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class _Transformer2D(nn.Module):
    def __init__(self, channels: int, heads: int, context_dim: int,
                 depth: int, groups: int, linear_projection: bool):
        super().__init__()
        self._linear_projection = linear_projection
        self.norm = nn.GroupNorm(groups, channels, pytorch_compatible=True)
        # Conv 1x1 and Linear are the same op on NHWC; the checkpoint stores
        # them with different rank, which the weight loader squeezes.
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = [
            _BasicTransformerBlock(channels, heads, context_dim)
            for _ in range(depth)
        ]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x, context):
        b, h, w, c = x.shape
        residual = x
        y = self.norm(x).reshape(b, h * w, c)
        y = self.proj_in(y)
        for blk in self.transformer_blocks:
            y = blk(y, context)
        y = self.proj_out(y)
        return y.reshape(b, h, w, c) + residual


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x):
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        b, h, w, c = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        return self.conv(x.reshape(b, h * 2, w * 2, c))


class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb, layers, groups, add_downsample,
                 attn: Optional[dict]):
        super().__init__()
        self.resnets = [
            _ResnetBlock(in_ch if i == 0 else out_ch, out_ch, temb, groups)
            for i in range(layers)
        ]
        if attn is not None:
            self.attentions = [_Transformer2D(out_ch, **attn)
                               for _ in range(layers)]
        if add_downsample:
            self.downsamplers = [_Downsample(out_ch)]

    def __call__(self, x, temb, context):
        outputs = []
        attns = self.attentions if "attentions" in self else None
        for i, res in enumerate(self.resnets):
            x = res(x, temb)
            if attns is not None:
                x = attns[i](x, context)
            outputs.append(x)
        if "downsamplers" in self:
            x = self.downsamplers[0](x)
            outputs.append(x)
        return x, outputs


class _UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, prev_ch, temb, layers, groups,
                 add_upsample, attn: Optional[dict]):
        super().__init__()
        resnets = []
        for i in range(layers):
            skip = in_ch if i == layers - 1 else out_ch
            resin = prev_ch if i == 0 else out_ch
            resnets.append(_ResnetBlock(resin + skip, out_ch, temb, groups))
        self.resnets = resnets
        if attn is not None:
            self.attentions = [_Transformer2D(out_ch, **attn)
                               for _ in range(layers)]
        if add_upsample:
            self.upsamplers = [_Upsample(out_ch)]

    def __call__(self, x, temb, context, skips):
        attns = self.attentions if "attentions" in self else None
        for i, res in enumerate(self.resnets):
            x = mx.concatenate([x, skips.pop()], axis=-1)
            x = res(x, temb)
            if attns is not None:
                x = attns[i](x, context)
        if "upsamplers" in self:
            x = self.upsamplers[0](x)
        return x


class _MidBlock(nn.Module):
    def __init__(self, channels, temb, groups, attn: dict):
        super().__init__()
        self.resnets = [_ResnetBlock(channels, channels, temb, groups),
                        _ResnetBlock(channels, channels, temb, groups)]
        self.attentions = [_Transformer2D(channels, **attn)]

    def __call__(self, x, temb, context):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class UNet2DConditionModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        chans: List[int] = cfg["block_out_channels"]
        groups = cfg.get("norm_num_groups", 32)
        layers = cfg.get("layers_per_block", 2)
        ctx_dim = cfg.get("cross_attention_dim", 768)
        down_types = cfg["down_block_types"]
        up_types = cfg["up_block_types"]
        temb_dim = chans[0] * 4
        self._freq_dim = chans[0]

        heads = cfg.get("attention_head_dim", 8)
        if isinstance(heads, int):
            heads = [heads] * len(chans)
        depths = cfg.get("transformer_layers_per_block", 1)
        if isinstance(depths, int):
            depths = [depths] * len(chans)
        linear_proj = cfg.get("use_linear_projection", False)

        self.conv_in = nn.Conv2d(cfg.get("in_channels", 4), chans[0], 3,
                                 padding=1)
        self.time_embedding = _TimestepEmbedding(chans[0], temb_dim)

        # SDXL conditions on pooled text + a 6-value size/crop vector.
        self._addition_time_embed_dim = cfg.get("addition_time_embed_dim")
        if cfg.get("addition_embed_type") == "text_time":
            self.add_embedding = _TimestepEmbedding(
                cfg["projection_class_embeddings_input_dim"], temb_dim
            )

        def attn_kwargs(i):
            return dict(heads=heads[i], context_dim=ctx_dim, depth=depths[i],
                        groups=groups, linear_projection=linear_proj)

        blocks = []
        for i, ch in enumerate(chans):
            blocks.append(_DownBlock(
                chans[max(i - 1, 0)], ch, temb_dim, layers, groups,
                i < len(chans) - 1,
                attn_kwargs(i) if "CrossAttn" in down_types[i] else None,
            ))
        self.down_blocks = blocks

        self.mid_block = _MidBlock(chans[-1], temb_dim, groups,
                                   attn_kwargs(len(chans) - 1))

        rchans = list(reversed(chans))
        rheads = list(reversed(heads))
        rdepths = list(reversed(depths))
        ups = []
        for i, ch in enumerate(rchans):
            prev = rchans[max(i - 1, 0)]
            skip_in = rchans[min(i + 1, len(rchans) - 1)]
            attn = None
            if "CrossAttn" in up_types[i]:
                attn = dict(heads=rheads[i], context_dim=ctx_dim,
                            depth=rdepths[i], groups=groups,
                            linear_projection=linear_proj)
            ups.append(_UpBlock(skip_in, ch, prev, temb_dim, layers + 1,
                                groups, i < len(rchans) - 1, attn))
        self.up_blocks = ups

        self.conv_norm_out = nn.GroupNorm(groups, chans[0],
                                          pytorch_compatible=True)
        self.conv_out = nn.Conv2d(chans[0], cfg.get("out_channels", 4), 3,
                                  padding=1)

    def __call__(self, x: mx.array, t: mx.array, context: mx.array,
                 added_cond: Optional[dict] = None) -> mx.array:
        temb = self.time_embedding(
            timestep_embedding(t, self._freq_dim).astype(x.dtype)
        )

        if "add_embedding" in self and added_cond is not None:
            time_ids = added_cond["time_ids"]          # (b, 6)
            flat = time_ids.reshape(-1)
            embeds = timestep_embedding(flat, self._addition_time_embed_dim)
            embeds = embeds.reshape(time_ids.shape[0], -1)
            add = mx.concatenate([added_cond["text_embeds"], embeds], axis=-1)
            temb = temb + self.add_embedding(add.astype(x.dtype))

        x = self.conv_in(x)
        skips = [x]
        for block in self.down_blocks:
            x, outs = block(x, temb, context)
            skips.extend(outs)

        x = self.mid_block(x, temb, context)

        for block in self.up_blocks:
            x = block(x, temb, context, skips)

        return self.conv_out(_silu(self.conv_norm_out(x)))
