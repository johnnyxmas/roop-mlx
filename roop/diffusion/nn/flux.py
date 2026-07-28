"""FluxTransformer2DModel in MLX -- the denoiser behind FLUX.1 Kontext.

The architecture is a rectified-flow DiT: 19 "double stream" blocks that
carry text and image tokens in separate residual streams while attending
jointly, then 38 "single stream" blocks over the concatenated sequence.
Conditioning is AdaLN-Zero driven by timestep + distilled guidance + the
pooled CLIP embedding; positions come from 3-axis RoPE, which is what lets
Kontext tell the reference image apart from the canvas (axis 0 is the image
index).
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from roop.diffusion.nn.unet import timestep_embedding


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self._eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self._eps)


class _LayerNormNoAffine(nn.Module):
    """LayerNorm with elementwise_affine=False -- carries no parameters."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self._eps)


# -- positional encoding ----------------------------------------------------
def rope_freqs(ids: mx.array, axes_dim: List[int],
               theta: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """3-axis RoPE tables for token positions *ids* of shape (seq, 3).

    Returns (cos, sin), each (seq, sum(axes_dim)), with every frequency
    repeated twice so it lines up with the interleaved real/imaginary
    pairing used when the rotation is applied.
    """
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(axes_dim):
        pos = ids[:, axis].astype(mx.float32)
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta ** scale)
        freqs = pos[:, None] * omega[None, :]           # (seq, dim/2)
        cos_parts.append(mx.repeat(mx.cos(freqs), 2, axis=1))
        sin_parts.append(mx.repeat(mx.sin(freqs), 2, axis=1))
    return (mx.concatenate(cos_parts, axis=-1),
            mx.concatenate(sin_parts, axis=-1))


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x is (b, heads, seq, head_dim); cos/sin are (seq, head_dim)."""
    b, h, l, d = x.shape
    pairs = x.reshape(b, h, l, d // 2, 2)
    x_real, x_imag = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(b, h, l, d)
    return x * cos[None, None] + rotated * sin[None, None]


# -- conditioning embedders -------------------------------------------------
class _TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x):
        return self.linear_2(_silu(self.linear_1(x)))


class _TextProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x):
        return self.linear_2(_silu(self.linear_1(x)))


class _CombinedTimestepTextEmbed(nn.Module):
    def __init__(self, dim: int, pooled_dim: int, with_guidance: bool,
                 freq_dim: int = 256):
        super().__init__()
        self._freq_dim = freq_dim
        self.timestep_embedder = _TimestepEmbedding(freq_dim, dim)
        if with_guidance:
            self.guidance_embedder = _TimestepEmbedding(freq_dim, dim)
        self.text_embedder = _TextProjection(pooled_dim, dim)

    def __call__(self, timestep, guidance, pooled):
        emb = self.timestep_embedder(
            timestep_embedding(timestep, self._freq_dim).astype(pooled.dtype)
        )
        if "guidance_embedder" in self and guidance is not None:
            emb = emb + self.guidance_embedder(
                timestep_embedding(guidance, self._freq_dim).astype(pooled.dtype)
            )
        return emb + self.text_embedder(pooled)


class _AdaLayerNormZero(nn.Module):
    """Six-way modulation: shift/scale/gate for attention and for the MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = _LayerNormNoAffine()

    def __call__(self, x, temb):
        params = self.linear(_silu(temb))
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = mx.split(
            params, 6, axis=-1
        )
        y = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return y, gate[:, None], shift_mlp, scale_mlp, gate_mlp[:, None]


class _AdaLayerNormZeroSingle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = _LayerNormNoAffine()

    def __call__(self, x, temb):
        shift, scale, gate = mx.split(self.linear(_silu(temb)), 3, axis=-1)
        y = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return y, gate[:, None]


class _AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = _LayerNormNoAffine()

    def __call__(self, x, temb):
        # diffusers emits scale before shift here (unlike AdaLayerNormZero).
        scale, shift = mx.split(self.linear(_silu(temb)), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = [_GELUProj(dim, dim * mult), nn.Identity(),
                    nn.Linear(dim * mult, dim)]

    def __call__(self, x):
        return self.net[2](self.net[0](x))


class _GELUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x):
        return nn.gelu_approx(self.proj(x))


# -- attention --------------------------------------------------------------
class _JointAttention(nn.Module):
    """Attention over text+image tokens, with per-stream projections."""

    def __init__(self, dim: int, heads: int, with_context: bool):
        super().__init__()
        self._heads = heads
        head_dim = dim // heads
        self.norm_q = _RMSNorm(head_dim)
        self.norm_k = _RMSNorm(head_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self._with_context = with_context
        if with_context:
            self.norm_added_q = _RMSNorm(head_dim)
            self.norm_added_k = _RMSNorm(head_dim)
            self.add_q_proj = nn.Linear(dim, dim)
            self.add_k_proj = nn.Linear(dim, dim)
            self.add_v_proj = nn.Linear(dim, dim)
            self.to_out = [nn.Linear(dim, dim)]
            self.to_add_out = nn.Linear(dim, dim)

    def _heads_first(self, x):
        b, l, _ = x.shape
        return x.reshape(b, l, self._heads, -1).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, context: Optional[mx.array],
                 rope: Tuple[mx.array, mx.array]):
        cos, sin = rope
        q = self.norm_q(self._heads_first(self.to_q(x)))
        k = self.norm_k(self._heads_first(self.to_k(x)))
        v = self._heads_first(self.to_v(x))

        ctx_len = 0
        if context is not None:
            ctx_len = context.shape[1]
            cq = self.norm_added_q(self._heads_first(self.add_q_proj(context)))
            ck = self.norm_added_k(self._heads_first(self.add_k_proj(context)))
            cv = self._heads_first(self.add_v_proj(context))
            # Text tokens come first in the joint sequence.
            q = mx.concatenate([cq, q], axis=2)
            k = mx.concatenate([ck, k], axis=2)
            v = mx.concatenate([cv, v], axis=2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scale = 1.0 / math.sqrt(q.shape[-1])
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        b = out.shape[0]
        out = out.transpose(0, 2, 1, 3).reshape(b, out.shape[2], -1)

        if context is None:
            return out, None
        ctx_out, img_out = out[:, :ctx_len], out[:, ctx_len:]
        return self.to_out[0](img_out), self.to_add_out(ctx_out)


# -- blocks -----------------------------------------------------------------
class _DoubleBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = _AdaLayerNormZero(dim)
        self.norm1_context = _AdaLayerNormZero(dim)
        self.attn = _JointAttention(dim, heads, with_context=True)
        self.norm2 = _LayerNormNoAffine()
        self.ff = _FeedForward(dim)
        self.norm2_context = _LayerNormNoAffine()
        self.ff_context = _FeedForward(dim)

    def __call__(self, x, context, temb, rope):
        nx, gate, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, temb)
        nc, c_gate, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            self.norm1_context(context, temb)

        attn_out, ctx_attn_out = self.attn(nx, nc, rope)

        x = x + gate * attn_out
        y = self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp * self.ff(y)

        context = context + c_gate * ctx_attn_out
        yc = (self.norm2_context(context) * (1 + c_scale_mlp[:, None])
              + c_shift_mlp[:, None])
        context = context + c_gate_mlp * self.ff_context(yc)
        return x, context


class _SingleBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm = _AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, dim * mlp_ratio)
        self.proj_out = nn.Linear(dim + dim * mlp_ratio, dim)
        self.attn = _JointAttention(dim, heads, with_context=False)

    def __call__(self, x, temb, rope):
        residual = x
        nx, gate = self.norm(x, temb)
        mlp = nn.gelu_approx(self.proj_mlp(nx))
        attn_out, _ = self.attn(nx, None, rope)
        out = self.proj_out(mx.concatenate([attn_out, mlp], axis=-1))
        return residual + gate * out


class FluxTransformer2DModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        heads = cfg.get("num_attention_heads", 24)
        head_dim = cfg.get("attention_head_dim", 128)
        dim = heads * head_dim
        self._dim = dim
        self._axes_dim = cfg.get("axes_dims_rope", [16, 56, 56])
        self._rope_theta = cfg.get("rope_theta", 10000)

        self.x_embedder = nn.Linear(cfg.get("in_channels", 64), dim)
        self.context_embedder = nn.Linear(
            cfg.get("joint_attention_dim", 4096), dim
        )
        self.time_text_embed = _CombinedTimestepTextEmbed(
            dim, cfg.get("pooled_projection_dim", 768),
            with_guidance=cfg.get("guidance_embeds", True),
        )
        self.transformer_blocks = [
            _DoubleBlock(dim, heads)
            for _ in range(cfg.get("num_layers", 19))
        ]
        self.single_transformer_blocks = [
            _SingleBlock(dim, heads)
            for _ in range(cfg.get("num_single_layers", 38))
        ]
        self.norm_out = _AdaLayerNormContinuous(dim)
        self.proj_out = nn.Linear(dim, cfg.get("in_channels", 64))

    def __call__(self, hidden: mx.array, encoder_hidden: mx.array,
                 pooled: mx.array, timestep: mx.array,
                 img_ids: mx.array, txt_ids: mx.array,
                 guidance: Optional[mx.array] = None) -> mx.array:
        x = self.x_embedder(hidden)
        context = self.context_embedder(encoder_hidden)
        temb = self.time_text_embed(timestep * 1000.0,
                                    None if guidance is None
                                    else guidance * 1000.0,
                                    pooled)

        ids = mx.concatenate([txt_ids, img_ids], axis=0)
        cos, sin = rope_freqs(ids, self._axes_dim, self._rope_theta)
        cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)

        for block in self.transformer_blocks:
            x, context = block(x, context, temb, (cos, sin))

        joint = mx.concatenate([context, x], axis=1)
        for block in self.single_transformer_blocks:
            joint = block(joint, temb, (cos, sin))
        x = joint[:, context.shape[1]:]

        return self.proj_out(self.norm_out(x, temb))
