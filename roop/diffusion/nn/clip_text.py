"""CLIP text encoder in MLX (diffusers CLIPTextModel / ...WithProjection).

Covers both encoders SD needs: CLIP-L (SD 1.5 and SDXL slot 1, quick_gelu)
and OpenCLIP bigG (SDXL slot 2, plain gelu + text_projection).
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


def _quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


_ACTS = {
    "quick_gelu": _quick_gelu,
    "gelu": nn.gelu,
    "gelu_new": nn.gelu_approx,
}


class CLIPConfig:
    def __init__(self, cfg: dict):
        self.hidden_size = cfg.get("hidden_size", 768)
        self.num_hidden_layers = cfg.get("num_hidden_layers", 12)
        self.num_attention_heads = cfg.get("num_attention_heads", 12)
        self.intermediate_size = cfg.get("intermediate_size", 3072)
        self.max_position_embeddings = cfg.get("max_position_embeddings", 77)
        self.vocab_size = cfg.get("vocab_size", 49408)
        self.hidden_act = cfg.get("hidden_act", "quick_gelu")
        self.projection_dim = cfg.get("projection_dim", self.hidden_size)
        self.eos_token_id = cfg.get("eos_token_id", 2)


class _Attention(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        b, l, d = x.shape
        h = self.num_heads
        q = self.q_proj(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(q.shape[-1])
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                   mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, d)
        return self.out_proj(out)


class _MLP(nn.Module):
    def __init__(self, dims: int, hidden: int, act):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)
        self._act = act

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._act(self.fc1(x)))


class _EncoderLayer(nn.Module):
    def __init__(self, cfg: CLIPConfig, act):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size)
        self.self_attn = _Attention(cfg.hidden_size, cfg.num_attention_heads)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = _MLP(cfg.hidden_size, cfg.intermediate_size, act)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _Embeddings(nn.Module):
    def __init__(self, cfg: CLIPConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings,
                                               cfg.hidden_size)


class _Encoder(nn.Module):
    def __init__(self, cfg: CLIPConfig, act):
        super().__init__()
        self.layers = [_EncoderLayer(cfg, act)
                       for _ in range(cfg.num_hidden_layers)]


class _TextTransformer(nn.Module):
    def __init__(self, cfg: CLIPConfig, act):
        super().__init__()
        self.embeddings = _Embeddings(cfg)
        self.encoder = _Encoder(cfg, act)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size)


class CLIPTextModel(nn.Module):
    """Returns (all hidden states, final-normed states, pooled projection)."""

    def __init__(self, cfg: dict, with_projection: bool = False):
        super().__init__()
        self.config = CLIPConfig(cfg)
        act = _ACTS.get(self.config.hidden_act, _quick_gelu)
        self.text_model = _TextTransformer(self.config, act)
        self.with_projection = with_projection
        if with_projection:
            self.text_projection = nn.Linear(
                self.config.hidden_size, self.config.projection_dim, bias=False
            )

    def __call__(self, tokens: mx.array):
        b, l = tokens.shape
        tm = self.text_model
        x = tm.embeddings.token_embedding(tokens)
        x = x + tm.embeddings.position_embedding.weight[:l][None]

        mask = nn.MultiHeadAttention.create_additive_causal_mask(l, x.dtype)

        hidden: List[mx.array] = [x]
        for layer in tm.encoder.layers:
            x = layer(x, mask)
            hidden.append(x)

        final = tm.final_layer_norm(x)

        # Pooled = state at the EOT token, which is the highest token id in
        # the CLIP vocabulary and therefore the argmax of the id sequence.
        eot = mx.argmax(tokens, axis=-1)
        pooled = final[mx.arange(b), eot]
        if self.with_projection:
            pooled = self.text_projection(pooled)

        return hidden, final, pooled

    def encode(self, tokens: mx.array, clip_skip: int = 1,
               penultimate: bool = False):
        """Text embedding used for cross-attention conditioning.

        SD 1.5 with clip_skip=n takes the n-th-from-last hidden state and
        still applies final_layer_norm; SDXL takes the penultimate state raw.
        """
        hidden, final, pooled = self(tokens)
        if penultimate:
            return hidden[-2], pooled
        if clip_skip > 1:
            return self.text_model.final_layer_norm(hidden[-clip_skip]), pooled
        return final, pooled
