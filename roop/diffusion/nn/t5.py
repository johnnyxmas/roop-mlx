"""T5 v1.1 XXL encoder in MLX -- the prompt encoder FLUX conditions on.

Only the encoder stack is implemented; FLUX never runs the decoder.  Two
T5 quirks matter for parity: attention scores are *not* divided by
sqrt(head_dim), and the relative position bias is computed once in block 0
and reused by every later block.
"""

import math
import mlx.core as mx
import mlx.nn as nn


class _RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self._eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self._eps)


_ACTS = {
    "gelu": nn.gelu_approx,       # T5 uses the tanh approximation
    "gelu_new": nn.gelu_approx,
    "relu": nn.relu,
    "silu": nn.silu,
}


def _act_fn(cfg: dict):
    """Feed-forward activation, from `feed_forward_proj` (e.g. "gated-gelu")."""
    name = cfg.get("dense_act_fn")
    if name is None:
        proj = cfg.get("feed_forward_proj", "relu")
        name = proj.split("gated-")[-1]
    return _ACTS.get(name, nn.gelu_approx)


def _is_gated(cfg: dict) -> bool:
    if "is_gated_act" in cfg:
        return bool(cfg["is_gated_act"])
    return str(cfg.get("feed_forward_proj", "relu")).startswith("gated-")


class _DenseGatedActDense(nn.Module):
    """T5 v1.1 feed-forward: wo(act(wi_0(x)) * wi_1(x))."""

    def __init__(self, d_model: int, d_ff: int, act):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self._act = act

    def __call__(self, x):
        return self.wo(self._act(self.wi_0(x)) * self.wi_1(x))


class _DenseActDense(nn.Module):
    """Original T5 v1.0 feed-forward: a single wi, no gate."""

    def __init__(self, d_model: int, d_ff: int, act):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self._act = act

    def __call__(self, x):
        return self.wo(self._act(self.wi(x)))


class _SelfAttention(nn.Module):
    def __init__(self, cfg: dict, has_bias: bool):
        super().__init__()
        d_model = cfg["d_model"]
        self._heads = cfg["num_heads"]
        inner = self._heads * cfg.get("d_kv", 64)
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)
        if has_bias:
            self.relative_attention_bias = nn.Embedding(
                cfg.get("relative_attention_num_buckets", 32), self._heads
            )

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        b, l, _ = x.shape
        h = self._heads
        q = self.q(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, l, h, -1).transpose(0, 2, 1, 3)
        # T5 deliberately omits the 1/sqrt(d) factor.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0,
                                                   mask=bias)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.o(out)


class _LayerSelfAttention(nn.Module):
    def __init__(self, cfg, has_bias):
        super().__init__()
        self.SelfAttention = _SelfAttention(cfg, has_bias)
        self.layer_norm = _RMSNorm(cfg["d_model"],
                                   cfg.get("layer_norm_epsilon", 1e-6))

    def __call__(self, x, bias):
        return x + self.SelfAttention(self.layer_norm(x), bias)


class _LayerFF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dense = _DenseGatedActDense if _is_gated(cfg) else _DenseActDense
        self.DenseReluDense = dense(cfg["d_model"], cfg["d_ff"], _act_fn(cfg))
        self.layer_norm = _RMSNorm(cfg["d_model"],
                                   cfg.get("layer_norm_epsilon", 1e-6))

    def __call__(self, x):
        return x + self.DenseReluDense(self.layer_norm(x))


class _Block(nn.Module):
    def __init__(self, cfg, has_bias):
        super().__init__()
        self.layer = [_LayerSelfAttention(cfg, has_bias), _LayerFF(cfg)]

    def __call__(self, x, bias):
        return self.layer[1](self.layer[0](x, bias))


def _relative_position_bucket(rel: mx.array, num_buckets: int = 32,
                              max_distance: int = 128) -> mx.array:
    """Bidirectional T5 bucketing (half the buckets per direction)."""
    num_buckets //= 2
    ret = (rel > 0).astype(mx.int32) * num_buckets
    n = mx.abs(rel)

    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (
        mx.log(n.astype(mx.float32) / max_exact + 1e-6)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(mx.int32)
    large = mx.minimum(large, num_buckets - 1)
    return ret + mx.where(is_small, n.astype(mx.int32), large)


class _Encoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.block = [_Block(cfg, has_bias=(i == 0))
                      for i in range(cfg["num_layers"])]
        self.final_layer_norm = _RMSNorm(cfg["d_model"],
                                         cfg.get("layer_norm_epsilon", 1e-6))


class T5EncoderModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        self.shared = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.encoder = _Encoder(cfg)
        self._num_buckets = cfg.get("relative_attention_num_buckets", 32)
        self._max_distance = cfg.get("relative_attention_max_distance", 128)

    def _position_bias(self, length: int, dtype) -> mx.array:
        pos = mx.arange(length, dtype=mx.int32)
        rel = pos[None, :] - pos[:, None]
        buckets = _relative_position_bucket(rel, self._num_buckets,
                                            self._max_distance)
        emb = self.encoder.block[0].layer[0].SelfAttention
        bias = emb.relative_attention_bias(buckets)      # (l, l, heads)
        return bias.transpose(2, 0, 1)[None].astype(dtype)

    def __call__(self, tokens: mx.array) -> mx.array:
        x = self.shared(tokens)
        bias = self._position_bias(tokens.shape[1], x.dtype)
        for block in self.encoder.block:
            x = block(x, bias)
        return self.encoder.final_layer_norm(x)
