"""Load diffusers-layout safetensors into the MLX module tree.

Key names in the MLX modules mirror the checkpoint exactly, so no rename
table is needed.  What *does* need fixing up is layout and rank:

* conv weights are (out, in, kH, kW) in the checkpoint and (out, kH, kW, in)
  in MLX;
* several 1x1 convs (proj_in/proj_out, quant_conv, resnet shortcuts) are
  modelled as Linear here, so their (out, in, 1, 1) weight is squeezed.

Both fixes are chosen by comparing against the shape the module actually
declares, which doubles as validation: anything that cannot be reconciled is
reported rather than silently reshaped.
"""

import glob
import json
import os
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class WeightMismatch(RuntimeError):
    pass


def load_config(root: str, subdir: str) -> dict:
    path = os.path.join(root, subdir, "config.json") if subdir else \
        os.path.join(root, "config.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def component_files(root: str, subdir: str, variant: str = "") -> List[str]:
    """Weight files for one component directory of a local snapshot."""
    d = os.path.join(root, subdir)
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {d}")

    if variant:
        suffixed = [f for f in files if f.endswith(f".{variant}.safetensors")]
        if suffixed:
            return suffixed
    plain = [f for f in files if os.path.basename(f).count(".") == 1]
    shards = [f for f in files if "-of-" in os.path.basename(f)
              and ".fp16." not in os.path.basename(f)]
    if shards:
        return shards
    return plain or files


def read_state_dict(paths: List[str]) -> Dict[str, mx.array]:
    state: Dict[str, mx.array] = {}
    for p in paths:
        state.update(mx.load(p))
    return state


# Checkpoints saved by older diffusers releases name the VAE mid-block
# attention projections query/key/value/proj_attn instead of to_q/to_k/to_v/
# to_out.0.  RealisticVision 5.1 is one of them.
_LEGACY_ALIASES = {
    ".query.": ".to_q.",
    ".key.": ".to_k.",
    ".value.": ".to_v.",
    ".proj_attn.": ".to_out.0.",
}


def _canonicalise(state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}
    for name, value in state.items():
        for old, new in _LEGACY_ALIASES.items():
            if old in name:
                name = name.replace(old, new)
                break
        out[name] = value
    return out


def _conv_weight_names(model: nn.Module) -> set:
    """Names of parameters that are Conv2d kernels.

    Shape alone cannot identify these.  The VAE encoder's `conv_in` is
    Conv2d(3 -> C, 3x3), so its checkpoint layout (C, 3, 3, 3) and its MLX
    layout (C, 3, 3, 3) are identical -- a shape-equality check silently
    accepts the un-transposed weight and scrambles the RGB channels against
    the kernel's width axis.  Structure survives, colour does not.
    """
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prefix = f"{module_name}." if module_name else ""
            names.add(f"{prefix}weight")
    return names


def _reconcile(name: str, src: mx.array, dst_shape: Tuple[int, ...],
               is_conv: bool = False) -> mx.array:
    # Conv kernels are always stored NCHW in the checkpoint, so transpose
    # first and never let an ambiguous shape match short-circuit it.
    if is_conv and src.ndim == 4:
        moved = src.transpose(0, 2, 3, 1)
        if tuple(moved.shape) == tuple(dst_shape):
            return moved

    if tuple(src.shape) == tuple(dst_shape):
        return src
    # conv: (out, in, kH, kW) -> (out, kH, kW, in)
    if src.ndim == 4 and len(dst_shape) == 4:
        moved = src.transpose(0, 2, 3, 1)
        if tuple(moved.shape) == tuple(dst_shape):
            return moved
    # 1x1 conv modelled as Linear
    if src.ndim == 4 and len(dst_shape) == 2 and src.shape[2:] == (1, 1):
        squeezed = src.reshape(src.shape[0], src.shape[1])
        if tuple(squeezed.shape) == tuple(dst_shape):
            return squeezed
    # Linear modelled as 1x1 conv (the reverse; some repos store it flat)
    if src.ndim == 2 and len(dst_shape) == 4 and dst_shape[1:3] == (1, 1):
        return src.reshape(dst_shape)
    raise WeightMismatch(
        f"{name}: checkpoint has {tuple(src.shape)}, model wants "
        f"{tuple(dst_shape)}"
    )


def load_into(model: nn.Module, state: Dict[str, mx.array],
              dtype: mx.Dtype = mx.float16,
              strict: bool = True) -> None:
    """Load *state* into *model* in place, adapting layout to each parameter."""
    expected = {k: v.shape for k, v in tree_flatten(model.parameters())}
    state = _canonicalise(state)
    conv_weights = _conv_weight_names(model)

    updates: List[Tuple[str, mx.array]] = []
    missing: List[str] = []
    for name, want in expected.items():
        src = state.get(name)
        if src is None:
            missing.append(name)
            continue
        adapted = _reconcile(name, src, want, is_conv=name in conv_weights)
        updates.append((name, adapted.astype(dtype)))

    unexpected = [k for k in state
                  if k not in expected
                  and not k.endswith("position_ids")]

    if strict and missing:
        raise WeightMismatch(
            f"{len(missing)} parameter(s) absent from the checkpoint, e.g. "
            f"{missing[:5]}"
        )
    if unexpected:
        # Extra keys are normal (EMA copies, logit scales, buffers); surface
        # them at a whisper rather than failing the load.
        print(f"[diffusion] ignoring {len(unexpected)} unused checkpoint "
              f"key(s), e.g. {unexpected[:3]}")

    model.update(tree_unflatten(updates))
    mx.eval(model.parameters())


def load_component(model: nn.Module, root: str, subdir: str,
                   variant: str = "", dtype: mx.Dtype = mx.float16,
                   strict: bool = True) -> None:
    files = component_files(root, subdir, variant)
    load_into(model, read_state_dict(files), dtype=dtype, strict=strict)
