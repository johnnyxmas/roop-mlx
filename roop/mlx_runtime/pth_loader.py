"""Load PyTorch ``.pth`` checkpoint weights into MLX arrays.

Used for models that were never converted to ONNX (DMDNet, CLIPSeg).
Requires ``torch`` only at load time; the returned dict is pure MLX.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def load_pth(path: str, map_location: str = "cpu") -> dict[str, mx.array]:
    """Load a PyTorch checkpoint and return weights as MLX arrays.

    Works with both ``torch.save(model.state_dict(), ...)`` and
    ``torch.save({"state_dict": ...}, ...)`` formats.

    Parameters
    ----------
    path:
        Path to the ``.pth`` file.
    map_location:
        Passed to ``torch.load``; keeps tensors on CPU.

    Returns
    -------
    dict mapping parameter name → ``mx.array``.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for loading .pth files. "
            "Install it with: pip install torch"
        ) from e

    ckpt = torch.load(path, map_location=map_location, weights_only=True)

    # Unwrap common checkpoint wrappers
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "params"):
            if key in ckpt:
                ckpt = ckpt[key]
                break

    result: dict[str, mx.array] = {}
    for k, v in ckpt.items():
        try:
            arr = v.float().numpy()
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            result[k] = mx.array(arr)
        except Exception:
            pass  # skip non-tensor entries (scalars, strings, etc.)

    return result
