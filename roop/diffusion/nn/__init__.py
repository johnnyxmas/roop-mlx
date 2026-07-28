"""Pure-MLX implementations of the diffusion building blocks.

Module and parameter names deliberately mirror the diffusers checkpoint
layout so a safetensors state dict loads straight in via tree_unflatten,
with no per-key rename table to drift out of sync.
"""
