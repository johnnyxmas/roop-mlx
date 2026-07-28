"""Diffusion models for the Stable Diffusion tab, implemented in MLX.

The whole stack -- CLIP/T5 text encoders, the SD UNet, the FLUX transformer
and the VAE -- is native MLX, so this package adds no torch or diffusers
dependency to the fork.  Checkpoints are read straight from diffusers-layout
safetensors on the HuggingFace Hub and fetched only on first use; see
roop/diffusion/hub.py.
"""
