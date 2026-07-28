"""Unit tests for the diffusion tab's model plumbing.

Everything here is network- and weight-free: it covers the pieces that
decide *what* gets downloaded and *how* a checkpoint is mapped onto the MLX
modules, which is where a wrong answer is both silent and expensive.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from roop.diffusion import hub, loader, registry, tokenizer
from roop.diffusion.nn.flux import apply_rope, rope_freqs
from roop.diffusion.pipeline_flux import (
    MAX_SEQUENCE_TOKENS,
    PREFERRED_KONTEXT_RESOLUTIONS,
    latent_ids,
    pack,
    reference_size,
    unpack,
)
from roop.diffusion.sampler import FlowMatchScheduler, SDScheduler


# -- registry ---------------------------------------------------------------
def test_every_model_is_addressable_by_label_and_key():
    for spec in registry.MODELS:
        assert registry.get(spec.label) is spec
        assert registry.get(spec.key) is spec


def test_overrides_only_target_declared_params():
    for spec in registry.MODELS:
        assert set(spec.overrides).issubset(set(spec.params)), spec.key


def test_edit_only_models_do_not_offer_denoise_strength():
    # Kontext-style models take the reference image directly; a strength
    # slider would imply a control the pipeline does not honour.
    for spec in registry.MODELS:
        if spec.needs_image:
            assert registry.P_STRENGTH not in spec.params, spec.key


def test_flow_models_expose_a_single_scheduler():
    for spec in registry.MODELS:
        names = registry.samplers_for(spec)
        if spec.arch == registry.ARCH_FLUX_KONTEXT:
            assert names == ["Flow Match Euler"]
        else:
            assert "Euler" in names and "DDIM" in names


# -- download file selection ------------------------------------------------
_SDXL_LISTING = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/model.onnx",
]


def test_variant_mirror_is_preferred_when_published():
    picked = hub._pick_weights(_SDXL_LISTING, "unet", "fp16")
    assert picked == ["unet/diffusion_pytorch_model.fp16.safetensors"]


def test_falls_back_to_full_precision_when_no_variant_exists():
    picked = hub._pick_weights(_SDXL_LISTING, "unet", "")
    assert picked == ["unet/diffusion_pytorch_model.safetensors"]


def test_sharded_weights_are_all_selected():
    listing = [
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ]
    picked = hub._pick_weights(listing, "transformer", "")
    assert len(picked) == 3
    assert all(p.endswith(".safetensors") for p in picked)


def test_no_weights_selected_for_absent_component():
    assert hub._pick_weights(_SDXL_LISTING, "text_encoder_2", "") == []


# -- weight reconciliation --------------------------------------------------
def test_conv_weights_are_transposed_to_nhwc():
    src = mx.zeros((320, 4, 3, 3))                 # torch (out, in, kH, kW)
    out = loader._reconcile("conv_in.weight", src, (320, 3, 3, 4))
    assert out.shape == (320, 3, 3, 4)


def test_1x1_conv_is_squeezed_for_linear_modules():
    src = mx.zeros((640, 320, 1, 1))
    out = loader._reconcile("proj_in.weight", src, (640, 320))
    assert out.shape == (640, 320)


def test_ambiguous_conv_kernel_is_still_transposed():
    """Regression: a 3-channel 3x3 conv has the same shape either way round.

    The VAE encoder's conv_in is Conv2d(3 -> C, 3x3), so checkpoint layout
    (C,3,3,3) and MLX layout (C,3,3,3) are indistinguishable by shape. Taking
    the weight as-is silently transposes RGB against the kernel width axis,
    which survives as plausible structure with the colour destroyed -- so it
    passes every shape check and only shows up in the pixels.
    """
    conv = nn.Conv2d(3, 8, 3, padding=1)
    torch_layout = mx.random.normal((8, 3, 3, 3))
    assert torch_layout.shape == conv.weight.shape       # the trap

    loader.load_into(conv, {"weight": torch_layout, "bias": mx.zeros((8,))},
                     dtype=mx.float32)
    assert mx.allclose(conv.weight, torch_layout.transpose(0, 2, 3, 1)).item()


def test_conv_weight_names_finds_nested_kernels():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(3, 8, 3)
            self.proj = nn.Linear(8, 8)

    names = loader._conv_weight_names(Net())
    assert "conv_in.weight" in names
    assert "proj.weight" not in names


def test_matching_shapes_pass_through_untouched():
    src = mx.zeros((768, 320))
    assert loader._reconcile("x", src, (768, 320)).shape == (768, 320)


def test_irreconcilable_shape_is_an_error_not_a_reshape():
    with pytest.raises(loader.WeightMismatch):
        loader._reconcile("x", mx.zeros((8, 9)), (10, 11))


def test_legacy_vae_attention_names_are_translated():
    # RealisticVision 5.1 ships the pre-0.18 diffusers naming.
    state = {
        "decoder.mid_block.attentions.0.query.weight": mx.zeros((4, 4)),
        "decoder.mid_block.attentions.0.proj_attn.bias": mx.zeros((4,)),
    }
    out = loader._canonicalise(state)
    assert "decoder.mid_block.attentions.0.to_q.weight" in out
    assert "decoder.mid_block.attentions.0.to_out.0.bias" in out


# -- tokenizer --------------------------------------------------------------
def test_prompts_are_wrapped_in_sot_eot_and_padded():
    ids = tokenizer.tokenize(["a red car"])
    assert ids.shape == (1, tokenizer.CONTEXT_LENGTH)
    assert ids[0, 0].item() == tokenizer.SOT
    assert tokenizer.EOT in np.array(ids[0]).tolist()


def test_overlong_prompts_are_truncated_not_rejected():
    ids = tokenizer.tokenize(["word " * 500])
    assert ids.shape == (1, tokenizer.CONTEXT_LENGTH)
    assert ids[0, -1].item() == tokenizer.EOT


def test_pad_token_is_configurable_per_encoder():
    # SDXL's second encoder pads with 0 rather than EOT.
    ids = tokenizer.tokenize(["cat"], pad_id=0)
    assert ids[0, -1].item() == 0


# -- FLUX latent packing ----------------------------------------------------
def test_pack_unpack_round_trips():
    x = mx.array(np.random.randn(2, 8, 12, 16).astype(np.float32))
    assert mx.allclose(unpack(pack(x), 8, 12, 16), x).item()


def test_packing_matches_the_reference_channel_ordering():
    # Reference: torch NCHW view/permute used by diffusers' _pack_latents.
    t = np.random.randn(1, 16, 4, 6).astype(np.float32)
    expected = t.reshape(1, 16, 2, 2, 3, 2).transpose(0, 2, 4, 1, 3, 5)
    expected = expected.reshape(1, 6, 64)
    got = np.array(pack(mx.array(t.transpose(0, 2, 3, 1))))
    assert np.allclose(expected, got)


def test_reference_image_tokens_carry_a_distinct_rope_axis():
    canvas = latent_ids(4, 6, index=0)
    reference = latent_ids(4, 6, index=1)
    assert np.array(canvas[:, 0]).max() == 0
    assert np.array(reference[:, 0]).min() == 1


def test_reference_image_is_snapped_to_a_trained_bucket():
    # A phone photo must not be encoded at its native size: reference tokens
    # are unbounded that way and attention is quadratic in them.
    assert reference_size(4000, 3000) in PREFERRED_KONTEXT_RESOLUTIONS
    assert reference_size(100, 100) == (1024, 1024)


def test_reference_bucket_preserves_orientation():
    landscape = reference_size(4000, 3000)
    portrait = reference_size(3000, 4000)
    assert landscape[0] > landscape[1]
    assert portrait[0] < portrait[1]


def test_every_bucket_stays_within_the_token_budget():
    # vae_scale 8 then 2x2 packing => (w/16)*(h/16) tokens per image.
    for w, h in PREFERRED_KONTEXT_RESOLUTIONS:
        canvas = (1536 // 16) ** 2          # largest output the UI allows
        reference = (w // 16) * (h // 16)
        assert canvas + reference + 512 < MAX_SEQUENCE_TOKENS, (w, h)


def test_rope_rotation_preserves_vector_norm():
    ids = latent_ids(4, 6, 0)
    cos, sin = rope_freqs(ids, [16, 56, 56])
    q = mx.array(np.random.randn(1, 3, ids.shape[0], 128).astype(np.float32))
    rotated = apply_rope(q, cos, sin)
    assert mx.allclose(mx.sum(q * q, -1), mx.sum(rotated * rotated, -1),
                       atol=1e-3).item()


# -- output size follows the uploaded image ---------------------------------
def _fit(model_key, w, h):
    from ui.tabs.diffusion_tab import fit_to_model
    return fit_to_model(registry.BY_KEY[model_key], w, h)


def test_output_size_keeps_the_uploaded_aspect_ratio():
    # A portrait photo must not come back square: edit models render a fresh
    # canvas at the requested size, so a square default resamples the subject.
    w, h = _fit("flux_kontext", 1536, 1920)
    assert h > w
    assert abs((w / h) - (1536 / 1920)) < 0.1


def test_output_size_targets_the_model_native_scale_not_the_upload():
    # A 12 MP phone photo must not drive SD 1.5 far off its training
    # resolution, nor SDXL far below its own.
    sd_w, sd_h = _fit("dreamshaper", 4000, 3000)
    xl_w, xl_h = _fit("sdxl", 4000, 3000)
    assert (sd_w * sd_h) ** 0.5 < (xl_w * xl_h) ** 0.5
    assert 384 <= sd_w <= 832
    assert 768 <= xl_w <= 1536


def test_output_size_snaps_to_the_slider_step_and_bounds():
    for key in ("dreamshaper", "sdxl", "flux_kontext"):
        spec = registry.BY_KEY[key]
        lo = spec.overrides.get(registry.P_WIDTH, {}).get("minimum", 256)
        hi = spec.overrides.get(registry.P_WIDTH, {}).get("maximum", 1536)
        for dims in [(4000, 3000), (100, 100), (1920, 1080), (640, 1600)]:
            w, h = _fit(key, *dims)
            assert w % 64 == 0 and h % 64 == 0, (key, dims, w, h)
            assert lo <= w <= hi and lo <= h <= hi, (key, dims, w, h)


def test_square_upload_gives_square_output():
    w, h = _fit("flux_kontext", 1024, 1024)
    assert w == h


def test_degenerate_size_is_ignored_rather_than_dividing_by_zero():
    assert _fit("sdxl", 0, 0) == (None, None)


# -- schedulers -------------------------------------------------------------
def _sd_config():
    return {"num_train_timesteps": 1000, "beta_start": 0.00085,
            "beta_end": 0.012, "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading", "steps_offset": 1}


@pytest.mark.parametrize("sampler", ["Euler", "Euler Ancestral", "DDIM"])
def test_sd_schedule_runs_high_noise_to_low(sampler):
    s = SDScheduler(_sd_config(), sampler)
    s.set_timesteps(20)
    assert len(s.timesteps) == 20
    assert s.timesteps[0] > s.timesteps[-1]
    assert s.sigmas[-1] == 0.0


def test_img2img_start_index_tracks_strength():
    s = SDScheduler(_sd_config(), "Euler")
    s.set_timesteps(20)
    assert s.start_index(20, 1.0) == 0        # full strength = full denoise
    assert s.start_index(20, 0.5) == 10
    assert s.start_index(20, 0.0) == 20       # no steps at all


def test_flow_schedule_descends_from_one_to_zero():
    s = FlowMatchScheduler({"use_dynamic_shifting": True})
    s.set_timesteps(28, seq_len=1024)
    assert len(s.sigmas) == 29
    assert s.sigmas[0] == pytest.approx(1.0, abs=1e-6)
    assert s.sigmas[-1] == 0.0
    assert all(np.diff(s.sigmas) < 0)


def test_flow_step_is_plain_euler_integration():
    s = FlowMatchScheduler({"use_dynamic_shifting": False, "shift": 3.0})
    s.set_timesteps(4)
    x = mx.ones((1, 4, 64))
    v = mx.ones((1, 4, 64))
    out = s.step(x, v, 0)
    expected = 1.0 + (s.sigmas[1] - s.sigmas[0])
    assert np.allclose(np.array(out), expected, atol=1e-5)
