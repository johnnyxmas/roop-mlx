"""Noise schedules and samplers.

Two families: the variance-preserving schedule SD 1.5 / SDXL are trained on
(Euler, Euler ancestral, DDIM), and the rectified-flow schedule FLUX uses.
Both expose the same three-call surface -- `set_timesteps`, `scale_input`,
`step` -- so the pipelines do not branch on sampler type.
"""

import math
from typing import List, Optional

import mlx.core as mx
import numpy as np


class SDScheduler:
    """Discrete VP schedule shared by SD 1.5 and SDXL."""

    def __init__(self, cfg: dict, sampler: str = "Euler"):
        self.sampler = sampler
        n = cfg.get("num_train_timesteps", 1000)
        beta_start = cfg.get("beta_start", 0.00085)
        beta_end = cfg.get("beta_end", 0.012)
        schedule = cfg.get("beta_schedule", "scaled_linear")
        self.prediction_type = cfg.get("prediction_type", "epsilon")
        self.timestep_spacing = cfg.get("timestep_spacing", "leading")
        self.steps_offset = cfg.get("steps_offset", 1)
        self.num_train_timesteps = n

        if schedule == "scaled_linear":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n,
                                dtype=np.float64) ** 2
        elif schedule == "linear":
            betas = np.linspace(beta_start, beta_end, n, dtype=np.float64)
        else:
            raise ValueError(f"unsupported beta_schedule {schedule!r}")

        self.alphas_cumprod = np.cumprod(1.0 - betas)
        self.sigmas_train = np.sqrt(
            (1.0 - self.alphas_cumprod) / self.alphas_cumprod
        )

        self.timesteps: np.ndarray = np.array([])
        self.sigmas: np.ndarray = np.array([])

    # -- schedule ---------------------------------------------------------
    def set_timesteps(self, steps: int) -> None:
        n = self.num_train_timesteps
        if self.timestep_spacing == "linspace":
            ts = np.linspace(0, n - 1, steps, dtype=np.float64)[::-1].copy()
        elif self.timestep_spacing == "trailing":
            step = n / steps
            ts = (np.arange(n, 0, -step) - 1).round().astype(np.float64)
        else:  # "leading"
            step = n // steps
            ts = (np.arange(0, steps) * step).astype(np.float64)[::-1].copy()
            ts += self.steps_offset

        sigmas = np.interp(ts, np.arange(n), self.sigmas_train)
        self.sigmas = np.concatenate([sigmas, [0.0]])
        self.timesteps = ts

    def start_index(self, steps: int, strength: float) -> int:
        """First index to denoise from for img2img at *strength*."""
        return max(steps - int(round(steps * strength)), 0)

    @property
    def init_sigma(self) -> float:
        """Scale applied to pure noise for a full txt2img run."""
        if self.sampler.startswith("Euler"):
            return float(self.sigmas[0])
        return 1.0

    # -- model I/O --------------------------------------------------------
    def scale_input(self, x: mx.array, i: int) -> mx.array:
        if self.sampler.startswith("Euler"):
            sigma = float(self.sigmas[i])
            return x / ((sigma ** 2 + 1) ** 0.5)
        return x

    def add_noise(self, latent: mx.array, noise: mx.array, i: int) -> mx.array:
        """Noise a clean latent to the level of step *i* (img2img entry)."""
        if self.sampler.startswith("Euler"):
            return latent + noise * float(self.sigmas[i])
        t = int(self.timesteps[i])
        ac = float(self.alphas_cumprod[t])
        return latent * (ac ** 0.5) + noise * ((1 - ac) ** 0.5)

    def _to_x0(self, x: mx.array, model_out: mx.array, i: int) -> mx.array:
        if self.sampler.startswith("Euler"):
            sigma = float(self.sigmas[i])
            if self.prediction_type == "v_prediction":
                return (model_out * (-sigma / (sigma ** 2 + 1) ** 0.5)
                        + x / (sigma ** 2 + 1))
            return x - sigma * model_out
        t = int(self.timesteps[i])
        ac = float(self.alphas_cumprod[t])
        if self.prediction_type == "v_prediction":
            return (ac ** 0.5) * x - ((1 - ac) ** 0.5) * model_out
        return (x - ((1 - ac) ** 0.5) * model_out) / (ac ** 0.5)

    # -- one denoising step ----------------------------------------------
    def step(self, x: mx.array, model_out: mx.array, i: int,
             key: Optional[mx.array] = None) -> mx.array:
        x0 = self._to_x0(x, model_out, i)

        if self.sampler == "Euler":
            sigma, sigma_next = float(self.sigmas[i]), float(self.sigmas[i + 1])
            derivative = (x - x0) / sigma
            return x + derivative * (sigma_next - sigma)

        if self.sampler == "Euler Ancestral":
            sigma, sigma_next = float(self.sigmas[i]), float(self.sigmas[i + 1])
            var = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2)
                   / max(sigma ** 2, 1e-12))
            sigma_up = min(sigma_next, math.sqrt(max(var, 0.0)))
            sigma_down = math.sqrt(max(sigma_next ** 2 - sigma_up ** 2, 0.0))
            derivative = (x - x0) / sigma
            x = x + derivative * (sigma_down - sigma)
            if sigma_up > 0 and key is not None:
                x = x + mx.random.normal(x.shape, key=key).astype(x.dtype) * sigma_up
            return x

        # DDIM (eta = 0)
        t = int(self.timesteps[i])
        prev_t = int(self.timesteps[i + 1]) if i + 1 < len(self.timesteps) else -1
        ac = float(self.alphas_cumprod[t])
        ac_prev = float(self.alphas_cumprod[prev_t]) if prev_t >= 0 else 1.0
        if self.prediction_type == "v_prediction":
            eps = (ac ** 0.5) * model_out + ((1 - ac) ** 0.5) * x
        else:
            eps = model_out
        return (ac_prev ** 0.5) * x0 + ((1 - ac_prev) ** 0.5) * eps


class FlowMatchScheduler:
    """Rectified-flow Euler schedule (FLUX and friends).

    `sigma` runs 1 -> 0; the transformer predicts the velocity field, so a
    step is a plain Euler integration with no rescaling of the input.
    """

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.num_train_timesteps = cfg.get("num_train_timesteps", 1000)
        self.base_shift = cfg.get("base_shift", 0.5)
        self.max_shift = cfg.get("max_shift", 1.15)
        self.base_seq_len = cfg.get("base_image_seq_len", 256)
        self.max_seq_len = cfg.get("max_image_seq_len", 4096)
        self.use_dynamic_shifting = cfg.get("use_dynamic_shifting", True)
        self.static_shift = cfg.get("shift", 3.0)
        self.sigmas: np.ndarray = np.array([])
        self.timesteps: np.ndarray = np.array([])

    def _mu(self, seq_len: int) -> float:
        m = (self.max_shift - self.base_shift) / (self.max_seq_len - self.base_seq_len)
        b = self.base_shift - m * self.base_seq_len
        return m * seq_len + b

    def set_timesteps(self, steps: int, seq_len: int = 1024) -> None:
        sigmas = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float64)
        if self.use_dynamic_shifting:
            mu = self._mu(seq_len)
            sigmas = np.exp(mu) / (np.exp(mu) + (1 / sigmas - 1))
        else:
            s = self.static_shift
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = np.concatenate([sigmas, [0.0]])

    def start_index(self, steps: int, strength: float) -> int:
        return max(steps - int(round(steps * strength)), 0)

    @property
    def init_sigma(self) -> float:
        return 1.0

    def scale_input(self, x: mx.array, i: int) -> mx.array:
        return x

    def add_noise(self, latent: mx.array, noise: mx.array, i: int) -> mx.array:
        sigma = float(self.sigmas[i])
        return (1.0 - sigma) * latent + sigma * noise

    def step(self, x: mx.array, model_out: mx.array, i: int,
             key: Optional[mx.array] = None) -> mx.array:
        dt = float(self.sigmas[i + 1]) - float(self.sigmas[i])
        return x + dt * model_out


def make_scheduler(cfg: dict, sampler: str):
    if sampler == "Flow Match Euler":
        return FlowMatchScheduler(cfg)
    return SDScheduler(cfg, sampler)


def sampler_names() -> List[str]:
    return ["Euler", "Euler Ancestral", "DDIM", "Flow Match Euler"]
