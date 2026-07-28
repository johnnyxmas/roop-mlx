"""On-demand weight fetching for the diffusion tab.

Deliberately separate from roop.model_manager: the face-swap models are
downloaded eagerly at startup, diffusion checkpoints are 2-32 GB each and are
only fetched the first time the user actually generates with that model.

Files land in models/diffusion/ using the standard HuggingFace cache layout,
so a repo already present in ~/.cache/huggingface is *not* re-downloaded --
HF_HOME/HUGGINGFACE_HUB_CACHE still wins if the user has set it.
"""

import glob
import os
from typing import Callable, Dict, List, Optional

from roop.utilities import resolve_relative_path

from roop.diffusion.registry import (
    ARCH_FLUX_KONTEXT,
    ARCH_SD15,
    ARCH_SDXL,
    ModelSpec,
)


ProgressFn = Optional[Callable[[float, str], None]]


class ModelAccessError(RuntimeError):
    """Repo exists but we are not allowed to read it (gated / private)."""


def cache_root() -> str:
    """Where diffusion checkpoints are stored, unless HF_HOME overrides it."""
    env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return env
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        # HF_HOME is the parent of the hub cache, not the cache itself.
        return os.path.join(hf_home, "hub")
    root = resolve_relative_path("../models/diffusion")
    os.makedirs(root, exist_ok=True)
    return root


# Config/tokenizer files each architecture needs alongside the weights.
_EXTRA_FILES = {
    ARCH_SD15: ["model_index.json", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "unet/config.json",
                "vae/config.json"],
    ARCH_SDXL: ["model_index.json", "scheduler/scheduler_config.json",
                "text_encoder/config.json", "text_encoder_2/config.json",
                "unet/config.json", "vae/config.json"],
    ARCH_FLUX_KONTEXT: ["model_index.json", "scheduler/scheduler_config.json",
                        "text_encoder/config.json",
                        "text_encoder_2/config.json",
                        "transformer/config.json", "vae/config.json",
                        "tokenizer_2/tokenizer.json",
                        "tokenizer_2/tokenizer_config.json",
                        "tokenizer_2/special_tokens_map.json"],
}

# Component subdirectories holding weights, per architecture.
_WEIGHT_DIRS = {
    ARCH_SD15: ["text_encoder", "unet", "vae"],
    ARCH_SDXL: ["text_encoder", "text_encoder_2", "unet", "vae"],
    ARCH_FLUX_KONTEXT: ["text_encoder", "text_encoder_2", "transformer", "vae"],
}


def _list_repo_files(spec: ModelSpec) -> List[str]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
    )

    try:
        info = HfApi().model_info(spec.repo, files_metadata=False)
    except GatedRepoError as exc:
        raise ModelAccessError(
            f"{spec.repo} is gated. Open https://huggingface.co/{spec.repo}, "
            "accept the licence, then run `hf auth login` (or set HF_TOKEN) "
            "and try again."
        ) from exc
    except RepositoryNotFoundError as exc:
        # HF returns 401 for private *and* nonexistent repos, so we cannot
        # distinguish them without a token.
        raise ModelAccessError(
            f"{spec.repo} could not be read -- it is private, gated or does "
            "not exist. If you have been granted access, set HF_TOKEN (or run "
            "`hf auth login`) and try again."
        ) from exc
    return [s.rfilename for s in info.siblings or []]


def _pick_weights(files: List[str], subdir: str, variant: str) -> List[str]:
    """Choose the weight file(s) for one component directory.

    Covers the layouts diffusers repos use: a single .safetensors, a
    variant-suffixed mirror (model.fp16.safetensors), and sharded weights
    (model-00001-of-00003.safetensors).  Shard index JSONs are not fetched --
    the loader simply reads every shard present in the directory.
    """
    prefix = f"{subdir}/"
    weights = [f for f in files
               if f.startswith(prefix) and f.endswith(".safetensors")]
    if not weights:
        return []

    if variant:
        suffix = f".{variant}.safetensors"
        preferred = [f for f in weights
                     if os.path.basename(f).endswith(suffix)]
        if preferred:
            return preferred

    # No variant requested or none published: keep the unsuffixed files,
    # which is either one file or the full shard set.
    plain = [f for f in weights if os.path.basename(f).count(".") == 1]
    return plain or weights[:1]


def required_files(spec: ModelSpec, files: List[str]) -> List[str]:
    wanted: List[str] = []
    for extra in _EXTRA_FILES.get(spec.arch, []):
        if extra in files:
            wanted.append(extra)
    for subdir in _WEIGHT_DIRS.get(spec.arch, []):
        wanted.extend(_pick_weights(files, subdir, spec.variant))
    # Kontext ships the autoencoder both as vae/ and as a bare ae.safetensors;
    # vae/ is picked up above, so nothing extra is needed here.
    return wanted


def local_snapshot(spec: ModelSpec) -> Optional[str]:
    """Path to a complete local snapshot of *spec*, or None. Never hits the network.

    The UI calls this on every model switch, so it must be cheap and must
    not raise for gated repos -- being unable to *list* a repo says nothing
    about whether it is already on disk.
    """
    repo_dir = os.path.join(
        cache_root(), "models--" + spec.repo.replace("/", "--"), "snapshots"
    )
    if not os.path.isdir(repo_dir):
        return None

    needed = _WEIGHT_DIRS.get(spec.arch, [])
    try:
        snaps = [os.path.join(repo_dir, d) for d in os.listdir(repo_dir)]
    except OSError:
        return None

    for snap in sorted(snaps, key=os.path.getmtime, reverse=True):
        if not os.path.isdir(snap):
            continue
        if all(
            os.path.isfile(os.path.join(snap, sub, "config.json"))
            and glob.glob(os.path.join(snap, sub, "*.safetensors"))
            for sub in needed
        ):
            return snap
    return None


def is_cached(spec: ModelSpec) -> bool:
    """True when a complete local snapshot exists (no network hit)."""
    return local_snapshot(spec) is not None


_FILE_LIST_CACHE: Dict[str, List[str]] = {}


def _cached_file_list(spec: ModelSpec) -> List[str]:
    """Repo file listing, memoised, falling back to the local snapshot."""
    if spec.repo in _FILE_LIST_CACHE:
        return _FILE_LIST_CACHE[spec.repo]
    listing = required_files(spec, _list_repo_files(spec))
    _FILE_LIST_CACHE[spec.repo] = listing
    return listing


def fetch(spec: ModelSpec, progress: ProgressFn = None) -> str:
    """Download whatever is missing and return the local snapshot directory.

    Raises ModelAccessError with actionable text when the repo is gated.
    """
    from huggingface_hub import hf_hub_download

    def report(frac: float, msg: str):
        if progress is not None:
            progress(frac, msg)

    # Already complete on disk: skip the network entirely so repeat runs
    # (and offline use) do not depend on the Hub being reachable.
    cached = local_snapshot(spec)
    if cached is not None:
        report(1.0, "Weights already downloaded")
        return cached

    report(0.0, f"Resolving {spec.repo}…")
    wanted = _cached_file_list(spec)
    if not wanted:
        raise ModelAccessError(
            f"{spec.repo} does not expose the safetensors layout this tab "
            "needs (diffusers-style component folders)."
        )

    local_paths: List[str] = []
    total = len(wanted)
    for i, fname in enumerate(wanted):
        report(i / total, f"Fetching {fname} ({i + 1}/{total})")
        local_paths.append(
            hf_hub_download(spec.repo, fname, cache_dir=cache_root())
        )
    report(1.0, "Weights ready")

    # Every file lives under <snapshot>/..., so strip back to the snapshot root
    # using the deepest known relative path.
    root = local_paths[0]
    for _ in range(wanted[0].count("/") + 1):
        root = os.path.dirname(root)
    return root
