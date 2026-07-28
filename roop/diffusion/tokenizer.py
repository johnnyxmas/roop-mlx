"""Prompt tokenisation for the CLIP text encoders.

Reuses the BPE tokenizer already vendored for CLIPSeg -- it is the same
49408-token CLIP vocabulary SD 1.5 and SDXL were trained with, so no
tokenizer files need downloading alongside the weights.
"""

from typing import List, Optional

import mlx.core as mx

from clip.simple_tokenizer import SimpleTokenizer

SOT = 49406
EOT = 49407
CONTEXT_LENGTH = 77

_TOKENIZER: Optional[SimpleTokenizer] = None


def _tokenizer() -> SimpleTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = SimpleTokenizer()
    return _TOKENIZER


def tokenize(prompts: List[str], pad_id: int = EOT,
             context_length: int = CONTEXT_LENGTH) -> mx.array:
    """Encode prompts to a (batch, 77) int32 array, truncating overlong text.

    *pad_id* differs between SDXL's two encoders: CLIP-L pads with the EOT
    token, OpenCLIP bigG pads with 0.
    """
    tok = _tokenizer()
    rows = []
    for text in prompts:
        ids = [SOT] + tok.encode(text or "")[: context_length - 2] + [EOT]
        ids = ids + [pad_id] * (context_length - len(ids))
        rows.append(ids)
    return mx.array(rows, dtype=mx.int32)
