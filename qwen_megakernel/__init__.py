"""Qwen Megakernel — single-kernel Qwen3-0.6B decode for RTX 5090."""

from qwen_megakernel.build import get_extension as _get_ext

_get_ext()

from qwen_megakernel.model import (  # noqa: E402
    load_weights,
    load_talker_weights,
    Decoder,
    TalkerDecoder,
    generate,
)

__all__ = ["load_weights", "load_talker_weights", "Decoder", "TalkerDecoder", "generate"]
