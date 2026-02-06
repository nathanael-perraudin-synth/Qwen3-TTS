# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Speech tokenizer for Qwen3-TTS standalone.

This module provides the speech tokenizer (audio codec) for encoding
and decoding audio waveforms to/from discrete tokens.
"""

from .speech_tokenizer import Qwen3TTSSpeechTokenizer
from .config import (
    Qwen3TTSTokenizerV2ConfigStandalone as Qwen3TTSSpeechTokenizerConfig,
)
from .model import (
    Qwen3TTSTokenizerV2ModelStandalone as Qwen3TTSSpeechTokenizerModel,
)

__all__ = [
    "Qwen3TTSSpeechTokenizer",
    "Qwen3TTSSpeechTokenizerConfig",
    "Qwen3TTSSpeechTokenizerModel",
]
