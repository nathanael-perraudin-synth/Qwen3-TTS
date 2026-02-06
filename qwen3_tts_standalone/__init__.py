# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Standalone Implementation

This package provides a fully standalone implementation of Qwen3-TTS that
minimizes dependencies on the transformers library.

Main components:
- TTS: The main text-to-speech model
- Talker: Generates audio codec tokens from text embeddings
- CodePredictor: Predicts higher codebook layers
- Qwen3TTSSpeechTokenizer: Encodes/decodes audio to/from discrete tokens

Usage:
    from qwen3_tts_standalone import Qwen3TTSModel
    
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-0.5B")
    audio, sample_rate = model.generate("Hello world!", speaker="Chelsie")
"""

# Core models
from .tts import TTS
from .talker import Talker
from .code_predictor import CodePredictor
from .speaker_encoder import Qwen3TTSSpeakerEncoderStandalone

# Configuration
from .configuration import (
    Qwen3TTSConfigStandalone,
    Qwen3TTSTalkerConfigStandalone,
    Qwen3TTSTalkerCodePredictorConfigStandalone,
    Qwen3TTSSpeakerEncoderConfigStandalone,
    BaseConfig,
)

# Tokenizer
from .tokenizer import (
    Qwen3TTSSpeechTokenizer,
    Qwen3TTSSpeechTokenizerConfig,
    Qwen3TTSSpeechTokenizerModel,
)

# Processor
from .processor import Qwen3TTSProcessor

# High-level inference API
from .inference import Qwen3TTSModelStandalone as Qwen3TTSModel

# Base model
from .base_model import StandalonePreTrainedModel

__all__ = [
    # Models
    "TTS",
    "Talker",
    "CodePredictor",
    "Qwen3TTSSpeakerEncoderStandalone",
    # Configuration
    "Qwen3TTSConfigStandalone",
    "Qwen3TTSTalkerConfigStandalone",
    "Qwen3TTSTalkerCodePredictorConfigStandalone",
    "Qwen3TTSSpeakerEncoderConfigStandalone",
    "BaseConfig",
    # Tokenizer
    "Qwen3TTSSpeechTokenizer",
    "Qwen3TTSSpeechTokenizerConfig",
    "Qwen3TTSSpeechTokenizerModel",
    # Processor
    "Qwen3TTSProcessor",
    # Inference
    "Qwen3TTSModel",
    # Base
    "StandalonePreTrainedModel",
]

__version__ = "0.1.0"
