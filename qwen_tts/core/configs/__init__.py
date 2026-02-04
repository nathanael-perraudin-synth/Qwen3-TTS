"""Qwen3 TTS configuration classes and conversion utilities."""
from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .convertion import (
    to_code_predictor_config,
    to_standalone_configs_from_tts,
    to_speaker_encoder_config,
    to_standalone_talker_config,
)
from .standalone_config import (
    CodePredictorConfig,
    SpeakerEncoderConfig,
    TalkerConfig,
)

__all__ = [
    # Original configs
    "Qwen3TTSConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTalkerConfig",
    # Standalone configs
    "CodePredictorConfig",
    "SpeakerEncoderConfig",
    "TalkerConfig",
    # Conversion
    "to_speaker_encoder_config",
    "to_code_predictor_config",
    "to_standalone_talker_config",
    "to_standalone_configs_from_tts",
]
