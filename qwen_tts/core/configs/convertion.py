"""Convert original (transformers) configs to standalone configs."""
from typing import Optional, Tuple

from .configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from .standalone_config import (
    CodePredictorConfig,
    SpeakerEncoderConfig,
    TalkerConfig,
)


def to_speaker_encoder_config(
    config: Qwen3TTSSpeakerEncoderConfig,
) -> SpeakerEncoderConfig:
    """Convert Qwen3TTSSpeakerEncoderConfig to StandaloneSpeakerEncoderConfig."""
    return SpeakerEncoderConfig(
        mel_dim=config.mel_dim,
        enc_dim=config.enc_dim,
        enc_channels=config.enc_channels,
        enc_kernel_sizes=config.enc_kernel_sizes,
        enc_dilations=config.enc_dilations,
        enc_attention_channels=config.enc_attention_channels,
        enc_res2net_scale=config.enc_res2net_scale,
        enc_se_channels=config.enc_se_channels,
        sample_rate=config.sample_rate,
    )


def to_code_predictor_config(
    config: Qwen3TTSTalkerCodePredictorConfig,
) -> CodePredictorConfig:
    """Convert Qwen3TTSTalkerCodePredictorConfig to StandaloneCodePredictorConfig."""
    rope_scaling = config.rope_scaling
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if "type" in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type", "default")
    return CodePredictorConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=config.initializer_range,
        rms_norm_eps=config.rms_norm_eps,
        use_cache=config.use_cache,
        rope_theta=config.rope_theta,
        rope_scaling=rope_scaling,
        attention_bias=config.attention_bias,
        use_sliding_window=config.use_sliding_window,
        sliding_window=config.sliding_window,
        max_window_layers=config.max_window_layers,
        layer_types=config.layer_types,
        attention_dropout=config.attention_dropout,
        num_code_groups=config.num_code_groups,
        pad_token_id=getattr(config, "pad_token_id", 0),
    )


def to_standalone_talker_config(
    config: Qwen3TTSTalkerConfig,
) -> TalkerConfig:
    """Convert Qwen3TTSTalkerConfig to StandaloneTalkerConfig."""
    rope_scaling = config.rope_scaling
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if "type" in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type", "default")
    out = TalkerConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None),
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=config.initializer_range,
        rms_norm_eps=config.rms_norm_eps,
        use_cache=config.use_cache,
        rope_theta=config.rope_theta,
        rope_scaling=rope_scaling,
        attention_bias=config.attention_bias,
        use_sliding_window=config.use_sliding_window,
        sliding_window=config.sliding_window,
        attention_dropout=config.attention_dropout,
        num_code_groups=config.num_code_groups,
        text_hidden_size=config.text_hidden_size,
        text_vocab_size=getattr(config, "text_vocab_size", 151936),
        pad_token_id=getattr(config, "pad_token_id", None),
        codec_eos_token_id=config.codec_eos_token_id,
        codec_think_id=config.codec_think_id,
        codec_nothink_id=config.codec_nothink_id,
        codec_think_bos_id=config.codec_think_bos_id,
        codec_think_eos_id=config.codec_think_eos_id,
        codec_pad_id=config.codec_pad_id,
        codec_bos_id=config.codec_bos_id,
        spk_id=config.spk_id if config.spk_id is not None else {"default": 0},
        spk_is_dialect=config.spk_is_dialect if config.spk_is_dialect is not None else {"default": False},
        codec_language_id=config.codec_language_id if config.codec_language_id is not None else {"en": 0},
    )
    # Set code predictor fields from sub-config
    cp = config.code_predictor_config
    out.code_predictor_vocab_size = cp.vocab_size
    out.code_predictor_hidden_size = cp.hidden_size
    out.code_predictor_intermediate_size = cp.intermediate_size
    out.code_predictor_num_layers = cp.num_hidden_layers
    out.code_predictor_num_heads = cp.num_attention_heads
    out.code_predictor_num_kv_heads = cp.num_key_value_heads
    return out


def to_standalone_configs_from_tts(
    config: Qwen3TTSConfig,
) -> Tuple[TalkerConfig, Optional[SpeakerEncoderConfig]]:
    """
    Convert Qwen3TTSConfig to standalone talker and speaker encoder configs.

    Returns:
        (talker_config, speaker_encoder_config). speaker_encoder_config may be None
        if the original config has no speaker encoder.
    """
    talker_config = to_standalone_talker_config(config.talker_config)
    speaker_encoder_config = None
    if config.speaker_encoder_config is not None:
        speaker_encoder_config = to_speaker_encoder_config(config.speaker_encoder_config)
    return talker_config, speaker_encoder_config


__all__ = [
    "to_speaker_encoder_config",
    "to_code_predictor_config",
    "to_standalone_talker_config",
    "to_standalone_configs_from_tts",
]
