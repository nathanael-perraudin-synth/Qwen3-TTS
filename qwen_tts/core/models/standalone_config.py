"""Standalone configuration classes without transformers dependency."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class StandaloneSpeakerEncoderConfig:
    """Configuration for standalone speaker encoder."""
    mel_dim: int = 128
    enc_dim: int = 1024
    enc_channels: List[int] = None
    enc_kernel_sizes: List[int] = None
    enc_dilations: List[int] = None
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128
    sample_rate: int = 24000

    def __post_init__(self):
        if self.enc_channels is None:
            self.enc_channels = [512, 512, 512, 512, 1536]
        if self.enc_kernel_sizes is None:
            self.enc_kernel_sizes = [5, 3, 3, 3, 1]
        if self.enc_dilations is None:
            self.enc_dilations = [1, 2, 3, 4, 1]


@dataclass
class StandaloneCodePredictorConfig:
    """Configuration for standalone code predictor model."""
    vocab_size: int = 2048
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 5
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 0.000001
    use_cache: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: Optional[int] = 4096
    max_window_layers: int = 28
    layer_types: Optional[List[str]] = None
    attention_dropout: float = 0.0
    num_code_groups: int = 32
    pad_token_id: Optional[int] = None
    output_attentions: bool = False
    output_hidden_states: bool = False
    _attn_implementation: str = "eager"  # Use eager attention (no flash attention)

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {"rope_type": "default"}
        elif "rope_type" not in self.rope_scaling:
            self.rope_scaling["rope_type"] = "default"
        
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.use_sliding_window and self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        
        if self.pad_token_id is None:
            self.pad_token_id = 0


@dataclass
class StandaloneTalkerConfig:
    """Configuration for standalone talker model."""
    vocab_size: int = 3072
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 0.000001
    use_cache: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: Optional[int] = 4096
    attention_dropout: float = 0.0
    num_code_groups: int = 32
    text_hidden_size: int = 2048
    text_vocab_size: int = 151936
    pad_token_id: Optional[int] = None
    codec_eos_token_id: int = 4198
    codec_think_id: int = 4202
    codec_nothink_id: int = 4203
    codec_think_bos_id: int = 4204
    codec_think_eos_id: int = 4205
    codec_pad_id: int = 4196
    codec_bos_id: int = 4197
    spk_id: Dict[str, int] = field(default_factory=lambda: {"default": 0})
    spk_is_dialect: Dict[str, bool] = field(default_factory=lambda: {"default": False})
    codec_language_id: Dict[str, int] = field(default_factory=lambda: {"en": 0})
    output_attentions: bool = False
    output_hidden_states: bool = False
    _attn_implementation: str = "eager"  # Use eager attention (no flash attention)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        if self.rope_scaling is None:
            self.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [self.head_dim // 3] * 3,  # Split into 3 for temporal, height, width
                "interleaved": False,
            }
        elif "rope_type" not in self.rope_scaling:
            self.rope_scaling["rope_type"] = "default"
        if "mrope_section" not in self.rope_scaling:
            # mrope_section should sum to head_dim/2 (not doubled)
            section_size = (self.head_dim // 2) // 3
            remainder = (self.head_dim // 2) % 3
            self.rope_scaling["mrope_section"] = [section_size] * 2 + [section_size + remainder]
        if "interleaved" not in self.rope_scaling:
            self.rope_scaling["interleaved"] = False
        
        if self.pad_token_id is None:
            self.pad_token_id = 0
