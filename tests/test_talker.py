"""Test for Qwen3TTSTalkerModel."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
from qwen_tts.core.configs import Qwen3TTSTalkerConfig


@pytest.fixture
def talker_config():
    """Create a talker config for testing."""
    return Qwen3TTSTalkerConfig(
        vocab_size=3072,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [11, 11, 10],  # head_dim=64, so mrope_section should sum to 64/2=32 (since it gets doubled)
            "interleaved": False,
        },
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        attention_dropout=0.0,
        num_code_groups=4,
        text_hidden_size=256,
        text_vocab_size=1000,
        pad_token_id=0,
    )


@pytest.fixture
def talker_model(talker_config, device):
    """Create a talker model for testing."""
    model = Qwen3TTSTalkerModel(talker_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_input_ids(device):
    """Create sample input ids."""
    batch_size = 2
    seq_len = 10
    return torch.randint(0, 3072, (batch_size, seq_len), device=device)


@pytest.fixture
def sample_position_ids(device):
    """Create sample position ids (3D for multimodal)."""
    batch_size = 2
    seq_len = 10
    # Shape: (3, batch, seq) for temporal, height, width
    return torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)


def test_talker_forward(talker_model, sample_input_ids, sample_position_ids):
    """Test forward pass of talker model."""
    # Use inputs_embeds instead of input_ids to avoid embed_tokens issue
    inputs_embeds = talker_model.codec_embedding(sample_input_ids)
    with torch.no_grad():
        output = talker_model(
            inputs_embeds=inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check output shape
    assert output.last_hidden_state.shape == (sample_input_ids.shape[0], sample_input_ids.shape[1], talker_model.config.hidden_size)
    assert output.last_hidden_state.dtype == inputs_embeds.dtype


def test_talker_with_cache(talker_model, sample_input_ids, sample_position_ids):
    """Test talker model with KV cache."""
    # Use inputs_embeds instead of input_ids to avoid embed_tokens issue
    inputs_embeds = talker_model.codec_embedding(sample_input_ids)
    with torch.no_grad():
        output = talker_model(
            inputs_embeds=inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
    
    assert output.past_key_values is not None
    assert output.last_hidden_state.shape[0] == sample_input_ids.shape[0]


def test_talker_output_range(talker_model, sample_input_ids, sample_position_ids):
    """Test that output values are reasonable."""
    # Use inputs_embeds instead of input_ids to avoid embed_tokens issue
    inputs_embeds = talker_model.codec_embedding(sample_input_ids)
    with torch.no_grad():
        output = talker_model(
            inputs_embeds=inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check that output is not NaN or Inf
    assert not torch.isnan(output.last_hidden_state).any()
    assert not torch.isinf(output.last_hidden_state).any()
