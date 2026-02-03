"""Test for Qwen3TTSTalkerCodePredictorModel."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModel
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig


@pytest.fixture
def code_predictor_config():
    """Create a code predictor config for testing."""
    return Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=1024,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=0,
        layer_types=None,
        attention_dropout=0.0,
        num_code_groups=4,
        pad_token_id=0,
    )


@pytest.fixture
def code_predictor_model(code_predictor_config, device):
    """Create a code predictor model for testing."""
    embedding_dim = 256
    model = Qwen3TTSTalkerCodePredictorModel(code_predictor_config, embedding_dim)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_inputs_embeds(device):
    """Create sample input embeddings."""
    # Batch size 2, sequence length 10, embedding dim 256
    batch_size = 2
    seq_len = 10
    embedding_dim = 256
    return torch.randn(batch_size, seq_len, embedding_dim, device=device)


@pytest.fixture
def sample_position_ids(device):
    """Create sample position ids."""
    batch_size = 2
    seq_len = 10
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


def test_code_predictor_forward(code_predictor_model, sample_inputs_embeds, sample_position_ids):
    """Test forward pass of code predictor."""
    with torch.no_grad():
        output = code_predictor_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check output shape
    assert output.last_hidden_state.shape == (sample_inputs_embeds.shape[0], sample_inputs_embeds.shape[1], code_predictor_model.config.hidden_size)
    assert output.last_hidden_state.dtype == sample_inputs_embeds.dtype


def test_code_predictor_with_cache(code_predictor_model, sample_inputs_embeds, sample_position_ids):
    """Test code predictor with KV cache."""
    with torch.no_grad():
        output = code_predictor_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
    
    assert output.past_key_values is not None
    assert output.last_hidden_state.shape[0] == sample_inputs_embeds.shape[0]


def test_code_predictor_output_range(code_predictor_model, sample_inputs_embeds, sample_position_ids):
    """Test that output values are reasonable."""
    with torch.no_grad():
        output = code_predictor_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check that output is not NaN or Inf
    assert not torch.isnan(output.last_hidden_state).any()
    assert not torch.isinf(output.last_hidden_state).any()
