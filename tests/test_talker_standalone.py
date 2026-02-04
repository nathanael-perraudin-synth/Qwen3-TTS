"""Test comparing original and standalone talker models."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerModel
from qwen_tts.core.configs import (
    Qwen3TTSTalkerConfig,
    TalkerConfig,
    to_standalone_talker_config,
)
from qwen_tts.core.models.standalone_talker import BigTransformer


@pytest.fixture
def original_config():
    """Create original talker config."""
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
            "mrope_section": [11, 11, 10],
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
def standalone_config(original_config):
    """Create standalone talker config from original."""
    return to_standalone_talker_config(original_config)


@pytest.fixture
def original_model(original_config, device):
    """Create original talker model."""
    model = Qwen3TTSTalkerModel(original_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def standalone_model(standalone_config, device):
    """Create standalone talker model."""
    model = BigTransformer(standalone_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_inputs_embeds(device, seed):
    """Create sample input embeddings."""
    torch.manual_seed(seed)
    batch_size = 2
    seq_len = 10
    embedding_dim = 256
    return torch.randn(batch_size, seq_len, embedding_dim, device=device)


@pytest.fixture
def sample_position_ids(device):
    """Create sample position ids (3D for multimodal)."""
    batch_size = 2
    seq_len = 10
    # Shape: (3, batch, seq) for temporal, height, width
    return torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, batch_size, -1)


def test_models_have_same_structure(original_model, standalone_model):
    """Test that both models have the same structure."""
    # Check that both models have the same number of layers
    assert len(original_model.layers) == len(standalone_model.layers)
    
    # Check that both models have the same hidden size
    assert original_model.config.hidden_size == standalone_model.config.hidden_size


def test_standalone_model_forward(standalone_model, sample_inputs_embeds, sample_position_ids):
    """Test standalone model forward pass."""
    with torch.no_grad():
        output = standalone_model(
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check output shape
    assert output.last_hidden_state.shape == (sample_inputs_embeds.shape[0], sample_inputs_embeds.shape[1], standalone_model.config.hidden_size)
    assert output.last_hidden_state.dtype == sample_inputs_embeds.dtype
    assert not torch.isnan(output.last_hidden_state).any()
    assert not torch.isinf(output.last_hidden_state).any()


def test_models_produce_identical_outputs(original_model, standalone_model, sample_inputs_embeds, sample_position_ids):
    """Test that both models produce identical outputs when weights are copied."""
    # Copy weights from original to standalone
    with torch.no_grad():
        # Copy embeddings
        standalone_model.codec_embedding.weight.data.copy_(original_model.codec_embedding.weight.data)
        standalone_model.text_embedding.weight.data.copy_(original_model.text_embedding.weight.data)
        
        # Copy layers
        for orig_layer, stand_layer in zip(original_model.layers, standalone_model.layers):
            # Copy attention weights
            stand_layer.self_attn.q_proj.weight.data.copy_(orig_layer.self_attn.q_proj.weight.data)
            if orig_layer.self_attn.q_proj.bias is not None:
                stand_layer.self_attn.q_proj.bias.data.copy_(orig_layer.self_attn.q_proj.bias.data)
            
            stand_layer.self_attn.k_proj.weight.data.copy_(orig_layer.self_attn.k_proj.weight.data)
            if orig_layer.self_attn.k_proj.bias is not None:
                stand_layer.self_attn.k_proj.bias.data.copy_(orig_layer.self_attn.k_proj.bias.data)
            
            stand_layer.self_attn.v_proj.weight.data.copy_(orig_layer.self_attn.v_proj.weight.data)
            if orig_layer.self_attn.v_proj.bias is not None:
                stand_layer.self_attn.v_proj.bias.data.copy_(orig_layer.self_attn.v_proj.bias.data)
            
            stand_layer.self_attn.o_proj.weight.data.copy_(orig_layer.self_attn.o_proj.weight.data)
            if orig_layer.self_attn.o_proj.bias is not None:
                stand_layer.self_attn.o_proj.bias.data.copy_(orig_layer.self_attn.o_proj.bias.data)
            
            # Copy attention norms
            stand_layer.self_attn.q_norm.weight.data.copy_(orig_layer.self_attn.q_norm.weight.data)
            stand_layer.self_attn.k_norm.weight.data.copy_(orig_layer.self_attn.k_norm.weight.data)
            
            # Copy MLP weights
            stand_layer.mlp.gate_proj.weight.data.copy_(orig_layer.mlp.gate_proj.weight.data)
            stand_layer.mlp.up_proj.weight.data.copy_(orig_layer.mlp.up_proj.weight.data)
            stand_layer.mlp.down_proj.weight.data.copy_(orig_layer.mlp.down_proj.weight.data)
            
            # Copy layer norms
            stand_layer.input_layernorm.weight.data.copy_(orig_layer.input_layernorm.weight.data)
            stand_layer.post_attention_layernorm.weight.data.copy_(orig_layer.post_attention_layernorm.weight.data)
        
        # Copy final norm
        standalone_model.norm.weight.data.copy_(original_model.norm.weight.data)
        
        # Copy rotary embedding (inv_freq)
        standalone_model.rotary_emb.inv_freq.data.copy_(original_model.rotary_emb.inv_freq.data)
    
    # Run both models
    with torch.no_grad():
        original_output = original_model(
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
        standalone_output = standalone_model(
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
    
    # Check that outputs are identical (with some tolerance for numerical differences)
    torch.testing.assert_close(
        original_output.last_hidden_state,
        standalone_output.last_hidden_state,
        rtol=1e-5,
        atol=1e-5
    )
