"""Tests for standalone code predictor generation model."""
import pytest
import torch
import numpy as np
from torch import nn

from qwen_tts.core.models.standalone_code_predictor_generation import StandaloneCodePredictorForConditionalGeneration
from qwen_tts.core.models.standalone_config import StandaloneCodePredictorConfig, StandaloneTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)


@pytest.fixture
def code_predictor_config():
    return StandaloneCodePredictorConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        num_code_groups=4,
    )


@pytest.fixture
def talker_config_for_predictor():
    return StandaloneTalkerConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=1024,
        num_code_groups=4,
    )


@pytest.fixture
def original_code_predictor_config():
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
        rms_norm_eps=1e-6,
        num_code_groups=4,
    )


@pytest.fixture
def original_talker_config():
    return Qwen3TTSTalkerConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=1024,
        num_code_groups=4,
    )


@pytest.fixture
def standalone_code_predictor(code_predictor_config, talker_config_for_predictor, device):
    model = StandaloneCodePredictorForConditionalGeneration(
        config=code_predictor_config,
        talker_config=talker_config_for_predictor,
        embedding_dim=talker_config_for_predictor.hidden_size,
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def original_code_predictor(original_code_predictor_config, original_talker_config, device):
    model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
        config=original_code_predictor_config,
        talker_config=original_talker_config,
    ).to(device)
    model.eval()
    return model


@pytest.fixture
def sample_inputs_embeds(device):
    # (batch_size, seq_len, hidden_size)
    return torch.randn(2, 5, 256, device=device)


def test_standalone_code_predictor_forward(standalone_code_predictor, sample_inputs_embeds):
    """Test that standalone code predictor can do forward pass."""
    outputs = standalone_code_predictor.forward_finetune(
        inputs_embeds=sample_inputs_embeds,
    )
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == sample_inputs_embeds.shape[0]
    assert outputs.logits.shape[1] == standalone_code_predictor.config.num_code_groups - 1


def test_standalone_code_predictor_generate(standalone_code_predictor, sample_inputs_embeds):
    """Test that standalone code predictor can generate."""
    torch.manual_seed(42)
    result = standalone_code_predictor.generate(
        inputs_embeds=sample_inputs_embeds,
        max_new_tokens=3,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature=0.8,
    )
    assert result.sequences is not None
    assert result.sequences.shape[0] == sample_inputs_embeds.shape[0]
    assert result.sequences.shape[1] <= 3


def test_models_have_same_structure(standalone_code_predictor, original_code_predictor):
    """Test that models have the same structure."""
    # Check that both models have the same components
    assert hasattr(standalone_code_predictor, 'model')
    assert hasattr(standalone_code_predictor, 'lm_head')
    assert hasattr(original_code_predictor, 'model')
    assert hasattr(original_code_predictor, 'lm_head')
    
    # Check vocab size
    assert standalone_code_predictor.vocab_size == original_code_predictor.vocab_size
    
    # Check number of lm heads
    assert len(standalone_code_predictor.lm_head) == len(original_code_predictor.lm_head)


def test_models_produce_identical_outputs(
    standalone_code_predictor,
    original_code_predictor,
    sample_inputs_embeds,
    device,
):
    """Test that models produce identical outputs when weights are copied."""
    # Copy weights from original to standalone
    with torch.no_grad():
        # Copy model weights
        standalone_code_predictor.model.load_state_dict(original_code_predictor.model.state_dict())
        
        # Copy lm_head weights
        for i in range(len(standalone_code_predictor.lm_head)):
            standalone_code_predictor.lm_head[i].weight.data.copy_(
                original_code_predictor.lm_head[i].weight.data
            )
        
        # Copy projection if exists
        if hasattr(standalone_code_predictor, 'small_to_mtp_projection'):
            if hasattr(original_code_predictor, 'small_to_mtp_projection'):
                if isinstance(standalone_code_predictor.small_to_mtp_projection, nn.Linear):
                    standalone_code_predictor.small_to_mtp_projection.load_state_dict(
                        original_code_predictor.small_to_mtp_projection.state_dict()
                    )
    
    # Set same seed for both models
    torch.manual_seed(42)
    np.random.seed(42)
    standalone_outputs = standalone_code_predictor.forward_finetune(
        inputs_embeds=sample_inputs_embeds,
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    original_outputs = original_code_predictor.forward_finetune(
        inputs_embeds=sample_inputs_embeds,
    )
    
    # Compare outputs
    assert torch.allclose(
        standalone_outputs.logits,
        original_outputs.logits,
        atol=1e-5,
        rtol=1e-5,
    )
