"""Tests for standalone talker generation model."""
import pytest
import torch
import numpy as np
from torch import nn

from qwen_tts.core.models.standalone_talker_generation import StandaloneTalkerForConditionalGeneration
from qwen_tts.core.configs import (
    Qwen3TTSTalkerConfig,
    TalkerConfig,
    to_standalone_talker_config,
)
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration


@pytest.fixture
def original_talker_config():
    return Qwen3TTSTalkerConfig(
        vocab_size=3072,
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
        text_hidden_size=256,
        text_vocab_size=151936,
        codec_eos_token_id=4198,
        codec_think_id=4202,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        rope_scaling={
            "rope_type": "dynamic",
            "factor": 2.0,
            "original_max_position_embeddings": 1024,
            "mrope_section": [10, 11, 11],
            "interleaved": False,
        },
    )


@pytest.fixture
def talker_config(original_talker_config):
    return to_standalone_talker_config(original_talker_config)


@pytest.fixture
def standalone_talker_gen(talker_config, device):
    model = StandaloneTalkerForConditionalGeneration(talker_config).to(device)
    model.eval()
    return model


@pytest.fixture
def original_talker_gen(original_talker_config, device):
    model = Qwen3TTSTalkerForConditionalGeneration(original_talker_config).to(device)
    model.eval()
    return model


@pytest.fixture
def sample_inputs_embeds(device):
    # List of embeddings for batch
    return [torch.randn(1, 10, 256, device=device) for _ in range(2)]


@pytest.fixture
def sample_attention_mask(device):
    return torch.ones(2, 10, device=device, dtype=torch.long)


def test_standalone_talker_gen_forward(standalone_talker_gen, sample_inputs_embeds, device):
    """Test that standalone talker generation can do forward pass."""
    # Create a simple forward pass
    inputs_embeds = torch.cat(sample_inputs_embeds, dim=0)
    attention_mask = torch.ones(2, 10, device=device, dtype=torch.long)
    trailing_text_hidden = torch.randn(2, 5, 256, device=device)
    tts_pad_embed = torch.randn(1, 1, 256, device=device)
    
    outputs = standalone_talker_gen.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        generation_step=-1,
    )
    
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == inputs_embeds.shape[0]
    assert outputs.logits.shape[1] == inputs_embeds.shape[1]
    assert outputs.logits.shape[2] == standalone_talker_gen.vocab_size


def test_standalone_talker_gen_get_rope_index(standalone_talker_gen, device):
    """Test get_rope_index method."""
    attention_mask = torch.ones(2, 10, device=device, dtype=torch.long)
    position_ids, rope_deltas = standalone_talker_gen.get_rope_index(attention_mask)
    
    assert position_ids.shape == (3, 2, 10)
    assert rope_deltas.shape == (2, 1)


def test_models_have_same_structure(standalone_talker_gen, original_talker_gen):
    """Test that models have the same structure."""
    # Check that both models have the same components
    assert hasattr(standalone_talker_gen, 'model')
    assert hasattr(standalone_talker_gen, 'codec_head')
    assert hasattr(standalone_talker_gen, 'text_projection')
    assert hasattr(standalone_talker_gen, 'code_predictor')
    assert hasattr(original_talker_gen, 'model')
    assert hasattr(original_talker_gen, 'codec_head')
    assert hasattr(original_talker_gen, 'text_projection')
    assert hasattr(original_talker_gen, 'code_predictor')
    
    # Check vocab size
    assert standalone_talker_gen.vocab_size == original_talker_gen.vocab_size


def test_models_produce_identical_outputs(
    standalone_talker_gen,
    original_talker_gen,
    device,
):
    """Test that models produce identical outputs when weights are copied."""
    # Create sample inputs
    inputs_embeds = torch.randn(2, 10, 256, device=device)
    attention_mask = torch.ones(2, 10, device=device, dtype=torch.long)
    trailing_text_hidden = torch.randn(2, 5, 256, device=device)
    tts_pad_embed = torch.randn(1, 1, 256, device=device)
    
    # Copy weights from original to standalone
    with torch.no_grad():
        # Copy model weights
        standalone_talker_gen.model.load_state_dict(original_talker_gen.model.state_dict())
        
        # Copy codec_head weights
        standalone_talker_gen.codec_head.weight.data.copy_(
            original_talker_gen.codec_head.weight.data
        )
        
        # Copy text_projection weights
        standalone_talker_gen.text_projection.load_state_dict(
            original_talker_gen.text_projection.state_dict()
        )
        
        # Copy code_predictor weights (if possible)
        try:
            standalone_talker_gen.code_predictor.model.load_state_dict(
                original_talker_gen.code_predictor.model.state_dict()
            )
            for i in range(len(standalone_talker_gen.code_predictor.lm_head)):
                standalone_talker_gen.code_predictor.lm_head[i].weight.data.copy_(
                    original_talker_gen.code_predictor.lm_head[i].weight.data
                )
        except Exception:
            pass  # Skip if structure differs
    
    # Set same seed for both models
    torch.manual_seed(42)
    np.random.seed(42)
    standalone_outputs = standalone_talker_gen.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        generation_step=-1,
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    original_outputs = original_talker_gen.forward(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        generation_step=-1,
    )
    
    # Compare outputs
    assert torch.allclose(
        standalone_outputs.logits,
        original_outputs.logits,
        atol=1e-4,
        rtol=1e-4,
    )
