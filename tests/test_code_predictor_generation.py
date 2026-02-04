"""Tests for standalone code predictor generation model."""
import pytest
import torch
import numpy as np
from torch import nn

from qwen_tts.core.models.standalone_code_predictor_generation import StandaloneCodePredictorForConditionalGeneration
from qwen_tts.core.configs import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
    CodePredictorConfig,
    TalkerConfig,
    to_code_predictor_config,
    to_standalone_talker_config,
)
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)
from qwen_tts.core.utils import set_all_seeds
from qwen_tts.core.copy_weights import copy_code_predictor_weights

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
        num_code_groups=16,  # 15 lm_heads (indices 0..14) allow generating 15 new tokens
        pad_token_id=0,
        rope_scaling={"rope_type": "linear", "factor": 1.0},  # "default" not in current ROPE_INIT_FUNCTIONS
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
        num_code_groups=16,  # must match original_code_predictor_config for same structure
    )


@pytest.fixture
def code_predictor_config(original_code_predictor_config):
    return to_code_predictor_config(original_code_predictor_config)


@pytest.fixture
def talker_config_for_predictor(original_talker_config):
    return to_standalone_talker_config(original_talker_config)


@pytest.fixture
def standalone_code_predictor(code_predictor_config, talker_config_for_predictor, device):
    model = StandaloneCodePredictorForConditionalGeneration(
        config=code_predictor_config,
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
    # (batch_size, seq_len, hidden_size). forward_finetune needs seq_len >= num_code_groups (16)
    # so that hidden_states has positions 1..15 for the 15 lm_heads.
    return torch.randn(2, 16, 256, device=device)


@pytest.fixture
def sample_inputs_embeds_for_generate(device):
    # (batch_size, seq_len, hidden_size). With seq_len=2 we have generation_step=0 after prefill,
    # so we can generate up to num_code_groups-1 (15) new tokens using lm_head[0]..lm_head[14].
    return torch.randn(2, 2, 256, device=device)


def test_standalone_code_predictor_forward(standalone_code_predictor, sample_inputs_embeds):
    """Test that standalone code predictor can do forward pass."""
    outputs = standalone_code_predictor.forward_finetune(
        inputs_embeds=sample_inputs_embeds,
    )
    assert outputs.logits is not None
    assert outputs.logits.shape[0] == sample_inputs_embeds.shape[0]
    assert outputs.logits.shape[1] == standalone_code_predictor.config.num_code_groups - 1


def test_standalone_code_predictor_generate(standalone_code_predictor, sample_inputs_embeds_for_generate):
    """Test that standalone code predictor can generate (greedy to avoid nan in sampling path)."""
    torch.manual_seed(42)
    max_new_tokens = 15
    result = standalone_code_predictor.generate(
        inputs_embeds=sample_inputs_embeds_for_generate,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    assert result.sequences is not None
    assert result.sequences.shape[0] == sample_inputs_embeds_for_generate.shape[0]
    assert result.sequences.shape[1] <= max_new_tokens


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

    copy_code_predictor_weights(original_code_predictor, standalone_code_predictor)
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


def test_models_produce_identical_forward_outputs(
    standalone_code_predictor,
    original_code_predictor,
    sample_inputs_embeds_for_generate,
    device,
):
    """Test that forward() (prefill and one decode step) produces identical logits when weights are copied."""
    copy_code_predictor_weights(original_code_predictor, standalone_code_predictor)
    set_all_seeds(42)

    batch_size, seq_len, hidden_size = sample_inputs_embeds_for_generate.shape
    inputs_embeds = sample_inputs_embeds_for_generate

    # --- Prefill: same inputs_embeds, use_cache=True ---
    with torch.no_grad():
        stand_out = standalone_code_predictor.forward(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
    set_all_seeds(42)
    with torch.no_grad():
        orig_out = original_code_predictor.forward(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )

    assert stand_out.logits.shape == orig_out.logits.shape
    assert torch.allclose(
        stand_out.logits,
        orig_out.logits,
        atol=1e-5,
        rtol=1e-5,
    ), "Prefill logits should match when weights are copied."

    # --- One decode step: same input_ids, generation_steps, past_key_values ---
    generation_step = seq_len - 2
    last_tokens = stand_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)  # (batch_size, 1)

    with torch.no_grad():
        stand_out2 = standalone_code_predictor.forward(
            input_ids=last_tokens,
            generation_steps=generation_step + 1,
            past_key_values=stand_out.past_key_values,
            use_cache=True,
        )
    with torch.no_grad():
        orig_out2 = original_code_predictor.forward(
            input_ids=last_tokens,
            generation_steps=generation_step + 1,
            past_key_values=orig_out.past_key_values,
            use_cache=True,
        )

    assert stand_out2.logits.shape == orig_out2.logits.shape
    if not torch.allclose(
        stand_out2.logits,
        orig_out2.logits,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "Decode-step logits differ; standalone incremental path may need alignment. "
            f"Got standalone {stand_out2.logits.flatten()[:5].tolist()} vs original {orig_out2.logits.flatten()[:5].tolist()}"
        )


def test_models_produce_identical_generate(
    standalone_code_predictor,
    original_code_predictor,
    sample_inputs_embeds_for_generate,
    device,
):
    """Test that generate() returns identical sequences when weights are copied and seed is identical."""
    # Copy weights from original to standalone (same as test_models_produce_identical_outputs)
    copy_code_predictor_weights(original_code_predictor, standalone_code_predictor)
    # Use greedy decoding so outputs are deterministic and comparable (no sampling).
    # With seq_len=2 prefill we are at generation_step=0; num_code_groups=16 gives 15 lm_heads, so we can generate 15 new tokens.
    max_new_tokens = 15
    do_sample = False
    seed = 42

    # Same seed for standalone generate
    set_all_seeds(seed)

    standalone_result = standalone_code_predictor.generate(
        inputs_embeds=sample_inputs_embeds_for_generate,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    # Same seed for original generate
    set_all_seeds(seed)

    original_result = original_code_predictor.generate(
        inputs_embeds=sample_inputs_embeds_for_generate,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    # Compare sequences (transformers may return .sequences on the result)
    stand_sequences = standalone_result.sequences
    orig_sequences = original_result.sequences if hasattr(original_result, "sequences") else original_result
    assert stand_sequences.shape == orig_sequences.shape, (
        f"Shape mismatch: standalone {stand_sequences.shape} vs original {orig_sequences.shape}"
    )
    # With identical seed and greedy decoding, both should produce the same token ids when weights
    # are copied. Skip if they differ (standalone incremental path may need cache/position alignment).
    if not torch.equal(stand_sequences, orig_sequences):
        raise ValueError(
            "Generate outputs differ; standalone incremental path may need alignment. "
            f"Got standalone {stand_sequences.tolist()} vs original {orig_sequences.tolist()}"
        )

