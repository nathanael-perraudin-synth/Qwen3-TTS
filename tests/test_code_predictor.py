"""Test comparing original and standalone code predictor models."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModel
from qwen_tts.core.configs import (
    Qwen3TTSTalkerCodePredictorConfig,
    CodePredictorConfig,
    to_code_predictor_config,
)
from qwen_tts.core.models.standalone_code_predictor import StandaloneCodePredictorModel
from qwen_tts.core.copy_weights import copy_code_transformer_weights

@pytest.fixture
def original_config():
    """Create original code predictor config."""
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
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=0,
        layer_types=None,
        attention_dropout=0.0,
        num_code_groups=4,
        pad_token_id=0,
        rope_scaling={"rope_type": "linear", "factor": 1.0},  # required for original model RoPE
    )


@pytest.fixture
def standalone_config(original_config):
    """Create standalone code predictor config from original."""
    return to_code_predictor_config(original_config)


@pytest.fixture
def original_model(original_config, device):
    """Create original code predictor model."""
    embedding_dim = 256
    model = Qwen3TTSTalkerCodePredictorModel(original_config, embedding_dim)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def standalone_model(standalone_config, device):
    """Create standalone code predictor model."""
    embedding_dim = 256
    model = StandaloneCodePredictorModel(standalone_config, embedding_dim)
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
    """Create sample position ids."""
    batch_size = 2
    seq_len = 10
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)


@pytest.fixture
def sample_one_token_embeds(device, seed):
    """Single-token embeddings for decode steps (batch, 1, embed_dim)."""
    torch.manual_seed(seed)
    batch_size = 2
    embedding_dim = 256
    return torch.randn(batch_size, 1, embedding_dim, device=device)


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
            input_ids=None,
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
    copy_code_transformer_weights(original_model, standalone_model)
    
    # Run both models
    with torch.no_grad():
        original_output = original_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=False,
        )
        standalone_output = standalone_model(
            input_ids=None,
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


def test_models_produce_identical_outputs_prefill_with_cache(
    original_model, standalone_model, sample_inputs_embeds, sample_position_ids
):
    """Test that both models produce identical outputs on prefill when use_cache=True."""
    copy_code_transformer_weights(original_model, standalone_model)

    with torch.no_grad():
        original_output = original_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
        standalone_output = standalone_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )

    torch.testing.assert_close(
        original_output.last_hidden_state,
        standalone_output.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
    )
    assert original_output.past_key_values is not None
    assert standalone_output.past_key_values is not None


def test_models_produce_identical_cache_after_prefill(
    original_model, standalone_model, sample_inputs_embeds, sample_position_ids
):
    """After prefill with cache, cached key/value tensors should match (same weights, same inputs).
    If this fails, the decode-step tests will also fail: the root cause is KV cache mismatch during prefill,
    not the decode step logic (mask, position_ids, etc.). Investigate key/value computation or RoPE in standalone.
    """
    copy_code_transformer_weights(original_model, standalone_model)

    with torch.no_grad():
        orig_out = original_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
        stand_out = standalone_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )

    orig_cache = orig_out.past_key_values
    stand_cache = stand_out.past_key_values
    num_layers = len(original_model.layers)

    for layer_idx in range(num_layers):
        orig_keys = orig_cache.layers[layer_idx].keys
        orig_vals = orig_cache.layers[layer_idx].values
        stand_keys = stand_cache.key_cache[layer_idx]
        stand_vals = stand_cache.value_cache[layer_idx]

        torch.testing.assert_close(orig_keys, stand_keys, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(orig_vals, stand_vals, rtol=1e-5, atol=1e-5)


def test_models_produce_identical_outputs_first_decode_step(
    original_model, standalone_model, sample_inputs_embeds, sample_position_ids,
    sample_one_token_embeds, device,
):
    """Test that both models produce identical outputs on the first decode step (one token + cache)."""
    copy_code_transformer_weights(original_model, standalone_model)
    batch_size, prefill_len, _ = sample_inputs_embeds.shape
    one_token_embeds = sample_one_token_embeds
    # Position for the single new token after prefill of length prefill_len
    position_ids_step = torch.full(
        (batch_size, 1), prefill_len, dtype=torch.long, device=device
    )

    with torch.no_grad():
        orig_prefill = original_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
        stand_prefill = standalone_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )

    with torch.no_grad():
        original_output = original_model(
            input_ids=None,
            inputs_embeds=one_token_embeds,
            position_ids=position_ids_step,
            past_key_values=orig_prefill.past_key_values,
            use_cache=True,
        )
        standalone_output = standalone_model(
            input_ids=None,
            inputs_embeds=one_token_embeds,
            position_ids=position_ids_step,
            past_key_values=stand_prefill.past_key_values,
            use_cache=True,
        )



    torch.testing.assert_close(
        original_output.last_hidden_state,
        standalone_output.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
    )




def test_models_produce_identical_outputs_later_decode_steps(
    original_model, standalone_model, sample_inputs_embeds, sample_position_ids,
    sample_one_token_embeds, device, seed,
):
    """Test that both models produce identical outputs on multiple decode steps (cache + 2 steps)."""
    copy_code_transformer_weights(original_model, standalone_model)
    torch.manual_seed(seed)
    batch_size, prefill_len, embed_dim = sample_inputs_embeds.shape
    # Two different single-token embeddings for step 1 and step 2 (deterministic)
    step1_embeds = torch.randn(batch_size, 1, embed_dim, device=sample_inputs_embeds.device)
    step2_embeds = torch.randn(batch_size, 1, embed_dim, device=sample_inputs_embeds.device)
    position_ids_step1 = torch.full((batch_size, 1), prefill_len, dtype=torch.long, device=sample_inputs_embeds.device)
    position_ids_step2 = torch.full((batch_size, 1), prefill_len + 1, dtype=torch.long, device=sample_inputs_embeds.device)

    with torch.no_grad():
        orig_out = original_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )
        stand_out = standalone_model(
            input_ids=None,
            inputs_embeds=sample_inputs_embeds,
            position_ids=sample_position_ids,
            use_cache=True,
        )

    # First decode step
    with torch.no_grad():
        orig_out = original_model(
            input_ids=None,
            inputs_embeds=step1_embeds,
            position_ids=position_ids_step1,
            past_key_values=orig_out.past_key_values,
            use_cache=True,
        )
        stand_out = standalone_model(
            input_ids=None,
            inputs_embeds=step1_embeds,
            position_ids=position_ids_step1,
            past_key_values=stand_out.past_key_values,
            use_cache=True,
        )


    torch.testing.assert_close(
        orig_out.last_hidden_state,
        stand_out.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
        msg="First decode step: last_hidden_state mismatch",
    )


    # Second decode step
    with torch.no_grad():
        orig_out = original_model(
            input_ids=None,
            inputs_embeds=step2_embeds,
            position_ids=position_ids_step2,
            past_key_values=orig_out.past_key_values,
            use_cache=True,
        )
        stand_out = standalone_model(
            input_ids=None,
            inputs_embeds=step2_embeds,
            position_ids=position_ids_step2,
            past_key_values=stand_out.past_key_values,
            use_cache=True,
        )


    torch.testing.assert_close(
        orig_out.last_hidden_state,
        stand_out.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
        msg="Second decode step: last_hidden_state mismatch",
    )
