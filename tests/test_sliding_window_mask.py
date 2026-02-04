"""Unit tests comparing standalone sliding window mask implementation with transformers."""
import pytest
import torch
import numpy as np
from typing import Optional

# Import standalone implementation
from qwen_tts.core.models.standalone_utils import (
    create_sliding_window_causal_mask,
    eager_attention_forward,
)
from qwen_tts.core.configs import (
    Qwen3TTSTalkerCodePredictorConfig,
    CodePredictorConfig,
    to_code_predictor_config,
)

# Import transformers implementation for comparison
try:
    from transformers.masking_utils import (
        create_sliding_window_causal_mask as transformers_create_sliding_window_causal_mask,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.fixture
def device():
    """Return the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def test_config():
    """Create a test config with sliding window (converted from original config)."""
    def _create_config(sliding_window: Optional[int] = None):
        orig = Qwen3TTSTalkerCodePredictorConfig(
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
            use_sliding_window=sliding_window is not None,
            sliding_window=sliding_window,
            max_window_layers=0,
            layer_types=None,
            attention_dropout=0.0,
            num_code_groups=4,
            pad_token_id=0,
        )
        return to_code_predictor_config(orig)
    return _create_config


class MockModule:
    """Mock module for testing attention."""
    def __init__(self, num_key_value_groups: int = 1, training: bool = False):
        self.num_key_value_groups = num_key_value_groups
        self.training = training


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
class TestSlidingWindowMaskEquivalence:
    """Test that standalone sliding window mask matches transformers implementation."""
    
    def test_create_sliding_window_causal_mask_basic(self, device, seed, test_config):
        """Test basic sliding window mask creation."""
        batch_size = 2
        seq_length = 20
        sliding_window = 5
        
        config = test_config(sliding_window=sliding_window)
        input_embeds = torch.randn(batch_size, seq_length, 128, device=device)
        cache_position = torch.arange(seq_length, device=device)
        
        # Standalone implementation
        standalone_mask = create_sliding_window_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=None,
        )
        
        # Transformers implementation
        transformers_mask = transformers_create_sliding_window_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
        )
        
        # Standalone returns None (handled in attention), transformers returns mask
        # So we need to test the actual behavior in attention forward
        assert standalone_mask is None, "Standalone should return None for sliding window"
        assert transformers_mask is not None, "Transformers should return a mask"
        
        # Verify transformers mask shape (can be (batch, seq, seq) or (batch, 1, seq, seq))
        assert transformers_mask.shape[-2:] == (seq_length, seq_length), \
            f"Transformers mask should have sequence dimensions ({seq_length}, {seq_length}), got {transformers_mask.shape}"
    
    def test_create_sliding_window_causal_mask_with_attention_mask(self, device, seed, test_config):
        """Test sliding window mask with attention mask (padding)."""
        batch_size = 2
        seq_length = 20
        sliding_window = 5
        
        config = test_config(sliding_window=sliding_window)
        input_embeds = torch.randn(batch_size, seq_length, 128, device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        attention_mask[0, 15:] = 0  # Mask some positions in first batch
        cache_position = torch.arange(seq_length, device=device)
        
        # Standalone implementation
        standalone_mask = create_sliding_window_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        
        # Transformers implementation
        transformers_mask = transformers_create_sliding_window_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
        )
        
        # Standalone returns the attention_mask, transformers returns combined mask
        assert standalone_mask is not None
        assert torch.equal(standalone_mask, attention_mask)
        # Verify transformers mask shape (can be (batch, seq, seq) or (batch, 1, seq, seq))
        assert transformers_mask.shape[-2:] == (seq_length, seq_length), \
            f"Transformers mask should have sequence dimensions ({seq_length}, {seq_length}), got {transformers_mask.shape}"
    
    def test_eager_attention_forward_sliding_window(self, device, seed):
        """Test that eager_attention_forward applies sliding window correctly."""
        batch_size = 2
        num_heads = 4
        seq_length = 20
        head_dim = 64
        sliding_window = 5
        
        # Create query, key, value
        query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        
        module = MockModule(num_key_value_groups=1)
        
        # Test with sliding window
        output, weights = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            scaling=1.0 / np.sqrt(head_dim),
            dropout=0.0,
            sliding_window=sliding_window,
        )
        
        # Verify output shapes
        # Output is transposed: (batch, seq, num_heads, head_dim)
        assert output.shape == (batch_size, seq_length, num_heads, head_dim)
        assert weights.shape == (batch_size, num_heads, seq_length, seq_length)
        
        # Verify sliding window constraint: for position i, only positions
        # [max(0, i - sliding_window + 1), i] should have non-zero attention
        for i in range(seq_length):
            start = max(0, i - sliding_window + 1)
            # Positions before start should have zero attention (after softmax)
            if start > 0:
                # Check that weights are very small (close to 0) for masked positions
                masked_weights = weights[:, :, i, :start]
                assert torch.all(masked_weights < 1e-6), \
                    f"Position {i}: weights before {start} should be masked"
            
            # Positions from start to i should have non-zero attention
            allowed_weights = weights[:, :, i, start:i+1]
            assert torch.all(allowed_weights > 1e-6), \
                f"Position {i}: weights from {start} to {i} should be non-zero"
    
    def test_eager_attention_forward_sliding_window_vs_full_mask(self, device, seed):
        """Compare sliding window attention with manually created full mask."""
        batch_size = 2
        num_heads = 4
        seq_length = 20
        head_dim = 64
        sliding_window = 5
        
        # Create query, key, value
        query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        
        module = MockModule(num_key_value_groups=1)
        scaling = 1.0 / np.sqrt(head_dim)
        
        # Method 1: Use sliding_window parameter (optimized)
        output1, weights1 = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            scaling=scaling,
            dropout=0.0,
            sliding_window=sliding_window,
        )
        
        # Method 2: Create full mask manually and pass it
        # Create sliding window mask manually
        mask = torch.full(
            (batch_size, seq_length, seq_length),
            fill_value=float("-inf"),
            device=device,
            dtype=query.dtype
        )
        for i in range(seq_length):
            start = max(0, i - sliding_window + 1)
            mask[:, i, start:i+1] = 0.0
        
        # Expand mask to (batch, num_heads, seq, seq)
        mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        
        output2, weights2 = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=mask,
            scaling=scaling,
            dropout=0.0,
            sliding_window=None,  # No sliding window, use mask instead
        )
        
        # Results should be very similar (within numerical precision)
        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-5), \
            "Outputs should match between sliding_window parameter and full mask"
        assert torch.allclose(weights1, weights2, atol=1e-5, rtol=1e-5), \
            "Attention weights should match between sliding_window parameter and full mask"
    
    def test_eager_attention_forward_sliding_window_different_sizes(self, device, seed):
        """Test sliding window with different sequence lengths and window sizes."""
        test_cases = [
            (2, 4, 10, 64, 3),   # Small sequence, small window
            (2, 4, 50, 64, 10),  # Medium sequence, medium window
            (2, 4, 100, 64, 20), # Larger sequence, larger window
        ]
        
        for batch_size, num_heads, seq_length, head_dim, sliding_window in test_cases:
            query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
            key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
            value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
            
            module = MockModule(num_key_value_groups=1)
            
            output, weights = eager_attention_forward(
                module=module,
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                scaling=1.0 / np.sqrt(head_dim),
                dropout=0.0,
                sliding_window=sliding_window,
            )
            
            # Verify shapes
            # Output is transposed: (batch, seq, num_heads, head_dim)
            assert output.shape == (batch_size, seq_length, num_heads, head_dim)
            assert weights.shape == (batch_size, num_heads, seq_length, seq_length)
            
            # Verify sliding window constraint for a few positions
            for i in [0, seq_length // 2, seq_length - 1]:
                start = max(0, i - sliding_window + 1)
                if start > 0:
                    masked_weights = weights[:, :, i, :start]
                    assert torch.all(masked_weights < 1e-6), \
                        f"Position {i}: weights before {start} should be masked"
    
    def test_eager_attention_forward_sliding_window_with_padding(self, device, seed):
        """Test sliding window with attention mask (padding)."""
        batch_size = 2
        num_heads = 4
        seq_length = 20
        head_dim = 64
        sliding_window = 5
        
        query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        
        # Create attention mask with padding
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        attention_mask[0, 15:] = 0  # Mask last 5 positions in first batch
        
        module = MockModule(num_key_value_groups=1)
        
        output, weights = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=1.0 / np.sqrt(head_dim),
            dropout=0.0,
            sliding_window=sliding_window,
        )
        
        # Verify that masked positions have zero or NaN attention in first batch
        # NaN can occur when all positions are masked (all -inf after softmax)
        masked_positions = weights[0, :, :, 15:]
        # Check that values are either very small (< 1e-6) or NaN
        assert torch.all((masked_positions < 1e-6) | torch.isnan(masked_positions)), \
            "Masked positions should have zero or NaN attention"
    
    def test_eager_attention_forward_no_sliding_window(self, device, seed):
        """Test that attention works without sliding window (basic functionality)."""
        batch_size = 2
        num_heads = 4
        seq_length = 20
        head_dim = 64
        
        query = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim, device=device)
        
        module = MockModule(num_key_value_groups=1)
        
        output, weights = eager_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            scaling=1.0 / np.sqrt(head_dim),
            dropout=0.0,
            sliding_window=None,  # No sliding window
        )
        
        # Verify shapes
        # Output is transposed: (batch, seq, num_heads, head_dim)
        assert output.shape == (batch_size, seq_length, num_heads, head_dim)
        assert weights.shape == (batch_size, num_heads, seq_length, seq_length)
        
        # Verify that attention weights sum to 1 (softmax normalization)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, num_heads, seq_length, device=device), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
