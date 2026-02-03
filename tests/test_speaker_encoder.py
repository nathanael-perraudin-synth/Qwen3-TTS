"""Test for Qwen3TTSSpeakerEncoder model."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig


@pytest.fixture
def speaker_encoder_config():
    """Create a speaker encoder config for testing."""
    return Qwen3TTSSpeakerEncoderConfig(
        mel_dim=128,
        enc_dim=1024,
        enc_channels=[512, 512, 512, 512, 1536],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=128,
        enc_res2net_scale=8,
        enc_se_channels=128,
        sample_rate=24000,
    )


@pytest.fixture
def speaker_encoder(speaker_encoder_config, device):
    """Create a speaker encoder model for testing."""
    model = Qwen3TTSSpeakerEncoder(speaker_encoder_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_input(device):
    """Create sample mel-spectrogram input."""
    # Batch size 2, sequence length 100, mel_dim 128
    # Input shape: (batch, seq_len, mel_dim) - will be transposed to (batch, mel_dim, seq_len) in forward
    batch_size = 2
    seq_len = 100
    mel_dim = 128
    return torch.randn(batch_size, seq_len, mel_dim, device=device)


def test_speaker_encoder_forward(speaker_encoder, sample_input):
    """Test forward pass of speaker encoder."""
    with torch.no_grad():
        output = speaker_encoder(sample_input)
    
    # Check output shape: (batch_size, enc_dim)
    assert output.shape == (sample_input.shape[0], speaker_encoder.fc.out_channels)
    assert output.dtype == sample_input.dtype


def test_speaker_encoder_output_range(speaker_encoder, sample_input):
    """Test that output values are reasonable."""
    with torch.no_grad():
        output = speaker_encoder(sample_input)
    
    # Check that output is not NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_speaker_encoder_different_batch_sizes(speaker_encoder, device):
    """Test speaker encoder with different batch sizes."""
    for batch_size in [1, 2, 4]:
        # Input shape: (batch, seq_len, mel_dim)
        input_tensor = torch.randn(batch_size, 100, 128, device=device)
        with torch.no_grad():
            output = speaker_encoder(input_tensor)
        assert output.shape[0] == batch_size


def test_speaker_encoder_different_seq_lengths(speaker_encoder, device):
    """Test speaker encoder with different sequence lengths."""
    batch_size = 2
    mel_dim = 128
    for seq_len in [50, 100, 200]:
        # Input shape: (batch, seq_len, mel_dim)
        input_tensor = torch.randn(batch_size, seq_len, mel_dim, device=device)
        with torch.no_grad():
            output = speaker_encoder(input_tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == speaker_encoder.fc.out_channels
