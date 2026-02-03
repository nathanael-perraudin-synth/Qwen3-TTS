"""Test comparing original and standalone speaker encoder."""
import torch
import pytest
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from qwen_tts.core.models.standalone_speaker_encoder import StandaloneSpeakerEncoder
from qwen_tts.core.models.standalone_config import StandaloneSpeakerEncoderConfig


@pytest.fixture
def original_config():
    """Create original speaker encoder config."""
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
def standalone_config():
    """Create standalone speaker encoder config."""
    return StandaloneSpeakerEncoderConfig(
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
def original_model(original_config, device):
    """Create original speaker encoder model."""
    model = Qwen3TTSSpeakerEncoder(original_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def standalone_model(standalone_config, device):
    """Create standalone speaker encoder model."""
    model = StandaloneSpeakerEncoder(standalone_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_input(device, seed):
    """Create sample mel-spectrogram input."""
    torch.manual_seed(seed)
    batch_size = 2
    seq_len = 100
    mel_dim = 128
    return torch.randn(batch_size, seq_len, mel_dim, device=device)


def test_models_have_same_structure(original_model, standalone_model):
    """Test that both models have the same structure."""
    # Check that both models have the same number of blocks
    assert len(original_model.blocks) == len(standalone_model.blocks)
    
    # Check that both models have the same output dimension
    assert original_model.fc.out_channels == standalone_model.fc.out_channels


def test_models_produce_identical_outputs(original_model, standalone_model, sample_input):
    """Test that both models produce identical outputs when weights are copied."""
    # Copy weights from original to standalone
    with torch.no_grad():
        # Copy initial TDNN block
        standalone_model.blocks[0].conv.weight.data.copy_(original_model.blocks[0].conv.weight.data)
        standalone_model.blocks[0].conv.bias.data.copy_(original_model.blocks[0].conv.bias.data)
        
        # Copy SE-Res2Net blocks
        for orig_block, stand_block in zip(original_model.blocks[1:], standalone_model.blocks[1:]):
            # Copy tdnn1
            stand_block.tdnn1.conv.weight.data.copy_(orig_block.tdnn1.conv.weight.data)
            stand_block.tdnn1.conv.bias.data.copy_(orig_block.tdnn1.conv.bias.data)
            
            # Copy res2net blocks
            for orig_res, stand_res in zip(orig_block.res2net_block.blocks, stand_block.res2net_block.blocks):
                stand_res.conv.weight.data.copy_(orig_res.conv.weight.data)
                stand_res.conv.bias.data.copy_(orig_res.conv.bias.data)
            
            # Copy tdnn2
            stand_block.tdnn2.conv.weight.data.copy_(orig_block.tdnn2.conv.weight.data)
            stand_block.tdnn2.conv.bias.data.copy_(orig_block.tdnn2.conv.bias.data)
            
            # Copy SE block
            stand_block.se_block.conv1.weight.data.copy_(orig_block.se_block.conv1.weight.data)
            stand_block.se_block.conv1.bias.data.copy_(orig_block.se_block.conv1.bias.data)
            stand_block.se_block.conv2.weight.data.copy_(orig_block.se_block.conv2.weight.data)
            stand_block.se_block.conv2.bias.data.copy_(orig_block.se_block.conv2.bias.data)
        
        # Copy MFA
        standalone_model.mfa.conv.weight.data.copy_(original_model.mfa.conv.weight.data)
        standalone_model.mfa.conv.bias.data.copy_(original_model.mfa.conv.bias.data)
        
        # Copy ASP
        standalone_model.asp.tdnn.conv.weight.data.copy_(original_model.asp.tdnn.conv.weight.data)
        standalone_model.asp.tdnn.conv.bias.data.copy_(original_model.asp.tdnn.conv.bias.data)
        standalone_model.asp.conv.weight.data.copy_(original_model.asp.conv.weight.data)
        standalone_model.asp.conv.bias.data.copy_(original_model.asp.conv.bias.data)
        
        # Copy final FC
        standalone_model.fc.weight.data.copy_(original_model.fc.weight.data)
        standalone_model.fc.bias.data.copy_(original_model.fc.bias.data)
    
    # Run both models
    with torch.no_grad():
        original_output = original_model(sample_input)
        standalone_output = standalone_model(sample_input)
    
    # Check that outputs are identical
    torch.testing.assert_close(original_output, standalone_output, rtol=1e-6, atol=1e-6)


def test_standalone_model_forward(standalone_model, sample_input):
    """Test standalone model forward pass."""
    with torch.no_grad():
        output = standalone_model(sample_input)
    
    # Check output shape: (batch_size, enc_dim)
    assert output.shape == (sample_input.shape[0], standalone_model.fc.out_channels)
    assert output.dtype == sample_input.dtype
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
