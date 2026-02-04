"""Unit tests for StandaloneQwen3TTSModel create_voice_clone_prompt method."""
import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch

import soundfile as sf

from qwen_tts.inference.standalone_qwen3_tts_model import StandaloneQwen3TTSModel, VoiceClonePromptItem


@pytest.fixture
def mock_model():
    """Create a mock standalone model."""
    model = Mock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    model.tts_model_type = "base"
    model.tokenizer_type = "qwen3_tts_tokenizer_12hz"
    model.tts_model_size = "1.7b"
    model.speaker_encoder_sample_rate = 24000
    
    # Mock speech tokenizer
    model.speech_tokenizer = Mock()
    
    # Mock speaker encoder extract method
    model.extract_speaker_embedding = Mock(return_value=torch.tensor([0.1, 0.2, 0.3, 0.4]))
    
    return model


@pytest.fixture
def mock_processor():
    """Create a mock processor."""
    processor = Mock()
    return processor


@pytest.fixture
def standalone_model(mock_model, mock_processor):
    """Create a StandaloneQwen3TTSModel instance for testing."""
    return StandaloneQwen3TTSModel(
        model=mock_model,
        processor=mock_processor,
        generate_defaults={}
    )


class TestCreateVoiceClonePrompt:
    """Test create_voice_clone_prompt method."""
    
    def test_create_prompt_single_audio_icl_mode(self, standalone_model):
        """Test creating prompt with single audio in ICL mode."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            # Mock speech tokenizer encode
            mock_enc = Mock()
            mock_enc.audio_codes = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
            standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
            
            # Mock resampling (since sr != 24000)
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                mock_resample.return_value = np.random.randn(24000).astype(np.float32)
                
                items = standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_path,
                    ref_text="Reference text",
                    x_vector_only_mode=False
                )
            
            assert isinstance(items, list)
            assert len(items) == 1
            assert isinstance(items[0], VoiceClonePromptItem)
            assert items[0].ref_code is not None
            assert items[0].ref_spk_embedding is not None
            assert items[0].x_vector_only_mode is False
            assert items[0].icl_mode is True
            assert items[0].ref_text == "Reference text"
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_single_audio_x_vector_only(self, standalone_model):
        """Test creating prompt with single audio in x-vector only mode."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            # Mock speech tokenizer encode
            mock_enc = Mock()
            mock_enc.audio_codes = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
            standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
            
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                mock_resample.return_value = np.random.randn(24000).astype(np.float32)
                
                items = standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_path,
                    ref_text=None,
                    x_vector_only_mode=True
                )
            
            assert len(items) == 1
            assert items[0].ref_code is None  # Should be None in x-vector only mode
            assert items[0].x_vector_only_mode is True
            assert items[0].icl_mode is False
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_batch_audio(self, standalone_model):
        """Test creating prompt with batch of audio files."""
        temp_paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    temp_paths.append(temp_path)
                    audio_data = np.random.randn(16000).astype(np.float32)
                    sf.write(temp_path, audio_data, 16000)
            
            # Mock speech tokenizer encode for batch
            mock_enc = Mock()
            mock_enc.audio_codes = [
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[4, 5, 6]])
            ]
            standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
            
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                mock_resample.return_value = np.random.randn(24000).astype(np.float32)
                
                items = standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_paths,
                    ref_text=["Text 1", "Text 2"],
                    x_vector_only_mode=False
                )
            
            assert len(items) == 2
            for item in items:
                assert isinstance(item, VoiceClonePromptItem)
                assert item.ref_code is not None
                assert item.x_vector_only_mode is False
                assert item.icl_mode is True
        finally:
            for path in temp_paths:
                os.unlink(path)
    
    def test_create_prompt_numpy_array(self, standalone_model):
        """Test creating prompt with numpy array input."""
        audio_data = np.random.randn(16000).astype(np.float32)
        
        # Mock speech tokenizer encode
        mock_enc = Mock()
        mock_enc.audio_codes = [torch.tensor([[1, 2, 3]])]
        standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
        
        with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
            mock_resample.return_value = np.random.randn(24000).astype(np.float32)
            
            items = standalone_model.create_voice_clone_prompt(
                ref_audio=(audio_data, 16000),
                ref_text="Reference text",
                x_vector_only_mode=False
            )
        
        assert len(items) == 1
        assert items[0].ref_code is not None
    
    def test_create_prompt_missing_ref_text_icl_mode_raises(self, standalone_model):
        """Test that missing ref_text in ICL mode raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            mock_enc = Mock()
            mock_enc.audio_codes = [torch.tensor([[1, 2, 3]])]
            standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
            
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                mock_resample.return_value = np.random.randn(24000).astype(np.float32)
                
                with pytest.raises(ValueError, match="ref_text is required when x_vector_only_mode=False"):
                    standalone_model.create_voice_clone_prompt(
                        ref_audio=temp_path,
                        ref_text=None,  # Missing ref_text
                        x_vector_only_mode=False  # ICL mode requires ref_text
                    )
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_batch_size_mismatch_raises(self, standalone_model):
        """Test that batch size mismatch raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            with pytest.raises(ValueError, match="Batch size mismatch"):
                standalone_model.create_voice_clone_prompt(
                    ref_audio=[temp_path, temp_path],  # 2 items
                    ref_text=["Text 1"],  # 1 item - mismatch
                    x_vector_only_mode=False
                )
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_wrong_model_type_raises(self, standalone_model):
        """Test that wrong model type raises ValueError."""
        standalone_model.model.tts_model_type = "custom_voice"  # Not "base"
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            with pytest.raises(ValueError, match="does not support create_voice_clone_prompt"):
                standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_path,
                    ref_text="Text",
                    x_vector_only_mode=False
                )
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_same_sample_rate_no_resample(self, standalone_model):
        """Test that audio with same sample rate doesn't trigger resampling."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(24000).astype(np.float32)  # Same as speaker_encoder_sample_rate
            sf.write(temp_path, audio_data, 24000)
        
        try:
            mock_enc = Mock()
            mock_enc.audio_codes = [torch.tensor([[1, 2, 3]])]
            standalone_model.model.speech_tokenizer.encode.return_value = mock_enc
            
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                items = standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_path,
                    ref_text="Text",
                    x_vector_only_mode=False
                )
                
                # Should not call resample when sr matches
                mock_resample.assert_not_called()
            
            assert len(items) == 1
        finally:
            os.unlink(temp_path)
    
    def test_create_prompt_different_sample_rates_batch(self, standalone_model):
        """Test creating prompt with batch of audio files with different sample rates."""
        temp_paths = []
        try:
            for i, sr in enumerate([16000, 22050]):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    temp_paths.append(temp_path)
                    audio_data = np.random.randn(sr).astype(np.float32)
                    sf.write(temp_path, audio_data, sr)
            
            # Mock speech tokenizer encode (called separately for each)
            def encode_side_effect(wav, sr=None):
                mock_enc = Mock()
                mock_enc.audio_codes = [torch.tensor([[1, 2, 3]])]
                return mock_enc
            
            standalone_model.model.speech_tokenizer.encode.side_effect = encode_side_effect
            
            with patch('qwen_tts.inference.standalone_qwen3_tts_model.librosa.resample') as mock_resample:
                mock_resample.return_value = np.random.randn(24000).astype(np.float32)
                
                items = standalone_model.create_voice_clone_prompt(
                    ref_audio=temp_paths,
                    ref_text=["Text 1", "Text 2"],
                    x_vector_only_mode=False
                )
            
            assert len(items) == 2
            # Should call resample for each (since sr != 24000)
            assert mock_resample.call_count == 2
        finally:
            for path in temp_paths:
                os.unlink(path)
