"""Unit tests for StandaloneQwen3TTSModel audio loading methods."""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import soundfile as sf

from qwen_tts.inference.standalone_qwen3_tts_model import StandaloneQwen3TTSModel


@pytest.fixture
def mock_model():
    """Create a mock standalone model."""
    model = Mock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
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


class TestIsProbablyBase64:
    """Test _is_probably_base64 method."""
    
    def test_base64_with_data_prefix(self, standalone_model):
        """Test base64 string with data:audio prefix."""
        assert standalone_model._is_probably_base64("data:audio/wav;base64,UklGRiQAAABXQVZFZm10")
    
    def test_base64_long_string(self, standalone_model):
        """Test long string without path separators."""
        long_string = "a" * 300
        assert standalone_model._is_probably_base64(long_string)
    
    def test_not_base64_path(self, standalone_model):
        """Test file path is not base64."""
        assert not standalone_model._is_probably_base64("/path/to/file.wav")
        assert not standalone_model._is_probably_base64("file.wav")
    
    def test_not_base64_short_string(self, standalone_model):
        """Test short string is not base64."""
        assert not standalone_model._is_probably_base64("short")


class TestIsURL:
    """Test _is_url method."""
    
    def test_http_url(self, standalone_model):
        """Test HTTP URL."""
        assert standalone_model._is_url("http://example.com/audio.wav")
    
    def test_https_url(self, standalone_model):
        """Test HTTPS URL."""
        assert standalone_model._is_url("https://example.com/audio.wav")
    
    def test_not_url_file_path(self, standalone_model):
        """Test file path is not URL."""
        assert not standalone_model._is_url("/path/to/file.wav")
        assert not standalone_model._is_url("file.wav")
    
    def test_not_url_no_scheme(self, standalone_model):
        """Test string without scheme is not URL."""
        assert not standalone_model._is_url("example.com/audio.wav")


class TestDecodeBase64:
    """Test _decode_base64_to_wav_bytes method."""
    
    def test_decode_base64_with_data_prefix(self, standalone_model):
        """Test decoding base64 with data: prefix."""
        import base64
        test_data = b"test audio data"
        encoded = base64.b64encode(test_data).decode()
        result = standalone_model._decode_base64_to_wav_bytes(f"data:audio/wav;base64,{encoded}")
        assert result == test_data
    
    def test_decode_base64_without_prefix(self, standalone_model):
        """Test decoding base64 without data: prefix."""
        import base64
        test_data = b"test audio data"
        encoded = base64.b64encode(test_data).decode()
        result = standalone_model._decode_base64_to_wav_bytes(encoded)
        assert result == test_data


class TestLoadAudioToNp:
    """Test _load_audio_to_np method."""
    
    def test_load_audio_from_file(self, standalone_model):
        """Test loading audio from file."""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            # Create a simple audio file
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            audio_data = np.random.randn(samples).astype(np.float32)
            sf.write(temp_path, audio_data, sample_rate)
        
        try:
            audio, sr = standalone_model._load_audio_to_np(temp_path)
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert sr == sample_rate
            assert len(audio) == samples
        finally:
            os.unlink(temp_path)
    
    @patch('qwen_tts.inference.standalone_qwen3_tts_model.urllib.request.urlopen')
    @patch('qwen_tts.inference.standalone_qwen3_tts_model.sf.read')
    def test_load_audio_from_url(self, mock_sf_read, mock_urlopen, standalone_model):
        """Test loading audio from URL."""
        # Mock URL response
        mock_response = MagicMock()
        mock_response.read.return_value = b"fake audio bytes"
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Mock soundfile read
        mock_sf_read.return_value = (np.array([0.1, 0.2, 0.3], dtype=np.float32), 16000)
        
        audio, sr = standalone_model._load_audio_to_np("http://example.com/audio.wav")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 16000
        mock_urlopen.assert_called_once()
    
    @patch('qwen_tts.inference.standalone_qwen3_tts_model.sf.read')
    def test_load_audio_from_base64(self, mock_sf_read, standalone_model):
        """Test loading audio from base64."""
        import base64
        # Create a simple audio file and encode it
        audio_data = np.random.randn(16000).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_data, 16000)
        
        try:
            with open(temp_path, 'rb') as f:
                wav_bytes = f.read()
            
            encoded = base64.b64encode(wav_bytes).decode()
            base64_string = f"data:audio/wav;base64,{encoded}"
            
            # Mock sf.read to return the audio data
            mock_sf_read.return_value = (audio_data, 16000)
            
            audio, sr = standalone_model._load_audio_to_np(base64_string)
            
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert sr == 16000
        finally:
            os.unlink(temp_path)


class TestNormalizeAudioInputs:
    """Test _normalize_audio_inputs method."""
    
    def test_normalize_single_file_path(self, standalone_model):
        """Test normalizing a single file path."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(temp_path, audio_data, 16000)
        
        try:
            result = standalone_model._normalize_audio_inputs(temp_path)
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], tuple)
            audio, sr = result[0]
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert sr == 16000
        finally:
            os.unlink(temp_path)
    
    def test_normalize_list_of_paths(self, standalone_model):
        """Test normalizing a list of file paths."""
        temp_paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                    temp_paths.append(temp_path)
                    audio_data = np.random.randn(16000).astype(np.float32)
                    sf.write(temp_path, audio_data, 16000)
            
            result = standalone_model._normalize_audio_inputs(temp_paths)
            assert isinstance(result, list)
            assert len(result) == 2
            for audio, sr in result:
                assert isinstance(audio, np.ndarray)
                assert audio.dtype == np.float32
                assert sr == 16000
        finally:
            for path in temp_paths:
                os.unlink(path)
    
    def test_normalize_numpy_array_with_sr(self, standalone_model):
        """Test normalizing numpy array with sample rate."""
        audio_data = np.random.randn(16000).astype(np.float32)
        result = standalone_model._normalize_audio_inputs((audio_data, 16000))
        
        assert isinstance(result, list)
        assert len(result) == 1
        audio, sr = result[0]
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 16000
        np.testing.assert_array_equal(audio, audio_data)
    
    def test_normalize_list_of_numpy_arrays(self, standalone_model):
        """Test normalizing a list of numpy arrays."""
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(24000).astype(np.float32)
        
        result = standalone_model._normalize_audio_inputs([
            (audio1, 16000),
            (audio2, 24000)
        ])
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][1] == 16000
        assert result[1][1] == 24000
    
    def test_normalize_numpy_array_without_sr_raises(self, standalone_model):
        """Test that numpy array without sr raises ValueError."""
        audio_data = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="For numpy waveform input"):
            standalone_model._normalize_audio_inputs(audio_data)
    
    def test_normalize_unsupported_type_raises(self, standalone_model):
        """Test that unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported audio input type"):
            standalone_model._normalize_audio_inputs(123)
    
    def test_normalize_stereo_to_mono(self, standalone_model):
        """Test that stereo audio is converted to mono."""
        # Create stereo audio (2 channels)
        stereo_audio = np.random.randn(2, 16000).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, stereo_audio.T, 16000)  # sf expects (samples, channels)
        
        try:
            result = standalone_model._normalize_audio_inputs(temp_path)
            audio, sr = result[0]
            # Should be mono (1D array)
            assert audio.ndim == 1
            assert len(audio) == 16000
        finally:
            os.unlink(temp_path)
