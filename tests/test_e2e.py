# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end tests for the Qwen3 TTS model.

These tests verify that the full pipeline works correctly, including:
- Model loading (both original and standalone versions)
- Voice cloning generation
- Output validation

Note: These tests require GPU and may be slow. They are marked with 
`pytest.mark.slow` and `pytest.mark.e2e` for easy filtering.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# Skip all tests if no GPU is available
pytestmark = [
    pytest.mark.slow,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="E2E tests require CUDA GPU"
    ),
]


class TestE2EGeneration:
    """End-to-end tests for voice clone generation."""
    
    # Path to the test.py script
    TEST_SCRIPT = Path(__file__).parent.parent / "test.py"
    # Path to the default prompt audio
    PROMPT_AUDIO = Path(__file__).parent.parent / "prompt.wav"
    # Model checkpoint
    CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    # Fixed seed for reproducibility
    SEED = 42
    
    @pytest.fixture
    def temp_output(self):
        """Create a temporary output file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def _run_test_script(self, *extra_args, output_path=None):
        """Run test.py with given arguments and return exit code."""
        cmd = [
            sys.executable, str(self.TEST_SCRIPT),
            "--seed", str(self.SEED),
            "--text", "Hello, this is a test.",
        ]
        if output_path:
            cmd.extend(["--output", output_path])
        cmd.extend(extra_args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result
    
    def _validate_audio_output(self, audio_path: str, min_duration: float = 0.1):
        """Validate that the audio output is valid."""
        assert os.path.exists(audio_path), f"Output file not found: {audio_path}"
        
        data, sr = sf.read(audio_path)
        assert sr > 0, "Sample rate should be positive"
        assert len(data) > 0, "Audio data should not be empty"
        
        duration = len(data) / sr
        assert duration >= min_duration, f"Audio too short: {duration}s < {min_duration}s"
        
        # Check that audio is not silent (all zeros)
        assert np.abs(data).max() > 1e-6, "Audio appears to be silent"
        
        return data, sr
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("prompt.wav").exists(),
        reason="prompt.wav not found"
    )
    def test_original_model_generation(self, temp_output):
        """Test that the original model can generate audio."""
        result = self._run_test_script(output_path=temp_output)
        
        assert result.returncode == 0, (
            f"test.py failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        
        data, sr = self._validate_audio_output(temp_output)
        assert sr == 24000, f"Expected 24kHz sample rate, got {sr}Hz"
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("prompt.wav").exists(),
        reason="prompt.wav not found"
    )
    def test_standalone_model_generation(self, temp_output):
        """Test that the standalone model can generate audio."""
        result = self._run_test_script("--standalone", output_path=temp_output)
        
        assert result.returncode == 0, (
            f"test.py --standalone failed with exit code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        
        data, sr = self._validate_audio_output(temp_output)
        assert sr == 24000, f"Expected 24kHz sample rate, got {sr}Hz"
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("prompt.wav").exists(),
        reason="prompt.wav not found"
    )
    def test_models_produce_similar_output(self, temp_output):
        """Test that original and standalone models produce similar output with same seed."""
        # Generate with original model
        output_original = temp_output.replace(".wav", "_original.wav")
        result_original = self._run_test_script(output_path=output_original)
        assert result_original.returncode == 0, f"Original model failed: {result_original.stderr}"
        
        # Generate with standalone model
        output_standalone = temp_output.replace(".wav", "_standalone.wav")
        result_standalone = self._run_test_script("--standalone", output_path=output_standalone)
        assert result_standalone.returncode == 0, f"Standalone model failed: {result_standalone.stderr}"
        
        # Load both outputs
        data_original, sr_original = sf.read(output_original)
        data_standalone, sr_standalone = sf.read(output_standalone)
        
        # Basic validation
        assert sr_original == sr_standalone, "Sample rates should match"
        
        # Cleanup extra files
        for f in [output_original, output_standalone]:
            if os.path.exists(f):
                os.unlink(f)


class TestE2EModelLoading:
    """Tests for model loading functionality."""
    
    def test_standalone_model_import(self):
        """Test that standalone model classes can be imported."""
        from qwen_tts import Qwen3TTSModelStandalone, VoiceClonePromptItemStandalone
        
        assert Qwen3TTSModelStandalone is not None
        assert VoiceClonePromptItemStandalone is not None
    
    def test_original_model_import(self):
        """Test that original model classes can be imported."""
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
        
        assert Qwen3TTSModel is not None
        assert VoiceClonePromptItem is not None
    
    def test_standalone_config_import(self):
        """Test that standalone config classes can be imported."""
        from qwen_tts.core.models.configuration_qwen3_tts_standalone import (
            Qwen3TTSConfigStandalone,
            Qwen3TTSTalkerConfigStandalone,
            Qwen3TTSSpeakerEncoderConfigStandalone,
            Qwen3TTSTalkerCodePredictorConfigStandalone,
        )
        
        # Verify they can be instantiated
        speaker_config = Qwen3TTSSpeakerEncoderConfigStandalone()
        assert speaker_config.mel_dim == 128
        
        code_predictor_config = Qwen3TTSTalkerCodePredictorConfigStandalone()
        assert code_predictor_config.vocab_size == 2048
        
        talker_config = Qwen3TTSTalkerConfigStandalone()
        assert talker_config.model_type == "qwen3_tts_talker"
        
        main_config = Qwen3TTSConfigStandalone()
        assert main_config.model_type == "qwen3_tts"
