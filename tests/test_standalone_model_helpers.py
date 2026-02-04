"""Unit tests for StandaloneQwen3TTSModel helper methods."""
import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from qwen_tts.inference.standalone_qwen3_tts_model import StandaloneQwen3TTSModel


@pytest.fixture
def mock_model():
    """Create a mock standalone model."""
    model = Mock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    model.tts_model_type = "base"
    model.tokenizer_type = "qwen3_tts_tokenizer_12hz"
    model.tts_model_size = "1.7b"
    model.speaker_encoder_sample_rate = 24000
    model.speech_tokenizer = Mock()
    model.get_supported_languages = Mock(return_value=["en", "zh", "auto"])
    model.get_supported_speakers = Mock(return_value=["default"])
    return model


@pytest.fixture
def mock_processor():
    """Create a mock processor."""
    processor = Mock()
    processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    return processor


@pytest.fixture
def standalone_model(mock_model, mock_processor):
    """Create a StandaloneQwen3TTSModel instance for testing."""
    return StandaloneQwen3TTSModel(
        model=mock_model,
        processor=mock_processor,
        generate_defaults={}
    )


class TestEnsureList:
    """Test _ensure_list method."""
    
    def test_ensure_list_with_list(self, standalone_model):
        """Test that list input is returned as-is."""
        result = standalone_model._ensure_list([1, 2, 3])
        assert result == [1, 2, 3]
        assert isinstance(result, list)
    
    def test_ensure_list_with_string(self, standalone_model):
        """Test that string input is converted to list."""
        result = standalone_model._ensure_list("hello")
        assert result == ["hello"]
        assert isinstance(result, list)
    
    def test_ensure_list_with_int(self, standalone_model):
        """Test that int input is converted to list."""
        result = standalone_model._ensure_list(42)
        assert result == [42]
        assert isinstance(result, list)
    
    def test_ensure_list_with_none(self, standalone_model):
        """Test that None input is converted to list."""
        result = standalone_model._ensure_list(None)
        assert result == [None]
        assert isinstance(result, list)


class TestBuildText:
    """Test text building methods."""
    
    def test_build_assistant_text(self, standalone_model):
        """Test _build_assistant_text."""
        text = "Hello world"
        result = standalone_model._build_assistant_text(text)
        expected = "<|im_start|>assistant\nHello world<|im_end|>\n<|im_start|>assistant\n"
        assert result == expected
        assert result.startswith("<|im_start|>assistant\n")
        assert result.endswith("<|im_start|>assistant\n")
    
    def test_build_ref_text(self, standalone_model):
        """Test _build_ref_text."""
        text = "Reference text"
        result = standalone_model._build_ref_text(text)
        expected = "<|im_start|>assistant\nReference text<|im_end|>\n"
        assert result == expected
        assert result.startswith("<|im_start|>assistant\n")
        assert result.endswith("<|im_end|>\n")
    
    def test_build_instruct_text(self, standalone_model):
        """Test _build_instruct_text."""
        instruct = "Make it sound happy"
        result = standalone_model._build_instruct_text(instruct)
        expected = "<|im_start|>user\nMake it sound happy<|im_end|>\n"
        assert result == expected
        assert result.startswith("<|im_start|>user\n")
        assert result.endswith("<|im_end|>\n")


class TestValidation:
    """Test validation methods."""
    
    def test_validate_languages_supported(self, standalone_model):
        """Test _validate_languages with supported languages."""
        # Should not raise
        standalone_model._validate_languages(["en", "zh"])
        standalone_model._validate_languages(["auto"])
    
    def test_validate_languages_unsupported(self, standalone_model):
        """Test _validate_languages with unsupported languages."""
        with pytest.raises(ValueError, match="Unsupported languages"):
            standalone_model._validate_languages(["fr", "de"])
    
    def test_validate_languages_none_supported(self, standalone_model):
        """Test _validate_languages when model returns None."""
        standalone_model.model.get_supported_languages.return_value = None
        # Should not raise when supported is None
        standalone_model._validate_languages(["any", "language"])
    
    def test_validate_speakers_supported(self, standalone_model):
        """Test _validate_speakers with supported speakers."""
        # Should not raise
        standalone_model._validate_speakers(["default"])
        standalone_model._validate_speakers([None, "default"])
    
    def test_validate_speakers_unsupported(self, standalone_model):
        """Test _validate_speakers with unsupported speakers."""
        with pytest.raises(ValueError, match="Unsupported speakers"):
            standalone_model._validate_speakers(["unknown_speaker"])
    
    def test_validate_speakers_none_supported(self, standalone_model):
        """Test _validate_speakers when model returns None."""
        standalone_model.model.get_supported_speakers.return_value = None
        # Should not raise when supported is None
        standalone_model._validate_speakers(["any", "speaker"])


class TestMergeGenerateKwargs:
    """Test _merge_generate_kwargs method."""
    
    def test_merge_with_user_values(self, standalone_model):
        """Test merging with user-provided values."""
        result = standalone_model._merge_generate_kwargs(
            do_sample=False,
            top_k=10,
            temperature=0.5,
            max_new_tokens=512
        )
        assert result["do_sample"] is False
        assert result["top_k"] == 10
        assert result["temperature"] == 0.5
        assert result["max_new_tokens"] == 512
    
    def test_merge_with_defaults(self, standalone_model):
        """Test merging with default values."""
        result = standalone_model._merge_generate_kwargs()
        assert result["do_sample"] is True  # hard default
        assert result["top_k"] == 50  # hard default
        assert result["top_p"] == 1.0  # hard default
        assert result["temperature"] == 0.9  # hard default
        assert result["max_new_tokens"] == 2048  # hard default
    
    def test_merge_with_generate_defaults(self, standalone_model):
        """Test merging with generate_defaults from model."""
        standalone_model.generate_defaults = {
            "do_sample": False,
            "top_k": 20,
            "temperature": 0.7
        }
        result = standalone_model._merge_generate_kwargs()
        assert result["do_sample"] is False  # from generate_defaults
        assert result["top_k"] == 20  # from generate_defaults
        assert result["temperature"] == 0.7  # from generate_defaults
        assert result["top_p"] == 1.0  # hard default (not in generate_defaults)
    
    def test_merge_user_overrides_defaults(self, standalone_model):
        """Test that user values override generate_defaults."""
        standalone_model.generate_defaults = {
            "do_sample": False,
            "top_k": 20
        }
        result = standalone_model._merge_generate_kwargs(
            do_sample=True,  # User overrides
            top_k=30  # User overrides
        )
        assert result["do_sample"] is True  # user value
        assert result["top_k"] == 30  # user value
        assert result["temperature"] == 0.9  # hard default
    
    def test_merge_with_extra_kwargs(self, standalone_model):
        """Test merging with extra kwargs."""
        result = standalone_model._merge_generate_kwargs(
            custom_param="value",
            another_param=123
        )
        assert result["custom_param"] == "value"
        assert result["another_param"] == 123
        assert result["do_sample"] is True  # still has defaults


class TestPromptItemsToVoiceClonePrompt:
    """Test _prompt_items_to_voice_clone_prompt method."""
    
    def test_prompt_items_to_dict(self, standalone_model):
        """Test converting prompt items to dict."""
        from qwen_tts.inference.standalone_qwen3_tts_model import VoiceClonePromptItem
        
        items = [
            VoiceClonePromptItem(
                ref_code=torch.tensor([1, 2, 3]),
                ref_spk_embedding=torch.tensor([0.1, 0.2, 0.3]),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text="Reference text"
            ),
            VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=torch.tensor([0.4, 0.5, 0.6]),
                x_vector_only_mode=True,
                icl_mode=False,
                ref_text=None
            )
        ]
        
        result = standalone_model._prompt_items_to_voice_clone_prompt(items)
        
        assert isinstance(result, dict)
        assert "ref_code" in result
        assert "ref_spk_embedding" in result
        assert "x_vector_only_mode" in result
        assert "icl_mode" in result
        
        assert len(result["ref_code"]) == 2
        assert result["ref_code"][0] is not None
        assert result["ref_code"][1] is None
        
        assert len(result["ref_spk_embedding"]) == 2
        assert len(result["x_vector_only_mode"]) == 2
        assert result["x_vector_only_mode"][0] is False
        assert result["x_vector_only_mode"][1] is True
        
        assert len(result["icl_mode"]) == 2
        assert result["icl_mode"][0] is True
        assert result["icl_mode"][1] is False
