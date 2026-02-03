# Qwen3-TTS Model Refactoring Tests

This directory contains tests for refactoring Qwen3-TTS models from transformers-based to pure PyTorch `nn.Module` implementations.

## Progress

### ✅ Completed: Qwen3TTSSpeakerEncoder

- **Status**: Fully completed
- **Files**:
  - `test_speaker_encoder.py`: Tests for the original model
  - `test_speaker_encoder_standalone.py`: Comparison tests between original and standalone versions
- **Standalone Implementation**: `qwen_tts/core/models/standalone_speaker_encoder.py`
- **Config**: `qwen_tts/core/models/standalone_config.py`
- **Tests**: All 7 tests passing ✅

### 🔄 In Progress: Remaining Models

The following models still need to be refactored:

1. **Qwen3TTSTalkerCodePredictorModel** - Complex transformer decoder model
   - Dependencies: `PreTrainedModel`, `BaseModelOutputWithPast`, `Cache`, `DynamicCache`, `create_causal_mask`, `create_sliding_window_causal_mask`
   - Components: `Qwen3TTSDecoderLayer`, `Qwen3TTSRMSNorm`, `Qwen3TTSRotaryEmbedding`

2. **Qwen3TTSTalkerModel** - Main talker transformer model
   - Dependencies: Similar to CodePredictorModel
   - Components: `Qwen3TTSTalkerDecoderLayer`, `Qwen3TTSTalkerRotaryEmbedding`

3. **Qwen3TTSForConditionalGeneration** - Top-level conditional generation model
   - Dependencies: `GenerationMixin`, all previous models
   - Components: Combines TalkerModel and SpeakerEncoder

## Testing Strategy

For each model:
1. Create test file to test the original model independently
2. Create standalone version without transformers
3. Create comparison test to ensure identical outputs
4. Test with `uv run python test.py` to verify end-to-end functionality

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_speaker_encoder.py -v

# Run with coverage
uv run pytest tests/ --cov=qwen_tts/core/models
```
