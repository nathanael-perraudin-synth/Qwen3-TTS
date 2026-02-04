"""Test that generate_voice_clone produces identical outputs for original and standalone models."""
import pytest
import torch
import numpy as np
import os
from pathlib import Path
from qwen_tts.core.copy_weights import copy_model_weights

# Skip this test if prompt.wav doesn't exist (it's needed for the test)
PROMPT_WAV = Path("prompt.wav")
pytestmark = pytest.mark.skipif(
    not PROMPT_WAV.exists(),
    reason=f"prompt.wav not found at {PROMPT_WAV.absolute()}"
)


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Set Python random seed if needed
    import random
    random.seed(seed)
    # Set deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module")
def device():
    """Get device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="module")
def test_params():
    """Test parameters matching test.py defaults."""
    return {
        "text": "Hello world",
        "ref_text": (
            "Good morning, John. So when you told me you were going to the "
            "United Nations, I was like, oh my God, that's a lot of travel "
            "right before Pitt tonight. I was right, but not as much as I "
            "thought."
        ),
        "audio": "prompt.wav",
        "language": "Auto",
        "x_vector_only": False,
    }


def test_generate_voice_clone_equivalence(device, test_params):
    """Test that original and standalone models produce identical generate_voice_clone outputs."""
    # Skip if CUDA not available and device is CUDA
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Import models
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.core.models.standalone_top_level import StandaloneQwen3TTSForConditionalGeneration
    from qwen_tts.core.configs import Qwen3TTSConfig, to_standalone_configs_from_tts
    
    checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    print(f"\n{'='*60}")
    print("Loading original model...")
    print(f"{'='*60}")
    
    # Load original model
    original_model_wrapper = Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=str(device),
        dtype=dtype,
        attn_implementation=None,  # Use eager attention
    )
    original_model = original_model_wrapper.model
    
    print(f"\n{'='*60}")
    print("Creating standalone model...")
    print(f"{'='*60}")
    
    # Create standalone model with same config
    original_config = original_model.config
    talker_config, speaker_encoder_config = to_standalone_configs_from_tts(original_config)
    
    standalone_model = StandaloneQwen3TTSForConditionalGeneration(
        talker_config=talker_config,
        speaker_encoder_config=speaker_encoder_config,
        tts_model_type=original_config.tts_model_type,
        tokenizer_type=original_config.tokenizer_type,
        tts_model_size=original_config.tts_model_size,
        tts_bos_token_id=original_config.tts_bos_token_id,
        tts_eos_token_id=original_config.tts_eos_token_id,
        tts_pad_token_id=original_config.tts_pad_token_id,
    ).to(device).to(dtype)
    standalone_model.eval()
    
    print(f"\n{'='*60}")
    print("Copying weights from original to standalone...")
    print(f"{'='*60}")
    
    # Copy weights
    copy_model_weights(original_model, standalone_model)
    
    # Create a wrapper for standalone model that mimics Qwen3TTSModel interface
    class StandaloneModelWrapper:
        def __init__(self, model, processor, speech_tokenizer, original_wrapper):
            self.model = model
            self.processor = processor
            self.speech_tokenizer = speech_tokenizer
            self.device = device
            self.original_wrapper = original_wrapper  # For helper methods
        
        def _ensure_list(self, x):
            """Helper to ensure list."""
            return x if isinstance(x, list) else [x]
        
        def _build_assistant_text(self, text):
            """Build assistant text format."""
            return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        def _build_ref_text(self, text):
            """Build reference text format."""
            return f"<|im_start|>assistant\n{text}<|im_end|>\n"
        
        def _tokenize_texts(self, texts):
            """Tokenize texts."""
            input_ids = []
            for text in texts:
                input = self.processor(text=text, return_tensors="pt", padding=True)
                input_id = input["input_ids"].to(self.device)
                input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
                input_ids.append(input_id)
            return input_ids
        
        def _merge_generate_kwargs(self, **kwargs):
            """Merge generation kwargs with defaults."""
            # Use same defaults as original
            defaults = {
                "do_sample": True,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "subtalker_dosample": True,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
                "subtalker_temperature": 0.9,
                "max_new_tokens": 2048,
            }
            merged = {**defaults, **kwargs}
            return merged
        
        def generate_voice_clone(self, text, language, ref_audio, ref_text, x_vector_only_mode=False, **kwargs):
            """Generate voice clone using standalone model."""
            # Use same logic as original wrapper but with standalone model
            texts = self._ensure_list(text)
            languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
            if len(languages) == 1 and len(texts) > 1:
                languages = languages * len(texts)
            
            # Create voice clone prompt using original wrapper (same processing)
            voice_clone_prompt_items = self.original_wrapper.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            if len(voice_clone_prompt_items) == 1 and len(texts) > 1:
                voice_clone_prompt_items = voice_clone_prompt_items * len(texts)
            
            # Convert to dict format
            voice_clone_prompt_dict = {
                "ref_code": [it.ref_code for it in voice_clone_prompt_items],
                "ref_spk_embedding": [it.ref_spk_embedding for it in voice_clone_prompt_items],
                "x_vector_only_mode": [it.x_vector_only_mode for it in voice_clone_prompt_items],
                "icl_mode": [it.icl_mode for it in voice_clone_prompt_items],
            }
            ref_texts_for_ids = [it.ref_text for it in voice_clone_prompt_items]
            
            # Tokenize input texts
            input_texts = [self._build_assistant_text(t) for t in texts]
            input_ids = self._tokenize_texts(input_texts)
            
            # Tokenize ref texts if needed
            ref_ids = None
            if ref_texts_for_ids is not None:
                ref_ids = []
                for rt in ref_texts_for_ids:
                    if rt is None or rt == "":
                        ref_ids.append(None)
                    else:
                        ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                        ref_ids.append(ref_tok)
            
            # Merge generation kwargs
            gen_kwargs = self._merge_generate_kwargs(**kwargs)
            
            # Generate using standalone model
            talker_codes_list, _ = self.model.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt_dict,
                languages=languages,
                non_streaming_mode=False,
                **gen_kwargs,
            )
            
            # Prepare codes for decoding
            codes_for_decode = []
            for i, codes in enumerate(talker_codes_list):
                ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
                if ref_code_list is not None and ref_code_list[i] is not None:
                    codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
                else:
                    codes_for_decode.append(codes)
            
            # Decode using speech tokenizer
            wavs_all, fs = self.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])
            
            # Process outputs (same as original)
            wavs_out = []
            for i, wav in enumerate(wavs_all):
                ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
                if ref_code_list is not None and ref_code_list[i] is not None:
                    ref_len = int(ref_code_list[i].shape[0])
                    total_len = int(codes_for_decode[i].shape[0])
                    cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                    wavs_out.append(wav[cut:])
                else:
                    wavs_out.append(wav)
            
            return wavs_out, fs
    
    standalone_wrapper = StandaloneModelWrapper(
        model=standalone_model,
        processor=original_model_wrapper.processor,
        speech_tokenizer=original_model_wrapper.model.speech_tokenizer,
        original_wrapper=original_model_wrapper,
    )
    
    print(f"\n{'='*60}")
    print("Running generation with fixed seeds...")
    print(f"{'='*60}")
    
    # Test parameters
    text = test_params["text"]
    ref_text = test_params["ref_text"]
    audio = test_params["audio"]
    language = test_params["language"]
    x_vector_only = test_params["x_vector_only"]
    
    # Generate with original model
    print("\nGenerating with original model...")
    set_all_seeds(42)
    # Limit max_new_tokens to avoid OOM and speed up test
    original_wavs, original_sr = original_model_wrapper.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
        max_new_tokens=512,  # Limit generation length for testing
    )
    
    # Generate with standalone model
    print("\nGenerating with standalone model...")
    set_all_seeds(42)
    # Use same parameters
    standalone_wavs, standalone_sr = standalone_wrapper.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only,
        max_new_tokens=512,  # Limit generation length for testing
    )
    
    print(f"\n{'='*60}")
    print("Comparing outputs...")
    print(f"{'='*60}")
    
    # Compare outputs
    assert original_sr == standalone_sr, f"Sample rates differ: {original_sr} vs {standalone_sr}"
    assert len(original_wavs) == len(standalone_wavs), f"Number of outputs differs: {len(original_wavs)} vs {len(standalone_wavs)}"
    
    # Compare audio waveforms
    for i, (orig_wav, stand_wav) in enumerate(zip(original_wavs, standalone_wavs)):
        assert orig_wav.shape == stand_wav.shape, f"Waveform shapes differ for output {i}: {orig_wav.shape} vs {stand_wav.shape}"
        
        # Compare with tolerance (some numerical differences may occur)
        max_diff = np.abs(orig_wav - stand_wav).max()
        mean_diff = np.abs(orig_wav - stand_wav).mean()
        
        print(f"\nOutput {i}:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Shape: {orig_wav.shape}")
        
        # Allow small numerical differences due to floating point operations
        # The models should produce very similar outputs
        assert max_diff < 1e-3, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-4, f"Mean difference too large: {mean_diff}"
    
    print(f"\n{'='*60}")
    print("✅ All outputs match!")
    print(f"{'='*60}\n")
