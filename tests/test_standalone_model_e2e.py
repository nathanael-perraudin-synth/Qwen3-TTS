"""End-to-end test comparing generate_voice_clone outputs between original and standalone models."""
import pytest
import torch
import numpy as np
from pathlib import Path
import random
import tempfile
import pickle
from qwen_tts.core.utils import set_all_seeds

# Skip this test if prompt.wav doesn't exist (it's needed for the test)
PROMPT_WAV = Path("prompt.wav")
pytestmark = [
    pytest.mark.skipif(
        not PROMPT_WAV.exists(),
        reason=f"prompt.wav not found at {PROMPT_WAV.absolute()}"
    ),
    pytest.mark.slow,  # Mark as slow test - requires GPU and long generation
]


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
    # Use shorter ref_text to reduce sequence length and memory usage
    return {
        "text": "Hello world",
        "ref_text": "Good morning, John.",  # Shorter text to reduce memory usage
        "audio": "prompt.wav",
        "language": "Auto",
        "x_vector_only": False,
    }


def test_generate_voice_clone_e2e_equivalence(device, test_params):
    """
    Test that original and standalone models produce identical generate_voice_clone outputs.
    
    This test uses a sequential approach to avoid OOM:
    1. Run original model and save output + weights
    2. Load standalone model with saved weights  
    3. Run standalone model and compare outputs
    
    This way only one model is in memory at a time.
    
    Note: This test may still encounter OOM errors during generation due to the sliding
    window attention mask creation for long sequences (seq_length > 1000). The mask creation
    tries to allocate (batch_size, seq_length, seq_length) which can be very large.
    
    The test structure is correct and will work once the mask creation is optimized
    or when run on a GPU with sufficient memory (>30GB recommended for full sequences).
    """
    # Skip if CUDA not available and device is CUDA
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Import models
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from qwen_tts.inference.standalone_qwen3_tts_model import StandaloneQwen3TTSModel
    from qwen_tts.core.models.standalone_top_level import StandaloneQwen3TTSForConditionalGeneration
    from qwen_tts.core.configs import to_standalone_configs_from_tts
    max_new_tokens = 64
    checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    # Test parameters
    text = test_params["text"]
    ref_text = test_params["ref_text"]
    audio = test_params["audio"]
    language = test_params["language"]
    x_vector_only = test_params["x_vector_only"]
    
    # Create temporary directory for saving results
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        original_output_path = tmpdir_path / "original_output.pkl"
        model_weights_path = tmpdir_path / "model_weights.pkl"
        config_path = tmpdir_path / "config.pkl"
        
        # ============================================================
        # STEP 1: Run original model and save output + weights
        # ============================================================
        print(f"\n{'='*60}")
        print("STEP 1: Running original model and saving results...")
        print(f"{'='*60}")
        
        # Load original model
        original_model_wrapper = Qwen3TTSModel.from_pretrained(
            checkpoint,
            device_map=str(device),
            dtype=dtype,
            attn_implementation=None,  # Use eager attention
        )
        original_model = original_model_wrapper.model
        
        # Generate with original model
        print("\nGenerating with original model...")
        set_all_seeds(42)
        original_wavs, original_sr = original_model_wrapper.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            max_new_tokens=max_new_tokens,  # Limit generation length for testing
        )
        
        # Save original output
        with open(original_output_path, 'wb') as f:
            pickle.dump({
                'wavs': original_wavs,
                'sr': original_sr,
            }, f)
        
        # Save model weights and config
        original_config = original_model.config
        model_weights = {
            'speaker_encoder': original_model.speaker_encoder.state_dict() if original_model.speaker_encoder is not None else None,
            'talker': original_model.talker.state_dict(),
        }
        
        with open(model_weights_path, 'wb') as f:
            pickle.dump(model_weights, f)
        
        # Save speech tokenizer reference (we'll reload it separately)
        speech_tokenizer = original_model.speech_tokenizer
        
        with open(config_path, 'wb') as f:
            pickle.dump({
                'original_config': original_config,
                'processor': original_model_wrapper.processor,
                'generate_defaults': original_model_wrapper.generate_defaults,
                'checkpoint': checkpoint,  # Save checkpoint path to reload speech tokenizer
            }, f)
        
        # Clear original model from memory (but keep speech_tokenizer reference)
        # We'll need to reload the speech tokenizer separately
        del original_model_wrapper
        del original_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # ============================================================
        # STEP 2: Load standalone model with saved weights
        # ============================================================
        print(f"\n{'='*60}")
        print("STEP 2: Creating standalone model and loading weights...")
        print(f"{'='*60}")
        
        # Load saved config
        with open(config_path, 'rb') as f:
            saved_data = pickle.load(f)
            original_config = saved_data['original_config']
            processor = saved_data['processor']
            generate_defaults = saved_data['generate_defaults']
            checkpoint = saved_data['checkpoint']
        
        # Create standalone model with same config
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
        
        # Load saved weights
        with open(model_weights_path, 'rb') as f:
            saved_weights = pickle.load(f)
        
        # Load weights into standalone model
        with torch.no_grad():
            if saved_weights['speaker_encoder'] is not None and standalone_model.speaker_encoder is not None:
                try:
                    standalone_model.speaker_encoder.load_state_dict(
                        saved_weights['speaker_encoder'], strict=False
                    )
                except Exception as e:
                    print(f"Warning: Could not load speaker encoder weights: {e}")
            
            # Load talker weights
            try:
                standalone_model.talker.load_state_dict(
                    saved_weights['talker'], strict=False
                )
            except Exception as e:
                print(f"Warning: Could not load all talker weights: {e}")
        
        # Load speech tokenizer - need to reload model temporarily to get it
        # This is still better than having both models in memory simultaneously
        print("\nLoading speech tokenizer (temporarily reloading model)...")
        temp_original = Qwen3TTSModel.from_pretrained(
            checkpoint,
            device_map=str(device),
            dtype=dtype,
            attn_implementation=None,
        )
        standalone_model.speech_tokenizer = temp_original.model.speech_tokenizer
        # Clear the temporary model but keep the speech tokenizer reference
        del temp_original.model
        del temp_original
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Create standalone wrapper
        standalone_wrapper = StandaloneQwen3TTSModel(
            model=standalone_model,
            processor=processor,
            generate_defaults=generate_defaults,
        )
        
        # ============================================================
        # STEP 3: Run standalone model and compare
        # ============================================================
        print(f"\n{'='*60}")
        print("STEP 3: Running standalone model and comparing...")
        print(f"{'='*60}")
        
        # Generate with standalone model
        print("\nGenerating with standalone model...")
        set_all_seeds(42)
        standalone_wavs, standalone_sr = standalone_wrapper.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            max_new_tokens=max_new_tokens,  # Limit generation length for testing
        )
        
        # Load original output for comparison
        with open(original_output_path, 'rb') as f:
            saved_output = pickle.load(f)
            original_wavs = saved_output['wavs']
            original_sr = saved_output['sr']
    
        # ============================================================
        # STEP 4: Compare outputs
        # ============================================================
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
            rms_diff = np.sqrt(np.mean((orig_wav - stand_wav) ** 2))
            
            print(f"\nOutput {i}:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  RMS difference: {rms_diff:.6f}")
            print(f"  Shape: {orig_wav.shape}")
            print(f"  Original range: [{orig_wav.min():.3f}, {orig_wav.max():.3f}]")
            print(f"  Standalone range: [{stand_wav.min():.3f}, {stand_wav.max():.3f}]")
            
            # Allow small numerical differences due to floating point operations
            # The models should produce very similar outputs
            # Using more lenient thresholds for end-to-end test
            assert max_diff < 1e-2, f"Max difference too large: {max_diff}"
            assert mean_diff < 1e-3, f"Mean difference too large: {mean_diff}"
            assert rms_diff < 1e-3, f"RMS difference too large: {rms_diff}"
        
        print(f"\n{'='*60}")
        print("✅ All outputs match within tolerance!")
        print(f"{'='*60}\n")
