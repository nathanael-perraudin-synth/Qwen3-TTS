"""
Simple test script for voice cloning with Qwen3 TTS using standalone models.
This version uses standalone models without transformers dependency.
"""
import argparse
import soundfile as sf
import torch
import os
import json

# Note: This is a simplified version that demonstrates using standalone models.
# For full functionality, you would need to implement model loading from pretrained weights.
# The standalone models are designed to work with the same weights as the original models.

def main():
    parser = argparse.ArgumentParser(
        description="Test voice cloning with Qwen3 TTS (Standalone Models)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help=(
            "Model checkpoint path or HuggingFace repo id "
            "(default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world",
        help="Target text to synthesize (default: Hello world)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="prompt.wav",
        help="Path to reference audio file (default: prompt.wav)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=(
            "Good morning, John. So when you told me you were going to the "
            "United Nations, I was like, oh my God, that's a lot of travel "
            "right before Pitt tonight. I was right, but not as much as I "
            "thought."
        ),
        help=(
            "Reference text (transcript of the audio). "
            "Optional if using x-vector-only mode."
        ),
    )
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use x-vector only mode (no ref_text needed, but lower quality)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_standalone.wav",
        help="Output audio file path (default: output_standalone.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to use (default: auto-detect, "
            "cuda:0 if available else cpu)"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Auto",
        help="Language for synthesis (default: Auto)",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("STANDALONE MODELS TEST")
    print("=" * 60)
    print("\nNote: This script demonstrates using standalone models.")
    print("For full functionality, you need to:")
    print("  1. Load pretrained weights into standalone models")
    print("  2. Implement from_pretrained for standalone models")
    print("  3. Handle model initialization and weight loading")
    print("\nThe standalone models are designed to be drop-in replacements")
    print("and can use the same weights as the original models.")
    print("=" * 60)
    
    print(f"\nLoading model from {args.checkpoint}...")
    print("(Using original model loading for now - standalone loading needs implementation)")
    
    # For now, use the original model loading
    # In a full implementation, you would:
    # 1. Load config from checkpoint
    # 2. Create standalone model instances
    # 3. Load weights into standalone models
    # 4. Use standalone models for inference
    
    from qwen_tts.inference.standalone_qwen3_tts_model import StandaloneQwen3TTSModel

    model = StandaloneQwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=device,
        dtype=dtype,
        attn_implementation=None,  # Use eager attention
    )

    print("Generating voice clone...")
    print(f"  Target text: {args.text}")
    print(f"  Reference audio: {args.audio}")
    print(f"  Reference text: {args.ref_text or '(none)'}")
    print(f"  X-vector only: {args.x_vector_only}")

    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
    )

    print(f"Saving output to {args.output}...")
    sf.write(args.output, wavs[0], sr)
    print("Done!")


if __name__ == "__main__":
    main()
