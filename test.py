#!/usr/bin/env python3
"""
Simple test script for voice cloning with Qwen3 TTS.
"""

import argparse
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(
        description="Test voice cloning with Qwen3 TTS"
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
        default="output.wav",
        help="Output audio file path (default: output.wav)",
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

    # Toggle flash attention based on GPU availability (CUDA only)
    use_gpu = device.startswith("cuda") if isinstance(device, str) else False
    if use_gpu:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = None
        if device != "cpu":
            print("Flash attention disabled (not available on MPS)")
        else:
            print("Flash attention disabled (no CUDA GPU detected)")

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.checkpoint}...")
    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
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
