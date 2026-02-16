# Qwen3-TTS: Simplified Standalone Implementation

<br>


**Qwen3-TTS** is a powerful speech generation system offering comprehensive support for voice cloning, voice design, ultra-high-quality human-like speech synthesis, and natural language-based voice control. This is a fork from the [original repo](https://github.com/QwenLM/Qwen3-TTS).

## What Makes This Implementation Special?

This repository provides a **simplified, standalone, human-friendly** implementation of Qwen3-TTS with the following improvements:

- **Standalone Architecture**: Minimal dependencies, easier to understand and modify
- **Simplified Codebase**: Refactored for clarity without sacrificing functionality
- **Fine-tuning Ready**: Built-in training functions for easy model customization
- **Full Compatibility**: Drop-in replacement for the original implementation with identical outputs
- **Comprehensive Tests**: 191+ tests ensuring reliability and equivalence with the original

Perfect for researchers, developers, and practitioners who want to:
- Understand the internals of modern TTS systems
- Fine-tune models on custom datasets
- Integrate TTS into production applications
- Experiment with neural speech synthesis


## News
* 2026.1.22: 🎉🎉🎉 We have released [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) series (0.6B/1.7B) based on Qwen3-TTS-Tokenizer-12Hz. Please check our [blog](https://qwen.ai/blog?id=qwen3tts-0115)!

## Why This Standalone Version?

### Simplification & Clarity
The original Qwen3-TTS implementation is tightly coupled with the HuggingFace Transformers library, making it challenging to understand, modify, or extend. This standalone version:

- **Removes unnecessary abstractions** - Direct, readable PyTorch code
- **Standalone components** - Each module (Talker, CodePredictor, SpeakerEncoder, Tokenizer) is self-contained
- **Clear architecture** - Easy to trace data flow from text input to audio output
- **Minimal dependencies** - Only PyTorch, no heavy framework requirements

### Fine-tuning & Customization
This implementation adds full fine-tuning support that wasn't easily accessible before:

- `forward_sub_talker_finetune()` - Train the CodePredictor on custom data
- `CodePredictor.forward_finetune()` - Direct access to training path
- Loss computation utilities - Standard causal LM loss with label shifting
- Training examples - See `tests/test_training.py` for complete examples

### Quality Assurance
Extensive testing ensures this simplified version maintains equivalence with the original:

- **191+ comprehensive tests** across 14 test files
- **Equivalence validation** - Numerically identical outputs to original implementation
- **Weight compatibility** - Load pretrained weights from official Qwen models
- **Training verification** - Overfitting tests confirm gradient flow

### Use Cases

This implementation is ideal for:

1. **Researchers** - Understand and experiment with modern TTS architectures
2. **ML Engineers** - Fine-tune models on domain-specific data (customer service voices, character voices, etc.)
3. **Production Teams** - Deploy with minimal dependencies and clear code paths
4. **Students** - Learn speech synthesis without framework complexity

### Comparison: Standalone vs Original

| Feature | Original Implementation | This Standalone Version |
|---------|------------------------|-------------------------|
| **Dependencies** | HuggingFace Transformers required | Minimal (PyTorch only for core) |
| **Code Complexity** | Tightly coupled with HF abstractions | Direct PyTorch, easy to understand |
| **Fine-tuning Support** | Requires custom training loops | Built-in `forward_finetune()` methods |
| **Model Loading** | HF AutoModel pattern | Simple `from_pretrained()` wrapper |
| **Testing** | Limited test coverage | 191+ comprehensive tests |
| **Equivalence** | N/A | Numerically identical outputs |
| **Documentation** | Scattered across HF docs | Self-contained in codebase |
| **Extensibility** | Modify HF framework | Modify straightforward PyTorch |
| **Production Ready** | Framework overhead | Lightweight deployment |
| **Learning Curve** | Steep (HF + TTS concepts) | Gentle (just TTS concepts) |

## Contents <!-- omit in toc -->

- [News](#news)
- [Why This Standalone Version?](#why-this-standalone-version)
  - [Simplification & Clarity](#simplification--clarity)
  - [Fine-tuning & Customization](#fine-tuning--customization)
  - [Quality Assurance](#quality-assurance)
  - [Use Cases](#use-cases)
  - [Comparison: Standalone vs Original](#comparison-standalone-vs-original)
- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Standalone Implementation Details](#standalone-implementation-details)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Quick Start Example](#quick-start-example)
  - [Python Package Usage](#python-package-usage)
    - [Custom Voice Generation](#custom-voice-generate)
    - [Voice Design](#voice-design)
    - [Voice Clone](#voice-clone)
      - [Option 1: `Qwen3TTSModel.generate_voice_clone()` (full-featured)](#option-1-qwen3ttsmodelgenerate_voice_clone-full-featured)
      - [Option 2: `VoiceCloner` (simplified, standalone)](#option-2-voicecloner-simplified-standalone)
    - [Voice Design then Clone](#voice-design-then-clone)
    - [Tokenizer Encode and Decode](#tokenizer-encode-and-decode)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [DashScope API Usage](#dashscope-api-usage)
- [vLLM Usage](#vllm-usage)
- [Fine Tuning](#fine-tuning)
  - [Training Functions](#training-functions)
  - [Training Features](#training-features)
- [Testing](#testing)
  - [Test Coverage](#test-coverage-191-tests)
  - [Running Tests](#running-tests)
  - [Test Features](#test-features)
  - [Test Organization](#test-organization)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
  - [Priority Areas](#priority-areas)
  - [How to Contribute](#how-to-contribute)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## Quickstart

### Environment Setup

Clone this repo and install in editable mode:

```bash
git clone https://github.com/nathanael-perraudin-synth/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

Or using [uv](https://docs.astral.sh/uv/) for faster dependency resolution:

```bash
git clone https://github.com/nathanael-perraudin-synth/Qwen3-TTS.git
cd Qwen3-TTS
uv pip install -e .
```

This installs the `qwen-tts` package and all its dependencies. The standalone module `qwen3_tts_standalone` is importable directly from the repository root.



This standalone implementation has minimal dependencies:
- **Core**: PyTorch (required)
- **Audio**: librosa, soundfile (for audio I/O)
- **Optional**: flash-attn (for memory efficiency)
- **Optional**: transformers (only for equivalence tests)

The core model code (`qwen3_tts_standalone/`) has zero dependency on HuggingFace Transformers. The original model remains in `qwen3_tts`.

#### Optional: Flash Attention 2

We recommend using FlashAttention 2 to reduce GPU memory usage:

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

FlashAttention 2 requires compatible hardware and works with `torch.float16` or `torch.bfloat16`. Read more in the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention).

### Quick Start Example

Here's how simple it is to generate speech with this standalone implementation:

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import Qwen3TTSModel

# Load model (automatic download from HuggingFace)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Optional: use Flash Attention
)

# Generate speech
wavs, sr = model.generate_custom_voice(
    text="Hello! This is a simplified standalone implementation of Qwen3-TTS.",
    language="English",
    speaker="Ryan",
    instruct="Speak with enthusiasm and clarity."  # Optional instruction
)

# Save audio
sf.write("output.wav", wavs[0], sr)
```

That's it! 

### Python Package Usage

After installation, you can import `Qwen3TTSModel` from `qwen3_tts_standalone` to run custom voice TTS, voice design, and voice clone. The model weights can be specified either as a Hugging Face model id (recommended) or as a local directory path you downloaded. For all the `generate_*` functions below, besides the parameters shown and explicitly documented, you can also pass generation kwargs such as `max_new_tokens`, `top_p`, etc.

#### Custom Voice Generate

For custom voice models (`Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice`), you just need to call `generate_custom_voice`, passing a single string or a batch list, along with `language`, `speaker`, and optional `instruct`. You can also call `model.get_supported_speakers()` and `model.get_supported_languages()` to see which speakers and languages the current model supports.

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
```

For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker’s native language for the best quality. Of course, each speaker can speak any language supported by the model.

| Speaker | Voice Description  |  Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

#### Voice Design

For the voice design model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`), you can use `generate_voice_design` to provide the target text and a natural-language `instruct` description.

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
sf.write("output_voice_design.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]
)
sf.write("output_voice_design_1.wav", wavs[0], sr)
sf.write("output_voice_design_2.wav", wavs[1], sr)
```

#### Voice Clone

This standalone implementation provides **two voice cloning APIs** for the Base model (`Qwen3-TTS-12Hz-1.7B/0.6B-Base`), each suited to different use cases:

| API | Class | Batch | Modes | Best for |
|-----|-------|-------|-------|----------|
| `Qwen3TTSModel.generate_voice_clone()` | `Qwen3TTSModel` | Yes | ICL + x-vector-only | Production use, batch inference, reusable prompts |
| `VoiceCloner.clone_voice()` | `VoiceCloner` | No | ICL only | Simplicity, learning, customization |

##### Option 1: `Qwen3TTSModel.generate_voice_clone()` (full-featured)

The `Qwen3TTSModel` wrapper provides the full voice cloning API with batch support, reusable prompts, and both ICL and x-vector-only modes. Provide a reference audio clip (`ref_audio`) along with its transcript (`ref_text`). `ref_audio` can be a local file path, a URL, a base64 string, or a `(numpy_array, sample_rate)` tuple. If you set `x_vector_only_mode=True`, only the speaker embedding is used so `ref_text` is not required, but cloning quality may be reduced.

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
```

If you need to reuse the same reference prompt across multiple generations (to avoid recomputing prompt features), build it once with `create_voice_clone_prompt` and pass it via `voice_clone_prompt`.

```python
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items,
)
sf.write("output_voice_clone_1.wav", wavs[0], sr)
sf.write("output_voice_clone_2.wav", wavs[1], sr)
```

For more examples of reusable voice clone prompts, batch cloning, and batch inference, please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py). With those examples and the `generate_voice_clone` function description, you can explore more advanced usage patterns.

##### Option 2: `VoiceCloner` (simplified, standalone)

The `VoiceCloner` class provides a simpler, more explicit voice cloning API using In-Context Learning (ICL). It loads and manages the individual model components (Talker, SpeakerEncoder, SpeechTokenizer) directly, making the generation pipeline transparent and easy to understand or modify.

Key differences from `Qwen3TTSModel.generate_voice_clone()`:
- **Single-sample only** -- no batch inference (returns a single `np.ndarray` instead of a list)
- **ICL mode only** -- no x-vector-only mode
- **Self-contained** -- loads components individually via its own `from_pretrained()`
- **Explicit pipeline** -- the ICL prompt construction logic is visible and hackable

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import VoiceCloner

cloner = VoiceCloner.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

audio, sr = cloner.clone_voice(
    text="Hello, how are you today?",
    ref_audio="speaker_sample.wav",
    ref_text="This is a sample of my voice.",
    language="English",
)
sf.write("output.wav", audio, sr)
```

`VoiceCloner.clone_voice()` also accepts generation parameters directly:

```python
audio, sr = cloner.clone_voice(
    text="Let me explain the concept step by step.",
    ref_audio="speaker_sample.wav",
    ref_text="This is a sample of my voice.",
    language="English",
    max_new_tokens=4096,
    temperature=0.9,
    top_k=50,
    repetition_penalty=1.05,
)
```

The `VoiceCloner` is ideal if you want to understand or modify the voice cloning internals -- for example, to change how the ICL prompt is constructed or to integrate custom speaker embedding logic.

#### Voice Design then Clone

If you want a designed voice that you can reuse like a cloned speaker, a practical workflow is: (1) use the **VoiceDesign** model to synthesize a short reference clip that matches your target persona, (2) feed that clip into `create_voice_clone_prompt` to build a reusable prompt, and then (3) call `generate_voice_clone` with `voice_clone_prompt` to generate new content without re-extracting features every time. This is especially useful when you want a consistent character voice across many lines.

```python
import torch
import soundfile as sf
from qwen3_tts_standalone import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)
```

#### Tokenizer Encode and Decode

If you only want to encode and decode audio for transport or training and so on, `Qwen3TTSTokenizer` supports encode/decode with paths, URLs, numpy waveforms, and dict/list payloads, for example:

```python
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

enc = tokenizer.encode("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
```

For more tokenizer examples (including different input formats and batch usage), please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_tokenizer_12hz.py). With those examples and the description for `Qwen3TTSTokenizer`, you can explore more advanced usage patterns.



## Fine Tuning

Please refer to [Qwen3-TTS-Finetuning](finetuning/) for detailed instructions on fine-tuning Qwen3-TTS.

This standalone implementation includes built-in fine-tuning support with the following features:

### Training Functions

**CodePredictor Fine-tuning**
```python
from qwen3_tts_standalone import CodePredictor, CodePredictorConfig

# Create model
config = CodePredictorConfig(vocab_size=2048, hidden_size=512, ...)
model = CodePredictor(config, embedding_dim=512)

# Training loop
for batch in dataloader:
    output = model.forward_finetune(inputs_embeds, labels)
    output.loss.backward()
    optimizer.step()
```

**Talker Sub-Model Fine-tuning**
```python
from qwen3_tts_standalone import Talker, TalkerConfig

talker = Talker(config)

# Extract hidden states and codec IDs from your data
logits, loss = talker.forward_sub_talker_finetune(codec_ids, hidden_states)
loss.backward()
optimizer.step()
```

### Training Features

- **Causal LM Loss** - Standard next-token prediction with label shifting
- **Gradient Flow** - Verified through overfitting tests on synthetic and real data
- **Weight Compatibility** - Load pretrained weights, fine-tune, and save
- **Flexible Training** - Train CodePredictor alone or full Talker end-to-end

See `tests/test_training.py` for complete training examples and validation.

## Testing

This implementation includes a comprehensive test suite to ensure correctness and equivalence with the original:

### Test Coverage (191+ Tests)

| Category | Tests | Coverage |
|----------|-------|----------|
| **Configuration** | 42 | Serialization, validation, compatibility |
| **Equivalence Testing** | 66 | Output matching with original implementation |
| **Training/Fine-tuning** | 18 | Loss computation, gradient flow, overfitting |
| **Code Predictor** | 11 | Generation, sampling, batch processing |
| **Voice Cloning** | 10 | API validation, output equivalence |
| **End-to-End** | 8 | Full pipeline testing |
| **Integration** | 36 | Package imports, utilities, edge cases |

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_training.py -v          # Training tests
pytest tests/test_equivalence_*.py -v     # Equivalence tests
pytest tests/test_e2e.py -v               # End-to-end tests

# Run with coverage report
pytest tests/ --cov=qwen3_tts_standalone --cov-report=html

# Skip slow tests (GPU required)
pytest tests/ -v -m "not slow"
```

### Test Features

- **Deterministic Testing** - Seeded random number generation for reproducibility
- **Equivalence Validation** - Numerical comparison with original implementation (atol=1e-5)
- **Weight Transfer Tests** - Verify loading weights from official checkpoints
- **Overfitting Tests** - Confirm model can learn (synthetic and real data)
- **Shape Validation** - Verify output shapes across all operations
- **Gradient Flow Tests** - Ensure backpropagation works correctly

### Test Organization

```
tests/
├── conftest.py                              # Shared fixtures and utilities
├── test_configuration.py                    # Config serialization/validation
├── test_equivalence_*.py (5 files)         # Original vs standalone comparison
├── test_training.py                         # Fine-tuning functions
├── test_code_predictor.py                   # CodePredictor generation
├── test_talker.py                           # Talker instantiation
├── test_voice_cloner.py                     # VoiceCloner API
├── test_e2e.py                              # Full pipeline tests
├── test_standalone_package.py               # Package imports
└── test_standalone_utils.py                 # Utility functions
```

## Citation

If you use this code, please consider giving a star :star: and citation :pencil: :)

**For the original Qwen3-TTS:**
```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```
