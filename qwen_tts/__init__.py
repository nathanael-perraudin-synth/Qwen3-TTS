# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
qwen_tts: Qwen-TTS package.
"""

__all__ = ["__version__", "Qwen3TTSModel", "VoiceClonePromptItem", "Qwen3TTSTokenizer"]


def __getattr__(name):
    """Lazy imports to avoid loading transformers when only using core/standalone."""
    if name == "Qwen3TTSModel":
        from .inference.qwen3_tts_model import Qwen3TTSModel
        return Qwen3TTSModel
    if name == "VoiceClonePromptItem":
        from .inference.qwen3_tts_model import VoiceClonePromptItem
        return VoiceClonePromptItem
    if name == "Qwen3TTSTokenizer":
        from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        return Qwen3TTSTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")