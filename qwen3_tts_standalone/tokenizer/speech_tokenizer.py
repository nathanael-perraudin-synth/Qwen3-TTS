# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Qwen3 TTS Speech Tokenizer (12Hz).

No dependency on the transformers package.
"""

import io
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import (
    SpeechTokenizerModel,
    SpeechTokenizerEncoderOutput,
    SpeechTokenizerDecoderOutput,
)


# =============================================================================
# Feature extraction (replaces transformers EncodecFeatureExtractor)
# =============================================================================

def extract_features(
    raw_audios: List[np.ndarray],
    sampling_rate: int = 24000,
) -> dict:
    """Pad a list of audio arrays to the same length and create a padding mask.

    This is a minimal replacement for HuggingFace EncodecFeatureExtractor.
    It does not normalize or transform the audio - just pads and batches.

    Args:
        raw_audios: List of 1-D float32 numpy arrays (mono audio).
        sampling_rate: Expected sampling rate (unused, kept for clarity).

    Returns:
        dict with:
          - "input_values": torch.FloatTensor of shape (batch, 1, max_length)
          - "padding_mask": torch.IntTensor of shape (batch, max_length)
    """
    max_length = max(len(a) for a in raw_audios)
    batch_size = len(raw_audios)

    input_values = torch.zeros(batch_size, 1, max_length, dtype=torch.float32)
    padding_mask = torch.zeros(batch_size, max_length, dtype=torch.int32)

    for i, audio in enumerate(raw_audios):
        length = len(audio)
        input_values[i, 0, :length] = torch.from_numpy(audio)
        padding_mask[i, :length] = 1

    return {"input_values": input_values, "padding_mask": padding_mask}


# =============================================================================
# Audio loading
# =============================================================================

def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio from a file path, resample to target_sr, return mono float32."""
    audio, sr = librosa.load(path, sr=None, mono=True)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


# =============================================================================
# Main tokenizer class
# =============================================================================

class SpeechTokenizer:
    """
    Standalone speech tokenizer for Qwen3-TTS 12Hz.

    Encodes audio waveforms to discrete codes and decodes back.
    No dependency on the transformers package.

    Usage:
        tokenizer = SpeechTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

        # Encode audio to codes
        encoded = tokenizer.encode("audio.wav")

        # Decode codes back to audio
        wavs, sr = tokenizer.decode(encoded)
    """

    def __init__(self, model: SpeechTokenizerModel, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SpeechTokenizer":
        """
        Load tokenizer from pretrained weights.

        Args:
            pretrained_model_name_or_path: HuggingFace repo id or local directory.
            device: Target device (e.g., "cuda:0", "cpu"). Defaults to CUDA if available.
            dtype: Target dtype (e.g., torch.float32). Defaults to float32.

        Returns:
            Initialized SpeechTokenizer.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        dtype = dtype or torch.float32

        model = SpeechTokenizerModel.from_pretrained(
            pretrained_model_name_or_path, device=device, dtype=dtype,
        )
        model.eval()

        return cls(model=model, device=device, dtype=dtype)

    def encode(
        self,
        audios: Union[str, np.ndarray, List[str], List[np.ndarray]],
        sr: Optional[int] = None,
    ) -> SpeechTokenizerEncoderOutput:
        """
        Encode audio to discrete codes.

        Args:
            audios: One of:
                - str: path to a wav file
                - np.ndarray: raw waveform (requires sr)
                - list[str]: list of wav paths
                - list[np.ndarray]: list of waveforms (requires sr)
            sr: Sample rate of numpy input. Required when audios is ndarray.

        Returns:
            SpeechTokenizerEncoderOutput with audio_codes as list of tensors,
            each of shape (codes_len, num_quantizers).
        """
        wavs = self._prepare_audio(audios, sr)
        target_sr = self.model.get_input_sample_rate()

        inputs = extract_features(wavs, sampling_rate=target_sr)
        input_values = inputs["input_values"].squeeze(1).to(self.device).to(self.dtype)
        padding_mask = inputs["padding_mask"].squeeze(1).to(self.device)

        with torch.inference_mode():
            return self.model.encode(input_values, padding_mask)

    def decode(
        self,
        encoded,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Decode codes back to waveform.

        Args:
            encoded: SpeechTokenizerEncoderOutput from encode(), or a dict
                with key "audio_codes", or a list of dicts.

        Returns:
            Tuple of (list of 1-D float32 numpy waveforms, sample_rate).
        """
        # Normalize to list of tensors
        if hasattr(encoded, "audio_codes"):
            audio_codes_list = encoded.audio_codes
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
        elif isinstance(encoded, list):
            audio_codes_list = [e["audio_codes"] for e in encoded]
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        # Pad to batch tensor
        if isinstance(audio_codes_list, torch.Tensor):
            t = audio_codes_list
            if t.dim() == 2:
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:
            tensors = [c if isinstance(c, torch.Tensor) else torch.from_numpy(np.asarray(c)).long() for c in audio_codes_list]
            audio_codes_padded = pad_sequence(tensors, batch_first=True, padding_value=-1).to(self.device)

        with torch.inference_mode():
            dec = self.model.decode(audio_codes_padded, return_dict=True)

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in dec.audio_values]
        return wavs, self.model.get_output_sample_rate()

    # -- Convenience properties --

    def get_model_type(self) -> str:
        return self.model.get_model_type()

    def get_input_sample_rate(self) -> int:
        return self.model.get_input_sample_rate()

    def get_output_sample_rate(self) -> int:
        return self.model.get_output_sample_rate()

    def get_encode_downsample_rate(self) -> int:
        return self.model.get_encode_downsample_rate()

    def get_decode_upsample_rate(self) -> int:
        return self.model.get_decode_upsample_rate()

    # -- Internal helpers --

    def _prepare_audio(
        self,
        audios: Union[str, np.ndarray, List[str], List[np.ndarray]],
        sr: Optional[int],
    ) -> List[np.ndarray]:
        """Normalize all input forms to a list of float32 numpy arrays at model sample rate."""
        target_sr = self.model.get_input_sample_rate()

        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        if len(audios) == 0:
            return []

        if isinstance(audios[0], str):
            return [load_audio(path, target_sr) for path in audios]

        # numpy arrays
        if sr is None:
            raise ValueError("For numpy waveform input, you must provide `sr`.")

        result = []
        for a in audios:
            if not isinstance(a, np.ndarray):
                raise TypeError("Mixed input types are not supported.")
            if a.ndim > 1:
                a = np.mean(a, axis=-1)
            if int(sr) != target_sr:
                a = librosa.resample(y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            result.append(a.astype(np.float32))
        return result


__all__ = ["SpeechTokenizer", "extract_features", "load_audio"]
