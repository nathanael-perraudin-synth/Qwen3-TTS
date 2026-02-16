# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for 12Hz encoder equivalence between original (transformers-based) and standalone.

These tests verify that the standalone Mimi encoder produces the same output as
the transformers-based MimiModel encoder at every stage of the pipeline.
"""

import pytest
import torch
import numpy as np

from tests.conftest import set_seed, copy_weights


# =============================================================================
# Component-level tests (with random weights, fast)
# =============================================================================


class TestSEANetEncoderEquivalence:
    """Test SEANet encoder (causal conv blocks) equivalence."""

    def test_mimi_causal_conv1d_basic(self):
        """Test MimiCausalConv1d matches HuggingFace MimiConv1d."""
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiCausalConv1d

        config = MimiConfig(use_causal_conv=True, pad_mode="constant")

        set_seed(42)
        orig = MimiConv1d(config, 64, 128, kernel_size=7, stride=1)
        standalone = MimiCausalConv1d(64, 128, kernel_size=7, stride=1, pad_mode="constant")
        copy_weights(orig, standalone)

        x = torch.randn(2, 64, 100)
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)

        assert out_orig.shape == out_standalone.shape
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_mimi_causal_conv1d_downsample(self):
        """Test MimiCausalConv1d with stride > 1 (downsampling conv)."""
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiCausalConv1d

        config = MimiConfig(use_causal_conv=True, pad_mode="constant")

        for stride, kernel in [(4, 8), (5, 10), (6, 12), (8, 16)]:
            set_seed(42)
            orig = MimiConv1d(config, 128, 256, kernel_size=kernel, stride=stride)
            standalone = MimiCausalConv1d(128, 256, kernel_size=kernel, stride=stride, pad_mode="constant")
            copy_weights(orig, standalone)

            x = torch.randn(2, 128, 500)
            with torch.no_grad():
                out_orig = orig(x)
                out_standalone = standalone(x)

            assert out_orig.shape == out_standalone.shape, f"Shape mismatch for stride={stride}"
            assert torch.allclose(out_orig, out_standalone, atol=1e-5), f"Value mismatch for stride={stride}"

    def test_mimi_causal_conv1d_replicate_pad(self):
        """Test MimiCausalConv1d with replicate padding (used by downsample)."""
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiCausalConv1d

        config = MimiConfig(use_causal_conv=True, pad_mode="replicate")

        set_seed(42)
        orig = MimiConv1d(config, 512, 512, kernel_size=4, stride=2, bias=False, pad_mode="replicate")
        standalone = MimiCausalConv1d(512, 512, kernel_size=4, stride=2, bias=False, pad_mode="replicate")
        copy_weights(orig, standalone)

        x = torch.randn(2, 512, 25)
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)

        assert out_orig.shape == out_standalone.shape
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_resnet_block(self):
        """Test MimiResnetBlock equivalence."""
        from transformers.models.mimi.modeling_mimi import MimiResnetBlock as MimiResnetBlockOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiResnetBlock

        config = MimiConfig(
            use_causal_conv=True, pad_mode="constant",
            residual_kernel_size=3, compress=2, use_conv_shortcut=False,
        )

        set_seed(42)
        orig = MimiResnetBlockOrig(config, dim=128, dilations=[1, 1])
        standalone = MimiResnetBlock(dim=128, dilation=1, compress=2, residual_kernel_size=3)
        copy_weights(orig, standalone)

        x = torch.randn(2, 128, 50)
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)

        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_full_seanet_encoder(self):
        """Test full SEANet encoder with tiny config."""
        from transformers.models.mimi.modeling_mimi import MimiEncoder as MimiEncoderOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import SEANetEncoder

        config = MimiConfig(
            audio_channels=1, num_filters=8, hidden_size=32,
            upsampling_ratios=[2, 2], num_residual_layers=1,
            dilation_growth_rate=2, compress=2, residual_kernel_size=3,
            kernel_size=7, last_kernel_size=3,
            use_causal_conv=True, pad_mode="constant",
        )

        set_seed(42)
        orig = MimiEncoderOrig(config)
        standalone = SEANetEncoder(
            audio_channels=1, num_filters=8, hidden_size=32,
            upsampling_ratios=(2, 2), num_residual_layers=1,
            dilation_growth_rate=2, compress=2, residual_kernel_size=3,
            kernel_size=7, last_kernel_size=3, pad_mode="constant",
        )
        copy_weights(orig, standalone)

        x = torch.randn(1, 1, 200)
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)

        assert out_orig.shape == out_standalone.shape
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)


class TestEncoderTransformerEquivalence:
    """Test encoder transformer (Mimi-style) equivalence."""

    def test_attention(self):
        """Test MimiAttention with causal masking."""
        from transformers.models.mimi.modeling_mimi import MimiAttention as MimiAttentionOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiAttention

        config = MimiConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
            head_dim=16, attention_bias=False, rope_theta=10000.0,
            max_position_embeddings=8000, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"

        set_seed(42)
        orig = MimiAttentionOrig(config, layer_idx=0)
        standalone = MimiAttention(
            hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
            rope_theta=10000.0, max_position_embeddings=8000,
        )
        copy_weights(orig, standalone)

        seq_len = 20
        x = torch.randn(2, seq_len, 64)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)

        # Build a manual causal mask (same format as transformers uses)
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=x.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # Shape: (batch, 1, seq_len, seq_len) for transformers attention
        causal_mask = causal_mask[None, None, :, :].expand(2, -1, -1, -1)

        with torch.no_grad():
            out_orig, _ = orig(x, attention_mask=causal_mask, position_ids=position_ids)
            out_standalone = standalone(x)

        assert torch.allclose(out_orig, out_standalone, atol=1e-4)

    def test_transformer_layer(self):
        """Test MimiTransformerLayer equivalence."""
        from transformers.models.mimi.modeling_mimi import MimiTransformerLayer as MimiTransformerLayerOrig, MimiConfig
        from transformers.masking_utils import create_causal_mask
        from qwen3_tts_standalone.tokenizer.encoder import MimiTransformerLayer

        config = MimiConfig(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
            head_dim=16, intermediate_size=128, hidden_act="gelu",
            layer_scale_initial_scale=0.01, norm_eps=1e-5,
            rope_theta=10000.0, max_position_embeddings=8000,
            attention_bias=False, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"

        set_seed(42)
        orig = MimiTransformerLayerOrig(config, layer_idx=0)
        standalone = MimiTransformerLayer(
            hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
            intermediate_size=128, layer_scale_initial_scale=0.01,
            norm_eps=1e-5, rope_theta=10000.0, max_position_embeddings=8000,
        )
        copy_weights(orig, standalone)

        seq_len = 20
        x = torch.randn(2, seq_len, 64)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
        cache_position = torch.arange(seq_len)
        causal_mask = create_causal_mask(
            config=config, input_embeds=x, attention_mask=None,
            cache_position=cache_position, past_key_values=None,
            position_ids=position_ids,
        )

        with torch.no_grad():
            out_orig = orig(x, attention_mask=causal_mask, position_ids=position_ids, cache_position=cache_position)
            out_standalone = standalone(x)

        assert torch.allclose(out_orig[0], out_standalone, atol=1e-4)


class TestEncoderQuantizerEquivalence:
    """Test encoder-side quantizer equivalence."""

    def test_euclidean_codebook_encode(self):
        """Test MimiEuclideanCodebook encode (nearest-neighbor lookup)."""
        from transformers.models.mimi.modeling_mimi import MimiEuclideanCodebook as MimiCodebookOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiEuclideanCodebook

        config = MimiConfig(codebook_size=64, codebook_dim=32)

        set_seed(42)
        orig = MimiCodebookOrig(config)
        standalone = MimiEuclideanCodebook(codebook_size=64, codebook_dim=32)

        # Match weights: orig uses embed_sum as buffer, standalone uses it too
        standalone.cluster_usage.copy_(orig.cluster_usage)
        standalone.embed_sum.copy_(orig.embed_sum)

        x = torch.randn(2, 10, 32)  # (batch, seq, dim)
        with torch.no_grad():
            indices_orig = orig.encode(x)
            indices_standalone = standalone.encode(x)

        assert torch.equal(indices_orig, indices_standalone)

    def test_vector_quantization_encode(self):
        """Test MimiVectorQuantization encode."""
        from transformers.models.mimi.modeling_mimi import MimiVectorQuantization as MimiVQOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiVectorQuantization

        config = MimiConfig(codebook_size=64, codebook_dim=32)

        set_seed(42)
        orig = MimiVQOrig(config)
        standalone = MimiVectorQuantization(codebook_size=64, codebook_dim=32)

        standalone.codebook.cluster_usage.copy_(orig.codebook.cluster_usage)
        standalone.codebook.embed_sum.copy_(orig.codebook.embed_sum)

        x = torch.randn(2, 32, 10)  # (batch, channels, time)
        with torch.no_grad():
            indices_orig = orig.encode(x)
            indices_standalone = standalone.encode(x)

        assert torch.equal(indices_orig, indices_standalone)

    def test_residual_vector_quantizer_encode(self):
        """Test MimiResidualVectorQuantizer encode."""
        from transformers.models.mimi.modeling_mimi import MimiResidualVectorQuantizer as MimiRVQOrig, MimiConfig
        from qwen3_tts_standalone.tokenizer.encoder import MimiResidualVectorQuantizer

        config = MimiConfig(
            codebook_size=64, codebook_dim=32,
            hidden_size=64, num_quantizers=4,
            vector_quantization_hidden_dimension=32,
        )

        set_seed(42)
        orig = MimiRVQOrig(config, num_quantizers=4)
        standalone = MimiResidualVectorQuantizer(
            num_quantizers=4, hidden_size=64,
            codebook_size=64, codebook_dim=32,
        )

        # Copy weights carefully: buffers in codebooks + projection weights
        standalone_sd = standalone.state_dict()
        orig_sd = orig.state_dict()
        for key in standalone_sd:
            if key in orig_sd:
                standalone_sd[key].copy_(orig_sd[key])
        standalone.load_state_dict(standalone_sd)

        x = torch.randn(2, 64, 10)  # (batch, hidden, time)
        with torch.no_grad():
            codes_orig = orig.encode(x)  # (n_q, batch, time)
            codes_standalone = standalone.encode(x)  # (n_q, batch, time)

        assert codes_orig.shape == codes_standalone.shape
        assert torch.equal(codes_orig, codes_standalone)


# =============================================================================
# Integration tests with pretrained weights
# =============================================================================


class TestPretrainedEncoderEquivalence:
    """Test full encoder equivalence with pretrained model weights."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load both models once."""
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from qwen3_tts_standalone.tokenizer import SpeechTokenizerModel

        self.device = torch.device("cpu")
        self.orig = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")
        self.standalone = SpeechTokenizerModel.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz", device=self.device,
        )

    def test_seanet_encoder_pretrained(self):
        """Test SEANet encoder with pretrained weights."""
        set_seed(42)
        audio = torch.randn(1, 1, 24000) * 0.1

        with torch.no_grad():
            orig_out = self.orig.model.encoder.encoder(audio)
            standalone_out = self.standalone.encoder.encoder(audio)

        assert orig_out.shape == standalone_out.shape
        assert torch.allclose(orig_out, standalone_out, atol=1e-5)

    def test_transformer_pretrained(self):
        """Test encoder transformer with pretrained weights."""
        set_seed(42)
        audio = torch.randn(1, 1, 24000) * 0.1

        with torch.no_grad():
            sea_out = self.orig.model.encoder.encoder(audio)
            orig_out = self.orig.model.encoder.encoder_transformer(
                sea_out.transpose(1, 2), return_dict=True,
            ).last_hidden_state
            standalone_out = self.standalone.encoder.encoder_transformer(
                sea_out.transpose(1, 2),
            )

        assert orig_out.shape == standalone_out.shape
        max_diff = (orig_out - standalone_out).abs().max().item()
        assert max_diff < 1e-4, f"Max diff: {max_diff}"

    def test_full_encode_pretrained(self):
        """Test full encode pipeline produces identical codes."""
        set_seed(42)
        audio = np.random.randn(24000).astype(np.float32) * 0.1

        # Encode with original
        orig_encoded = self.orig.encode(audio, sr=24000)
        orig_codes = orig_encoded.audio_codes[0]

        # Encode with standalone
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        mask = torch.ones(1, len(audio), dtype=torch.int32)
        with torch.no_grad():
            standalone_encoded = self.standalone.encode(audio_t, mask)
        standalone_codes = standalone_encoded.audio_codes[0]

        assert orig_codes.shape == standalone_codes.shape, (
            f"Shape mismatch: {orig_codes.shape} vs {standalone_codes.shape}"
        )
        assert torch.equal(orig_codes, standalone_codes), (
            f"Code mismatch! Match rate: {(orig_codes == standalone_codes).float().mean():.4f}"
        )

    def test_full_encode_longer_audio(self):
        """Test encode with longer audio (2 seconds)."""
        set_seed(123)
        audio = np.random.randn(48000).astype(np.float32) * 0.1

        orig_encoded = self.orig.encode(audio, sr=24000)
        orig_codes = orig_encoded.audio_codes[0]

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        mask = torch.ones(1, len(audio), dtype=torch.int32)
        with torch.no_grad():
            standalone_encoded = self.standalone.encode(audio_t, mask)
        standalone_codes = standalone_encoded.audio_codes[0]

        assert orig_codes.shape == standalone_codes.shape
        assert torch.equal(orig_codes, standalone_codes)

    def test_batch_encode_pretrained(self):
        """Test batch encoding produces identical codes for each sample."""
        set_seed(42)
        audio1 = np.random.randn(24000).astype(np.float32) * 0.1
        audio2 = np.random.randn(16000).astype(np.float32) * 0.1

        # Encode individually with original
        orig_codes1 = self.orig.encode(audio1, sr=24000).audio_codes[0]
        orig_codes2 = self.orig.encode(audio2, sr=24000).audio_codes[0]

        # Encode individually with standalone
        for audio_np, orig_codes in [(audio1, orig_codes1), (audio2, orig_codes2)]:
            audio_t = torch.from_numpy(audio_np).unsqueeze(0)
            mask = torch.ones(1, len(audio_np), dtype=torch.int32)
            with torch.no_grad():
                standalone_encoded = self.standalone.encode(audio_t, mask)
            standalone_codes = standalone_encoded.audio_codes[0]

            assert torch.equal(orig_codes, standalone_codes)

    def test_roundtrip_pretrained(self):
        """Test encode -> decode roundtrip produces same result for both implementations."""
        set_seed(42)
        audio = np.random.randn(24000).astype(np.float32) * 0.1

        # Encode with original, decode with both
        orig_encoded = self.orig.encode(audio, sr=24000)
        orig_wavs, orig_sr = self.orig.decode(orig_encoded)

        # Encode with standalone
        from qwen3_tts_standalone.tokenizer import SpeechTokenizer
        standalone_tokenizer = SpeechTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz", device="cpu",
        )
        standalone_encoded = standalone_tokenizer.encode(audio, sr=24000)
        standalone_wavs, standalone_sr = standalone_tokenizer.decode(standalone_encoded)

        assert orig_sr == standalone_sr
        assert len(orig_wavs) == len(standalone_wavs)
        for ow, sw in zip(orig_wavs, standalone_wavs):
            assert ow.shape == sw.shape
            max_diff = np.abs(ow - sw).max()
            assert max_diff < 0.02, f"Roundtrip max diff: {max_diff}"


class TestFeatureExtraction:
    """Test that standalone feature extraction matches HuggingFace."""

    def test_single_audio(self):
        """Test feature extraction with single audio."""
        from transformers import AutoFeatureExtractor
        from qwen3_tts_standalone.tokenizer.speech_tokenizer import extract_features

        fe = AutoFeatureExtractor.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

        audio = np.random.randn(16000).astype(np.float32)
        hf_out = fe(raw_audio=[audio], sampling_rate=24000, return_tensors="pt")
        standalone_out = extract_features([audio], sampling_rate=24000)

        assert torch.allclose(hf_out["input_values"], standalone_out["input_values"])
        assert torch.equal(hf_out["padding_mask"], standalone_out["padding_mask"])

    def test_batch_padding(self):
        """Test feature extraction pads shorter sequences correctly."""
        from transformers import AutoFeatureExtractor
        from qwen3_tts_standalone.tokenizer.speech_tokenizer import extract_features

        fe = AutoFeatureExtractor.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")

        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(8000).astype(np.float32)
        hf_out = fe(raw_audio=[audio1, audio2], sampling_rate=24000, return_tensors="pt")
        standalone_out = extract_features([audio1, audio2], sampling_rate=24000)

        assert torch.allclose(hf_out["input_values"], standalone_out["input_values"])
        assert torch.equal(hf_out["padding_mask"], standalone_out["padding_mask"])


__all__ = [
    "TestSEANetEncoderEquivalence",
    "TestEncoderTransformerEquivalence",
    "TestEncoderQuantizerEquivalence",
    "TestPretrainedEncoderEquivalence",
    "TestFeatureExtraction",
]
