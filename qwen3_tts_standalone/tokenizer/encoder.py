# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Mimi encoder for the 12Hz speech tokenizer.

This replicates the encode path of HuggingFace MimiModel without
depending on the transformers package. Only the components needed
for encoding are included (no decoder, no streaming).

Architecture overview (encode path):
  audio waveform
    -> SEANet encoder (causal conv blocks, downsample 960x)
    -> Encoder transformer (8 layers, 512 hidden)
    -> Downsample conv (2x, from 25 Hz to 12.5 Hz)
    -> Split residual vector quantizer (encode to discrete codes)
  = audio_codes of shape (batch, num_quantizers, code_length)
"""

import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


# =============================================================================
# Causal Conv1d (Mimi-style, used by SEANet encoder and downsample)
# =============================================================================

class MimiCausalConv1d(nn.Module):
    """Causal conv1d with asymmetric padding, matching HuggingFace MimiConv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, groups=groups, bias=bias,
        )
        # Effective kernel size with dilation
        eff_kernel = (kernel_size - 1) * dilation + 1
        self.padding_total = eff_kernel - stride

    def _get_extra_padding(self, hidden_states: torch.Tensor) -> int:
        length = hidden_states.shape[-1]
        eff_kernel = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        stride = self.conv.stride[0]
        padding_total = eff_kernel - stride
        n_frames = math.ceil((length - eff_kernel + padding_total) / stride + 1) - 1
        ideal_length = n_frames * stride + eff_kernel - padding_total
        return int(ideal_length - length)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding(hidden_states)
        # Causal: all padding on the left
        if self.pad_mode == "constant":
            hidden_states = F.pad(hidden_states, (self.padding_total, extra_padding), mode="constant", value=0)
        elif self.pad_mode == "replicate":
            hidden_states = self._pad_replicate(hidden_states, (self.padding_total, extra_padding))
        else:
            hidden_states = F.pad(hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(hidden_states)

    @staticmethod
    def _pad_replicate(x: torch.Tensor, paddings: tuple) -> torch.Tensor:
        """Replicate padding that handles small inputs."""
        pad_left, pad_right = paddings
        length = x.shape[-1]
        max_pad = max(pad_left, pad_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        x = F.pad(x, (pad_left, pad_right), mode="replicate")
        if extra_pad > 0:
            x = x[..., : x.shape[-1] - extra_pad]
        return x


# =============================================================================
# SEANet Encoder (causal conv blocks that downsample audio)
# =============================================================================

class MimiResnetBlock(nn.Module):
    """Residual block: ELU -> Conv(dilated) -> ELU -> Conv(1x1), with identity shortcut."""

    def __init__(self, dim: int, dilation: int, compress: int = 2, residual_kernel_size: int = 3):
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleList([
            nn.ELU(),
            MimiCausalConv1d(dim, hidden, residual_kernel_size, dilation=dilation),
            nn.ELU(),
            MimiCausalConv1d(hidden, dim, kernel_size=1),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return residual + hidden_states


class SEANetEncoder(nn.Module):
    """
    SEANet encoder: causal convolutions that progressively downsample the audio.

    For upsampling_ratios=[8,6,5,4], num_filters=64, hidden_size=512:
      Input: (batch, 1, T)
      -> Conv1d(1, 64, 7)
      -> [ResnetBlock(64) + ELU + Conv1d(64, 128, 8, stride=4)]   # ratio=4
      -> [ResnetBlock(128) + ELU + Conv1d(128, 256, 10, stride=5)] # ratio=5
      -> [ResnetBlock(256) + ELU + Conv1d(256, 512, 12, stride=6)] # ratio=6
      -> [ResnetBlock(512) + ELU + Conv1d(512, 1024, 16, stride=8)] # ratio=8
      -> ELU + Conv1d(1024, 512, 3)
      Output: (batch, 512, T // 960)
    """

    def __init__(
        self,
        audio_channels: int = 1,
        num_filters: int = 64,
        hidden_size: int = 512,
        upsampling_ratios: tuple = (8, 6, 5, 4),
        num_residual_layers: int = 1,
        dilation_growth_rate: int = 2,
        compress: int = 2,
        residual_kernel_size: int = 3,
        kernel_size: int = 7,
        last_kernel_size: int = 3,
        pad_mode: str = "constant",
    ):
        super().__init__()
        layers = [MimiCausalConv1d(audio_channels, num_filters, kernel_size, pad_mode=pad_mode)]
        scaling = 1

        for ratio in reversed(upsampling_ratios):
            current_scale = scaling * num_filters
            for j in range(num_residual_layers):
                layers.append(
                    MimiResnetBlock(current_scale, dilation_growth_rate ** j, compress, residual_kernel_size)
                )
            layers.append(nn.ELU())
            layers.append(
                MimiCausalConv1d(current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio, pad_mode=pad_mode)
            )
            scaling *= 2

        layers.append(nn.ELU())
        layers.append(MimiCausalConv1d(scaling * num_filters, hidden_size, last_kernel_size, pad_mode=pad_mode))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# =============================================================================
# Encoder Transformer (processes encoded features before quantization)
# =============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # (B, 1, T, D)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class MimiRotaryEmbedding(nn.Module):
    """Rotary position embeddings for encoder transformer."""

    def __init__(self, head_dim: int, max_position_embeddings: int = 8000, rope_theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # inv_freq: (head_dim/2,), position_ids: (B, T)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ pos.float()).transpose(1, 2)  # (B, T, head_dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, T, head_dim)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MimiLayerScale(nn.Module):
    """Learnable diagonal rescaling of residual outputs."""

    def __init__(self, hidden_size: int, initial_scale: float = 0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), initial_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class MimiMLP(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MimiAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = 1.0 / math.sqrt(head_dim)

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = MimiRotaryEmbedding(head_dim, max_position_embeddings, rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Expand kv for grouped-query attention
        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, seq_len, self.head_dim)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # Causal mask: prevent attending to future positions
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        attn_weights = attn_weights + causal_mask[None, None, :, :]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class MimiTransformerLayer(nn.Module):
    """One transformer layer: LayerNorm -> Attention -> LayerScale + residual -> LayerNorm -> MLP -> LayerScale + residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        layer_scale_initial_scale: float,
        norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.self_attn = MimiAttention(hidden_size, num_heads, num_kv_heads, head_dim, rope_theta, max_position_embeddings)
        self.mlp = MimiMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.self_attn_layer_scale = MimiLayerScale(hidden_size, layer_scale_initial_scale)
        self.mlp_layer_scale = MimiLayerScale(hidden_size, layer_scale_initial_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)
        return hidden_states


class MimiEncoderTransformer(nn.Module):
    """Stack of transformer layers for the encoder."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        layer_scale_initial_scale: float = 0.01,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8000,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MimiTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                layer_scale_initial_scale=layer_scale_initial_scale,
                norm_eps=norm_eps,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# =============================================================================
# Vector Quantizer (for encoding audio to discrete codes)
# =============================================================================

class MimiEuclideanCodebook(nn.Module):
    """Codebook with Euclidean-distance nearest-neighbor lookup."""

    def __init__(self, codebook_size: int, codebook_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embed_sum", torch.zeros(codebook_size, codebook_dim))
        self._embed = None

    @property
    def embed(self) -> torch.Tensor:
        if self._embed is None:
            self._embed = self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return self._embed

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Find nearest codebook entry for each vector.

        Args:
            hidden_states: (batch, seq_len, dim)
        Returns:
            indices: (batch, seq_len)
        """
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        dists = torch.cdist(flat[None].float(), self.embed[None].float(), p=2)[0]
        indices = dists.argmin(dim=-1)
        return indices.view(*shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)


class MimiVectorQuantization(nn.Module):
    """Single-codebook vector quantization."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = MimiEuclideanCodebook(codebook_size, codebook_dim)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode: (B, C, T) -> indices (B, T)."""
        return self.codebook.encode(hidden_states.permute(0, 2, 1))

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode: indices (B, T) -> (B, C, T)."""
        return self.codebook.decode(indices).permute(0, 2, 1)


class MimiResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with input/output projections."""

    def __init__(
        self,
        num_quantizers: int,
        hidden_size: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([MimiVectorQuantization(codebook_size, codebook_dim) for _ in range(num_quantizers)])

        # Project from hidden_size to codebook_dim and back
        if codebook_dim != hidden_size:
            self.input_proj = nn.Conv1d(hidden_size, codebook_dim, 1, bias=False)
            self.output_proj = nn.Conv1d(codebook_dim, hidden_size, 1, bias=False)
        else:
            self.input_proj = None
            self.output_proj = None

    def encode(self, embeddings: torch.Tensor, num_quantizers: Optional[int] = None) -> torch.Tensor:
        """Encode embeddings to codes using residual quantization.

        Args:
            embeddings: (B, C, T) - continuous embeddings
            num_quantizers: how many quantizers to use (default: all)
        Returns:
            codes: (num_quantizers, B, T)
        """
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)

        n_q = num_quantizers if num_quantizers is not None else self.num_quantizers
        residual = embeddings
        all_indices = []
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        return torch.stack(all_indices)


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Split RVQ: separate semantic (1 codebook) and acoustic (rest) quantizers."""

    def __init__(
        self,
        num_quantizers: int = 16,
        num_semantic_quantizers: int = 1,
        hidden_size: int = 512,
        codebook_size: int = 2048,
        codebook_dim: int = 256,
    ):
        super().__init__()
        self.num_semantic_quantizers = num_semantic_quantizers
        self.num_acoustic_quantizers = num_quantizers - num_semantic_quantizers

        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            num_semantic_quantizers, hidden_size, codebook_size, codebook_dim,
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            self.num_acoustic_quantizers, hidden_size, codebook_size, codebook_dim,
        )

    def encode(self, embeddings: torch.Tensor, num_quantizers: int) -> torch.Tensor:
        """Encode to codes.

        Args:
            embeddings: (B, C, T)
            num_quantizers: total number of codebooks to use
        Returns:
            codes: (B, num_quantizers, T)
        """
        codes = self.semantic_residual_vector_quantizer.encode(embeddings)
        if num_quantizers > self.num_semantic_quantizers:
            acoustic_codes = self.acoustic_residual_vector_quantizer.encode(
                embeddings, num_quantizers=num_quantizers - self.num_semantic_quantizers,
            )
            codes = torch.cat([codes, acoustic_codes], dim=0)
        # Transpose from (num_quantizers, B, T) to (B, num_quantizers, T)
        return codes.transpose(0, 1)


# =============================================================================
# Full Encoder Model
# =============================================================================

class MimiEncoderModel(nn.Module):
    """
    Complete standalone Mimi encoder for the 12Hz tokenizer.

    Encodes audio waveforms to discrete codes without any transformers dependency.
    """

    def __init__(
        self,
        # SEANet encoder params
        audio_channels: int = 1,
        num_filters: int = 64,
        hidden_size: int = 512,
        upsampling_ratios: tuple = (8, 6, 5, 4),
        num_residual_layers: int = 1,
        dilation_growth_rate: int = 2,
        compress: int = 2,
        residual_kernel_size: int = 3,
        kernel_size: int = 7,
        last_kernel_size: int = 3,
        pad_mode: str = "constant",
        # Transformer params
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        layer_scale_initial_scale: float = 0.01,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8000,
        # Downsample params (from SEANet frame rate to target frame rate)
        encodec_frame_rate: int = 25,
        frame_rate: float = 12.5,
        upsample_groups: int = 512,
            # Quantizer params
            num_quantizers: int = 16,
        num_semantic_quantizers: int = 1,
        codebook_size: int = 2048,
        codebook_dim: int = 256,
    ):
        super().__init__()
        self.encoder = SEANetEncoder(
            audio_channels=audio_channels,
            num_filters=num_filters,
            hidden_size=hidden_size,
            upsampling_ratios=upsampling_ratios,
            num_residual_layers=num_residual_layers,
            dilation_growth_rate=dilation_growth_rate,
            compress=compress,
            residual_kernel_size=residual_kernel_size,
            kernel_size=kernel_size,
            last_kernel_size=last_kernel_size,
            pad_mode=pad_mode,
        )

        self.encoder_transformer = MimiEncoderTransformer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            layer_scale_initial_scale=layer_scale_initial_scale,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

        # Downsample from encodec_frame_rate to frame_rate
        ds_kernel = 2 * int(encodec_frame_rate / frame_rate)
        self.downsample = MimiCausalConv1d(
            hidden_size, hidden_size,
            kernel_size=ds_kernel, stride=2, bias=False,
            pad_mode="replicate",
        )

        self.quantizer = MimiSplitResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_semantic_quantizers=num_semantic_quantizers,
            hidden_size=hidden_size,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    def encode(self, input_values: torch.Tensor, num_quantizers: int) -> torch.Tensor:
        """Encode audio waveform to discrete codes.

        Args:
            input_values: (batch, channels, time) - raw audio waveform
            num_quantizers: number of codebooks to use

        Returns:
            codes: (batch, num_quantizers, code_length)
        """
        # SEANet encoder
        embeddings = self.encoder(input_values)
        # Transformer
        embeddings = self.encoder_transformer(embeddings.transpose(1, 2)).transpose(1, 2)
        # Downsample from 25Hz to 12.5Hz
        embeddings = self.downsample(embeddings)
        # Quantize to discrete codes
        codes = self.quantizer.encode(embeddings, num_quantizers)
        return codes


__all__ = ["MimiEncoderModel"]
