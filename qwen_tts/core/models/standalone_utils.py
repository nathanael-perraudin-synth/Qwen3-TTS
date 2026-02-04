"""Standalone utilities to replace transformers dependencies."""
import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class TransformerOutput:
    """Simple output dataclass to replace transformers ModelOutput."""
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None


class SimpleCache:
    """Simple KV cache implementation to replace transformers Cache."""
    
    def __init__(self):
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: int, cache_kwargs: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a given layer."""
        if layer_idx in self.key_cache:
            # Concatenate new states
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        else:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self) -> int:
        """Get the sequence length of cached keys."""
        if not self.key_cache:
            return 0
        # Get length from first cached key
        first_key = next(iter(self.key_cache.values()))
        return first_key.shape[2] if len(first_key.shape) > 2 else 0

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> Tuple[int, int]:
        """Return (kv_length, kv_offset) for mask creation (align with transformers Cache)."""
        past_len = self.get_seq_length() if layer_idx in self.key_cache else 0
        q_len = cache_position.shape[0]
        kv_length = past_len + q_len
        kv_offset = 0
        return kv_length, kv_offset


class DynamicCache(SimpleCache):
    """Dynamic cache (same as SimpleCache for now)."""
    pass


def create_causal_mask(
    config: Any,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values: Optional[SimpleCache] = None,
    **kwargs
) -> torch.Tensor:
    """Create causal attention mask.
    
    Args:
        config: Model config
        input_embeds: Input embeddings tensor
        attention_mask: Optional attention mask
        cache_position: Optional cache position tensor
        past_key_values: Optional past key values cache
    
    Returns:
        Causal attention mask
    """
    batch_size, seq_length = input_embeds.shape[:2]
    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    kv_length = past_length + seq_length

    # Create causal mask: (batch, seq_length, kv_length) so query can attend to past + current
    causal_mask = torch.full(
        (batch_size, seq_length, kv_length),
        fill_value=float("-inf"),
        device=input_embeds.device,
        dtype=input_embeds.dtype,
    )
    # Causal: query q can attend to key k iff k <= past_length + q
    q_idx = torch.arange(seq_length, device=input_embeds.device, dtype=torch.long)[:, None]
    k_idx = torch.arange(kv_length, device=input_embeds.device, dtype=torch.long)[None, :]
    allow = (past_length + q_idx) >= k_idx  # (seq_length, kv_length)
    causal_mask = causal_mask.masked_fill(allow.unsqueeze(0).expand(batch_size, -1, -1), 0)
    
    # Apply attention mask if provided
    if attention_mask is not None:
        # Expand attention mask to match causal mask shape
        if attention_mask.dim() == 2:
            # (batch, seq) -> (batch, 1, seq)
            attention_mask = attention_mask.unsqueeze(1)
        if attention_mask.dim() == 3:
            # (batch, seq, seq) -> (batch, 1, seq, seq)
            attention_mask = attention_mask.unsqueeze(1)
        
        # Combine masks: where attention_mask is 0, set to -inf
        causal_mask = causal_mask.masked_fill(attention_mask == 0, float("-inf"))
    
    return causal_mask


def create_sliding_window_causal_mask(
    config: Any,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values: Optional[SimpleCache] = None,
    **kwargs
) -> Optional[torch.Tensor]:
    """Create sliding window causal attention mask.
    
    For memory efficiency, this returns None and the sliding window logic
    is applied directly in eager_attention_forward using the sliding_window parameter.
    Only returns a mask if attention_mask is provided (for padding).
    
    Args:
        config: Model config with sliding_window attribute
        input_embeds: Input embeddings tensor
        attention_mask: Optional attention mask (for padding)
        cache_position: Optional cache position tensor
        past_key_values: Optional past key values cache
    
    Returns:
        None (sliding window handled in attention) or attention_mask if provided
    """
    # For sliding window, we don't create the full mask here
    # The sliding window logic is applied directly in eager_attention_forward
    # Only return attention_mask if provided (for padding handling)
    if attention_mask is not None:
        # Return a lightweight version for padding only
        # The sliding window constraint will be applied in eager_attention_forward
        return attention_mask
    return None


def can_return_tuple(func):
    """Decorator to allow functions to return tuples (no-op for standalone)."""
    return func


def default_rope_init(config: Any, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, float]:
    """Initialize default RoPE (Rotary Position Embedding).
    
    Args:
        config: Model config with rope_theta and head_dim/hidden_size/num_attention_heads
        device: Optional device to create tensors on
    
    Returns:
        Tuple of (inv_freq, attention_scaling)
    """
    rope_theta = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        num_heads = getattr(config, "num_attention_heads", None)
        assert hidden_size is not None and num_heads is not None, "hidden_size and num_attention_heads must be set"
        head_dim = hidden_size // num_heads
    
    # Create inverse frequencies
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    attention_scaling = 1.0
    
    return inv_freq, attention_scaling


def dynamic_rope_update(func):
    """Decorator for dynamic RoPE update (no-op for standalone)."""
    return func


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    """Eager attention forward pass with memory-efficient sliding window support."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # Apply sliding window constraint directly to attn_weights (memory efficient)
    if sliding_window is not None:
        # attn_weights shape: (batch, num_heads, seq_q, seq_k)
        batch_size, num_heads, seq_q, seq_k = attn_weights.shape
        
        # Vectorized sliding window mask: create a mask using broadcasting
        # For query position q, allow key positions k where: k >= max(0, q - sliding_window + 1) and k <= q
        # We mask positions where: k < max(0, q - sliding_window + 1) OR k > q
        # Note: k > q is already handled by causal masking (should be done separately if needed)
        
        # Create indices for query and key positions
        q_pos = torch.arange(seq_q, device=attn_weights.device, dtype=torch.long)
        k_pos = torch.arange(seq_k, device=attn_weights.device, dtype=torch.long)
        
        # Shape: (seq_q, seq_k)
        q_pos_expanded = q_pos[:, None]  # (seq_q, 1)
        k_pos_expanded = k_pos[None, :]  # (1, seq_k)
        
        # Calculate window start for each query position
        window_start = torch.clamp(q_pos_expanded - sliding_window + 1, min=0)
        
        # Mask positions outside sliding window: k < window_start OR k > q (causal)
        # For sliding window: mask where k < window_start
        window_mask = k_pos_expanded < window_start
        
        # Also apply causal mask: k > q (keys after query position)
        causal_mask = k_pos_expanded > q_pos_expanded
        
        # Combine masks
        combined_mask = window_mask | causal_mask
        
        # Apply mask: set to -inf where outside window or after query position
        # Expand to match attn_weights shape: (batch, num_heads, seq_q, seq_k)
        combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_q, seq_k)
        attn_weights = attn_weights.masked_fill(combined_mask, float("-inf"))
    
    # Handle attention_mask (for padding, causal mask, etc.)
    if attention_mask is not None:
        # Handle different mask shapes
        # attn_weights shape: (batch, num_heads, seq_q, seq_k)
        # attention_mask can be: (batch, seq, seq), (batch, num_heads, seq, seq), or (batch, seq)
        if attention_mask.dim() == 4:
            # Already (batch, num_heads, seq, seq) or (batch, 1, seq, seq)
            if attention_mask.shape[1] == 1:
                # Expand to match number of heads
                attention_mask = attention_mask.expand(-1, query.shape[1], -1, -1)
        elif attention_mask.dim() == 3:
            # (batch, seq, seq) -> (batch, 1, seq, seq) -> expand to num_heads
            attention_mask = attention_mask.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        elif attention_mask.dim() == 2:
            # (batch, seq) -> (batch, 1, 1, seq) -> expand to num_heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, query.shape[1], -1, -1)
        
        # Ensure mask matches attention weights shape
        # attn_weights: (batch, num_heads, seq_q, seq_k)
        # attention_mask should be: (batch, num_heads, seq_q, seq_k)
        if attention_mask.shape[-1] != key_states.shape[-2]:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        if attention_mask.shape[-2] != query.shape[-2]:
            attention_mask = attention_mask[:, :, : query.shape[-2], :]
        
        # Ensure batch and head dimensions match
        # If mask has fewer heads than query, expand it
        if attention_mask.shape[1] < attn_weights.shape[1]:
            # Expand from (batch, mask_heads, seq, seq) to (batch, query_heads, seq, seq)
            num_heads_to_expand = attn_weights.shape[1] // attention_mask.shape[1]
            attention_mask = attention_mask.repeat(1, num_heads_to_expand, 1, 1)
        elif attention_mask.shape[1] > attn_weights.shape[1]:
            # Truncate if mask has more heads
            attention_mask = attention_mask[:, : attn_weights.shape[1]]
        
        # Apply mask: add mask values to attn_weights
        # The mask can contain -inf (for causal/padding) or 0 (for padding only)
        # If mask has -inf, add it directly; if mask has 0, set those positions to -inf
        if torch.isinf(attention_mask).any():
            # Mask already contains -inf values (causal mask), add it directly
            attn_weights = attn_weights + attention_mask
        else:
            # Mask contains 0/1 values (padding mask), set 0 positions to -inf
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))

    # Avoid softmax(-inf)=NaN: replace -inf with large negative before softmax
    attn_weights = torch.where(
        torch.isinf(attn_weights),
        torch.full_like(attn_weights, -1e9, dtype=attn_weights.dtype),
        attn_weights,
    )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    
    Args:
        q: The query tensor with shape (batch, num_heads, seq, head_dim).
        k: The key tensor with shape (batch, num_heads, seq, head_dim).
        cos: The cosine part of the rotary embedding with shape (3, batch, seq, head_dim) or (batch, seq, head_dim).
        sin: The sine part of the rotary embedding with shape (3, batch, seq, head_dim) or (batch, seq, head_dim).
        mrope_section: Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        mrope_interleaved: Whether to use interleaved mode.
        unsqueeze_dim: The dimension along which to unsqueeze cos and sin.
    
    Returns:
        Tuple of (q_embed, k_embed) rotated using the Rotary Position Embedding.
    """
    # Handle case where cos/sin have shape (3, batch, seq, head_dim) from multimodal RoPE
    # The original code expects this shape and processes it by:
    # 1. Splitting along head_dim into sections
    # 2. For each section, taking modality[i % 3]
    # 3. Concatenating all sections
    if cos.dim() == 4 and cos.shape[0] == 3:
        # cos/sin shape: (3, batch, seq, head_dim)
        if mrope_interleaved:
            def apply_interleaved_rope(x, modality_num):
                x_t = x[0].clone()
                index_ranges = []
                for i, n in enumerate(mrope_section[1:], 1):
                    beg_idx = i
                    end_idx = n * modality_num
                    index_ranges.append((beg_idx, end_idx))
                for beg_idx, end_idx in index_ranges:
                    x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
                return x_t

            dim = cos.shape[-1]
            modality_num = len(mrope_section)
            cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1)
            sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1)
        else:
            # Non-interleaved: split along head_dim, then take modality[i % 3] for each section
            # mrope_section sums to head_dim // 2, so we repeat it to get sections for cos/sin pairs
            # mrope_section * 2 repeats the list: [11, 11, 10] * 2 = [11, 11, 10, 11, 11, 10]
            mrope_section = mrope_section * 2
            # Split cos along last dim: (3, batch, seq, head_dim) -> list of (3, batch, seq, section_size)
            # Then for each section m, take m[i % 3] to get (batch, seq, section_size)
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    else:
        # cos/sin already have shape (batch, seq, head_dim) - process normally
        if mrope_interleaved:
            def apply_interleaved_rope(x, modality_num):
                x_t = x[0].clone()
                index_ranges = []
                for i, n in enumerate(mrope_section[1:], 1):
                    beg_idx = i
                    end_idx = n * modality_num
                    index_ranges.append((beg_idx, end_idx))
                for beg_idx, end_idx in index_ranges:
                    x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
                return x_t

            dim = cos.shape[-1]
            modality_num = len(mrope_section)
            cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1)
            sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1)
        else:
            # cos and sin should now be (batch, seq, head_dim)
            # mrope_section sums to head_dim // 2, so we repeat it to get sections for cos/sin pairs
            mrope_section = mrope_section * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    
    # Unsqueeze to match q/k shape: (batch, num_heads, seq, head_dim)
    # cos/sin should be (batch, seq, head_dim) -> (batch, 1, seq, head_dim) after unsqueeze
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
