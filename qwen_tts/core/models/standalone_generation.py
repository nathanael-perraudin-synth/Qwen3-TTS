"""Standalone generation utilities to replace transformers GenerationMixin."""
import torch
from torch import nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .standalone_utils import TransformerOutput, DynamicCache


@dataclass
class GenerateOutput:
    """Output from generation."""
    sequences: torch.Tensor
    hidden_states: Optional[List[Tuple]] = None


def sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Sample from logits using top-k and/or top-p (nucleus) sampling.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        top_k: Keep only top k logits
        top_p: Keep tokens with cumulative probability <= top_p
        temperature: Temperature for sampling
        filter_value: Value to set for filtered tokens
    
    Returns:
        Sampled token indices of shape (batch_size,)
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k is not None and top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    # Sample from the filtered distribution
    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    
    return next_tokens


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """Apply repetition penalty to logits.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        input_ids: Previous token ids of shape (batch_size, seq_len)
        penalty: Repetition penalty factor (>1.0 penalizes, <1.0 encourages)
    
    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits
    
    # Create penalty mask
    for prev_id in input_ids.view(-1):
        if prev_id >= 0 and prev_id < logits.shape[-1]:
            if penalty > 1.0:
                logits[:, prev_id] /= penalty
            else:
                logits[:, prev_id] *= penalty
    
    return logits


def generate(
    model: nn.Module,
    inputs_embeds: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    min_new_tokens: int = 0,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    suppress_tokens: Optional[List[int]] = None,
    output_hidden_states: bool = False,
    return_dict_in_generate: bool = True,
    **kwargs,
) -> GenerateOutput:
    """Standalone generation function.
    
    Args:
        model: Model with forward method that returns logits
        inputs_embeds: Input embeddings (batch, seq, hidden)
        input_ids: Input token ids (batch, seq) - used if inputs_embeds is None
        max_new_tokens: Maximum number of tokens to generate
        min_new_tokens: Minimum number of tokens to generate
        do_sample: Whether to use sampling
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        temperature: Sampling temperature
        eos_token_id: End-of-sequence token ID
        repetition_penalty: Repetition penalty factor
        suppress_tokens: List of token IDs to suppress
        output_hidden_states: Whether to output hidden states
        return_dict_in_generate: Whether to return a dict-like object
        **kwargs: Additional arguments passed to model forward
    
    Returns:
        GenerateOutput with sequences and optional hidden_states
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Initialize sequences
    if inputs_embeds is not None:
        batch_size = inputs_embeds.shape[0]
        # For now, we'll track generated token IDs
        generated_ids = []
        past_key_values = None
        current_inputs_embeds = inputs_embeds
    elif input_ids is not None:
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.tolist()
        past_key_values = None
        current_inputs_embeds = None
    else:
        raise ValueError("Either inputs_embeds or input_ids must be provided")
    
    hidden_states_list = [] if output_hidden_states else None
    
    # Generation loop
    for step in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            if hasattr(model, 'forward'):
                # Handle different model signatures
                forward_kwargs = {
                    "inputs_embeds": current_inputs_embeds,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "output_hidden_states": output_hidden_states,
                    **kwargs,
                }
                
                # Remove None values
                forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}
                
                outputs = model(**forward_kwargs)
                
                # Extract logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get logits for next token (last position)
                if logits.dim() == 3:
                    next_token_logits = logits[:, -1, :]  # (batch, vocab)
                else:
                    next_token_logits = logits  # Already (batch, vocab)
                
                # Store hidden states if requested
                if output_hidden_states and hasattr(outputs, 'hidden_states'):
                    hidden_states_list.append(outputs.hidden_states)
                elif output_hidden_states and hasattr(outputs, 'last_hidden_state'):
                    hidden_states_list.append((outputs.last_hidden_state,))
                
                # Update past_key_values
                if hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
            else:
                raise ValueError("Model must have a forward method")
        
        # Apply repetition penalty
        if repetition_penalty != 1.0 and len(generated_ids) > 0:
            prev_ids = torch.tensor([gen[-1] for gen in generated_ids], device=device)
            next_token_logits = apply_repetition_penalty(next_token_logits, prev_ids.unsqueeze(0), repetition_penalty)
        
        # Suppress tokens
        if suppress_tokens is not None:
            next_token_logits[:, suppress_tokens] = -float("Inf")
        
        # Sample next token
        if do_sample:
            next_tokens = sample_top_k_top_p(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # Convert to list and append
        next_tokens_list = next_tokens.cpu().tolist()
        for i, token_id in enumerate(next_tokens_list):
            if i >= len(generated_ids):
                generated_ids.append([])
            generated_ids[i].append(token_id)
        
        # Check for EOS
        if eos_token_id is not None:
            all_eos = all(gen[-1] == eos_token_id for gen in generated_ids if len(gen) > 0)
            if all_eos and step >= min_new_tokens - 1:
                break
        
        # Prepare next iteration
        # For models that use inputs_embeds, we need to get embeddings for next tokens
        if inputs_embeds is not None:
            # Get embeddings for next tokens
            if hasattr(model, 'get_input_embeddings'):
                embeds = model.get_input_embeddings()
                if isinstance(embeds, nn.ModuleList):
                    # For code predictor with multiple embeddings
                    if hasattr(model, 'config') and hasattr(model.config, 'generation_steps'):
                        step_idx = model.config.generation_steps if hasattr(model.config, 'generation_steps') else step
                        if step_idx < len(embeds):
                            next_embeds = embeds[step_idx](next_tokens.unsqueeze(1))
                        else:
                            break  # No more embeddings
                    else:
                        next_embeds = embeds[0](next_tokens.unsqueeze(1))
                elif isinstance(embeds, nn.Module):
                    next_embeds = embeds(next_tokens.unsqueeze(1))
                else:
                    next_embeds = embeds(next_tokens.unsqueeze(1))
                
                current_inputs_embeds = next_embeds
            else:
                # Fallback: use input_ids if model supports it
                current_input_ids = torch.tensor(generated_ids, device=device)
                forward_kwargs = {
                    "input_ids": current_input_ids,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                }
                outputs = model(**forward_kwargs)
                if hasattr(outputs, 'last_hidden_state'):
                    current_inputs_embeds = outputs.last_hidden_state[:, -1:, :]
                else:
                    break
        else:
            # Update input_ids for next iteration
            current_input_ids = torch.tensor(generated_ids, device=device)
            forward_kwargs = {
                "input_ids": current_input_ids,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            outputs = model(**forward_kwargs)
            if hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values
    
    # Convert to tensor
    sequences = torch.tensor(generated_ids, device=device, dtype=torch.long)
    
    return GenerateOutput(
        sequences=sequences,
        hidden_states=hidden_states_list if output_hidden_states else None,
    )
