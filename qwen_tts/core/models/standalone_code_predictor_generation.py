"""Standalone code predictor for conditional generation without transformers."""
import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .standalone_code_predictor import StandaloneCodePredictorModel
from ..configs.standalone_config import CodePredictorConfig
from .standalone_utils import TransformerOutput, can_return_tuple
from .standalone_generation import generate, sample_top_k_top_p, apply_repetition_penalty


@dataclass
class CodePredictorOutputWithPast:
    """Output from code predictor."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None
    generation_steps: Optional[int] = None


class StandaloneCodePredictorForConditionalGeneration(nn.Module):
    """Standalone code predictor for conditional generation."""
    
    def __init__(
        self,
        config: CodePredictorConfig,
        embedding_dim: int,
    ):
        super().__init__()
        self.config = config
        self.model = StandaloneCodePredictorModel(config, embedding_dim)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        
        if config.hidden_size != embedding_dim:
            self.small_to_mtp_projection = nn.Linear(embedding_dim, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def forward_finetune(
        self,
        inputs_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CodePredictorOutputWithPast:
        """Forward pass for fine-tuning."""
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        
        logits = []
        for i in range(1, self.config.num_code_groups):
            if i < hidden_states.shape[1]:
                logits.append(self.lm_head[i-1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1) if logits else None
        
        loss = None
        if labels is not None and logits is not None:
            # Simple cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape for loss calculation
            if logits.dim() == 3:  # (batch, seq, vocab)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return CodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        generation_steps: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CodePredictorOutputWithPast:
        """Forward pass."""
        # Prefill stage
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        elif input_ids is not None and generation_steps is not None:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
        else:
            raise ValueError("Either inputs_embeds with seq_len > 1 or (input_ids, generation_steps) must be provided")
        
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head[generation_steps](hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return CodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )
    
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 10,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        suppress_tokens: Optional[list] = None,
        output_hidden_states: bool = False,
        return_dict_in_generate: bool = True,
        **kwargs,
    ):
        """Generate tokens using the code predictor.
        
        This handles the special case where we have multiple embeddings (one per code group).
        """
        self.eval()
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        batch_size = inputs_embeds.shape[0]
        generated_ids = []
        hidden_states_list = [] if output_hidden_states else None
        
        # Initial prefill
        # The inputs_embeds shape[1] represents the sequence length
        # For prefill, we process all embeddings at once
        # For generation, we process one token at a time
        past_key_values = None
        generation_step = 0
        
        # Prefill stage - process the initial inputs_embeds
        prefill_last_hidden = None  # used for first token logits without re-running model
        if inputs_embeds.shape[1] > 1:
            # This is the prefill stage
            current_embeds = self.small_to_mtp_projection(inputs_embeds)
            outputs = self.model(
                inputs_embeds=current_embeds,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )
            past_key_values = outputs.past_key_values
            # Save last hidden state for first token: original model uses prefill logits, does not re-run last token
            prefill_last_hidden = outputs.last_hidden_state[:, -1:, :]  # (batch, 1, hidden)
            if output_hidden_states:
                hidden_states_list.append(outputs.hidden_states)
            generation_step = inputs_embeds.shape[1] - 2
        else:
            # Generation stage - single token
            current_embeds = self.small_to_mtp_projection(inputs_embeds)
        
        # Generation loop
        for step in range(max_new_tokens):
            if generation_step >= len(self.lm_head):
                break
            
            with torch.no_grad():
                # First token after prefill: use prefill last hidden state (align with transformers: no extra forward)
                if prefill_last_hidden is not None:
                    hidden_states = prefill_last_hidden
                    prefill_last_hidden = None
                else:
                    # Forward pass: one new token with cache
                    step_embeds = current_embeds[:, -1:, :] if current_embeds.shape[1] > 1 else current_embeds
                    outputs = self.model(
                        inputs_embeds=step_embeds,
                        use_cache=True,
                        past_key_values=past_key_values,
                        output_hidden_states=output_hidden_states,
                    )
                    hidden_states = outputs.last_hidden_state
                    past_key_values = outputs.past_key_values
                    if output_hidden_states:
                        hidden_states_list.append(outputs.hidden_states)
                
                # Get logits for current generation step
                logits = self.lm_head[generation_step](hidden_states[:, -1, :])  # (batch, vocab)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and len(generated_ids) > 0:
                    prev_ids = torch.tensor([gen[-1] if len(gen) > 0 else 0 for gen in generated_ids], device=device)
                    logits = apply_repetition_penalty(logits, prev_ids.unsqueeze(0), repetition_penalty)
                
                # Suppress tokens
                if suppress_tokens is not None:
                    logits[:, suppress_tokens] = -float("Inf")
                
                # Sample next token
                if do_sample:
                    next_tokens = sample_top_k_top_p(
                        logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Convert to list and append
                next_tokens_list = next_tokens.cpu().tolist()
                for i, token_id in enumerate(next_tokens_list):
                    if i >= len(generated_ids):
                        generated_ids.append([])
                    generated_ids[i].append(token_id)
                
                # Check for EOS
                if eos_token_id is not None:
                    all_eos = all(gen[-1] == eos_token_id for gen in generated_ids if len(gen) > 0)
                    if all_eos:
                        break
                
                # Get embeddings for next iteration (project like original forward)
                if generation_step + 1 < len(self.model.get_input_embeddings()):
                    next_embeds = self.model.get_input_embeddings()[generation_step](next_tokens.unsqueeze(1))
                    next_embeds = self.small_to_mtp_projection(next_embeds)
                    current_embeds = next_embeds
                    generation_step += 1
                else:
                    break
        
        # Convert to tensor
        max_len = max(len(gen) for gen in generated_ids) if generated_ids else 0
        if max_len > 0:
            sequences = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
            for i, gen in enumerate(generated_ids):
                sequences[i, :len(gen)] = torch.tensor(gen, device=device, dtype=torch.long)
        else:
            sequences = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
        
        # Create output similar to transformers GenerateDecoderOnlyOutput
        class GenerateOutput:
            def __init__(self, sequences, hidden_states=None):
                self.sequences = sequences
                self.hidden_states = hidden_states
        
        return GenerateOutput(
            sequences=sequences,
            hidden_states=hidden_states_list if output_hidden_states else None,
        )
