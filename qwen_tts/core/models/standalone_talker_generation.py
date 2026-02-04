"""Standalone talker for conditional generation without transformers."""
import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .standalone_talker import BigTransformer, StandaloneResizeMLP
from ..configs.standalone_config import TalkerConfig
from .standalone_utils import TransformerOutput, can_return_tuple, DynamicCache
from .standalone_code_predictor_generation import StandaloneCodePredictorForConditionalGeneration
from .standalone_generation import sample_top_k_top_p, apply_repetition_penalty


@dataclass
class TalkerOutputWithPast:
    """Output from talker conditional generation."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Tuple] = None
    attentions: Optional[Tuple] = None
    past_hidden: Optional[torch.Tensor] = None
    generation_step: Optional[int] = None
    trailing_text_hidden: Optional[torch.Tensor] = None
    tts_pad_embed: Optional[torch.Tensor] = None


class StandaloneTalkerForConditionalGeneration(nn.Module):
    """Standalone talker for conditional generation."""
    
    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        self.model = BigTransformer(config)
        self.vocab_size = config.vocab_size
        self.text_projection = StandaloneResizeMLP(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size, config.hidden_act, bias=True
        )
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Create code predictor config from talker config
        from ..configs.standalone_config import CodePredictorConfig
        # Use code predictor vocab_size if available, otherwise fall back to talker vocab_size
        code_predictor_vocab_size = getattr(config, 'code_predictor_vocab_size', None)
        if code_predictor_vocab_size is None:
            code_predictor_vocab_size = config.vocab_size
        code_predictor_hidden_size = getattr(config, 'code_predictor_hidden_size', None)
        if code_predictor_hidden_size is None:
            code_predictor_hidden_size = config.hidden_size
        code_predictor_intermediate_size = getattr(config, 'code_predictor_intermediate_size', None)
        if code_predictor_intermediate_size is None:
            code_predictor_intermediate_size = config.intermediate_size
        code_predictor_num_layers = getattr(config, 'code_predictor_num_layers', None)
        if code_predictor_num_layers is None:
            code_predictor_num_layers = config.num_hidden_layers
        code_predictor_num_heads = getattr(config, 'code_predictor_num_heads', None)
        if code_predictor_num_heads is None:
            code_predictor_num_heads = config.num_attention_heads
        code_predictor_num_kv_heads = getattr(config, 'code_predictor_num_kv_heads', None)
        if code_predictor_num_kv_heads is None:
            code_predictor_num_kv_heads = config.num_key_value_heads
        
        code_predictor_config = CodePredictorConfig(
            vocab_size=code_predictor_vocab_size,
            hidden_size=code_predictor_hidden_size,
            intermediate_size=code_predictor_intermediate_size,
            num_hidden_layers=code_predictor_num_layers,
            num_attention_heads=code_predictor_num_heads,
            num_key_value_heads=code_predictor_num_kv_heads,
            head_dim=config.head_dim,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            num_code_groups=config.num_code_groups,
        )
        self.code_predictor = StandaloneCodePredictorForConditionalGeneration(
            config=code_predictor_config,
            embedding_dim=config.hidden_size,
        )
        self.rope_deltas = None
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def get_text_embeddings(self):
        return self.model.get_text_embeddings()
    
    def get_output_embeddings(self):
        return self.codec_head
    
    def forward_sub_talker_finetune(self, codec_ids, talker_hidden_states):
        """Forward pass for sub-talker fine-tuning."""
        assert len(codec_ids.shape) == 2
        assert len(talker_hidden_states.shape) == 2
        assert codec_ids.shape[0] == talker_hidden_states.shape[0]
        assert talker_hidden_states.shape[1] == self.config.hidden_size
        assert codec_ids.shape[1] == self.config.num_code_groups
        
        sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]
        
        for i in range(self.config.num_code_groups - 1):
            if i == 0:
                sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))
            else:
                sub_talker_inputs_embeds.append(self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1]))
        sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)
        
        sub_talker_outputs = self.code_predictor.forward_finetune(
            inputs_embeds=sub_talker_inputs_embeds,
            labels=codec_ids[:, 1:],
        )
        
        sub_talker_logits = sub_talker_outputs.logits
        sub_talker_loss = sub_talker_outputs.loss
        return sub_talker_logits, sub_talker_loss
    
    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ) -> TalkerOutputWithPast:
        """Forward pass."""
        # Prefill
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generate
        else:
            # this is where the higher codebook are predicted
            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1]) 
                   for i in range(self.config.num_code_groups - 1)],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)
            
            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed
        
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return TalkerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.hidden_states, codec_ids),
            attentions=outputs.attentions,
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )
    
    def get_rope_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the 3D rope index."""
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        
        return position_ids, mrope_position_deltas
    
    def generate(
        self,
        inputs_embeds: List[List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        trailing_text_hidden: Optional[torch.Tensor] = None,
        tts_pad_embed: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4096,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        suppress_tokens: Optional[List[int]] = None,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs,
    ):
        """Generate tokens using the talker model.
        
        This is a simplified generation that handles the special requirements of Qwen3TTS.
        """
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        batch_size = len(inputs_embeds)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'codec_eos_token_id', None)
        
        # Concatenate input embeddings
        talker_input_embeds = [torch.cat(embeds, dim=1) if isinstance(embeds, list) else embeds 
                               for embeds in inputs_embeds]
        max_len = max(emb.shape[1] for emb in talker_input_embeds)
        
        # Pad to same length
        padded_embeds = []
        attention_masks = []
        for emb in talker_input_embeds:
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                emb = torch.cat([emb, tts_pad_embed.expand(emb.shape[0], pad_len, -1)], dim=1)
            padded_embeds.append(emb)
            mask = torch.ones((emb.shape[0], emb.shape[1]), device=device, dtype=torch.float32)
            if pad_len > 0:
                mask[:, -pad_len:] = 0
            attention_masks.append(mask)
        
        talker_input_embeds = torch.cat(padded_embeds, dim=0)
        talker_attention_mask = torch.cat(attention_masks, dim=0)
        
        # Initialize generation state
        generated_ids = []
        hidden_states_list = [] if output_hidden_states else None
        past_key_values = None
        generation_step = 0
        self.rope_deltas = None
        
        # Generation loop
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Prepare inputs for this step
                if step == 0:
                    # First step: use all input embeddings
                    current_embeds = talker_input_embeds
                    current_mask = talker_attention_mask
                else:
                    # Subsequent steps: use only the last generated token
                    if len(generated_ids) == 0:
                        break
                    # Get last token embedding
                    last_ids = torch.tensor([[gen[-1]] for gen in generated_ids], device=device)
                    last_embeds = self.get_input_embeddings()(last_ids)
                    current_embeds = last_embeds
                    # Update attention mask
                    current_mask = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
                
                # Forward pass
                outputs = self.forward(
                    inputs_embeds=current_embeds,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    trailing_text_hidden=trailing_text_hidden,
                    tts_pad_embed=tts_pad_embed,
                    generation_step=generation_step,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_temperature=subtalker_temperature,
                    use_cache=True,
                    output_hidden_states=output_hidden_states,
                )
                
                logits = outputs.logits[:, -1, :]  # (batch, vocab)
                past_key_values = outputs.past_key_values
                
                if output_hidden_states:
                    hidden_states_list.append(outputs.hidden_states)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0 and len(generated_ids) > 0:
                    prev_ids = torch.tensor([[gen[-1] if len(gen) > 0 else 0] for gen in generated_ids], device=device)
                    logits = apply_repetition_penalty(logits, prev_ids, repetition_penalty)
                
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
                    if all_eos and step >= min_new_tokens - 1:
                        break
                
                generation_step += 1
        
        # Convert to tensor
        max_len = max(len(gen) for gen in generated_ids) if generated_ids else 0
        if max_len > 0:
            sequences = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
            for i, gen in enumerate(generated_ids):
                sequences[i, :len(gen)] = torch.tensor(gen, device=device, dtype=torch.long)
        else:
            sequences = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
        
        # Create output similar to transformers
        class GenerateOutput:
            def __init__(self, sequences, hidden_states=None):
                self.sequences = sequences
                self.hidden_states = hidden_states
        
        return GenerateOutput(
            sequences=sequences,
            hidden_states=hidden_states_list if output_hidden_states else None,
        )
