"""Standalone top-level Qwen3TTS model without transformers dependency."""
import torch
from torch import nn
from typing import Optional, List, Dict, Any, Tuple

from .standalone_talker_generation import StandaloneTalkerForConditionalGeneration
from .standalone_speaker_encoder import StandaloneSpeakerEncoder
from .standalone_config import StandaloneTalkerConfig, StandaloneSpeakerEncoderConfig


class StandaloneQwen3TTSForConditionalGeneration(nn.Module):
    """Standalone top-level Qwen3TTS model for conditional generation."""
    
    def __init__(
        self,
        talker_config: StandaloneTalkerConfig,
        speaker_encoder_config: Optional[StandaloneSpeakerEncoderConfig] = None,
        tts_model_type: str = "base",
        tokenizer_type: Optional[str] = None,
        tts_model_size: Optional[str] = None,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        tts_pad_token_id: int = 151671,
    ):
        super().__init__()
        self.config = type('Config', (), {
            'talker_config': talker_config,
            'speaker_encoder_config': speaker_encoder_config,
            'tts_model_type': tts_model_type,
            'tokenizer_type': tokenizer_type,
            'tts_model_size': tts_model_size,
            'tts_bos_token_id': tts_bos_token_id,
            'tts_eos_token_id': tts_eos_token_id,
            'tts_pad_token_id': tts_pad_token_id,
        })()
        
        self.talker = StandaloneTalkerForConditionalGeneration(talker_config)
        
        if tts_model_type == "base" and speaker_encoder_config is not None:
            self.speaker_encoder = StandaloneSpeakerEncoder(speaker_encoder_config)
        else:
            self.speaker_encoder = None
        
        self.speech_tokenizer = None
        self.generate_config = None
        
        self.supported_speakers = talker_config.spk_id.keys() if hasattr(talker_config, 'spk_id') else set()
        self.supported_languages = ["auto"]
        if hasattr(talker_config, 'codec_language_id'):
            for language_id in talker_config.codec_language_id.keys():
                if "dialect" not in language_id:
                    self.supported_languages.append(language_id)
        
        if speaker_encoder_config is not None:
            self.speaker_encoder_sample_rate = speaker_encoder_config.sample_rate
        else:
            self.speaker_encoder_sample_rate = 24000
        
        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type
    
    def load_speech_tokenizer(self, speech_tokenizer):
        """Load speech tokenizer."""
        self.speech_tokenizer = speech_tokenizer
    
    def load_generate_config(self, generate_config):
        """Load generation config."""
        self.generate_config = generate_config
    
    def get_supported_speakers(self):
        """Get supported speakers."""
        return self.supported_speakers
    
    def get_supported_languages(self):
        """Get supported languages."""
        return self.supported_languages
    
    @torch.no_grad()
    def extract_speaker_embedding(self, audio, sr):
        """Extract speaker embedding from audio."""
        assert sr == 24000, "Only support 24kHz audio"
        # Import mel_spectrogram function
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        device = next(self.speaker_encoder.parameters()).device
        dtype = next(self.speaker_encoder.parameters()).dtype
        speaker_embedding = self.speaker_encoder(mels.to(device).to(dtype))[0]
        return speaker_embedding
    
    @torch.no_grad()
    def generate_speaker_prompt(self, voice_clone_prompt: List[Dict]):
        """Generate speaker prompt from voice clone prompt."""
        voice_clone_spk_embeds = []
        device = next(self.talker.parameters()).device
        dtype = next(self.talker.parameters()).dtype
        
        for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
            ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index].to(device).to(dtype)
            voice_clone_spk_embeds.append(ref_spk_embedding)
        
        return voice_clone_spk_embeds
    
    @torch.no_grad()
    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        """Generate in-context learning prompt."""
        # text embed (ref id + text id + eos) 1 T1 D
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        
        # codec embed (codec bos + codec) 1 T2 D
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i-1](ref_code[:, i:i+1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat([
            self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_bos_id]],
                    device=self.talker.get_input_embeddings().weight.device,
                    dtype=text_id.dtype,
                )
            ),
            codec_embed
        ], dim=1)
        
        if non_streaming_mode:
            return text_embed + codec_embed, tts_pad_embed
        else:
            return text_embed + codec_embed, tts_pad_embed
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[List[torch.Tensor]] = None,
        instruct_ids: Optional[List[torch.Tensor]] = None,
        ref_ids: Optional[List[torch.Tensor]] = None,
        voice_clone_prompt: Optional[List[Dict]] = None,
        languages: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
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
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate tokens using the full model."""
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id
            if eos_token_id is not None
            else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": kwargs.get("output_hidden_states", True),
            "return_dict_in_generate": kwargs.get("return_dict_in_generate", True),
        }
        
        talker_input_embeds = [[] for _ in range(len(input_ids))]
        
        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        
        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        self.talker.text_projection(
                            self.talker.get_text_embeddings()(instruct_id)
                        )
                    )
        
        # tts text prompt generate
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        
        device = next(self.talker.parameters()).device
        dtype = next(self.talker.parameters()).dtype
        
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:  # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                            torch.tensor(
                                [[spk_id]],
                                device=device,
                                dtype=input_id.dtype,
                            )
                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None
            
            assert language is not None
            
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]
            
            if (
                language.lower() in ["chinese", "auto"]
                and speaker != ""
                and speaker is not None
                and self.config.talker_config.spk_is_dialect.get(speaker.lower(), False) is not False
            ):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]
            
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]
            
            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [[
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                ]]
            else:
                codec_prefill_list = [[
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    language_id,
                    self.config.talker_config.codec_think_eos_id,
                ]]
            
            codec_input_embedding_0 = self.talker.get_input_embeddings()(
                torch.tensor(
                    codec_prefill_list,
                    device=device,
                    dtype=input_id.dtype,
                )
            )
            codec_input_embedding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [[
                        self.config.talker_config.codec_pad_id,
                        self.config.talker_config.codec_bos_id,
                    ]],
                    device=device,
                    dtype=input_id.dtype,
                )
            )
            
            if speaker_embed is None:
                codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
            else:
                codec_input_embedding = torch.cat([
                    codec_input_embedding_0,
                    speaker_embed.view(1, 1, -1),
                    codec_input_embedding_1
                ], dim=1)
            
            # '<|im_start|>assistant\n我叫通义千问，是阿里云的开源大模型。<|im_end|>\n<|im_start|>assistant\n'
            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            
            # tts_pad * 4 + tts_bos
            _talker_input_embed = torch.cat((
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                tts_bos_embed,
            ), dim=1) + codec_input_embedding[:, :-1]
            
            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)
            
            if voice_clone_prompt is not None and voice_clone_prompt.get("ref_code") is not None and voice_clone_prompt["icl_mode"][index]:
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                # tts_text_first_token
                talker_input_embed = torch.cat([
                    talker_input_embed,
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 3:4])
                    ) + codec_input_embedding[:, -1:]
                ], dim=1)
                
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]  # Remove the text token
                    talker_input_embed = torch.cat([
                        talker_input_embed,
                        torch.cat((
                            self.talker.text_projection(
                                self.talker.get_text_embeddings()(input_id[:, 3:-5])
                            ),
                            tts_eos_embed
                        ), dim=1) + self.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                device=device,
                                dtype=input_id.dtype,
                            )
                        ),
                        tts_pad_embed + self.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.config.talker_config.codec_bos_id]],
                                device=device,
                                dtype=input_id.dtype,
                            )
                        )
                    ], dim=1)
                    trailing_text_hidden = tts_pad_embed
                else:
                    # 叫通义千问，是阿里云的开源大模型。
                    trailing_text_hidden = torch.cat((
                        self.talker.text_projection(
                            self.talker.get_text_embeddings()(input_id[:, 4:-5])
                        ),
                        tts_eos_embed
                    ), dim=1)
            
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
        
        # Concatenate all input embeddings
        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)
        
        # Batch inference - pad sequences
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds], device=device)
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])
        
        # Generate mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len, device=device).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(device)
        
        # Pad trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0
        )
        arange_tensor = torch.arange(
            max(trailing_text_original_lengths),
            device=padded_hiddens.device
        ).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens
        
        # Forward
        talker_result = self.talker.generate(
            inputs_embeds=[talker_input_embeds[i] for i in range(batch_size)],
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )
        
        # Process results
        if talker_result.hidden_states is not None and len(talker_result.hidden_states) > 0:
            talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
            talker_hidden_states = torch.cat([hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1)[:, :-1]
            
            first_codebook = talker_codes[:, :, 0]
            is_stop_token = (first_codebook == self.config.talker_config.codec_eos_token_id)
            stop_indices = torch.argmax(is_stop_token.int(), dim=1)
            has_stop_token = is_stop_token.any(dim=1)
            effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
            
            talker_codes_list = [talker_codes[i, :length, ] for i, length in enumerate(effective_lengths)]
            talker_hidden_states_list = [talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)]
        else:
            # Fallback if hidden_states not available
            talker_codes_list = [torch.empty(0, self.config.talker_config.num_code_groups, device=device, dtype=torch.long) for _ in range(batch_size)]
            talker_hidden_states_list = [torch.empty(0, self.config.talker_config.hidden_size, device=device) for _ in range(batch_size)]
        
        return talker_codes_list, talker_hidden_states_list
