import torch


def copy_code_transformer_weights(code_transformer, target_code_transformer):
    with torch.no_grad():
        # Copy model weights
        target_code_transformer.load_state_dict(code_transformer.state_dict(), strict=True)

def copy_code_predictor_weights(original_code_predictor, target_code_predictor):
    # Copy weights from original to standalone
    copy_code_transformer_weights(original_code_predictor.model, target_code_predictor.model)
    
    with torch.no_grad():
        # Copy lm_head weights
        for i in range(len(target_code_predictor.lm_head)):
            target_code_predictor.lm_head[i].weight.data.copy_(
                original_code_predictor.lm_head[i].weight.data
            )
        
        # Copy projection 
        target_code_predictor.small_to_mtp_projection.load_state_dict(
            original_code_predictor.small_to_mtp_projection.state_dict(),
            strict=True
        )
        
def copy_talker_weights(original_talker, target_talker):
    with torch.no_grad():
        # Copy talker model
        target_talker.model.load_state_dict(
            original_talker.model.state_dict(), strict=True
        )
        # Re-initialize rotary embedding from target config so cos/sin head_dim matches
        # the decoder layers (state_dict may have copied different head_dim inv_freq).
        if hasattr(target_talker.model, "rotary_emb") and hasattr(
            target_talker.model.rotary_emb, "reinit_from_config"
        ):
            target_talker.model.rotary_emb.reinit_from_config(
                device=next(target_talker.model.parameters()).device
            )
        # Copy text projection
        target_talker.text_projection.load_state_dict(
            original_talker.text_projection.state_dict(), strict=True
        )

        # Copy codec head
        target_talker.codec_head.weight.data.copy_(
            original_talker.codec_head.weight.data
        )
            

        # Copy code predictor
        copy_code_predictor_weights(original_talker.code_predictor, target_talker.code_predictor)
        

def copy_model_weights(source_model, target_model):
    """Copy weights from source model to target model."""
    with torch.no_grad():

        # Copy speaker encoder
        target_model.speaker_encoder.load_state_dict(
            source_model.speaker_encoder.state_dict(), strict=True
        )



    # Copy talker weights
    copy_talker_weights(source_model.talker, target_model.talker)
