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


def copy_model_weights(source_model, target_model):
    """Copy weights from source model to target model."""
    with torch.no_grad():

        # Copy speaker encoder
        target_model.speaker_encoder.load_state_dict(
            source_model.speaker_encoder.state_dict(), strict=True
        )

        # Copy talker model
        target_model.talker.model.load_state_dict(
            source_model.talker.model.state_dict(), strict=True
        )

        # Copy text projection
        target_model.talker.text_projection.load_state_dict(
            source_model.talker.text_projection.state_dict(), strict=True
        )

        # Copy codec head
        target_model.talker.codec_head.weight.data.copy_(
            source_model.talker.codec_head.weight.data
        )
            
    # Copy code predictor
    copy_code_predictor_weights(source_model.talker.code_predictor, target_model.talker.code_predictor)
