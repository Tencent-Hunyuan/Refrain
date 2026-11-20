import torch

@torch.no_grad()
def load_audio_encoders(enc_type: str, device: torch.device):
    enc_names = enc_type.split(',')
    encoders = []

    for enc_name in enc_names:
        if enc_name == "audiossl":
            from .audiossl.asl import ASL
            #TODO: using config
            ckpt_path="assets/local/resource_212d813261.ckpt"
            encoder = ASL(ckpt_path=ckpt_path, n_blocks=1)
            encoder = encoder.to(device)
            encoder.eval()
        elif enc_name == "eat":
            from transformers import AutoModel

            model_id = "worstchan/EAT-large_epoch20_pretrain"
            model_path = "assets/local/resource_b2487c054c" # 请将此路径替换为您的模型所在的实际路径
            encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True, local_files_only=True, cache_dir=model_path)

            encoder = encoder.to(device)
            encoder.eval()
        else:
            raise NotImplementedError(f"Encoder {enc_name} not implemented.")
        

        encoders.append(encoder)
    
    return encoders


