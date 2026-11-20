from .model import FrameATST
from .common import MinMax
import torch
import torchaudio
from torchvision import transforms


class ASL(torch.nn.Module):
    def __init__(self, ckpt_path, n_blocks=None):
        super().__init__()

        self.N_BLOCKS = 12
        self.n_blocks = n_blocks

        self.melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        self.normalize = MinMax(min=-79.6482,max=50.6842)

        self.model = self.load_model(ckpt_path)
    
    def load_model(self, ckpt_path):
        def load_checkpoint(model, checkpoint_state_dict):
            # Create a new state_dict with modified keys
            new_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                if k.startswith("model."):
                    new_key = k[len("model."):]  # Remove "model." prefix
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            # Load the modified state_dict
            # It's often good to use strict=False if you're not sure all keys will match,
            # or if you're intentionally loading a partial model.
            # However, the original code uses strict=False later, so we can be consistent.
            # For a helper function, it might be better to be more explicit or allow `strict` as a parameter.
            # For now, let's assume the main loading logic will handle strictness.
            model.load_state_dict(new_state_dict, strict=False)
            return model
        
        import os
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist"
            
        s = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hyper_param = s['hyper_parameters']
        pretrained_model = FrameATST(**hyper_param)

        # pretrained_model.load_state_dict(s['state_dict'], strict=False)
        pretrained_model = load_checkpoint(pretrained_model, s['state_dict'])
        pretrained_encoder = pretrained_model.teacher.encoder
        pretrained_encoder.hyper_param = s['hyper_parameters']

        pretrained_encoder.sample_rate = 16000
        pretrained_encoder.scene_embedding_size = pretrained_encoder.embed_dim*2*self.N_BLOCKS
        pretrained_encoder.timestamp_embedding_size = pretrained_encoder.embed_dim*self.N_BLOCKS

        pretrained_encoder.eval()
        pretrained_encoder.transform = transforms.Compose([
            self.melspec_t,
            self.to_db,
            self.normalize])

        return pretrained_encoder


    def forward(self, x: torch.Tensor):
        timestamp_embedding, timestamps = self.get_timestamp_embedding(x, n_blocks=self.n_blocks)
        return timestamp_embedding


    def get_timestamp_embedding(self, audio, n_blocks=None):
        """
        Extract frame-level embeddings from an audio clip 
        ==================================================
        args:
            audio: torch.tensor in the shape of [1,N] or [B,1,N] 
            model: the pretrained encoder returned by load_model 
        return:
            emb: retured embedding in the shape of [1,T,N_BLOCKS*emb_size] or [B,T,N_BLOCKS,emb_size], where emb_size is 768 for base model and 384 for small model.
            timestamps: timestamps in miliseconds
        """
        n_blocks = n_blocks if n_blocks is not None else self.N_BLOCKS
        
        if len(audio.shape)==2: 
            audio = audio.unsqueeze(1)
        else:
            assert len(audio.shape) == 3

        mel = self.model.transform(audio)
        output=[]

        chunk_len=1001 #10 secnods, consistent with the length of positional embedding

        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        for i in range(num_chunks):

            start = i*chunk_len
            end = (i+1) * chunk_len
            if end > total_len:
                end = total_len
            if end>start:
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(self.device)

                output_chunk = self.model.get_intermediate_layers(mel_chunk,len_chunk,n=n_blocks,scene=False)

                output.append(output_chunk)
        output=torch.cat(output,dim=1)
        length=output.shape[1]
        timestamps= (torch.arange(length)*40).float().unsqueeze(0).expand(mel.shape[0],-1)
        return output, timestamps

    # device
    @property
    def device(self):
        return next(self.parameters()).device
    
    # dtype
    @property
    def dtype(self):
        return next(self.parameters()).dtype


    # np.save("emb_timestamp.npy", emb_timestamp.cpu().numpy())    
