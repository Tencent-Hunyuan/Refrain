from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
import torch.nn as nn

# from hymm.metrics_video.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from hymm.metrics_video.fvd.videogpt.fvd import load_i3d_pretrained
from hymm.metrics_video.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
from hymm.metrics_video.clip_score.clip import clip
from hymm.metrics_video.clip_score.clip_score import forward_modality

from hymm.metrics_video.inception_score.inception_score import load_inception_pretrained, get_inception_score
from hymm.metrics_video.fid.fid_score import load_fid_pretrained, get_fid_score

class VideoEvaluator(nn.Module):
    def __init__(
        self,
        evaluator_type=None,
        device=None,
        pretrained_model_base_path=None,
        clip_model="ViT-B/32",
        inception_resize=True,
        inception_batch_size=16,
        fid_dims=2048,
        fid_batch_size=16,
    ):
        super().__init__()
        self.evaluator_type = evaluator_type
        self.device = device

        if self.evaluator_type == "fvd":
            self.fvd_model = load_i3d_pretrained(device=self.device,
                                                 pretrained_model_base_path=pretrained_model_base_path)
        elif self.evaluator_type == 'clip':
            self.clip_model, _ = clip.load(clip_model, device=self.device, download_root=pretrained_model_base_path)
        elif self.evaluator_type == 'inception':
            self.inception_model = load_inception_pretrained(device=self.device, pretrained_model_base_path=pretrained_model_base_path)
            self.inception_resize = nn.Upsample(size=(299, 299), mode='bilinear') if inception_resize else None
            self.inception_batch_size = inception_batch_size
        elif self.evaluator_type == 'fid':
            self.fid_model = load_fid_pretrained(device=self.device, pretrained_model_base_path=pretrained_model_base_path, dims=fid_dims)
            self.fid_batch_size = fid_batch_size
        else:
            raise ValueError("The evaluator type is not supported !")

    def fvd_forward(self, video):
        return get_fvd_feats(video.cpu(), i3d=self.fvd_model, device=self.device)

    def fid_forward(self, video):
        video_length = video.shape[2]

        n_rounds = video_length // self.fid_batch_size + 1 \
            if video_length % self.fid_batch_size != 0 \
            else video_length // self.fid_batch_size

        video = video.permute(0, 2, 1, 3, 4)[0].to(device=self.device)
        output_list = []

        for i in range(n_rounds):
            start_index = i * self.fid_batch_size
            end_index = min((i + 1) * self.fid_batch_size, video_length)
            video_clip = video[start_index:end_index]
            output = get_fid_score(video_clip, self.fid_model)
            output_list.append(output)

        output = torch.concat(output_list, dim=0).unsqueeze(0)  # torch.Size([1, t, 1000])

        return output

    def clip_forward(self, video, text_emb):
        assert text_emb is not None, "clip need input text_emb"

        logit_scale = self.clip_model.logit_scale.exp()
        output_list = []
        for t in range(video.shape[2]):
            video_clip = video[:, :, t]

            img_features = forward_modality(self.clip_model, video_clip, "img")
            txt_features = forward_modality(self.clip_model, text_emb, "txt")

            img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
            txt_features = txt_features / txt_features.norm(dim=1, keepdim=True).to(torch.float32)

            output = logit_scale * (img_features * txt_features).sum()
            output_list.append(torch.tensor([output.cpu().item()]))

        output = torch.concat(output_list, dim=0).unsqueeze(0)  # torch.Size([1, t])
        return output

    def inception_forward(self, video):
        video_length = video.shape[2]
        n_rounds = video_length // self.inception_batch_size + 1 \
            if video_length % self.inception_batch_size != 0 \
            else video_length // self.inception_batch_size

        video = video.permute(0, 2, 1, 3, 4)[0].to(device=self.device)

        output_list = []
        for i in range(n_rounds):
            start_index = i * self.inception_batch_size
            end_index = min((i + 1) * self.inception_batch_size, video_length)
            video_clip = video[start_index:end_index]
            output = get_inception_score(video_clip, self.inception_model, self.inception_resize)
            output_list.append(output)

        output = torch.concat(output_list, dim=0).unsqueeze(0)  # torch.Size([1, t, 1000])
        return output

    def forward(self, video, text_emb=None):

        if self.evaluator_type == "fvd":
            output = self.fvd_forward(video)
        elif self.evaluator_type == "fid":
            output = self.fid_forward(video)
        elif self.evaluator_type == "clip":
            output = self.clip_forward(video, text_emb)
        elif self.evaluator_type == "inception":
            output = self.inception_forward(video)
        else:
            raise ValueError("The evaluator type is not supported !")

        return output

