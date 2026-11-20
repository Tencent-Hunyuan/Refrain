import os

import torch.utils.data as data
import torchvision.transforms as T
import random
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import torch
import numpy as np
import pyarrow.ipc as ipc
from pandas import DataFrame
import pyarrow as pa

class VideoDataset(data.Dataset):
    def __init__(self,
                 resolution_h=None,
                 resolution_w=None,
                 evaluator_type=None,
                 video_length=None,
                 is_gt=None,
                 gt_sample_index_path=None,
                 gt_arrow_file=None,
                 gen_video_path=None,
                 clip_tokenizer=None,
                 clip_caption_type=None,
                 clip_context_length=None,
                 timeout_seconds=60.0,
                 is_category_1=False,
                 is_category_2=False,
                 logger=None,
                 dataset_num=None
                 ):

        self.resolution_h = resolution_h
        self.resolution_w = resolution_w
        self.evaluator_type = evaluator_type
        self.video_length = video_length
        self.gen_video_path = gen_video_path
        self.gt_sample_index_path = gt_sample_index_path
        self.clip_tokenizer =clip_tokenizer
        self.timeout_seconds = timeout_seconds
        self.is_category_1 = is_category_1
        self.is_category_2 = is_category_2
        self.clip_caption_type = clip_caption_type
        self.clip_context_length = clip_context_length
        self.is_gt = is_gt

        if self.evaluator_type == "clip" and (not (self.clip_caption_type == "name" or self.clip_caption_type == "rename")):
            with open(clip_caption_type, 'rb') as f:
                reader = ipc.open_file(f)
                self.val_structure_caption_table = reader.read_all()
            self.val_structure_caption_table = self.val_structure_caption_table.slice(0, dataset_num)

        self.transform_ToPILImage = T.ToPILImage()

        if self.evaluator_type == "clip":
            self.resolution_h, self.resolution_w = 224, 224
            self.transform_Norm_clip = T.Compose([T.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                              (0.26862954, 0.26130258, 0.27577711))])

        self.transform_Ten_Re = T.Compose([
            T.ToTensor(),
            T.Resize(min(self.resolution_h, self.resolution_w), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
        ])
        self.transform_CCrop = T.CenterCrop((self.resolution_h, self.resolution_w))
        self.transform_Norm = T.Compose([T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        with open(gt_arrow_file, 'rb') as f:
            reader = ipc.open_file(f)
            self.gt_table = reader.read_all()

        if dataset_num is not None:
            self.gt_table = self.gt_table.slice(0, dataset_num)

        if not self.is_gt:
            self.get_gen_table()
            if dataset_num is not None:
                self.gen_table = self.gen_table.slice(0, dataset_num)

        if logger is not None:
            if self.is_gt:
                logger.info(f'[VideoDataset] number of videos: {len(self.gt_table)}')
            else:
                logger.info(f'[VideoDataset] number of videos: {len(self.gen_table)}')
        # ==============================================================================================================

        self.dynamic_fps = False
        self.temporal_transform = "rand_clips"
        self.get_gt_IdxToSampleIndex_mapping()

    def get_gen_table(self):
        names = [name for name in os.listdir(self.gen_video_path) if ".mp4" in name]
        nums = [int(name.split('.')[0]) for name in names]
        nums.sort()

        idx_list = []
        video_path_list = []
        for num in nums:
            idx_list.append(num)
            video_path_list.append(f"{self.gen_video_path}/{num}.mp4")

        data = {'idx': idx_list,
                "video_path": video_path_list}

        df = DataFrame(data)
        self.gen_table = pa.Table.from_pandas(df)

    def __len__(self):
        if self.is_gt:
            return len(self.gt_table)
        else:
            return len(self.gen_table)

    def get_gt_IdxToSampleIndex_mapping(self):
        if self.is_gt and self.gt_sample_index_path is not None:
            self.gt_IdxToSampleIndex_mapping = {}
            dictionary = np.load(f"{self.gt_sample_index_path}", allow_pickle=True).item()
            idx_np = dictionary["idx"]
            sample_index_np = dictionary["sample_index"]
            assert len(list(sample_index_np[0])) == self.video_length
            for i in range(idx_np.shape[0]):
                self.gt_IdxToSampleIndex_mapping[int(idx_np[i])] = list(sample_index_np[i])

    def get_sample_index(self, vr, idx):
        if self.is_gt:
            if self.gt_sample_index_path is not None:
                sample_index = self.gt_IdxToSampleIndex_mapping[idx]
            else:
                if self.dynamic_fps:
                    self.frame_stride = random.randint(2, 15)
                else:
                    self.frame_stride = 1
                framelst = list(range(0, len(vr), self.frame_stride))

                if len(framelst) < self.video_length:
                    self.frame_stride = len(vr) // (self.video_length + 1)
                    framelst = list(range(0, len(vr), self.frame_stride))

                if self.temporal_transform == 'rand_clips':
                    if len(framelst) > self.video_length:
                        valid_firstidx_list = [i for i in range(0, len(framelst) - self.video_length)]
                        valid_firstidx_list_len = len(valid_firstidx_list)
                        if valid_firstidx_list_len % 2 == 1:
                            firstidx = valid_firstidx_list_len // 2
                        else:
                            firstidx = valid_firstidx_list_len // 2 - 1
                        # firstidx = random.randint(0, len(framelst) - self.video_length)
                    else:
                        firstidx = 0
                else:
                    firstidx = 0

                sample_index = framelst[firstidx:firstidx + self.video_length]
        else:
            sample_index = [i for i in range(len(vr))]
            sample_index = sample_index[:self.video_length]

        return sample_index

    def transform_Norm_Evaluator_Type(self, frame):
        if self.evaluator_type == "fvd" or self.evaluator_type == "fid":
            pass
        elif self.evaluator_type == "inception":
            frame = self.transform_Norm(frame)
        elif self.evaluator_type == "clip":
            frame = self.transform_Norm_clip(frame)

        return frame

    def __getitem__(self, index):

        assert index == self.gt_table["idx"][index].as_py()
        if not self.is_gt:
            assert index == self.gen_table["idx"][index].as_py(), \
                f"index: {index}, gt: {self.gt_table['idx'][index].as_py()}, gen: {self.gen_table['idx'][index].as_py()}, " \
                f"gen_video_path: {self.gen_video_path}, gen_table: {self.gen_table}"

        if self.is_gt:
            idx = self.gt_table["idx"][index].as_py()
            video_path = self.gt_table["video_path"][index].as_py()
            vr = decord.VideoReader(video_path)
        else:
            idx = self.gen_table["idx"][index].as_py()
            video_path = self.gen_table["video_path"][index].as_py()
            vr = decord.VideoReader(video_path)

        sample_index = self.get_sample_index(vr, idx)

        frames = vr.get_batch(sample_index)
        frames = rearrange(frames, "f h w c -> f c h w")

        frames_tensor = []
        for frame in frames:
            frame_pil = self.transform_ToPILImage(frame.squeeze(0))
            frame = self.transform_Ten_Re(frame_pil)
            frame = self.transform_CCrop(frame)
            frame = self.transform_Norm_Evaluator_Type(frame)
            frame = frame.view(frame.size(0), 1, frame.size(1), frame.size(2))
            frames_tensor.append(frame)
        frames = torch.cat(frames_tensor, 1)

        example = dict()
        example["idx"] = idx
        example["frames"] = frames

        if self.is_gt:
            example["sample_index"] = torch.tensor(sample_index)

        if self.evaluator_type == "clip":

            if not (self.clip_caption_type == "name" or self.clip_caption_type == "rename"):
                assert self.gt_table["idx"][index].as_py() == self.val_structure_caption_table["idx"][index].as_py()
                prompt = self.val_structure_caption_table["structure_prompt"][index].as_py()
            else:
                if self.clip_caption_type == "name":
                    prompt = self.gt_table["name"][index].as_py()
                elif self.clip_caption_type == "rename":
                    prompt = self.gt_table["rename"][index].as_py()
                else:
                    raise ValueError("clip_caption_type must be name or rename !")

            text_emb = self.clip_tokenizer(prompt, context_length=self.clip_context_length, truncate=True).squeeze()
            example["text_emb"] = text_emb

        return example

