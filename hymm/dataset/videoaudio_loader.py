import random
import os
import io
import sys
import torch
import numpy as np
import json
import urllib.request
import math
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from pathlib import Path
from typing import List

import torchvision.transforms as T
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import ConcatDataset, ChainDataset
from omegaconf import OmegaConf, DictConfig

import traceback
import time
import requests
from socket import timeout

from decord import VideoReader
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from hymm.utils.data_utils import get_closest_ratio, generate_crop_size_list, get_temporal_closest_ratio, generate_crop_size_w_temporal_list
from hymm.dataset.format_prompt import MultilingualPromptFormat
from hymm.utils.distributed_utils import rank0_print
import pandas as pd
import re


def sum_aspect_ratio_nums(aspect_ratios_nums):
    return sum(num for ratio_dict in aspect_ratios_nums.values() for num in ratio_dict.values())


class VideoAudioArrowDataset(Dataset):
    def __init__(self, 
                arrow_file,                # arrow info存放路径
                audio_arrow_file,          # audio_arrow_file
                video_folder: str='',      # 视频存放 dir
                cache_folder: str='',      # vae latent 存放dir的list
                audio_cache_folder: str='', # audio vae latent 存放dir的list
                sample_size: list = None,   # 视频采样尺寸
                sample_n_frames: int=16,    # 帧采样帧数
                sample_stride: int=1,       # 帧采样间隔
                is_image: bool = False,     # 是否抽帧训图
                multireso: bool = False,    # 是否使用多分辨率桶
                multitemp: bool = False,    # 是否使用多时长桶
                bucket_hw_bucket_stride: int=16,         # 用于分桶计算 token 数
                bucket_hw_base_size: int=256, # 用于分桶的base size
                bucket_temporal_min_length: int=49,  # 用于时长分桶的最小长度
                bucket_temporal_max_length: int=65,  # 用于时长分桶的最小长度
                bucket_temporal_interval: int=1,  # 用于时长分桶的采样间隔
                aspect_ratio_index: dict=None, # 用于记录aspect ratio 每个桶所含index
                aspect_ratios_nums: dict=None, # 用于记录aspect ratio 每个桶所对应样本数量
                motion_threshold=10000.0,         # motion score 阈值
                text_encoder=None,
                text_encoder_2=None,
                text_len=256,               # text 长度
                video_uncond_p=0.0,         # 
                audio_uncond_p=0.0,         # 
                use_caption_prob=0.0,       # 使用caption的概率
                timeout_seconds=60.0,       # 下载视频的超时时间
                chunk_size=1024,            # 下载视频的chunk_size
                vae_suffix: str='.npy',     # vae latent 后缀
                vae_spatial_ratio: float=8, # vae latent 空间压缩比例
                args=None,                  # 其他参数
                caption_sample_ratio=None,  # caption 采样比例
                ocr_only_long_caption=False,
                text_remove_rate=0.0,
                num_replace_rate=0.0,       # 数字替换比例
                use_latent_shape=False,     # 是否使用 latent 的 shape 作为 bucket 的 shape
                filtered_video_ids=None,    # 过滤的video id list json file
                byt5_tokenizer=None,
                byt5_max_length=None,
                byt5_uncond_p=None,
                debug: bool = False,   
                logger=None,
                reserved_video_ids_trans_score=None,
                video_caption_col_probs: dict=None, # 多语言caption采样概率
                video_caption_col: str='structure_caption', # 英文caption列名
                video_caption_col_2: str='structure_caption_cn', # 中文caption列名
                skip_sources: list=None,    # 要跳过的source列表
                # audio settings
                audio_latent_dim=128,
                audio_sr=48000,
                use_audio_vae_cache=False,
                audio_vae_suffix=".npy",
                audio_wave_suffix=".m4a",
                audio_frame_rate=50,
                audio_caption_col='caption',
                use_merged_caption=True,
                ) -> None:
        self.args = args
        self.arrow_file = arrow_file
        self.audio_arrow_file = audio_arrow_file
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        self.multireso = multireso
        self.multitemp = multitemp
        self.bucket_hw_bucket_stride = bucket_hw_bucket_stride
        self.video_folder = video_folder
        self.cache_folder = cache_folder
        self.audio_cache_folder = audio_cache_folder
        self.sample_n_frames = sample_n_frames 
        self.sample_stride = sample_stride
        self.motion_threshold = motion_threshold
        self.is_image = is_image
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_len = text_len
        self.video_uncond_p = video_uncond_p
        self.audio_uncond_p = audio_uncond_p
        self.vae_suffix = vae_suffix
        self.use_caption_prob = use_caption_prob
        self.timeout_seconds = timeout_seconds
        self.caption_sample_ratio = json.dumps(OmegaConf.to_container(caption_sample_ratio)) if isinstance(caption_sample_ratio, DictConfig) else caption_sample_ratio
        self.ocr_only_long_caption = ocr_only_long_caption
        self.text_remove_rate = text_remove_rate
        self.num_replace_rate = num_replace_rate
        self.use_latent_shape = use_latent_shape
        self.chunk_size = chunk_size
        self.vae_spatial_ratio = vae_spatial_ratio
        self.byt5_tokenizer = byt5_tokenizer
        self.byt5_max_length = byt5_max_length
        self.byt5_uncond_p = byt5_uncond_p
        self.debug = debug
        # extra audio settings
        self.audio_latent_dim = audio_latent_dim
        self.audio_sr = audio_sr
        self.use_audio_vae_cache = use_audio_vae_cache
        self.audio_vae_suffix = audio_vae_suffix
        self.audio_wave_suffix = audio_wave_suffix
        self.audio_frame_rate = audio_frame_rate
        self.audio_caption_col = audio_caption_col

        self.use_merged_caption = use_merged_caption


        if logger is None:
            from loguru import logger
        self.logger = logger

        self.video_caption_col_probs = video_caption_col_probs
        self.video_caption_col = video_caption_col
        self.video_caption_col_2 = video_caption_col_2
        self.skip_sources = skip_sources if skip_sources is not None else []
        if self.video_caption_col_probs is not None:
            self.multi_caption_cols = []
            self.multi_caption_probs = []
            for col, prob in self.video_caption_col_probs.items():
                self.multi_caption_cols.append(getattr(self, col.strip('_prob'))) # map to video_caption_col
                self.multi_caption_probs.append(prob)

        if self.caption_sample_ratio is not None:
            self.use_structure_caption = True
            video_caption_processor = args.video_caption_processor
            # caption_process_v3 不需要替换下划线为空格
            if video_caption_processor == "caption_process":
                self.caption_sample_ratio = self.caption_sample_ratio.replace('_', ' ')
                kwargs={
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'text_remove_rate': self.text_remove_rate,
                }
            elif video_caption_processor == "caption_process_v3":
                kwargs = {
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'num_replace_rate': self.num_replace_rate,
                }
            else:
                raise NotImplementedError(f"video_caption_processor {video_caption_processor} not implemented")
            
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            from hymm.dataset.caps import load_caption_processor
            self.caption_aug = load_caption_processor(name=video_caption_processor,
                                                     caption_sample_ratio=self.caption_sample_ratio,
                                                     logger=self.logger,
                                                     kwargs=kwargs)
        else:
            self.use_structure_caption = False

        self.use_vae_cache = True
        self.cache_folder = cache_folder
        self.video_folder = video_folder
        # if cache_folder!='': self.use_vae_cache = True
        
        # load filtered video ids
        self.filtered_video_ids_df = None
        if filtered_video_ids is not None:
            assert filtered_video_ids.endswith('.jsonl'), f"filtered_video_ids must be jsonl file, but got {filtered_video_ids}"
            rank0_print(logger, f"load filtered video ids from {filtered_video_ids}")
            data = []
            with open(filtered_video_ids, 'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line))
            filtered_video_ids_df = pd.DataFrame(data)     
            filtered_video_ids_df['video_id'] = filtered_video_ids_df['video_id'].apply(lambda x: x[0])
            filtered_video_ids_df['url'] = filtered_video_ids_df['url'].apply(lambda x: x[0])
            self.filtered_video_ids_df =filtered_video_ids_df

        if reserved_video_ids_trans_score is not None:
            import pickle as pkl
            with open(reserved_video_ids_trans_score, 'rb') as f:
                self.reserved_videoids_filtered_by_trans = pkl.load(f)
        else:
            self.reserved_videoids_filtered_by_trans = None
            
        # load arrow
        self.table = self._load_table()
        s_time = time.time()
        rank0_print(logger, f"load {self.arrow_file} \t cost {time.time() - s_time} s \t total length {len(self.table)}")
        
        if multitemp and multireso:
            # 多时长+多分辨率桶
            # generate temporal bucket
            assert bucket_temporal_max_length>bucket_temporal_min_length, f"temporal_max_length must be larger than temporal_min_length, but got {bucket_temporal_max_length}, {bucket_temporal_min_length}"
            assert (bucket_temporal_min_length-1)%4==0, f"bucket_temporal_min_length-1 must be multiple of 4, but got {bucket_temporal_min_length}"
            assert (bucket_temporal_max_length-1)%4==0, f"bucket_temporal_max_length-1 must be multiple of 4, but got {bucket_temporal_max_length}"
            if bucket_temporal_max_length>sample_n_frames: bucket_temporal_max_length = sample_n_frames
            
            video_length_list = [fn+1 for fn in list(range(bucket_temporal_min_length-1, bucket_temporal_max_length-1, bucket_temporal_interval))]+[bucket_temporal_max_length]
            self.video_length_buckets = np.array(video_length_list)

            # 多分辨率桶
            assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"
            crop_size_list, spatial_crop_size_list = generate_crop_size_w_temporal_list(bucket_hw_base_size, bucket_hw_bucket_stride, temporal_list=self.video_length_buckets)
            assert sample_size in spatial_crop_size_list, f"sample_size must be in crop_size_list, current sample_size: {sample_size}"
            self.crop_size_list = crop_size_list
            self.spatial_crop_size_list = spatial_crop_size_list
            self.aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in spatial_crop_size_list])
            self.aspect_ratio_index = aspect_ratio_index
            self.aspect_ratios_nums = aspect_ratios_nums
            if aspect_ratio_index is None or aspect_ratios_nums is None:
                temporal_aspect_ratio_index = {}
                temporal_aspect_ratios_nums = {}
                for temp_bucket in self.video_length_buckets:
                    if int(temp_bucket) not in temporal_aspect_ratio_index: temporal_aspect_ratio_index[int(temp_bucket)] = {}
                    if int(temp_bucket) not in temporal_aspect_ratios_nums: temporal_aspect_ratios_nums[int(temp_bucket)] = {}
                    for k in self.aspect_ratios:
                        temporal_aspect_ratio_index[int(temp_bucket)][float(k)] = []     # used for self.getitem
                        temporal_aspect_ratios_nums[int(temp_bucket)][float(k)] = 0      # used for batch-sampler
                    
                    # scan the dataset for ratio static
                    for idx in tqdm.tqdm(range(len(self.table))):
                        if use_latent_shape:
                            latent_shape = self.table['latent_shape'][idx].as_py()
                            if isinstance(latent_shape, str):
                                latent_shape = eval(latent_shape)
                            assert isinstance(latent_shape, list), "latent_shape must be list"
                            num_frames = latent_shape[-3]
                            height = latent_shape[-2]
                            width = latent_shape[-1]
                            num_frames = (num_frames-1)*4+1
                        else:
                            height = int(float(self.table['height'][idx].as_py()))
                            width = int(float(self.table['width'][idx].as_py()))
                            fps = float(self.table['fps'][idx].as_py())
                            duration = float(self.table['duration'][idx].as_py())
                            num_frames = int(fps*duration)
                        closest_temp, closest_size, closest_ratio = get_temporal_closest_ratio(num_frames, height, width, self.aspect_ratios, self.video_length_buckets, self.spatial_crop_size_list)
                        temporal_aspect_ratios_nums[closest_temp][float(closest_ratio)] += 1
                        temporal_aspect_ratio_index[closest_temp][float(closest_ratio)].append(idx)
                
                self.aspect_ratio_index = temporal_aspect_ratio_index
                self.aspect_ratios_nums = temporal_aspect_ratios_nums
            else:
                # convert key type
                new_aspect_ratio_index = {
                    int(temp): {float(aspect_ratio): indexs for aspect_ratio, indexs in aspect_ratios2index.items()}
                    for temp, aspect_ratios2index in aspect_ratio_index.items()
                }

                new_aspect_ratios_nums = {
                    int(temp): {float(aspect_ratio): nums for aspect_ratio, nums in aspect_ratios2num.items()}
                    for temp, aspect_ratios2num in aspect_ratios_nums.items()
                }

                self.aspect_ratio_index = new_aspect_ratio_index
                self.aspect_ratios_nums = new_aspect_ratios_nums
                    
        elif multireso:
            assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"
            crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
            assert sample_size in crop_size_list, f"sample_size must be in crop_size_list, current sample_size: {sample_size}"
            self.crop_size_list = crop_size_list
            self.aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
            self.aspect_ratio_index = aspect_ratio_index
            self.aspect_ratios_nums = aspect_ratios_nums
            if aspect_ratio_index is None or aspect_ratios_nums is None:
                self.aspect_ratio_index = {}
                self.aspect_ratios_nums = {}
                for k in self.aspect_ratios:
                    self.aspect_ratio_index[float(k)] = []     # used for self.getitem
                    self.aspect_ratios_nums[float(k)] = 0      # used for batch-sampler
                
                # scan the dataset for ratio static
                rank0_print(logger, f"Scan the dataset for aspect ratio static")
                for idx in tqdm.tqdm(range(len(self.table))):
                    height = int(float(self.table['height'][idx].as_py()))
                    width = int(float(self.table['width'][idx].as_py()))
        
                    closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratios, self.crop_size_list)
                    self.aspect_ratios_nums[closest_ratio] += 1
                    if len(self.aspect_ratio_index[closest_ratio]) < 10:
                        self.aspect_ratio_index[closest_ratio].append(idx)
            else:
                # convert key type
                self.aspect_ratio_index = {float(k):v for k, v in aspect_ratio_index.items()}
                self.aspect_ratios_nums = {float(k):v for k, v in aspect_ratios_nums.items()}


        self.logic2physical_index_map, self.physical2logic_index_map = self.generate_index_mapping()
            
        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        # self.prompt_format = MultilingualPromptFormat(font_path=self.args.multilingual_prompt_format_font_path,
        #                                               color_path=self.args.multilingual_prompt_format_color_path)
    
    def remap_path(self, path):
        # if path.startswith('assets/local/resource_f460d99100'):
        #     return path.replace('assets/local/resource_f460d99100', 'assets/local/resource_3c7ad9bc5d')
        # elif path.startswith('assets/local/resource_ec6a0c0b4e'):
        #     return path.replace('assets/local/resource_ec6a0c0b4e', 'assets/local/resource_cf46f8492a')
        # else:
        return path
    
    def _load_table(self):
        self.arrow_file = self.remap_path(self.arrow_file)
        table = pa.memory_map(self.arrow_file, "r")
        table = pa.ipc.RecordBatchFileReader(table).read_all()  # 打印第一条数据，0.0013秒
        return table

    def generate_index_mapping(self):
        logic2physical_index_mapping = {}
        physical2logic_index_mapping = {}
        count = 0
        
        if self.multitemp and self.multireso:
            for _, ratio_dict in self.aspect_ratio_index.items():
                for _, indices in ratio_dict.items():
                    for idx in indices:
                        logic2physical_index_mapping[count] = idx
                        physical2logic_index_mapping[idx] = count
                        count += 1
        elif self.multireso:
            for _, indices in self.aspect_ratio_index.items():
                for idx in indices:
                    logic2physical_index_mapping[count] = idx
                    physical2logic_index_mapping[idx] = count
                    count += 1
        else:
            for idx in range(len(self.table)):
                logic2physical_index_mapping[count] = idx
                physical2logic_index_mapping[idx] = count
                count += 1
        
        return logic2physical_index_mapping, physical2logic_index_mapping
    
    def __len__(self):
        if self.debug:
            return min(len(self.logic2physical_index_map), 40960)
        return len(self.logic2physical_index_map)
    
    def get_data_info(self, index, index_type="logic"): 
        '''
        index_type: logic or physical
        '''
        if index_type == "logic":
            # convert to physical index
            index = self.logic2physical_index_map[index]

        if self.use_latent_shape:
            latent_shape = self.table['latent_shape'][index].as_py()
            if isinstance(latent_shape, str):
                latent_shape = eval(latent_shape)
            assert isinstance(latent_shape, list), "latent_shape must be list"
            num_frames = latent_shape[-3]
            height = latent_shape[-2]
            width = latent_shape[-1]
            num_frames = (num_frames-1)*4+1
            motion_score = float(self.table['max_motion_score_v2'][index].as_py()) if 'max_motion_score_v2' in self.table.column_names else None
        else:
            height = int(float(self.table['height'][index].as_py()))
            width = int(float(self.table['width'][index].as_py()))
            fps = float(self.table['fps'][index].as_py())
            duration = float(self.table['duration'][index].as_py())
            num_frames = int(fps*duration)
            motion_score = float(self.table['max_motion_score_v2'][index].as_py()) if 'max_motion_score_v2' in self.table.column_names else None
            # motion_score = float(self.table['motion_score'][idx].as_py())
        return {'height': height, 
                'width': width,
                'num_frames': num_frames,
                'motion_score': motion_score}
        
    @staticmethod
    def get_text_tokens(text_encoder, description, text_len):
        text_inputs = text_encoder.text2tokens(description, data_type='video', max_length=text_len)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask
    
    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    def get_raw_video(self, videoid, video_folder, cos_url, sample_stride, sample_n_frames, is_image=False):
        if video_folder == '':
            cos_signed_url = cos_url
            if cos_signed_url == '':
                self.logger.info(f'Wrong data item!')
                raise Exception(f'cosurl无效:{videoid}')
            
            # step1: 优先 load resized dataset
            video_path = 'assets/local/resource_73a37c647c' + videoid+'.mp4'
            
            # step2: dop data
            if not os.path.exists(video_path) or os.stat(video_path).st_size == 0:
                video_root = 'assets/local/resource_c2645c471d'
                video_name = cos_signed_url.split('train_data')[-1].split('?')[0]
                video_path = video_root + video_name
                
            # step3: cos_signed_url
            if not os.path.exists(video_path):
                video_path = cos_signed_url
        else:
            # local
            video_path  = os.path.join(video_folder, f"{videoid}.mp4")
        
        if video_path.startswith('http'): 
            session = requests.Session()
            data = b''
            with session.get(video_path, stream=True, timeout=10) as response:
                response.raise_for_status()  # 检查请求是否成功
                stime = time.time()
                for chunk in response.iter_content(chunk_size=1024):
                    # print(chunk)
                    if chunk:  # 过滤掉保持活动连接的新块
                        data += chunk
                    cost = time.time() - stime
                    if cost > self.timeout_seconds:
                        raise timeout(f"Downloading video timeout: {video_path}")
            video_reader = VideoReader(io.BytesIO(data))
        else:
            video_reader = VideoReader(video_path)

        video_length = len(video_reader)
            
        if not is_image:
            framelst = list(range(0, video_length, sample_stride))
            if len(framelst) < sample_n_frames:
                framelst = list(range(0, video_length, 1))
                if len(framelst) < sample_n_frames:
                    framelst = list(np.linspace(0, video_length-4, sample_n_frames, dtype=int))

            if len(framelst)>sample_n_frames:
                start_idx = random.randint(0,len(framelst)-sample_n_frames)
            else:
                start_idx = 0
            batch_index = framelst[start_idx:start_idx+sample_n_frames]
        else:
            batch_index = [random.randint(0, video_length - 1)]
        try:
            video_images = video_reader.get_batch(batch_index)
        except Exception as e:
            raise Exception(f'wrong video length: {str(e)}, {video_length}, {batch_index}')

        pixel_values = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values/255.
            
        del video_reader

        if is_image:
            pixel_values = pixel_values[0]
        return pixel_values

    def prompt_valid_check(self, prompt, another_prompt):
        if prompt == "\\N" or prompt == "":
            prompt = another_prompt
            if prompt == "\\N" or prompt == "":
                self.logger.info(f'Empty caption: {prompt}')
                raise Exception('Wrong Caption')
        return prompt
    
    def path_remap(self, path):
        if path.startswith('assets/local/resource_f460d99100'):
            return path.replace('assets/local/resource_f460d99100', 'assets/local/resource_3c7ad9bc5d')
        elif path.startswith('assets/local/resource_ec6a0c0b4e'):
            return path.replace('assets/local/resource_ec6a0c0b4e', 'assets/local/resource_cf46f8492a')
        else:
            return path

    def get_batch(self, idx, index_type="logic"):
        '''
        inputs:
            idx: index
            index_type: logic or physical
        '''
        if index_type == "logic":
            idx = self.logic2physical_index_map[idx]

        # 根据索引获取记录
        if 'videoid' in self.table.schema.names:
            video_key = 'videoid'
        elif 'source_id' in self.table.schema.names:
            video_key = 'source_id'
        elif 'video_id' in self.table.schema.names:
            video_key = 'video_id'
        else:
            raise Exception(f'videoid or source_id not in self.table.schema.names')

        self.video_key = video_key
        videoid = self.table[video_key][idx].as_py()
        cos_url = ''
        schema = self.table.schema
        if schema.get_field_index('cos_url')==1: cos_url = self.table['cos_url'][idx].as_py()
        elif schema.get_field_index('contentUrl')==1: cos_url = self.table['contentUrl'][idx].as_py()

        pixel_values = torch.tensor(0)
        latents = torch.tensor(0)
        pixel_transforms = None
        sample_n_frames = self.sample_n_frames
        if self.multitemp and self.multireso:
            data_info = self.get_data_info(idx, index_type="physical")
            num_frames, height, width = data_info['num_frames'], data_info['height'], data_info['width']
            closest_temp, closest_size, closest_ratio = get_temporal_closest_ratio(num_frames, height, width, self.aspect_ratios, self.video_length_buckets, self.spatial_crop_size_list)
            self.closest_temp = closest_temp
            self.closest_ratio = closest_ratio
            sample_n_frames = closest_temp
            
            if data_info['motion_score'] is not None and data_info['motion_score'] > self.motion_threshold:
                ms = data_info['motion_score']
                raise ValueError(f'motion score {ms} too high')
            
            closest_size = list(map(lambda x: int(x), closest_size))
            if closest_size[0] / height > closest_size[1] / width:
                resize_size = closest_size[0], int(width * closest_size[0] / height)
            else:
                resize_size = int(height * closest_size[1] / width), closest_size[1]
            
            pixel_transforms = transforms.Compose([
                transforms.Resize(resize_size, antialias=True),
                transforms.CenterCrop(closest_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            
        elif self.multireso:
            data_info = self.get_data_info(idx, index_type="physical")
            num_frames, height, width = data_info['num_frames'], data_info['height'], data_info['width']
            closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratios, self.crop_size_list)
            closest_size = list(map(lambda x: int(x), closest_size))
            self.closest_ratio = closest_ratio
            
            if data_info['motion_score'] is not None and data_info['motion_score'] > self.motion_threshold:
                ms = data_info['motion_score']
                raise ValueError(f'motion score {ms} too high')
            
            if closest_size[0] / height > closest_size[1] / width:
                resize_size = closest_size[0], int(width * closest_size[0] / height)
            else:
                resize_size = int(height * closest_size[1] / width), closest_size[1]
            
            pixel_transforms = transforms.Compose([
                transforms.Resize(resize_size, antialias=True),
                transforms.CenterCrop(closest_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        
        # filter video after online bucketing, otherwise the self.closest_temp and self.closest_ratio is wrong
        if self.filtered_video_ids_df is not None and videoid in self.filtered_video_ids_df['video_id'].values:
            raise Exception(f'Video id {videoid} is filtered')
        
        if self.reserved_videoids_filtered_by_trans is not None and (not videoid in self.reserved_videoids_filtered_by_trans):
            raise Exception(f'Video id {videoid} is filtered by transition score')
        
        # filter by source field
        if self.skip_sources and 'source' in self.table.schema.names:
            source_value = self.table['source'][idx].as_py()
            if source_value in self.skip_sources:
                raise Exception(f'Video with source {source_value} is skipped')

        # get image / latent
        if self.use_vae_cache: # 如果使用 vae cache
            cache_folder = self.cache_folder
            if 'latent_cos_path' in self.table.schema.names:
                npy_videoid=self.table['latent_cos_path'][idx].as_py()
            else:
                npy_videoid=self.table['npy_videoid'][idx].as_py()
            if not npy_videoid.endswith('npy'):
                cache_key=npy_videoid+'.npy'
            else:
                cache_key=npy_videoid
            cache_path = f'{cache_folder}/{cache_key}'
            latents = torch.from_numpy(np.load(cache_path)).squeeze(0)
            # 检查是否包含NaN值
            if torch.isnan(latents).any():
                self.logger.warning(f"videoid: {videoid} contains NaN values in cached latents at {cache_path}")
                raise Exception(f'videoid: {videoid} has NaN values in cached latents')
            sample_n_latent = (sample_n_frames - 1) // 4 + 1
            start_idx = 0 # 目前默认 start_idx 从首帧开始，而非随机的一个起始点
            latents = latents[:,start_idx:start_idx+sample_n_latent,...]
            # 如果采用多分辨率分桶，latent的尺寸和桶的尺寸不一致，则报异常
            if self.multireso and (latents.shape[-2]*self.vae_spatial_ratio != closest_size[0] or latents.shape[-1]*self.vae_spatial_ratio != closest_size[1]):
                raise Exception(f' videoid: {videoid} has wrong cache data for aspect ratio of shape {latents.shape}, expected shape {closest_size}')
            # 如果采用时长分桶，但是由于fps*duration计算的帧数得到的，buckets和latent长度不一样，则报异常
            if latents.shape[1] < sample_n_latent:
                raise Exception(f' videoid: {videoid} has wrong cache data for temporal buckets of shape {latents.shape}, expected length: {sample_n_latent}')
        else:
            video_folder = self.video_folder
            pixel_values = self.get_raw_video(videoid, video_folder, cos_url, self.sample_stride, sample_n_frames, self.is_image)
            if pixel_transforms is None: pixel_values = self.pixel_transforms(pixel_values)
            else: pixel_values = pixel_transforms(pixel_values)

        # get prompt
        if self.use_structure_caption:
            video_caption_processor = self.args.video_caption_processor
            if video_caption_processor == "caption_process_v3":
                assert self.video_caption_col in self.table.schema.names, f"{self.video_caption_col} not in self.table.schema.names"
                has_zh_caption = self.video_caption_col_2 in self.table.schema.names

                lang = random.choices(
                    self.multi_caption_cols, weights=self.multi_caption_probs, k=1
                )[0]
                lang = 'zh' if '_cn' in lang else 'en' # e.g., structure_caption_cn -> zh, structure_caption -> en
                text_en = self.table[self.video_caption_col][idx].as_py()
                text_zh = self.table[self.video_caption_col_2][idx].as_py() if has_zh_caption else text_en
                camera_movement = self.table['camera_movement'][idx].as_py() if 'camera_movement' in self.table.schema.names else None
                prompt = self.caption_aug.caption_aug(text_en, text_zh, lang, camera_movement=camera_movement)
            else:
                assert "structure_caption" in self.table.schema.names, f"structure_caption not in self.table.schema.names"
                structure_caption_str = self.table['structure_caption'][idx].as_py()
                prompt = self.caption_aug.caption_aug(structure_caption_str)
        else:
            name = self.table['name'][idx].as_py()
            rename = self.table['rename'][idx].as_py()

            # get text
            prompt = rename
            ori_prompt = name

            if ori_prompt != '':
                prompt = prompt if random.random() < self.use_caption_prob else ori_prompt

            prompt = self.prompt_valid_check(prompt, ori_prompt)

        # if random.random() < self.video_uncond_p:
        #     prompt = ''
            
        # # get text token id
        # text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt, self.text_len)
        
        # # get byt5 text token id
        # byt5_text_ids = torch.zeros((self.byt5_max_length), dtype=torch.int64)
        # byt5_text_mask = torch.zeros((self.byt5_max_length), dtype=torch.int64)
        # byt5_text_valid = torch.tensor(0, dtype=torch.int64)
        # if self.args.glyph_byT5_v2:

        #     pattern = r'\"(.*?)\"|"(.+?)"'
        #     matches = re.findall(pattern, prompt)
        #     glyph_byT5_text_list = [match[0] or match[1] for match in matches]
        #     glyph_byT5_text_list = list(dict.fromkeys(glyph_byT5_text_list)) if len(glyph_byT5_text_list) > 1 else glyph_byT5_text_list

        #     if len(glyph_byT5_text_list) > 0:
        #         if random.random() < self.byt5_uncond_p:
        #             glyph_byT5_text_list = ["" for _ in range(len(glyph_byT5_text_list))]

        #         text_prompt_style_list = [{'color': None, 'font-family': None} for _ in range(len(glyph_byT5_text_list))]
        #         glyph_byT5_text_formatted = self.prompt_format.format_prompt(glyph_byT5_text_list, text_prompt_style_list)

        #         byt5_text_ids, byt5_text_mask = self.get_byt5_text_tokens(self.byt5_tokenizer, self.byt5_max_length,
        #                                                                   glyph_byT5_text_formatted)
        #         byt5_text_ids = byt5_text_ids.squeeze(0)
        #         byt5_text_mask = byt5_text_mask.squeeze(0)
        #         byt5_text_valid = torch.tensor(1, dtype=torch.int64)

        # kwargs = {
        #     "text": prompt,
        #     "index": idx,
        #     "type": 'video',
        #     "dataset_type": 'video',
        #     'bucket': [num_frames, height, width],
        #     "videoid": videoid,
        # }
        # if self.args.glyph_byT5_v2:
        #     if self.text_encoder_2 is None:
        #         return (
        #             pixel_values,
        #             latents,
        #             audio_latents,
        #             text_ids.clone(),
        #             text_mask.clone(),
        #             byt5_text_ids.clone(),
        #             byt5_text_mask.clone(),
        #             byt5_text_valid,
        #             {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        #         )
        #     else:
        #         text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt, self.text_len)
        #         return (
        #             pixel_values,
        #             latents,
        #             audio_latents,
        #             text_ids.clone(),
        #             text_mask.clone(),
        #             text_ids_2.clone(),
        #             text_mask_2.clone(),
        #             byt5_text_ids.clone(),
        #             byt5_text_mask.clone(),
        #             byt5_text_valid,
        #             {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        #         )
        # else:
        #     if self.text_encoder_2 is None:
        #         return (
        #             pixel_values,
        #             latents,
        #             audio_latents,
        #             text_ids.clone(),
        #             text_mask.clone(),
        #             {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        #         )
        #     else:
        #         text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt, self.text_len)
        #         return (
        #             pixel_values,
        #             latents,
        #             audio_latents,
        #             text_ids.clone(),
        #             text_mask.clone(),
        #             text_ids_2.clone(),
        #             text_mask_2.clone(),
        #             {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
        #         )
        
        # if 'latent_cos_path' in self.table.schema.names:
        #     audioid=self.table['latent_cos_path'][idx].as_py()
        # else:
        #     audioid=self.table['npy_videoid'][idx].as_py()
        if "npy_audioid" in self.table.schema.names:
            audioid = self.table["npy_audioid"][idx].as_py()
        else:
            audioid = videoid
        audio_file_path = self.table["audio_path"][idx].as_py()
        # try:
        audio_file_path = self.remap_path(audio_file_path) # comment out if no multi gy cephs
        # except Exception as e:
        #     set_trace(term_size=(200, 52))
        #     raise Exception(f'Remap path error: {e}.')
        """
        calculate audio latent length from sample_n_frames
        audio_latent_len = round((sample_n_frames - 1) / 16(fps) * 50(hz))
        audio_latent_len_sampling_points = round((sample_n_frames - 1) / 16(fps) * 48000)
        """
        audio_latent_len_in_second = (sample_n_frames - 1) / (self.args.video_fps * 1.0)
        video_best_frame = self.table["best_frame_index"][idx].as_py()
        video_frame_rate = self.table["frame_rate"][idx].as_py()
        audio_start_frame_in_second = video_best_frame / video_frame_rate
        if self.use_audio_vae_cache: 
            audio_latent_len = 400 # TODO: automatically calculate audio latent length from audio_latent_len_in_second
            audio_cache_mean_path = os.path.join(self.audio_cache_folder, audioid+ '_mean' + self.audio_vae_suffix)
            audio_cache_std_path = os.path.join(self.audio_cache_folder, audioid + '_std' + self.audio_vae_suffix)
            # 使用 .clone().detach() 确保张量有独立的存储空间
            audio_mean = torch.from_numpy(np.load(audio_cache_mean_path)).clone()
            audio_std = torch.from_numpy(np.load(audio_cache_std_path)).clone()
            # sample from vae
            # audio_latents = audio_mean + audio_std * torch.randn(audio_std.shape, dtype=torch.float32) # [D, T]
            # audio_latents = audio_latents.clone().detach()
            # 确保audio_latents有固定的形状 [D, audio_latent_len]
            if audio_mean.shape[1] >= audio_latent_len:
                # 如果音频长度足够，截取固定长度
                audio_latents = audio_mean[:, :audio_latent_len]
            else:
                # 如果音频长度不够，进行padding
                audio_latents = torch.zeros(audio_mean.shape[0], audio_latent_len, dtype=torch.float32)
                audio_latents[:, :audio_mean.shape[1]] = audio_mean
        else:
            import torchaudio
            from torchaudio.transforms import Resample
            
            try:
                audio_caption = self.table[self.audio_caption_col][idx].as_py()
                audio_caption = "" if audio_caption is None else audio_caption
                transcript_text = self.table["asr_text"][idx].as_py() if "asr_text" in self.table.schema.names else ""
                transcript_text = "" if transcript_text is None else transcript_text
                for col in ["language", "asr_language"]:
                    if col in self.table.schema.names:
                        language_tag = self.table[col][idx].as_py()
                        break
                language_tag = "en" if language_tag is None else language_tag

                # FIXME: support only en & zh for now
                if language_tag not in ["en", "zh", "ja", "ko"]:
                    language_tag = "en"
                    transcript_text = ""

                # if random.random() < self.audio_uncond_p:
                #     audio_caption = ""
                #     transcript_text = ""
                waveform, sample_rate = torchaudio.load(audio_file_path)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sample_rate != self.audio_sr:
                    resampler = Resample(orig_freq=sample_rate, new_freq=self.audio_sr)
                    waveform = resampler(waveform)
                
                # 转换为一维tensor
                audio = waveform.squeeze(0)
                audio_start_frame = round(audio_start_frame_in_second * self.audio_sr)
                audio_end_frame = round(audio_start_frame + audio_latent_len_in_second * self.audio_sr)
                audio = audio[audio_start_frame:audio_end_frame]
                if len(audio) != (audio_end_frame - audio_start_frame):
                    if (audio_end_frame - audio_start_frame) - len(audio) < self.audio_sr * 1.0 and len(audio) > self.audio_sr * 0.5:
                        # pad to target length
                        audio = torch.cat([audio, torch.zeros(audio_end_frame - audio_start_frame - len(audio))], dim=-1)
                    else:
                        self.logger.warning(f"Audio length mismatch: {len(audio)} != {audio_end_frame - audio_start_frame}")
                        self.logger.warning(f"Index info: {idx}, {self.table[self.video_key][idx]}, {self.arrow_file}, {self.cache_folder}")
                        raise ValueError(f"Audio length mismatch: {len(audio)} != {audio_end_frame - audio_start_frame}")
            except Exception as e:
                raise Exception(f'Loading audio error: {e}.')
            
            if self.use_merged_caption:
                video_caption = audio_caption = prompt + " " + audio_caption
            else:
                video_caption = prompt
                audio_caption = audio_caption

            if random.random() < self.video_uncond_p:
                video_caption = ""
                audio_caption = ""

            # Add caption:
            audio_caption = "Audio capition: " + audio_caption

            results = {
                "video_latents": latents,
                "waveform": audio.unsqueeze(0),
                "video_caption": video_caption,
                "audio_caption": audio_caption,
                "language_tag": language_tag,
                "transcript_text": transcript_text,
                "videoid": videoid,
            }
            return results
            
    def __getitem__(self, idx): # logic index
        return self.get_batch(idx, index_type="logic")


class VideoAudioArrowConcatDataset:
    def __init__(self,
                args,                       # 其他参数
                index_file,                 # arrow info存放路径json或json列表
                index_file_weights: list = None,  # index_file对应的权重列表
                video_size: list = None,    # 视频采样尺寸
                sample_n_frames: int=16,    # 帧采样帧数
                sample_stride: int=1,       # 帧采样间隔
                is_image: bool = False,     # 是否抽帧训图
                multireso: bool = False,    # 是否使用多分辨率桶
                multitemp: bool = False,    # 是否使用多时长桶
                bucket_hw_bucket_stride: int=16,         # 用于分桶计算 token 数
                bucket_hw_base_size: int=256,  # 用于分桶的base尺寸数，分别为256, 480, 512, 720, 960
                bucket_temporal_min_length: int=49,  # 用于时长分桶的最小长度
                bucket_temporal_max_length: int=65,  # 用于时长分桶的最小长度
                bucket_temporal_interval: int=1,  # 用于时长分桶的采样间隔
                text_encoder=None,
                text_encoder_2=None,
                text_len=256,               # text 长度
                video_uncond_p=0.1,         # 
                audio_uncond_p=0.1,         # 
                use_caption_prob=0.0,       # 使用原始caption的概率
                timeout_seconds=60.0,       # 下载视频的超时时间
                chunk_size=1024,            # 下载视频的chunk_size
                vae_spatial_ratio: float=8, # vae latent 空间压缩比例
                caption_sample_ratio=None,  # caption 采样比例
                text_remove_rate=0.0,
                num_replace_rate=0.0,       # 数字替换比例
                use_latent_shape=False,     # 是否使用 latent 的 shape 作为 bucket 的 shape
                filtered_video_ids=None,    # 过滤的video id list json file
                motion_threshold=10000.0,
                reserved_video_ids_trans_score=None,
                ocr_only_long_caption=False,
                byt5_tokenizer=None,
                byt5_max_length=None,
                byt5_uncond_p=None,
                debug: bool = False,
                logger=None,
                video_caption_col_probs: dict=None,
                video_caption_col: str='structure_caption',
                video_caption_col_2: str='structure_caption_cn',
                skip_sources: list=None,    # 要跳过的source列表
                index_file_sample_seed=42,  # 对index file采样时的seed
                # extra audio settings
                audio_latent_dim=128,
                audio_sr=48000,
                use_audio_vae_cache=False,
                audio_vae_suffix=".npy",
                audio_wave_suffix=".m4a",
                audio_frame_rate=50,
                audio_caption_col='caption',
                use_merged_caption=True,
                ) -> None:
        self.args = args
        self.use_latent_shape = use_latent_shape
        self.index_file_sample_seed = index_file_sample_seed

        index_file = list(index_file)

        # 处理多个index_file输入
        if isinstance(index_file, str):
            self.index_files = [index_file]
            self.index_weights = [1.0] if index_file_weights is None else [index_file_weights]
        elif isinstance(index_file, list):
            self.index_files = index_file
            if index_file_weights is None:
                self.index_weights = [1.0] * len(index_file)
            elif isinstance(index_file_weights, list):
                self.index_weights = index_file_weights
            else:
                raise TypeError("index_file_weights must be a list when index_file is a list")
        else:
            raise TypeError("index_file must be a string or list")

        all_arrow_lists = []
        all_cache_folder_lists = []
        all_video_folder_lists = []
        all_audio_cache_folder_lists = []
        all_audio_arrow_lists = []
        all_aspect_ratio_index_lists = []
        all_aspect_ratios_nums_lists = []

        def convert_arrow_to_audio_cache_path(arrow_path):
            filename = os.path.splitext(os.path.basename(arrow_path))[0]
            audio_cache_path = arrow_path.replace('/arrow_index/all/', '/arrow_index_audio/')
            audio_cache_path = os.path.splitext(audio_cache_path)[0]
            return audio_cache_path

        def convert_arrow_to_audio_info_path(arrow_path):
            filename = os.path.splitext(os.path.basename(arrow_path))[0]
            audio_cache_path = arrow_path.replace('/arrow_index/all/', '/arrow_index_audioinfo/')
            audio_cache_path = os.path.splitext(audio_cache_path)[0]
            return audio_cache_path

        # 读取所有index文件内容
        for idx, (index_file_path, weight) in enumerate(zip(self.index_files, self.index_weights)):
            if not index_file_path.endswith('.json'):
                raise ValueError(f'index_file {index_file_path} must be json file')

            with open(index_file_path) as f:
                d = json.load(f)
                assert 'arrow_list' in d, f'index_file {index_file_path} must contain arrow_list'
                arrow_list = d['arrow_list']
                cache_folder_list = d.get('cache_folder_list', ["" for _ in arrow_list])
                video_folder_list = d.get('video_folder_list', ["" for _ in arrow_list])

                if "audio_cache_folder_list" in d:
                    audio_cache_folder_list = d["audio_cache_folder_list"]
                else:
                    audio_cache_folder_list = [convert_arrow_to_audio_cache_path(arrow_path) for arrow_path in arrow_list]

                if "audio_arrow_list" in d:
                    audio_arrow_list = d["audio_arrow_list"]
                else:
                    audio_arrow_list = [convert_arrow_to_audio_info_path(arrow_path) for arrow_path in arrow_list]

                if multireso:
                    aspect_ratio_index_list = d.get('aspect_ratio_index_list', [None for _ in arrow_list])
                    aspect_ratios_nums_list = d.get('aspect_ratios_nums_list', [None for _ in arrow_list])
                else:
                    aspect_ratio_index_list = [None for _ in arrow_list]
                    aspect_ratios_nums_list = [None for _ in arrow_list]

                # 根据权重采样
                if weight == 1:
                    choose_indices = list(range(len(arrow_list)))
                else:
                    sample_count = int(len(arrow_list) * weight)
                    replace = True if weight > 1.0 else False
                    rnd = random.Random(self.index_file_sample_seed)
                    if replace:
                        choose_indices = rnd.choices(range(len(arrow_list)), k=sample_count)
                    else:
                        choose_indices = rnd.sample(range(len(arrow_list)), k=sample_count)

                all_arrow_lists.extend([arrow_list[i] for i in choose_indices])
                all_cache_folder_lists.extend([cache_folder_list[i] for i in choose_indices])
                all_video_folder_lists.extend([video_folder_list[i] for i in choose_indices])
                all_audio_cache_folder_lists.extend([audio_cache_folder_list[i] for i in choose_indices])
                all_audio_arrow_lists.extend([audio_arrow_list[i] for i in choose_indices])
                all_aspect_ratio_index_lists.extend([aspect_ratio_index_list[i] for i in choose_indices])
                all_aspect_ratios_nums_lists.extend([aspect_ratios_nums_list[i] for i in choose_indices])
            
        self.multireso = multireso
        self.multitemp = multitemp
        self.arrow_file = all_arrow_lists
        sample_size = tuple(video_size) if not isinstance(video_size, int) else (video_size, video_size)
        self.sample_size = sample_size

        if logger is None: from loguru import logger
        self.logger = logger
        
        
        if multitemp and multireso:
            # 多时长+多分辨率桶
            # generate temporal bucket
            assert bucket_temporal_max_length>bucket_temporal_min_length, f"temporal_max_length must be larger than temporal_min_length, but got {bucket_temporal_max_length}, {bucket_temporal_min_length}"
            assert (bucket_temporal_min_length-1)%4==0, f"bucket_temporal_min_length-1 must be multiple of 4, but got {bucket_temporal_min_length}"
            assert (bucket_temporal_max_length-1)%4==0, f"bucket_temporal_max_length-1 must be multiple of 4, but got {bucket_temporal_max_length}"
            if bucket_temporal_max_length>sample_n_frames: bucket_temporal_max_length = sample_n_frames
            
            video_length_list = [fn+1 for fn in list(range(bucket_temporal_min_length-1, bucket_temporal_max_length-1, bucket_temporal_interval))]+[bucket_temporal_max_length]
            rank0_print(self.logger, f"List of temporal length buckets: {video_length_list}")
            video_length_buckets = np.array(video_length_list)
            
            # 多分辨率桶
            assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"
                
            crop_size_list, spatial_crop_size_list = generate_crop_size_w_temporal_list(bucket_hw_base_size, bucket_hw_bucket_stride, temporal_list=video_length_buckets)
            rank0_print(self.logger, "List of crop sizes:")
            for i in range(0, len(crop_size_list), 6):
                rank0_print(self.logger, " " + "".join([f"{f'{t} x {h} x {w}':20s}" for t, h, w in crop_size_list[i: i + 6]]))
                
            aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in spatial_crop_size_list])
            self.aspect_ratios_buckets = aspect_ratios
            self.video_length_buckets = video_length_buckets
            self.spatial_crop_size_list = spatial_crop_size_list
            
            temporal_aspect_ratios_nums = {}
            for temp_bucket in video_length_buckets:
                if int(temp_bucket) not in temporal_aspect_ratios_nums: temporal_aspect_ratios_nums[int(temp_bucket)] = {}
                for k in aspect_ratios:
                    temporal_aspect_ratios_nums[int(temp_bucket)][float(k)] = 0      # used for batch-sampler
            self.temporal_aspect_ratios_nums = temporal_aspect_ratios_nums
            
        elif multireso:
            # 多分辨率桶
            assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"
            crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
            rank0_print(self.logger, "List of crop sizes:")
            for i in range(0, len(crop_size_list), 6):
                rank0_print(self.logger, " " + "".join([f"{f'{h} x {w}':14s}" for h, w in crop_size_list[i: i + 6]]))
            assert sample_size in crop_size_list, f"sample_size must be in crop_size_list, current sample_size: {sample_size}"
            self.crop_size_list = crop_size_list
            self.aspect_ratios_buckets = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
            self.aspect_ratios_nums = {}
            for k in self.aspect_ratios_buckets:
                self.aspect_ratios_nums[float(k)] = 0      # used for batch-sampler

        s_time = time.time()
        self.datasets = []
        self.table = []
        total_length = 0
        
        ignored_cache_folders = [
            'assets/local/resource_dcb15256b5',
            'assets/local/resource_d18fb7ca6e'
        ]        
        
        # 如果有多个index文件，对所有列表进行shuffle以混合数据源
        if len(self.index_files) > 1:
            rnd = random.Random(self.index_file_sample_seed)
            random_indexs = list(range(len(all_arrow_lists)))
            rnd.shuffle(random_indexs)
            all_arrow_lists = [all_arrow_lists[i] for i in random_indexs]
            all_cache_folder_lists = [all_cache_folder_lists[i] for i in random_indexs]
            all_video_folder_lists = [all_video_folder_lists[i] for i in random_indexs]
            all_audio_cache_folder_lists = [all_audio_cache_folder_lists[i] for i in random_indexs]
            all_audio_arrow_lists = [all_audio_arrow_lists[i] for i in random_indexs]
            all_aspect_ratio_index_lists = [all_aspect_ratio_index_lists[i] for i in random_indexs]
            all_aspect_ratios_nums_lists = [all_aspect_ratios_nums_lists[i] for i in random_indexs]

        # 准备有效的数据项（过滤掉被忽略的 cache_folder）
        valid_data_items = []
        for i, (arrow_file, audio_arrow_file, video_folder, cache_folder, audio_cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict) in enumerate(zip(all_arrow_lists, all_audio_arrow_lists, all_video_folder_lists, all_cache_folder_lists, all_audio_cache_folder_lists, all_aspect_ratio_index_lists, all_aspect_ratios_nums_lists)):
            if cache_folder not in ignored_cache_folders:
                valid_data_items.append((i, arrow_file, audio_arrow_file, video_folder, cache_folder, audio_cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict))
        
        def create_dataset(data_item):
            """创建单个dataset的函数，用于并发执行"""
            original_index, arrow_file, audio_arrow_file, video_folder, cache_folder, audio_cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict = data_item
            
            dataset = VideoAudioArrowDataset(
                arrow_file                  = arrow_file,
                audio_arrow_file            = audio_arrow_file,
                video_folder                = video_folder,
                cache_folder                = cache_folder,
                audio_cache_folder          = audio_cache_folder,
                sample_size                 = sample_size,
                sample_stride               = sample_stride,
                sample_n_frames             = sample_n_frames,
                is_image                    = is_image,
                multireso                   = multireso,
                multitemp                   = multitemp,
                bucket_hw_base_size         = bucket_hw_base_size,
                bucket_temporal_min_length  = bucket_temporal_min_length,
                bucket_temporal_max_length  = bucket_temporal_max_length,
                bucket_temporal_interval    = bucket_temporal_interval,
                aspect_ratio_index          = aspect_ratio_index_dict,
                aspect_ratios_nums          = aspect_ratios_nums_dict,
                bucket_hw_bucket_stride     = bucket_hw_bucket_stride,
                text_encoder                = text_encoder,
                text_encoder_2              = text_encoder_2,
                text_len                    = text_len,
                video_uncond_p              = video_uncond_p,
                audio_uncond_p              = audio_uncond_p,
                use_caption_prob            = use_caption_prob,
                timeout_seconds             = timeout_seconds,
                chunk_size                  = chunk_size,
                vae_spatial_ratio           = vae_spatial_ratio,
                caption_sample_ratio        = caption_sample_ratio,
                ocr_only_long_caption       = ocr_only_long_caption,
                text_remove_rate            = text_remove_rate,
                num_replace_rate            = num_replace_rate,
                use_latent_shape            = use_latent_shape,
                filtered_video_ids          = filtered_video_ids,
                byt5_tokenizer              = byt5_tokenizer,
                byt5_max_length             = byt5_max_length,
                byt5_uncond_p               = byt5_uncond_p,
                args                        = args,
                debug                       = debug,
                logger                      = logger,
                motion_threshold            = motion_threshold,
                reserved_video_ids_trans_score = reserved_video_ids_trans_score,
                video_caption_col_probs     = video_caption_col_probs,
                video_caption_col           = video_caption_col,
                video_caption_col_2         = video_caption_col_2,
                skip_sources                = skip_sources,
                audio_latent_dim            = audio_latent_dim,
                audio_sr                    = audio_sr,
                use_audio_vae_cache         = use_audio_vae_cache,
                audio_vae_suffix            = audio_vae_suffix,
                audio_wave_suffix           = audio_wave_suffix,
                audio_frame_rate            = audio_frame_rate,
                audio_caption_col           = audio_caption_col,
                use_merged_caption          = use_merged_caption,
            )
            return original_index, dataset
        
        max_workers = min(len(valid_data_items), os.cpu_count(), 4)  # 限制线程数
        rank0_print(logger, f"Creating {len(valid_data_items)} datasets concurrently with {max_workers} workers...")
        
        # 存储结果，保证顺序
        dataset_results = [None] * len(valid_data_items)
        
        # 记录并发创建开始时间
        concurrent_start_time = time.time()
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            task_submit_time = time.time()
            future_to_index = {executor.submit(create_dataset, item): idx for idx, item in enumerate(valid_data_items)}
            rank0_print(logger, f"All {len(valid_data_items)} tasks submitted in {time.time() - task_submit_time:.3f}s")
            
            # 收集结果并保持顺序
            for future in as_completed(future_to_index):
                result_index = future_to_index[future]
                try:
                    original_index, dataset = future.result()
                    dataset_results[result_index] = dataset
                    completed_count += 1
                    elapsed_time = time.time() - concurrent_start_time
                    rank0_print(logger, f"Dataset {completed_count}/{len(valid_data_items)} created successfully (elapsed: {elapsed_time:.3f}s)")
                except Exception as exc:
                    rank0_print(logger, f"Dataset creation failed for index {result_index}: {exc}")
                    raise exc
        
        concurrent_total_time = time.time() - concurrent_start_time
        rank0_print(logger, f"All {len(valid_data_items)} datasets created concurrently in {concurrent_total_time:.3f}s")
        
        # 按原始顺序处理结果
        post_process_start_time = time.time()
        # rank0_print(logger, f"Starting post-processing of {len(valid_data_items)} datasets...")

        # 建立全局索引到子数据集的映射
        self.index_to_dataset = {}  # {global_index: (dataset_idx, local_index)}
        cumulative_length = 0
        
        # 初始化全局的temporal_aspect_ratios_index
        if multitemp and multireso:
            self.temporal_aspect_ratios_index = {}
            for temp_bucket in self.video_length_buckets:
                if int(temp_bucket) not in self.temporal_aspect_ratios_index:
                    self.temporal_aspect_ratios_index[int(temp_bucket)] = {}
                for aspect_ratio in self.aspect_ratios_buckets:
                    self.temporal_aspect_ratios_index[int(temp_bucket)][float(aspect_ratio)] = []
        elif multireso:
            self.aspect_ratios_index = {}
            for aspect_ratio in self.aspect_ratios_buckets:
                self.aspect_ratios_index[float(aspect_ratio)] = []
        
        for dataset in dataset_results:
            if dataset is not None:
                dataset_idx = len(self.datasets)
                dataset_length = len(dataset)
                
                # 为当前数据集的所有样本建立映射
                for local_idx in range(dataset_length):
                    global_idx = cumulative_length + local_idx
                    self.index_to_dataset[global_idx] = (dataset_idx, local_idx)
                
                self.datasets.append(dataset)
                self.table.append(dataset.table)
                total_length += dataset_length
                cumulative_length += dataset_length
                
                if multitemp and multireso:
                    # 构建全局temporal_aspect_ratios_index, bucket中存储全局逻辑索引
                    for temp, aspect_ratios2index in dataset.aspect_ratio_index.items():
                        for aspect_ratio, local_physical_indices in aspect_ratios2index.items():
                            #  here are physical index, need to convert it to logic index
                            local_logic_indices = [dataset.physical2logic_index_map[local_physical_idx] for local_physical_idx in local_physical_indices] 
                            # 将本地索引转换为全局索引
                            global_indices = [cumulative_length - dataset_length + local_idx for local_idx in local_logic_indices]
                            self.temporal_aspect_ratios_index[int(temp)][float(aspect_ratio)].extend(global_indices)
                    
                    # 更新数量统计
                    for temp, aspect_ratios2num in dataset.aspect_ratios_nums.items():
                        for aspect_ratio, nums in aspect_ratios2num.items():
                            self.temporal_aspect_ratios_nums[int(temp)][aspect_ratio] += nums
                        
                elif multireso:
                    # 构建全局aspect_ratios_index
                    for aspect_ratio, local_physical_indices in dataset.aspect_ratio_index.items():
                        # 将本地物理索引转换为本地逻辑索引
                        local_logic_indices = [dataset.physical2logic_index_map[local_physical_idx] for local_physical_idx in local_physical_indices]
                        # 将本地逻辑索引转换为全局逻辑索引
                        global_indices = [cumulative_length - dataset_length + local_idx for local_idx in local_logic_indices]
                        self.aspect_ratios_index[float(aspect_ratio)].extend(global_indices)
                    
                    # 更新数量统计
                    for k, v in dataset.aspect_ratios_nums.items():
                        self.aspect_ratios_nums[float(k)] += v
        
        post_process_time = time.time() - post_process_start_time
        rank0_print(logger, f"Post-processing completed in {post_process_time:.3f}s")
        
        # PyArrow表合并耗时
        table_concat_start_time = time.time()
        # self.table = pa.concat_tables(self.table)
        table_concat_time = time.time() - table_concat_start_time
        
        # 总耗时统计
        total_cost_time = time.time() - s_time
        # 内存使用统计
        index_mapping_size = len(self.index_to_dataset)
        bucket_indices_total = 0
        if hasattr(self, 'temporal_aspect_ratios_index'):
            bucket_indices_total = sum(len(indices) for temp_dict in self.temporal_aspect_ratios_index.values() 
                                     for indices in temp_dict.values())
        elif hasattr(self, 'aspect_ratios_index'):
            bucket_indices_total = sum(len(indices) for indices in self.aspect_ratios_index.values())
        
        rank0_print(logger, f'=== Dataset Loading Summary ===')
        rank0_print(logger, f'Total datasets: {len(valid_data_items)}')
        rank0_print(logger, f'Total data length: {total_length}')
        rank0_print(logger, f'Index mapping entries: {index_mapping_size}')
        rank0_print(logger, f'Bucket indices total: {bucket_indices_total}')
        rank0_print(logger, f'Dataset creation time: {concurrent_total_time:.3f}s')
        rank0_print(logger, f'Post-processing time: {post_process_time:.3f}s') 
        rank0_print(logger, f'Table concat time: {table_concat_time:.3f}s')
        rank0_print(logger, f'Total loading time: {total_cost_time:.3f}s')
        rank0_print(logger, f'Average time per dataset: {concurrent_total_time/len(valid_data_items):.3f}s')
        rank0_print(logger, f'================================')
        self.total_length = total_length
        self.concat_dataset = ConcatDataset(self.datasets)
        self.dataset_type = 'video' # 预留字段
        
        # 初始化错误处理时需要的属性
        self.closest_temp = None
        self.closest_ratio = None
        
        # 可选：懒加载标志，用于大数据集优化
        self._lazy_loading = False
        if total_length > 10000000:  # 超过1000万样本启用懒加载优化
            rank0_print(logger, f"Large dataset detected ({total_length} samples), consider enabling lazy loading optimizations")
            self._lazy_loading = True
        
    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx): # logic index
        try_times = 10000
        original_idx = idx
        failed_indices = set()  # 记录失败的索引，避免重复尝试
        for i in range(try_times):
            try:
                return self.get_batch(idx)
            except Exception as e:
                print(e)
                failed_indices.add(idx)
                
                # 优先从相同桶中选择未失败的索引
                if self.multitemp and self.multireso and self.closest_temp is not None and self.closest_ratio is not None:
                    bucket_list = self.temporal_aspect_ratios_index.get(self.closest_temp, {}).get(self.closest_ratio, [])
                    available_indices = [x for x in bucket_list if x not in failed_indices]
                    if available_indices:
                        idx = random.choice(available_indices)
                    else:
                        idx = np.random.randint(len(self))
                        while idx in failed_indices and len(failed_indices) < len(self):
                            idx = np.random.randint(len(self))
                    continue
                elif self.multireso and self.closest_ratio is not None:
                    bucket_list = self.aspect_ratios_index.get(self.closest_ratio, [])
                    available_indices = [x for x in bucket_list if x not in failed_indices]
                    if available_indices:
                        idx = random.choice(available_indices)
                    else:
                        idx = np.random.randint(len(self))
                        while idx in failed_indices and len(failed_indices) < len(self):
                            idx = np.random.randint(len(self))
                    continue
                else:
                    idx = np.random.randint(len(self))
                    while idx in failed_indices and len(failed_indices) < len(self):
                        idx = np.random.randint(len(self))
                    continue
        raise RuntimeError(f'Too many bad data. Original index: {original_idx}, failed attempts: {len(failed_indices)}')

    def get_batch(self, idx): 
        '''
        inputs:
            idx: logic index
        '''
        # 根据全局逻辑索引找到对应的子数据集和局部逻辑索引
        dataset_idx, local_idx = self.index_to_dataset[idx]

        item = self.datasets[dataset_idx].get_batch(local_idx)
        # 更新 closest_ratio 和 closest_temp, 调用子dataset的get_batch会更新这两个值
        self.closest_temp = getattr(self.datasets[dataset_idx], 'closest_temp', None)
        self.closest_ratio = getattr(self.datasets[dataset_idx], 'closest_ratio', None)

        return item
        
        
    def get_data_info(self, index, index_type="logic"):
        '''
        index_type: logic or physical
        '''
        # 根据全局索引找到对应的子数据集和本地索引
        if index_type == "logic":
            dataset_idx, local_index = self.index_to_dataset[index]
            # 调用对应子数据集的get_data_info方法
            return self.datasets[dataset_idx].get_data_info(local_index, index_type="logic")
        else:
            # 兜底方案：使用原来的合并表方式（保持向后兼容）
            if self.use_latent_shape:
                latent_shape = self.table['latent_shape'][index].as_py()
                if isinstance(latent_shape, str):
                    latent_shape = eval(latent_shape)
                assert isinstance(latent_shape, list), "latent_shape must be list"
                num_frames = latent_shape[-3]
                height = latent_shape[-2]
                width = latent_shape[-1]
                num_frames = (num_frames-1)*4+1
                motion_score = float(self.table['max_motion_score_v2'][index].as_py()) if 'max_motion_score_v2' in self.table.column_names else None
            else:
                height = int(float(self.table['height'][index].as_py()))
                width = int(float(self.table['width'][index].as_py()))
                fps = float(self.table['fps'][index].as_py())
                duration = float(self.table['duration'][index].as_py())
                num_frames = int(fps*duration)
                motion_score = float(self.table['max_motion_score_v2'][index].as_py()) if 'max_motion_score_v2' in self.table.column_names else None
            return {'height': height,
                    'width': width,
                    'num_frames': num_frames,
                    'motion_score': motion_score}


def collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples]).contiguous()
    latents = torch.stack([sample["latents"] for sample in samples]).contiguous()
    text_ids = torch.stack([sample["text_ids"] for sample in samples]).contiguous()
    text_mask = torch.stack([sample["text_mask"] for sample in samples]).contiguous()
    # target_size = torch.tensor([sample["bucket"] for sample in samples])
    text = [sample["text"] for sample in samples]

    batch = {
        "pixel_values": pixel_values,
        "latents": latents,
        "text_ids": text_ids,
        "text_mask": text_mask,
        # "target_size": target_size,
        "text": text,
        "type": "video",
    }
    return batch

        # print(f"transcript_text: {data['transcript_text']}")
        # print(f"audio_caption: {data['audio_caption']}")
        # print(f"language_tag: {data['language_tag']}")
        # print('-'*100)

    # {"short caption": "A group of people sing on a stage, followed by four women modeling formal gowns.",
    #  "long caption": "Four young adults with light skin stand on a stage with silver curtains behind them, singing into microphones. They are wearing casual clothing; two wear white jackets, one wears a black top and green shorts, and another wears a gray shirt and dark pants. Later, five women with various hair colors and styles walk across a stage while posing for the camera. They are all wearing floor-length formal gowns in various colors, including pink, purple, blue, teal, and gold.",
    #  "background": "Silver foil backdrop",
    #  "light": "Brightly lit",
    #  "atmosphere": "Lively",
    #  "shot type": "Medium shot",
    #  "style": "Realistic",
    #  "camera movement": "Other"}

    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=3,
    #     num_workers=8,
    #     pin_memory=True,
    #     collate_fn=collate_fn
    # )
    # cnt = 0
    # frequency = defaultdict(int)
    # start = time.time()
    # for sample in dataloader:
    #     print(f'XXXXX{cnt}')
    #     print(sample)
    #
    #     cnt += 1
    #     if cnt >= 100:
    #         break
    # print(frequency)
    # print(f"batch per second: {cnt / (time.time() - start)}")
