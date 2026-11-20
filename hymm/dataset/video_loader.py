import random
import os
import io
import copy
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

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms as transforms
from torch.utils.data import Dataset, get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import ConcatDataset, ChainDataset

import traceback
import time
import requests
from socket import timeout

from decord import VideoReader
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from hymm.utils.data_utils import get_closest_ratio, generate_crop_size_list, get_temporal_closest_ratio, generate_crop_size_w_temporal_list
from hymm.utils.recreation_mask_utils import create_clustered_masks, parse_recreation_result, convert_recreation_mask_to_latent_space
from hymm.dataset.format_prompt import MultilingualPromptFormat
from hymm.utils.distributed_utils import rank0_print
import pandas as pd
import re
from hymm.dataset.caps.caption_process_v4 import SEPARATORS
from hymm.utils.temporal_resample import resample_latent_temporal
from hymm.utils.physrope_prompt_rewrite import (
    rewrite_segment_infos_for_reference_conditions,
    validate_reference_prompt_segment_identity,
)
from hymm.utils.native_reference import as_bool
from hymm.utils.reference_temporal_pack import (
    REFERENCE_TIMELINE,
    normalize_temporal_pack_mode,
    scatter_reference_timeline_latent,
)
from hymm.utils.prompt_cleaning import strip_identity_tags


def _canonicalize_bucket_key(value, valid_keys, cast_type=float):
    target_value = cast_type(value)
    if not valid_keys:
        return target_value
    normalized_keys = [cast_type(key) for key in valid_keys]
    if target_value in normalized_keys:
        return target_value
    return min(normalized_keys, key=lambda key: abs(key - target_value))


def _build_temporal_ratio_template(valid_temporal_keys, valid_ratio_keys, value_factory):
    return {
        int(temp): {float(ratio): value_factory() for ratio in valid_ratio_keys}
        for temp in valid_temporal_keys
    }


def _log_bucket_remap(logger, context, remapped_temps=None, remapped_ratios=None):
    remapped_temps = remapped_temps or {}
    remapped_ratios = remapped_ratios or {}
    if not remapped_temps and not remapped_ratios:
        return
    messages = []
    if remapped_temps:
        temp_items = ", ".join([f"{src}->{dst}" for src, dst in sorted(remapped_temps.items())[:10]])
        messages.append(f"temporal buckets: {temp_items}")
    if remapped_ratios:
        ratio_items = ", ".join([f"{src}->{dst}" for src, dst in list(remapped_ratios.items())[:10]])
        messages.append(f"aspect ratios: {ratio_items}")
    rank0_print(logger, f"[bucket_remap][{context}] " + " | ".join(messages))

def open_image_retry(image_path, retry=3):
    for i in range(retry):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ret = img.copy()  # VERY
        except OSError:
            # incase cos rate exceed
            print(f"fail load {image_path}")
            time.sleep(random.random() + i)
            continue
        break
    return ret

class VideoArrowDataset(Dataset):
    def __init__(self,
                arrow_file,                # arrow info存放路径
                video_folder: str='',      # 视频存放 dir
                cache_folder: str='',      # vae latent 存放dir的list
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
                uncond_p=0.0,               # 使用原始空字符串的概率
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
                recreation_mask_col: str='recreation_mask_data', #  recreation mask 列名
                skip_sources: list=None,    # 要跳过的source列表
                video_caption_i2v_instruction_enable=None,
                video_caption_i2v_instruction_col_probs=None,
                video_caption_i2v_instruction_col=None,
                video_caption_i2v_instruction_col_2=None,
                video_caption_i2v_instruction_sub_col_probs=None,
                video_caption_i2v_instruction_sub_col=None,
                video_caption_i2v_instruction_sub_col_2=None,
                video_caption_i2v_instruction_sub_col_3=None,
                local_seed=None,
                ) -> None:
        self.args = args
        self.arrow_file = arrow_file
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        self.multireso = multireso
        self.multitemp = multitemp
        self.bucket_hw_bucket_stride = bucket_hw_bucket_stride
        self.video_folder = video_folder
        self.cache_folder = cache_folder
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.motion_threshold = motion_threshold
        self.is_image = is_image
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_len = text_len
        self.uncond_p = uncond_p
        self.vae_suffix = vae_suffix
        self.use_caption_prob = use_caption_prob
        self.timeout_seconds = timeout_seconds
        self.caption_sample_ratio = caption_sample_ratio
        self.ocr_only_long_caption = ocr_only_long_caption
        self.text_remove_rate = text_remove_rate
        self.num_replace_rate = num_replace_rate
        self.use_latent_shape = use_latent_shape
        self.chunk_size = chunk_size
        self.vae_spatial_ratio = vae_spatial_ratio
        self.byt5_tokenizer = byt5_tokenizer
        self.byt5_max_length = byt5_max_length
        self.byt5_uncond_p = byt5_uncond_p
        self._physrope_prompt_debug_count = 0
        self._physrope_condition_debug_count = 0

        self.pil_image_to_tensor = T.Compose(
            [
                T.ToTensor(),  # convert to tensor and normalize to [0, 1]
                T.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.debug = debug
        if logger is None:
            from loguru import logger
        self.logger = logger

        # auto detect fl2v_enable
        fl2v_prob = self.args.multitask_mask_training_args.get("fl2v", {}).get("prob", 0.0)
        self.fl2v_enable = fl2v_prob > 0
        #rank0_print(logger, f"fl2v_enable: {self.fl2v_enable} with prob={fl2v_prob}")

        self.recreation_mask_col = recreation_mask_col
        self.video_caption_col_probs = video_caption_col_probs
        self.video_caption_col = video_caption_col
        self.video_caption_col_2 = video_caption_col_2
        self.skip_sources = skip_sources if skip_sources is not None else []
        if self.video_caption_col_probs is not None:
            self.multi_caption_cols = []
            self.multi_caption_probs = []
            for col, prob in self.video_caption_col_probs.items():
                self.multi_caption_cols.append(getattr(self, col)) # map to video_caption_col
                self.multi_caption_probs.append(prob)

        self.video_caption_i2v_instruction_enable = video_caption_i2v_instruction_enable
        self.video_caption_i2v_instruction_col_probs = video_caption_i2v_instruction_col_probs
        self.video_caption_i2v_instruction_col = video_caption_i2v_instruction_col
        self.video_caption_i2v_instruction_col_2 = video_caption_i2v_instruction_col_2
        if self.video_caption_i2v_instruction_enable:
            self.multi_caption_i2v_instruction_cols = []
            self.multi_caption_i2v_instruction_probs = []
            for col, prob in self.video_caption_i2v_instruction_col_probs.items():
                self.multi_caption_i2v_instruction_cols.append(getattr(self, col)) # map to video_caption_col
                self.multi_caption_i2v_instruction_probs.append(prob)

        self.video_caption_i2v_instruction_sub_col_probs = video_caption_i2v_instruction_sub_col_probs
        self.video_caption_i2v_instruction_sub_col = video_caption_i2v_instruction_sub_col
        self.video_caption_i2v_instruction_sub_col_2 = video_caption_i2v_instruction_sub_col_2
        self.video_caption_i2v_instruction_sub_col_3 = video_caption_i2v_instruction_sub_col_3
        if self.video_caption_i2v_instruction_enable:
            self.multi_caption_i2v_instruction_sub_cols = []
            self.multi_caption_i2v_instruction_sub_probs = []
            for col, prob in self.video_caption_i2v_instruction_sub_col_probs.items():
                self.multi_caption_i2v_instruction_sub_cols.append(getattr(self, col)) # map to video_caption_col
                self.multi_caption_i2v_instruction_sub_probs.append(prob)

           # rank0_print(logger, f"self.video_caption_i2v_instruction_col_probs: {self.video_caption_i2v_instruction_col_probs}")
           # rank0_print(logger, f"self.multi_caption_i2v_instruction_cols: {self.multi_caption_i2v_instruction_cols}")
           # rank0_print(logger, f"self.multi_caption_i2v_instruction_probs: {self.multi_caption_i2v_instruction_probs}")
           # rank0_print(logger, f"self.video_caption_i2v_instruction_sub_col_probs: {self.video_caption_i2v_instruction_sub_col_probs}")
           # rank0_print(logger, f"self.multi_caption_i2v_instruction_sub_cols: {self.multi_caption_i2v_instruction_sub_cols}")
           # rank0_print(logger, f"self.multi_caption_i2v_instruction_sub_probs: {self.multi_caption_i2v_instruction_sub_probs}")

        self.local_seed = local_seed
        rank0_print(logger, f"self.local_seed: {self.local_seed}")

        if self.caption_sample_ratio is not None:
            self.use_structure_caption = True
            video_caption_processor = args.video_caption_processor
            # caption_process_v4 & caption_process_v3 不需要替换下划线为空格
            if video_caption_processor == "caption_process":
                self.caption_sample_ratio = self.caption_sample_ratio.replace('_', ' ')
                kwargs={
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'text_remove_rate': self.text_remove_rate,
                }
            elif video_caption_processor in ["caption_process_v4", "caption_process_v3"]:
                kwargs = {
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'num_replace_rate': self.num_replace_rate,
                    'time_range_mode': getattr(args, 'physical_rope_time_range_mode', 'second'),
                    'video_fps': getattr(args, 'video_fps', 24),
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
            raise NotImplementedError(f"use_structure_caption must be True when video_caption_processor is caption_process_v4 or caption_process_v3")

        self.use_vae_cache = False
        self.cache_folder = cache_folder
        self.video_folder = video_folder
        if cache_folder!='': self.use_vae_cache = True

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
        if self.video_caption_i2v_instruction_enable:
            assert self.multi_caption_i2v_instruction_cols[0] in self.table.column_names and self.multi_caption_i2v_instruction_cols[1] in self.table.column_names, \
                f"{self.multi_caption_i2v_instruction_cols[0]} and {self.multi_caption_i2v_instruction_cols[1]} not in self.table.column_names: {self.table.column_names}"
            assert (self.multi_caption_i2v_instruction_probs[0] + self.multi_caption_i2v_instruction_probs[1]) == 1, \
                f"{self.multi_caption_i2v_instruction_probs[0]} + {self.multi_caption_i2v_instruction_probs[1]} != 1"
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
                        latent_shape = eval(self.table['latent_shape'][idx].as_py())
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
                    temporal_aspect_ratios_nums[int(closest_temp)][float(closest_ratio)] += 1
                    temporal_aspect_ratio_index[int(closest_temp)][float(closest_ratio)].append(idx)

                self.aspect_ratio_index = temporal_aspect_ratio_index
                self.aspect_ratios_nums = temporal_aspect_ratios_nums
            else:
                # convert key type
                valid_temporal_keys = [int(temp) for temp in self.video_length_buckets]
                valid_ratio_keys = [float(k) for k in self.aspect_ratios]
                new_aspect_ratio_index = _build_temporal_ratio_template(
                    valid_temporal_keys, valid_ratio_keys, list
                )
                new_aspect_ratios_nums = _build_temporal_ratio_template(
                    valid_temporal_keys, valid_ratio_keys, lambda: 0
                )
                remapped_temps = {}
                remapped_ratios = {}

                for temp, aspect_ratios2index in (aspect_ratio_index or {}).items():
                    mapped_temp = _canonicalize_bucket_key(temp, valid_temporal_keys, int)
                    raw_temp = int(float(temp))
                    if mapped_temp != raw_temp:
                        remapped_temps[raw_temp] = mapped_temp
                    for aspect_ratio, indexs in (aspect_ratios2index or {}).items():
                        mapped_ratio = _canonicalize_bucket_key(
                            aspect_ratio, valid_ratio_keys, float
                        )
                        raw_ratio = float(aspect_ratio)
                        if mapped_ratio != raw_ratio:
                            remapped_ratios[raw_ratio] = mapped_ratio
                        new_aspect_ratio_index[mapped_temp][mapped_ratio].extend(list(indexs))

                for temp, aspect_ratios2num in (aspect_ratios_nums or {}).items():
                    mapped_temp = _canonicalize_bucket_key(temp, valid_temporal_keys, int)
                    raw_temp = int(float(temp))
                    if mapped_temp != raw_temp:
                        remapped_temps[raw_temp] = mapped_temp
                    for aspect_ratio, nums in (aspect_ratios2num or {}).items():
                        mapped_ratio = _canonicalize_bucket_key(
                            aspect_ratio, valid_ratio_keys, float
                        )
                        raw_ratio = float(aspect_ratio)
                        if mapped_ratio != raw_ratio:
                            remapped_ratios[raw_ratio] = mapped_ratio
                        new_aspect_ratios_nums[mapped_temp][mapped_ratio] += nums

                self.aspect_ratio_index = new_aspect_ratio_index
                self.aspect_ratios_nums = new_aspect_ratios_nums
                _log_bucket_remap(
                    logger,
                    os.path.basename(self.arrow_file),
                    remapped_temps=remapped_temps,
                    remapped_ratios=remapped_ratios,
                )

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
                valid_ratio_keys = [float(k) for k in self.aspect_ratios]
                self.aspect_ratio_index = {float(k): [] for k in valid_ratio_keys}
                self.aspect_ratios_nums = {float(k): 0 for k in valid_ratio_keys}
                remapped_ratios = {}

                for aspect_ratio, indexs in (aspect_ratio_index or {}).items():
                    mapped_ratio = _canonicalize_bucket_key(
                        aspect_ratio, valid_ratio_keys, float
                    )
                    raw_ratio = float(aspect_ratio)
                    if mapped_ratio != raw_ratio:
                        remapped_ratios[raw_ratio] = mapped_ratio
                    self.aspect_ratio_index[mapped_ratio].extend(list(indexs))

                for aspect_ratio, nums in (aspect_ratios_nums or {}).items():
                    mapped_ratio = _canonicalize_bucket_key(
                        aspect_ratio, valid_ratio_keys, float
                    )
                    raw_ratio = float(aspect_ratio)
                    if mapped_ratio != raw_ratio:
                        remapped_ratios[raw_ratio] = mapped_ratio
                    self.aspect_ratios_nums[mapped_ratio] += nums

                _log_bucket_remap(
                    logger,
                    os.path.basename(self.arrow_file),
                    remapped_ratios=remapped_ratios,
                )

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        if self.args.glyph_byT5_v2:
            self.prompt_format = MultilingualPromptFormat(font_path=self.args.multilingual_prompt_format_font_path,
                                                      color_path=self.args.multilingual_prompt_format_color_path)

    def _load_table(self):
        table = pa.memory_map(self.arrow_file, "r")
        table = pa.ipc.RecordBatchFileReader(table).read_all()  # 打印第一条数据，0.0013秒
        return table

    def __len__(self):
        if self.debug:
            return min(len(self.table), 40960)
        return len(self.table)

    def get_data_info(self, index):
        if self.use_latent_shape:
            latent_shape = eval(self.table['latent_shape'][index].as_py())
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

        if isinstance(video_images, torch.Tensor):
            pixel_values = video_images.permute(0, 3, 1, 2).contiguous()
        elif hasattr(video_images, "asnumpy"):
            pixel_values = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2).contiguous()
        else:
            pixel_values = torch.as_tensor(video_images).permute(0, 3, 1, 2).contiguous()
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

    def _create_recreation_mask(self, latents, recreation_mask_data, original_height, original_width):
        """
        Create recreation mask for video latents using real recreation data.
        Args:
            latents: torch.Tensor of shape [C, T, H, W]
            recreation_data: Parsed recreation data dictionary
            original_height: Original video height
            original_width: Original video width
        Returns:
            torch.Tensor: Binary mask of shape [T, H, W] where 1 indicates masked regions, or None if failed
        """
        if len(latents.shape) != 4:
            # If latents is not 4D, return None (no mask)
            return None

        C, T, H, W = latents.shape

        try:
            # Extract recreation boxes
            original_boxes = recreation_mask_data
            if not original_boxes:
                return None

            # Get processed dimensions from latent shape
            target_height = H * self.vae_spatial_ratio
            target_width = W * self.vae_spatial_ratio

            mapped_boxes = original_boxes

            if not mapped_boxes:
                return None

            # Get clustering parameters from args
            eps = getattr(self.args, 'recreation_clustering_eps', 2.0)
            margin_ratio = getattr(self.args, 'recreation_mask_margin_ratio', 0.05)

            # Get separate margin ratios for width and height
            margin_ratio_w = getattr(self.args, 'recreation_mask_margin_ratio_w', None)
            margin_ratio_h = getattr(self.args, 'recreation_mask_margin_ratio_h', None)

            # Get minimum margin pixels to ensure visibility in latent space
            min_margin_pixels = getattr(self.args, 'recreation_min_margin_pixels', 0)

            # Create clustered mask with separate margin control
            combined_mask = create_clustered_masks(
                mapped_boxes, target_height, target_width,
                eps=eps, margin_ratio=margin_ratio,
                margin_ratio_w=margin_ratio_w,
                margin_ratio_h=margin_ratio_h,
                min_margin_pixels=min_margin_pixels
            )
            # Convert to latent space
            # 注意：这里直接使用latent的实际尺寸H, W，而不是target_height, target_width
            latent_mask = convert_recreation_mask_to_latent_space(
                combined_mask, target_height, target_width, T, H, W, vae_spatial_stride=int(self.vae_spatial_ratio)
            )
            # print(f"latents shape: {latents.shape}, original_height: {original_height}, original_width: {original_width}, target_height: {target_height}, target_width: {target_width}\noriginal_boxes: {original_boxes}，combined_mask: {combined_mask}, latent_mask: {latent_mask}")

            return latent_mask

        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Failed to create OCR mask from real data: {e}")
            return None

    def sample_i2v_instruction(self, text1, text2, text3, prob1, prob2, prob3):
        texts = [text1, text2]
        probs = [prob1, prob2]

        # 过滤非空字符串及其对应概率
        valid_texts = []
        valid_probs = []
        for text, prob in zip(texts, probs):
            if len(text) > 0:
                valid_texts.append(text)
                valid_probs.append(prob)

        # 如果没有有效字符串，返回空字符串
        if not valid_texts:
            return None

        # 重新归一化概率
        total_prob = sum(valid_probs)
        if total_prob == 0:
            # 如果所有有效概率之和为0，则均匀分配概率
            normalized_probs = [1 / len(valid_probs)] * len(valid_probs)
        else:
            normalized_probs = [p / total_prob for p in valid_probs]

        # 根据归一化后的概率采样
        sampled_text = random.choices(valid_texts, weights=normalized_probs, k=1)[0]

        if len(text3) > 0:
            if random.random() <= prob3:
                sampled_text = sampled_text.rstrip()
                chinese_punctuation = "，。！？；：「」『』（）【】《》"
                if len(sampled_text) == 0:
                    sampled_text = text3
                else:
                    sampled_text_last_char = sampled_text[-1]
                    if sampled_text_last_char in chinese_punctuation:
                        sampled_text = sampled_text + text3
                    else:
                        # 对于英文标点或其他字符，添加空格
                        sampled_text = sampled_text + " " + text3

        return sampled_text

    def get_camera_movement_for_i2v_instruction(self, prompt, processed_camera_movement_en, processed_camera_movement_zh,
                                                i2v_instruction_col_idx):

        random_num = random.random()
        if random_num > self.caption_sample_ratio["camera_movement"] or processed_camera_movement_en is None or processed_camera_movement_zh is None:
            return prompt

        def strip_zh(text):
            text = text.replace("。，", "，").replace("，。", "。").replace("。。", "。").replace("，，", "，")
            text = re.sub(r'\s+,', ' ', text).strip()
            text = text.lstrip("，。 ")
            text = text if text.endswith('。') else text + '。'
            return text
        def strip_en(text):
            text = text.replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
            text = re.sub(r'\s+,', ' ', text).strip()
            text = text.lstrip(",. ")
            text = text if text.endswith('.') else text + '.'
            return text

        lang_flag = self.multi_caption_i2v_instruction_cols[i2v_instruction_col_idx]
        assert len(self.multi_caption_i2v_instruction_cols) == 2, "multi_caption_i2v_instruction_cols must be length 2"
        lang = "en" if i2v_instruction_col_idx == 0 else "zh" # 第一列英文，第二列中文

        if lang == "zh":
            # 随机替换关键词与文本描述的顺序
            final_string = [processed_camera_movement_zh, prompt]
            random.shuffle(final_string)
            return strip_zh(SEPARATORS[lang]['period'].join(final_string))
        elif lang == "en":
            final_string = [processed_camera_movement_en, prompt]
            random.shuffle(final_string)
            return strip_en(SEPARATORS[lang]['period'].join(final_string))

    def get_last_frame_tensor(self, idx, latent_size):
        if "last_frame_cache_image" in self.table.schema.names:
            img_path = self.table["last_frame_cache_image"][idx].as_py()
        else:
            raise ValueError("last_frame_cache_image not in table")

        def latent_size_to_image_size(latent_size):
            # hard code for vae spatial downsample 16
            h = latent_size[-2] * 16
            w = latent_size[-1] * 16
            return h, w
        try:
            img = open_image_retry(img_path)
            img = self.pil_image_to_tensor(img)
        except Exception as e:
            raise ValueError(f"Failed to load last frame image {img_path}: {e}")

        try:
            h,w = latent_size_to_image_size(latent_size)
            assert img.size(-1)==w and img.size(-2)==h
        except Exception as e:
            raise ValueError(f"Last frame size mismatch {img_path}: {img.size()}, latent_size: {latent_size}")

        return img

    def get_batch(self, idx):
        # 根据索引获取记录
        if 'videoid' in self.table.schema.names:
            video_key = 'videoid'
        elif 'video_id' in self.table.schema.names:
            video_key = 'video_id'
        elif 'md5' in self.table.schema.names:
            video_key = 'md5'
        elif 'npy_videoid' in self.table.schema.names:
            video_key = 'npy_videoid'
        else:
            raise Exception(f'videoid or md5 not in self.table.schema.names')

        self.video_key = video_key
        videoid = self.table[video_key][idx].as_py()
        cos_url = ''
        schema = self.table.schema
        if schema.get_field_index('cos_url') >= 0:
            cos_url = self.table['cos_url'][idx].as_py()
        elif schema.get_field_index('contentUrl') >= 0:
            cos_url = self.table['contentUrl'][idx].as_py()

        pixel_values = torch.tensor(0)
        latents = torch.tensor(0)
        pixel_transforms = None
        sample_n_frames = self.sample_n_frames
        sample_n_latent = (sample_n_frames - 1) // 4 + 1
        def _table_value(name, default=None):
            if name not in self.table.schema.names:
                return default
            value = self.table[name][idx].as_py()
            return default if value is None else value

        def _fit_latents_to_target_t(raw_latents, target_t):
            if raw_latents.ndim != 4:
                return raw_latents
            target_t = int(target_t)
            src_t = int(raw_latents.shape[1])
            if src_t <= 0 or target_t <= 0:
                return raw_latents
            if src_t == target_t:
                return raw_latents
            if src_t > target_t:
                # Bucket fitting is a start-anchored tail crop. Never slow down or
                # interpolate the whole target latent merely to hit a bucket length.
                return raw_latents[:, :target_t]
            raise ValueError(
                f"main latent T={src_t} is shorter than requested target bucket T={target_t}; "
                "temporal interpolation/padding is forbidden"
            )

        if self.multitemp and self.multireso:
            data_info = self.get_data_info(idx)
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
            data_info = self.get_data_info(idx)
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

        sample_n_latent = (sample_n_frames - 1) // 4 + 1

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
            cache_path = cache_key if os.path.isabs(cache_key) else f'{cache_folder}/{cache_key}'
            latents = torch.from_numpy(np.load(cache_path)).squeeze(0)
            # 检查是否包含NaN值
            if torch.isnan(latents).any():
                self.logger.warning(f"videoid: {videoid} contains NaN values in cached latents at {cache_path}")
                raise Exception(f'videoid: {videoid} has NaN values in cached latents')
            start_idx = 0 # 目前默认 start_idx 从首帧开始，而非随机的一个起始点
            # Special-transition samples must point to already generated offline
            # latents in the arrow. Do not insert extra frames during training:
            # doing it here changes the real sequence length after bucketing and
            # can push 241f jobs over the expected memory envelope.
            latents = latents[:, start_idx:, ...]
            latents = _fit_latents_to_target_t(latents, sample_n_latent)
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

        # Create recreation mask if enabled
        recreation_mask = None
        if hasattr(self.args, 'enable_recreation_mask') and self.args.enable_recreation_mask:
            if self.recreation_mask_col not in self.table.schema.names:
                if not getattr(self, "_missing_recreation_mask_col_warned", False):
                    rank0_print(
                        self.logger,
                        f"recreation_mask_col {self.recreation_mask_col} not in schema, skip recreation mask for this dataset",
                    )
                    self._missing_recreation_mask_col_warned = True
            else:
                recreation_mask_data = None
                original_height = None
                original_width = None

                try:
                    recreation_mask_data_str = self.table[self.recreation_mask_col][idx].as_py()
                    if recreation_mask_data_str and recreation_mask_data_str != '\\N':
                        recreation_mask_data = parse_recreation_result(recreation_mask_data_str)
                except Exception as e:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.debug(f"Failed to parse recreation mask data: {e}")

                if recreation_mask_data is not None:
                    if len(recreation_mask_data) == 0: # no mask data detected
                        recreation_mask_data = [[0, 0, 0, 0]] # default to no mask
                        recreation_mask = torch.zeros(latents.shape[1], latents.shape[2], latents.shape[3], dtype=torch.bool)
                    else:
                        if 'shape_l' in self.table.schema.names and 'shape_w' in self.table.schema.names:
                            original_height = int(float(self.table['shape_l'][idx].as_py()))
                            original_width = int(float(self.table['shape_w'][idx].as_py()))

                        recreation_mask = self._create_recreation_mask(latents, recreation_mask_data, original_height, original_width)
                        # Apply mask to latents - set masked regions to 0
                        # if recreation_mask is not None:
                        #     # recreation_mask shape: [T, H, W], latents shape: [C, T, H, W], expand mask to match latents dimensions
                        #     recreation_mask_expanded = recreation_mask.unsqueeze(0).expand(latents.shape[0], -1, -1, -1)

                        #     # Set masked latent regions to 0
                        #     latents_dtype = latents.dtype
                        #     latents = latents * (~recreation_mask_expanded).float().to(latents_dtype)
                        # else:
                        #     if hasattr(self, 'logger') and self.logger:
                        #         self.logger.debug(f"No valid recreation mask data found, skipping mask creation")

        # get prompt
        prompt_i2v_instruction = None
        valid_i2v_instruction = "false"
        segment_infos = None  # PhysRoPE: will be set if visual_events are available
        physrope_prompt_presegmented = False

        # for HunyuanVideo-1.5 video dataset only
        part_version = self.table["part_version"][idx].as_py() if "part_version" in self.table.schema.names else None

        if self.use_structure_caption:
            video_caption_processor = self.args.video_caption_processor
            assert video_caption_processor in ["caption_process_v4", "caption_process_v3"], f"video_caption_processor: {video_caption_processor} is not supported"
            if video_caption_processor in ["caption_process_v4", "caption_process_v3"]:
                # use | as separator to get multiple captions(mask caption and no mask caption)
                video_caption_cols = self.video_caption_col.split('|')
                video_caption_cols_2 = self.video_caption_col_2.split('|')

                def get_t2v_caption(video_caption_cols, video_caption_cols_2):
                    force_long_caption = False
                    schema_names = set(self.table.schema.names)

                    def _select_caption_cols(en_candidates, zh_candidates):
                        selected_en = next((col for col in en_candidates if col in schema_names), None)
                        selected_zh = next((col for col in zh_candidates if col in schema_names), None)
                        if selected_en is None:
                            raise AssertionError(
                                f"None of caption columns {en_candidates} in self.table.schema.names"
                            )
                        if selected_zh is None:
                            selected_zh = selected_en
                        return selected_en, selected_zh

                    def _caption_candidates(configured_cols, fallback_cols):
                        return [col for col in configured_cols if col] + [
                            col for col in fallback_cols if col not in configured_cols
                        ]

                    # Mixed PhysRoPE emergence training combines multishot data
                    # with original single-shot Refrain arrows.  The latter use
                    # t2v_caption_*_v2_3_* columns and have no aiv_caption_*.
                    en_fallback_cols = [
                        "t2v_caption_en_v2_3_nomask",
                        "t2v_caption_en_v2_3_mask",
                    ]
                    zh_fallback_cols = [
                        "t2v_caption_cn_v2_3_nomask",
                        "t2v_caption_cn_v2_3_mask",
                    ]
                    if part_version == "part1":
                        en_col, zh_col = _select_caption_cols(
                            _caption_candidates([video_caption_cols[0]], en_fallback_cols),
                            _caption_candidates([video_caption_cols_2[0]], zh_fallback_cols),
                        )
                        text_en = self.table[en_col][idx].as_py()
                        text_zh = self.table[zh_col][idx].as_py() if zh_col in self.table.schema.names else text_en
                    elif part_version in ["part2", "part3"]:
                        en_col, zh_col = _select_caption_cols(
                            _caption_candidates([video_caption_cols[1]], en_fallback_cols),
                            _caption_candidates([video_caption_cols_2[1]], zh_fallback_cols),
                        )
                        text_en = self.table[en_col][idx].as_py()
                        text_zh = self.table[zh_col][idx].as_py() if zh_col in self.table.schema.names else text_en
                        # force long caption if has mask and part_version is part2 and part3
                        has_mask = self.table['has_mask'][idx].as_py() if 'has_mask' in self.table.schema.names else None
                        force_long_caption = has_mask
                    else:
                        en_col, zh_col = _select_caption_cols(
                            _caption_candidates([video_caption_cols[0]], en_fallback_cols),
                            _caption_candidates([video_caption_cols_2[0]], zh_fallback_cols),
                        )
                        text_en = self.table[en_col][idx].as_py()
                        text_zh = self.table[zh_col][idx].as_py()

                    assert "en" in self.video_caption_col, f"en not in {self.video_caption_col}, col1 should be en"
                    assert "cn" in self.video_caption_col_2 or "zh" in self.video_caption_col_2, f"zh not in {self.video_caption_col_2}, col2 should be cn or zh"
                    assert text_en != '' and text_zh != '', f"text_en: {text_en}, text_zh: {text_zh} is empty"
                    lang = random.choices(
                        self.multi_caption_cols, weights=self.multi_caption_probs, k=1
                    )[0]
                    lang = 'zh' if '_cn' in lang or '_zh' in lang else 'en' # e.g., structure_caption_cn -> zh, structure_caption -> en

                    return text_en, text_zh, lang, force_long_caption

                text_en, text_zh, lang, force_long_caption = get_t2v_caption(video_caption_cols, video_caption_cols_2)

                camera_movement = self.table['camera_movement'][idx].as_py() if 'camera_movement' in self.table.schema.names else None
                special_camera_movement_classification = self.table['special_camera_movement_classification'][idx].as_py() if 'special_camera_movement_classification' in self.table.schema.names else None

                def _parse_bracket_physrope_prompt(raw_prompt):
                    text = str(raw_prompt or "").strip()
                    pattern = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*\]")
                    matches = list(pattern.finditer(text))
                    if not matches:
                        return None
                    segments = []
                    for seg_i, match in enumerate(matches):
                        body_start = match.end()
                        body_end = matches[seg_i + 1].start() if seg_i + 1 < len(matches) else len(text)
                        body = text[body_start:body_end].strip()
                        try:
                            start = float(match.group(1))
                            end = float(match.group(2))
                        except (TypeError, ValueError):
                            continue
                        if end < start:
                            start, end = end, start
                        segments.append({
                            "segment_id": seg_i + 1,
                            "body_text": body,
                            "text": body,
                            "time_range": (float(start), float(end)),
                            "time_range_mode": "latent",
                        })
                    return segments or None

                # Check for aiv_caption columns (visual_events / multishot data)
                has_aiv_caption = (
                    'aiv_caption_en_v1_0' in self.table.schema.names
                    and 'aiv_caption_cn_v1_0' in self.table.schema.names
                    and getattr(self.args, 'use_visual_events', False)
                )
                segment_infos = None

                if has_aiv_caption:
                    raw_aiv_caption_en = self.table['aiv_caption_en_v1_0'][idx].as_py()
                    raw_aiv_caption_zh = self.table['aiv_caption_cn_v1_0'][idx].as_py()
                    prompt, processed_camera_movement_en, processed_camera_movement_zh, segment_infos = \
                        self.caption_aug.caption_aug_with_visual_events(
                            text_en, text_zh,
                            raw_aiv_caption_en, raw_aiv_caption_zh,
                            lang,
                            camera_movement=camera_movement,
                            special_camera_movement_classification=special_camera_movement_classification,
                            force_long_caption=force_long_caption,
                            data_type="video",
                        )
                else:
                    prompt, processed_camera_movement_en, processed_camera_movement_zh = self.caption_aug.caption_aug(text_en, text_zh, lang,
                                                                                    camera_movement=camera_movement,
                                                                                    special_camera_movement_classification = special_camera_movement_classification,
                                                                                    force_long_caption = force_long_caption,
                                                                                    data_type="video")

                if "physrope_prompt" in self.table.schema.names:
                    raw_physrope_prompt = self.table["physrope_prompt"][idx].as_py()
                    if raw_physrope_prompt and str(raw_physrope_prompt).strip():
                        prompt = strip_identity_tags(str(raw_physrope_prompt).strip())
                        parsed_segment_infos = _parse_bracket_physrope_prompt(prompt)
                        if parsed_segment_infos is not None:
                            segment_infos = parsed_segment_infos
                            physrope_prompt_presegmented = True

                if self.video_caption_i2v_instruction_enable:
                    random_num = random.random()
                    if random_num < self.multi_caption_i2v_instruction_probs[0]:
                        i2v_instruction_col_idx = 0
                    else:
                        i2v_instruction_col_idx = 1

                    text_i2v_instruction_structure = self.table[self.multi_caption_i2v_instruction_cols[i2v_instruction_col_idx]][idx].as_py()
                    text_i2v_instruction_structure = json.loads(text_i2v_instruction_structure)

                    text_i2v_instruction = text_i2v_instruction_structure[self.multi_caption_i2v_instruction_sub_cols[0]]
                    text_i2v_instruction_2 = text_i2v_instruction_structure[self.multi_caption_i2v_instruction_sub_cols[1]]
                    text_i2v_instruction_3 = text_i2v_instruction_structure[self.multi_caption_i2v_instruction_sub_cols[2]]

                    if len(text_i2v_instruction) > 0 or len(text_i2v_instruction_2) > 0:

                        prob_i2v_instruction = self.multi_caption_i2v_instruction_sub_probs[0]
                        prob_i2v_instruction_2 = self.multi_caption_i2v_instruction_sub_probs[1]
                        prob_i2v_instruction_3 = self.multi_caption_i2v_instruction_sub_probs[2]

                        prompt_i2v_instruction = self.sample_i2v_instruction(text_i2v_instruction, text_i2v_instruction_2, text_i2v_instruction_3,
                                                                             prob_i2v_instruction, prob_i2v_instruction_2, prob_i2v_instruction_3)
                        if len(prompt_i2v_instruction) == 0:
                            prompt_i2v_instruction = prompt

                        prompt_i2v_instruction = self.get_camera_movement_for_i2v_instruction(prompt_i2v_instruction,
                                                                                              processed_camera_movement_en,
                                                                                              processed_camera_movement_zh,
                                                                                              i2v_instruction_col_idx)
                        valid_i2v_instruction = "true"
            else:
                raise NotImplementedError(f"video_caption_processor: {video_caption_processor} is not supported")
        else:
            raise NotImplementedError(f"use_structure_caption must be True")

        if self.video_caption_i2v_instruction_enable and prompt_i2v_instruction is None:
            prompt_i2v_instruction = prompt
            valid_i2v_instruction = "false"

        is_uncond = False
        orig_prompt_i2v_instruction = prompt_i2v_instruction
        physrope_time_range_mode = str(
            getattr(self.args, 'physical_rope_time_range_mode', 'second')
        ).strip().lower()
        clip_fps = float(getattr(self.args, 'video_fps', 24.0))
        if clip_fps <= 0:
            clip_fps = 24.0
        fps_prefix = None

        if self.args.video_fps_col_name in self.table.schema.names:
            clip_fps = float(self.table[self.args.video_fps_col_name][idx].as_py())
            if clip_fps <= 0:
                clip_fps = float(getattr(self.args, 'video_fps', 24.0))
            fps_prefix = f"FPS:{round(clip_fps)}, "
        elif self.args.video_fps_col_name is not None:
            raise Exception(f"video-fps-col-name: {self.args.video_fps_col_name} not in self.table.schema.names")

        def _parse_segment_time_range(time_range):
            if time_range is None:
                return None
            if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                try:
                    start = float(time_range[0])
                    end = float(time_range[1])
                except (TypeError, ValueError):
                    return None
                if end < start:
                    start, end = end, start
                return (float(start), float(end))
            if isinstance(time_range, (int, float)):
                value = float(time_range)
                return (value, value)
            text = str(time_range).strip()
            if not text:
                return None
            if "-" in text:
                start_text, end_text = text.split("-", 1)
                try:
                    start = float(start_text)
                    end = float(end_text)
                except ValueError:
                    return None
                if end < start:
                    start, end = end, start
                return (float(start), float(end))
            try:
                value = float(text)
            except ValueError:
                return None
            return (value, value)

        def _clone_segment_infos(raw_segment_infos):
            return [dict(seg) for seg in (raw_segment_infos or [])]

        def _get_total_physrope_range_end():
            return (
                max(sample_n_frames - 1, 0) / clip_fps
                if physrope_time_range_mode in ("second", "seconds")
                else (
                    float((max(sample_n_frames - 1, 0)) // 4 + 1)
                    if physrope_time_range_mode == "latent"
                    else float(max(sample_n_frames - 1, 0))
                )
            )

        def _prepare_physrope_segment_infos(raw_segment_infos):
            if not raw_segment_infos:
                return None
            prepared_segment_infos = _clone_segment_infos(raw_segment_infos)
            total_range_end = _get_total_physrope_range_end()

            def _seconds_to_latent(sec_val):
                """Convert seconds to latent frame index.

                3D causal VAE: first pixel frame is independent, remaining
                frames are compressed by temporal_stride (default 4 for
                16164 VAE).  pixel_frame = round(sec * fps).
                latent_idx = 0 if pixel_frame == 0
                           else (pixel_frame - 1) // temporal_stride + 1
                """
                pixel_frame = int(round(sec_val * clip_fps))
                if pixel_frame <= 0:
                    return 0
                # temporal_stride from VAE: 4 for 16164, 8 for 16168
                temporal_stride = 4
                return (pixel_frame - 1) // temporal_stride + 1

            for seg in prepared_segment_infos:
                try:
                    seg_id = int(seg.get("segment_id", -1))
                except (TypeError, ValueError):
                    seg_id = -1
                parsed_time = _parse_segment_time_range(seg.get("time_range"))
                if seg_id == 0:
                    seg["time_range"] = (0.0, float(total_range_end))
                elif parsed_time is None:
                    seg["time_range"] = None
                else:
                    start, end = parsed_time
                    # Data source time_range is in seconds; convert for latent mode
                    source_range_mode = str(seg.get("time_range_mode", "")).strip().lower()
                    if physrope_time_range_mode == "latent" and source_range_mode not in ("latent", "latent_index", "latent_frame"):
                        start = _seconds_to_latent(start)
                        end = _seconds_to_latent(end)
                    start = min(max(start, 0.0), float(total_range_end))
                    end = min(max(end, 0.0), float(total_range_end))
                    if end < start:
                        start, end = end, start
                    seg["time_range"] = (float(start), float(end))
            # Ensure first event segment starts at 0, last ends at total_range_end
            event_segs = [s for s in prepared_segment_infos
                          if int(s.get("segment_id", -1)) > 0 and s.get("time_range") is not None]
            if event_segs:
                event_segs.sort(key=lambda s: int(s["segment_id"]))
                fs, fe = event_segs[0]["time_range"]
                if fs > 0:
                    event_segs[0]["time_range"] = (0.0, fe)
                ls, le = event_segs[-1]["time_range"]
                if le < total_range_end:
                    event_segs[-1]["time_range"] = (ls, float(total_range_end))
            return prepared_segment_infos

        def _prefix_segment_prompt(raw_segment_infos, prefix):
            if not raw_segment_infos or not prefix:
                return raw_segment_infos
            prefixed_segment_infos = _clone_segment_infos(raw_segment_infos)
            for seg in prefixed_segment_infos:
                seg_text = str(seg.get("text", ""))
                if seg_text:
                    seg["text"] = f"{prefix}{seg_text}"
                    break
            return prefixed_segment_infos

        def _build_single_shot_segment_infos(single_prompt):
            total_range_end = _get_total_physrope_range_end()
            return [{
                "segment_id": 1,
                "body_text": single_prompt,
                "text": single_prompt,
                "time_range": (0.0, float(total_range_end)),
            }]

        segment_infos_for_physrope = _prepare_physrope_segment_infos(segment_infos)
        if (
            segment_infos_for_physrope is None
            and
            getattr(self.args, "physrope_single_shot_full_range", False)
            and len(str(prompt).strip()) > 0
        ):
            segment_infos_for_physrope = _build_single_shot_segment_infos(prompt)

        # ===================== Per-segment condition loading (TIV2V) =====================
        # Read segment_conditions JSON column when present. Training only keeps a
        # random subset of video-condition segments so we do not accidentally turn
        # the whole clip into condition frames and leave no generation target.
        # Each active segment is kept as a separate slot in
        # cond_seg_latents_packed[MAX, C, T, H, W], with
        # cond_seg_dst_ranges[MAX, 2] tracking target [start, end).
        # Per-seg latent files may be [1, C, T_src, H, W] or [C, T_src, H, W].
        # T_src may differ from T_dst=end-start for L4. Legacy packing resamples
        # the reference latent to T_dst; native T concat preserves the raw T_src.
        # cond_seg_src_ranges always records the original source temporal span.
        # Slots not used are zero-padded with dst_range=(-1,-1). MAX_SEGS bounds
        # the per-sample budget.
        # 120-arrow datasets without this column → all slots empty → t2v behavior.
        per_seg_cond_enable = getattr(self.args, 'per_segment_condition_enable', False)
        if isinstance(per_seg_cond_enable, str):
            per_seg_cond_enable = per_seg_cond_enable.strip().lower() in ('true', 'on', '1', 'yes')
        max_segs = int(getattr(self.args, 'per_segment_condition_max', 4))
        assert max_segs >= 1, f"per_segment_condition_max must be >= 1, got {max_segs}"
        has_main_latents = torch.is_tensor(latents) and latents.ndim == 4
        if has_main_latents:
            main_latent_shape = tuple(int(v) for v in latents.shape)
            main_latent_dtype = latents.dtype
        else:
            raw_main_latent_shape = _table_value("latent_shape", None)
            main_latent_shape = None
            if raw_main_latent_shape is not None:
                try:
                    parsed_main_latent_shape = eval(raw_main_latent_shape) if isinstance(raw_main_latent_shape, str) else raw_main_latent_shape
                    if isinstance(parsed_main_latent_shape, (list, tuple)):
                        if len(parsed_main_latent_shape) == 5:
                            main_latent_shape = tuple(int(v) for v in parsed_main_latent_shape[1:])
                        elif len(parsed_main_latent_shape) == 4:
                            main_latent_shape = tuple(int(v) for v in parsed_main_latent_shape)
                except Exception:
                    main_latent_shape = None
            if main_latent_shape is None:
                raise Exception(
                    f"videoid: {videoid} needs latent_shape metadata for raw-video per-segment conditions"
                )
            main_latent_dtype = torch.float32
        main_latent_c, main_latent_t, main_latent_h, main_latent_w = main_latent_shape
        native_t_concat = as_bool(getattr(self.args, 'range_condition_native_t_concat', False))
        temporal_pack_mode = normalize_temporal_pack_mode(
            getattr(self.args, 'range_condition_temporal_pack_mode', 'legacy_trilinear')
        )
        reference_timeline = temporal_pack_mode == REFERENCE_TIMELINE
        if reference_timeline and native_t_concat:
            raise ValueError(
                "range_condition_temporal_pack_mode=reference_timeline requires "
                "range_condition_native_t_concat=false"
            )
        cond_seg_count_int = 0
        cond_seg_dst_ranges = torch.full((max_segs, 2), -1, dtype=torch.long)
        cond_seg_physical_dst_ranges = torch.full((max_segs, 2), -1, dtype=torch.long)
        # cond_seg_src_ranges records source provenance even when source and
        # target lengths differ. ReferenceTimeline uses it for exact-frame selection
        # and debug only; it never changes the target PhysRoPE timeline.
        cond_seg_src_ranges = torch.full((max_segs, 2), -1, dtype=torch.long)
        packed_target_t = 1 if native_t_concat else main_latent_t
        cond_seg_latents_packed = torch.zeros(
            (max_segs, main_latent_c, packed_target_t, main_latent_h, main_latent_w),
            dtype=main_latent_dtype,
        )
        cond_seg_native_latents_by_slot = [None] * max_segs
        cond_seg_mask = torch.zeros((max_segs, main_latent_t), dtype=torch.float32)
        # 0=inactive, 1=video segment condition, 2=in-video image anchor.
        cond_seg_kind = torch.zeros((max_segs,), dtype=torch.long)
        # Prompt-shot identity for each video condition. Image anchors use -1.
        cond_seg_prompt_ids = torch.full((max_segs,), -1, dtype=torch.long)
        cond_seg_sparse_mappings = []
        active_seg_ids = []
        active_video_ranges = []
        cond_debug_raw_candidates = 0
        cond_debug_prob_dropped = 0
        cond_debug_bad_skipped = 0
        cond_debug_video_slots = 0
        cond_debug_anchor_slots = 0
        cond_debug_l4_slots = 0
        l4_resample_mode = str(getattr(self.args, 'l4_temporal_resample_mode', 'trilinear'))
        assert l4_resample_mode in ('trilinear',), (
            f"l4_temporal_resample_mode must be 'trilinear' for [C,T,H,W] VAE latents, got {l4_resample_mode}"
        )
        l4_align_raw = getattr(self.args, 'l4_temporal_resample_align_corners', False)
        if isinstance(l4_align_raw, str):
            l4_align_corners = l4_align_raw.strip().lower() in ('true', 'on', '1', 'yes')
        else:
            l4_align_corners = bool(l4_align_raw)

        def _log_condition_skip(reason, cond=None, detail=""):
            nonlocal cond_debug_bad_skipped
            cond_debug_bad_skipped += 1
            seg_id = None
            if isinstance(cond, dict):
                seg_id = cond.get('seg_id', None)
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            print(
                f"[Per-Segment TIV2V SKIP] rank={os.environ.get('RANK', '0')} "
                f"worker={worker_id} idx={idx} videoid={videoid} seg_id={seg_id} "
                f"reason={reason}"
                + (f" detail={detail}" if detail else ""),
                flush=True,
            )

        def _log_condition_warn(reason, cond=None, detail=""):
            seg_id = None
            if isinstance(cond, dict):
                seg_id = cond.get('seg_id', None)
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            print(
                f"[Per-Segment TIV2V WARN] rank={os.environ.get('RANK', '0')} "
                f"worker={worker_id} idx={idx} videoid={videoid} seg_id={seg_id} "
                f"reason={reason}"
                + (f" detail={detail}" if detail else ""),
                flush=True,
            )

        def _parse_latent_range(value):
            if isinstance(value, str):
                text = value.strip()
                if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
                    text = text[1:-1]
                sep = "," if "," in text else "-"
                value = [item.strip() for item in text.split(sep) if item.strip()]
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                return None
            try:
                start = int(round(float(value[0])))
                end = int(round(float(value[1])))
            except (TypeError, ValueError):
                return None
            if end <= start:
                return None
            return start, end

        def _validate_condition_prompt_segment(cond):
            validation = validate_reference_prompt_segment_identity(
                segment_infos,
                segment_infos_for_physrope,
                cond,
                require_explicit_ranges=as_bool(
                    getattr(
                        self.args,
                        "per_segment_condition_require_explicit_ranges",
                        True,
                    )
                ),
            )
            if not validation.get("valid", False):
                return None, validation
            return int(validation["declared_seg_id"]), validation

        if per_seg_cond_enable and 'segment_conditions' in self.table.schema.names:
            raw_seg_cond = None
            try:
                raw_seg_cond = self.table['segment_conditions'][idx].as_py()
            except Exception as exc:
                _log_condition_skip("segment_conditions_read_failed", detail=repr(exc))
                raw_seg_cond = None
            seg_list = []
            if raw_seg_cond:
                try:
                    seg_list = json.loads(raw_seg_cond)
                except Exception as exc:
                    _log_condition_skip("segment_conditions_json_parse_failed", detail=str(raw_seg_cond)[:200] + f" exc={repr(exc)}")
                    seg_list = []
            per_seg_p = float(getattr(self.args, 'per_segment_condition_prob', 0.3))
            video_seg_cap = min(max_segs, 3)
            selected_seg_entries = []
            if seg_list:
                seg_list = list(seg_list)
                random.shuffle(seg_list)
            for cond in seg_list:
                cond_debug_raw_candidates += 1
                if len(selected_seg_entries) >= video_seg_cap:
                    break
                if not isinstance(cond, dict):
                    _log_condition_skip("condition_entry_not_dict", detail=str(type(cond)))
                    continue
                if cond.get('edit_type') == 'failed':
                    _log_condition_skip("edit_type_failed", cond=cond)
                    continue
                if random.random() >= per_seg_p:
                    cond_debug_prob_dropped += 1
                    continue
                dst_range = cond.get('dst_range', None)
                if not (isinstance(dst_range, (list, tuple)) and len(dst_range) == 2):
                    _log_condition_skip("missing_or_invalid_dst_range", cond=cond, detail=str(dst_range))
                    continue
                try:
                    ds = int(dst_range[0]); de = int(dst_range[1])
                except Exception as exc:
                    _log_condition_skip("dst_range_parse_failed", cond=cond, detail=f"{dst_range} exc={repr(exc)}")
                    continue
                if ds < 0 or ds >= de:
                    _log_condition_skip(
                        "dst_range_out_of_bounds",
                        cond=cond,
                        detail=f"dst_range=({ds},{de}) latent_T={main_latent_t}",
                    )
                    continue
                write_ds = ds
                write_de = min(de, int(main_latent_t))
                if write_ds >= write_de:
                    _log_condition_skip(
                        "dst_range_no_visible_overlap",
                        cond=cond,
                        detail=f"physical_dst_range=({ds},{de}) latent_T={main_latent_t}",
                    )
                    continue
                lp = cond.get('latent_path', '')
                if not lp:
                    _log_condition_skip("empty_latent_path", cond=cond)
                    continue
                try:
                    seg_npy = np.load(lp)
                except Exception as exc:
                    _log_condition_skip("latent_load_failed", cond=cond, detail=f"path={lp} exc={repr(exc)}")
                    continue
                if seg_npy.ndim == 5:
                    if seg_npy.shape[0] != 1:
                        _log_condition_skip(
                            "latent_batch_dim_not_one",
                            cond=cond,
                            detail=f"path={lp} shape={tuple(seg_npy.shape)}",
                        )
                        continue
                    seg_npy = seg_npy[0]
                if seg_npy.ndim != 4:
                    _log_condition_skip(
                        "latent_ndim_not_4_after_squeeze",
                        cond=cond,
                        detail=f"path={lp} shape={tuple(seg_npy.shape)}",
                    )
                    continue
                c_src, t_src, h_src, w_src = seg_npy.shape
                expected_c, expected_h, expected_w = main_latent_c, main_latent_h, main_latent_w
                if c_src != expected_c:
                    _log_condition_skip(
                        "latent_channel_mismatch",
                        cond=cond,
                        detail=f"path={lp} got_C={c_src} expected_C={expected_c} shape={tuple(seg_npy.shape)}",
                    )
                    continue
                if h_src != expected_h or w_src != expected_w:
                    _log_condition_skip(
                        "latent_spatial_mismatch",
                        cond=cond,
                        detail=f"path={lp} got_HW=({h_src},{w_src}) expected_HW=({expected_h},{expected_w})",
                    )
                    continue
                if t_src <= 0:
                    _log_condition_skip("latent_temporal_empty", cond=cond, detail=f"path={lp} shape={tuple(seg_npy.shape)}")
                    continue
                dst_len = de - ds

                # L4 source span. Explicit latent-coordinate source ranges win;
                # otherwise the raw latent length defines the source duration and
                # is anchored at dst_start on the target timeline.
                explicit_src = cond.get('src_range', cond.get('src_latent_range', None))
                if explicit_src is None and ('src_start' in cond or 'src_end' in cond):
                    explicit_src = [cond.get('src_start', ds), cond.get('src_end', ds + int(t_src))]
                parsed_src = _parse_latent_range(explicit_src)
                if explicit_src is not None and parsed_src is None:
                    _log_condition_skip(
                        "src_range_parse_failed",
                        cond=cond,
                        detail=f"src_range={explicit_src} path={lp}",
                    )
                    continue
                if parsed_src is not None:
                    ss_lat, se_lat = parsed_src
                    if se_lat - ss_lat != int(t_src):
                        # The guarded segment-latent cache can differ from metadata
                        # by a few latent frames after boundary rounding/trimming.
                        # Tensor packing below already resamples raw T_src to T_dst;
                        # keep the explicit source start but trust the actual latent T.
                        se_lat = ss_lat + int(t_src)
                        src_range_origin = "explicit_src_range_adjusted_to_raw_t"
                    else:
                        src_range_origin = "explicit_src_range"
                elif t_src != dst_len:
                    ss_lat, se_lat = ds, ds + int(t_src)
                    src_range_origin = "raw_latent_temporal_length"
                else:
                    ss_lat, se_lat = ds, de
                    src_range_origin = "dst_equal"
                try:
                    declared_seg_id = int(cond.get('seg_id', -1))
                except (TypeError, ValueError):
                    _log_condition_warn(
                        "seg_id_parse_failed_using_minus_one",
                        cond=cond,
                        detail=f"seg_id={cond.get('seg_id', None)}",
                    )
                    declared_seg_id = -1
                seg_id, seg_resolution = _validate_condition_prompt_segment(cond)
                if seg_id is None:
                    _log_condition_skip(
                        "condition_prompt_shot_identity_invalid",
                        cond=cond,
                        detail=(
                            f"declared={declared_seg_id} dst=({ds},{de}) "
                            f"src_time_range_sec={cond.get('src_time_range_sec')} "
                            f"validation={seg_resolution}"
                        ),
                    )
                    continue

                selected_seg_entries.append({
                    "ds": ds,
                    "de": de,
                    "write_ds": write_ds,
                    "write_de": write_de,
                    "ss": ss_lat,
                    "se": se_lat,
                    "t_src": int(t_src),
                    "src_range_origin": src_range_origin,
                    "seg_id": seg_id,
                    "declared_seg_id": declared_seg_id,
                    "seg_resolution": seg_resolution,
                    "latent_path": lp,
                    "seg_npy": seg_npy,
                })

            if selected_seg_entries:
                random.shuffle(selected_seg_entries)
                pruned_seg_entries = []
                covered_frames = 0
                for entry in selected_seg_entries:
                    seg_len = int(entry["write_de"] - entry["write_ds"])
                    if not native_t_concat and covered_frames + seg_len >= main_latent_t:
                        _log_condition_skip(
                            "pruned_to_keep_generation_frames",
                            cond={"seg_id": entry["seg_id"]},
                            detail=(
                                f"physical_dst=({entry['ds']},{entry['de']}) "
                                f"visible_dst=({entry['write_ds']},{entry['write_de']}) "
                                f"covered={covered_frames} latent_T={main_latent_t}"
                            ),
                        )
                        continue
                    pruned_seg_entries.append(entry)
                    covered_frames += seg_len
                    if len(pruned_seg_entries) >= video_seg_cap:
                        break

                if not pruned_seg_entries and not native_t_concat:
                    fallback_choices = [
                        entry for entry in selected_seg_entries
                        if int(entry["write_de"] - entry["write_ds"]) < main_latent_t
                    ]
                    if fallback_choices:
                        pruned_seg_entries = [min(
                            fallback_choices,
                            key=lambda entry: (int(entry["write_de"] - entry["write_ds"]), int(entry["write_ds"]))
                        )]

                pruned_seg_entries.sort(key=lambda entry: int(entry["write_ds"]))
                overlap_checked_entries = []
                last_end = -1
                for entry in pruned_seg_entries:
                    write_ds_i = int(entry["write_ds"])
                    write_de_i = int(entry["write_de"])
                    if write_ds_i < last_end:
                        _log_condition_skip(
                            "dst_range_overlap",
                            cond={"seg_id": entry["seg_id"]},
                            detail=(
                                f"physical_dst=({entry['ds']},{entry['de']}) "
                                f"visible_dst=({write_ds_i},{write_de_i}) overlaps previous_end={last_end}; "
                                f"path={entry['latent_path']}"
                            ),
                        )
                        continue
                    overlap_checked_entries.append(entry)
                    last_end = write_de_i
                pruned_seg_entries = overlap_checked_entries

                for entry in pruned_seg_entries:
                    slot = cond_seg_count_int
                    ds = int(entry["ds"]); de = int(entry["de"])
                    write_ds = int(entry["write_ds"]); write_de = int(entry["write_de"])
                    ss_lat = int(entry["ss"]); se_lat = int(entry["se"])
                    # Load condition latent and apply L4 temporal resampling if needed
                    raw_t_src = int(entry["t_src"])
                    write_len = write_de - write_ds
                    # The condition metadata is expressed on the original
                    # segment timeline, but this training sample may be a
                    # shorter temporal bucket. Pack only the visible dst span
                    # into the current latent tensor; keep the source range
                    # below so PhysRoPE can still encode src_len != dst_len.
                    dst_len = write_len
                    assert write_len > 0, (
                        f"internal error: visible condition span is empty for "
                        f"physical_dst=({ds},{de}) visible_dst=({write_ds},{write_de})"
                    )
                    try:
                        seg_latent = torch.from_numpy(entry["seg_npy"]).to(
                            cond_seg_latents_packed.dtype
                        )
                        sparse_mapping = None
                        if reference_timeline:
                            seg_latent, sparse_mask, sparse_mapping = scatter_reference_timeline_latent(
                                seg_latent,
                                target_t=main_latent_t,
                                physical_dst_range=(ds, de),
                                visible_dst_range=(write_ds, write_de),
                            )
                            if not sparse_mapping["visible_target_indices"]:
                                _log_condition_skip(
                                    "reference_timeline_no_visible_reference_frame",
                                    cond={"seg_id": entry["seg_id"]},
                                    detail=(
                                        f"source_T={raw_t_src} physical_dst=({ds},{de}) "
                                        f"visible_dst=({write_ds},{write_de})"
                                    ),
                                )
                                continue
                        elif not native_t_concat:
                            # Legacy packing aligns the clean reference channel to
                            # the target canvas. Native T concat keeps this tensor
                            # untouched and packs it later as a separate T prefix.
                            if raw_t_src != dst_len:
                                seg_latent = resample_latent_temporal(
                                    seg_latent,
                                    src_frames=raw_t_src,
                                    dst_frames=dst_len,
                                    align_corners=l4_align_corners,
                                    mode=l4_resample_mode,
                                )
                            visible_offset = write_ds - ds
                            assert 0 <= visible_offset and visible_offset + write_len <= seg_latent.shape[1], (
                                f"visible slice out of packed condition latent: "
                                f"physical_dst=({ds},{de}) visible_dst=({write_ds},{write_de}) "
                                f"packed_shape={tuple(seg_latent.shape)}"
                            )
                            seg_latent = seg_latent[:, visible_offset:visible_offset + write_len]
                    except (TypeError, ValueError, RuntimeError) as exc:
                        _log_condition_skip(
                            "latent_pack_or_resample_failed",
                            cond={"seg_id": entry["seg_id"]},
                            detail=(
                                f"path={entry['latent_path']} raw_shape={tuple(entry['seg_npy'].shape)} "
                                f"raw_t_src={raw_t_src} dst=({ds},{de}) src=({ss_lat},{se_lat}) "
                                f"exc={repr(exc)}"
                            ),
                        )
                        continue
                    expected_t = raw_t_src if native_t_concat else (
                        main_latent_t if reference_timeline else write_len
                    )
                    expected_seg_shape = (main_latent_c, expected_t, main_latent_h, main_latent_w)
                    if tuple(seg_latent.shape) != expected_seg_shape:
                        _log_condition_skip(
                            "latent_pack_shape_mismatch",
                            cond={"seg_id": entry["seg_id"]},
                            detail=(
                                f"packed_shape={tuple(seg_latent.shape)} expected={expected_seg_shape} "
                                f"raw_t_src={raw_t_src} dst=({ds},{de}) src=({ss_lat},{se_lat}) "
                                f"path={entry['latent_path']}"
                            ),
                        )
                        continue

                    cond_seg_dst_ranges[slot, 0] = write_ds
                    cond_seg_dst_ranges[slot, 1] = write_de
                    cond_seg_physical_dst_ranges[slot, 0] = ds
                    cond_seg_physical_dst_ranges[slot, 1] = de
                    cond_seg_src_ranges[slot, 0] = ss_lat
                    cond_seg_src_ranges[slot, 1] = se_lat
                    if native_t_concat:
                        cond_seg_native_latents_by_slot[slot] = seg_latent
                    elif reference_timeline:
                        cond_seg_latents_packed[slot] = seg_latent
                        cond_seg_mask[slot] = sparse_mask
                        cond_seg_sparse_mappings.append({
                            "slot": slot,
                            "seg_id": int(entry["seg_id"]),
                            "kind": 1,
                            **sparse_mapping,
                        })
                    else:
                        cond_seg_latents_packed[slot, :, write_ds:write_de] = seg_latent
                        cond_seg_mask[slot, write_ds:write_de] = 1.0
                    cond_seg_kind[slot] = 1
                    cond_seg_prompt_ids[slot] = int(entry["seg_id"])
                    cond_debug_video_slots += 1
                    if (se_lat - ss_lat) != (de - ds):
                        cond_debug_l4_slots += 1
                    active_seg_ids.append(int(entry["seg_id"]))
                    active_video_ranges.append((ds, de))
                    cond_seg_count_int += 1

        # ============================================================================

        # ============== L1/L2 In-Video Image Anchor (training-side) ==============
        # Sample K random latent frames from the GT itself; each frame becomes a
        # 1-frame "image anchor" condition with dst_range=(t,t+1), src_range=(t,t+1).
        # Trains DreaMontage-style anchor binding: model learns "image at t* must
        # be the frame generated at t*". Composes with video-segment slots — each
        # anchor takes one slot up to MAX_SEGS. No external data needed.
        image_anchor_enable = getattr(self.args, 'image_anchor_enable', False)
        if isinstance(image_anchor_enable, str):
            image_anchor_enable = image_anchor_enable.strip().lower() in ('true', 'on', '1', 'yes')
        if image_anchor_enable and has_main_latents and cond_seg_count_int < max_segs:
            forced_anchor_conditions = []
            if "physrope_anchor_conditions" in self.table.schema.names:
                raw_anchor_conditions = self.table["physrope_anchor_conditions"][idx].as_py()
                if raw_anchor_conditions:
                    try:
                        parsed_anchor_conditions = json.loads(raw_anchor_conditions)
                    except Exception as exc:
                        _log_condition_skip(
                            "anchor_conditions_json_parse_failed",
                            detail=str(raw_anchor_conditions)[:200] + f" exc={repr(exc)}",
                        )
                        parsed_anchor_conditions = []
                    if isinstance(parsed_anchor_conditions, list):
                        forced_anchor_conditions = [
                            item for item in parsed_anchor_conditions
                            if isinstance(item, dict)
                        ]
            occupied = set()
            for slot_i in range(cond_seg_count_int):
                ds_i = int(cond_seg_dst_ranges[slot_i, 0].item())
                de_i = int(cond_seg_dst_ranges[slot_i, 1].item())
                if ds_i >= 0:
                    for ti in range(ds_i, min(de_i, main_latent_t)):
                        occupied.add(ti)
            for anchor in forced_anchor_conditions:
                if cond_seg_count_int >= max_segs:
                    break
                try:
                    t_star = int(anchor.get("frame_idx", anchor.get("anchor_frame")))
                    source_t = int(anchor.get("source_frame", t_star))
                except (TypeError, ValueError):
                    _log_condition_skip("anchor_frame_parse_failed", detail=str(anchor))
                    continue
                if t_star < 0 or t_star >= main_latent_t:
                    _log_condition_skip(
                        "anchor_frame_out_of_bounds",
                        detail=f"anchor={anchor} latent_T={main_latent_t}",
                    )
                    continue
                source_t_for_latent = source_t
                if source_t_for_latent < 0 or source_t_for_latent >= main_latent_t:
                    _log_condition_skip(
                        "anchor_source_frame_out_of_bounds",
                        detail=(
                            f"anchor={anchor} mapped_source={source_t_for_latent} "
                            f"latent_T={main_latent_t}"
                        ),
                    )
                    continue
                if t_star in occupied or len(occupied) >= max(0, main_latent_t - 1):
                    _log_condition_skip(
                        "anchor_overlaps_or_no_generation_frame",
                        detail=f"anchor={anchor} occupied={len(occupied)} latent_T={main_latent_t}",
                    )
                    continue
                slot = cond_seg_count_int
                cond_seg_dst_ranges[slot, 0] = t_star
                cond_seg_dst_ranges[slot, 1] = t_star + 1
                cond_seg_physical_dst_ranges[slot, 0] = t_star
                cond_seg_physical_dst_ranges[slot, 1] = t_star + 1
                cond_seg_src_ranges[slot, 0] = source_t
                cond_seg_src_ranges[slot, 1] = source_t + 1
                anchor_latent = latents[:, source_t_for_latent:source_t_for_latent+1].detach().clone()
                if native_t_concat:
                    cond_seg_native_latents_by_slot[slot] = anchor_latent
                else:
                    cond_seg_latents_packed[slot, :, t_star:t_star+1] = anchor_latent
                cond_seg_mask[slot, t_star:t_star+1] = 1.0
                cond_seg_kind[slot] = 2
                if reference_timeline:
                    cond_seg_sparse_mappings.append({
                        "slot": slot,
                        "seg_id": -1,
                        "kind": 2,
                        "source_t": 1,
                        "physical_dst_range": (t_star, t_star + 1),
                        "visible_dst_range": (t_star, t_star + 1),
                        "physical_dst_t": 1,
                        "relation": "equal",
                        "source_indices": [0],
                        "target_indices": [t_star],
                        "visible_source_indices": [0],
                        "visible_target_indices": [t_star],
                    })
                occupied.add(t_star)
                cond_debug_anchor_slots += 1
                cond_seg_count_int += 1
            anchor_prob = float(getattr(self.args, 'image_anchor_prob', 0.0))
            if not forced_anchor_conditions and anchor_prob > 0.0 and random.random() < anchor_prob:
                anchor_max = int(getattr(self.args, 'image_anchor_max', 2))
                anchor_max = min(anchor_max, max_segs - cond_seg_count_int)
                T_full = main_latent_t
                if T_full >= 1 and anchor_max >= 1:
                    # Pick how many anchors: 60% → 1, 30% → 2, 10% → 3+ (capped)
                    roll = random.random()
                    if roll < 0.6:
                        n_anchors = 1
                    elif roll < 0.9:
                        n_anchors = min(2, anchor_max)
                    else:
                        n_anchors = min(3, anchor_max)
                    n_anchors = min(n_anchors, anchor_max, T_full)
                    # Reject sample anchor positions that overlap existing dst_ranges
                    occupied = set()
                    for slot_i in range(cond_seg_count_int):
                        ds_i = int(cond_seg_dst_ranges[slot_i, 0].item())
                        de_i = int(cond_seg_dst_ranges[slot_i, 1].item())
                        if ds_i >= 0:
                            for ti in range(ds_i, min(de_i, T_full)):
                                occupied.add(ti)
                    available = [t for t in range(T_full) if t not in occupied]
                    if available:
                        # Keep at least one generation frame after adding anchors.
                        n_anchors = min(n_anchors, max(0, len(available) - 1))
                        if n_anchors <= 0:
                            chosen = []
                        else:
                            chosen = random.sample(available, k=n_anchors)
                        for t_star in chosen:
                            if cond_seg_count_int >= max_segs:
                                break
                            slot = cond_seg_count_int
                            cond_seg_dst_ranges[slot, 0] = t_star
                            cond_seg_dst_ranges[slot, 1] = t_star + 1
                            cond_seg_physical_dst_ranges[slot, 0] = t_star
                            cond_seg_physical_dst_ranges[slot, 1] = t_star + 1
                            cond_seg_src_ranges[slot, 0] = t_star
                            cond_seg_src_ranges[slot, 1] = t_star + 1
                            # Copy the GT latent frame into the anchor slot — the
                            # model sees this clean frame at (t*, t*+1) and must
                            # generate the rest. latents is [C, T, H, W].
                            anchor_latent = latents[:, t_star:t_star+1].detach().clone()
                            if native_t_concat:
                                cond_seg_native_latents_by_slot[slot] = anchor_latent
                            else:
                                cond_seg_latents_packed[slot, :, t_star:t_star+1] = anchor_latent
                            cond_seg_mask[slot, t_star:t_star+1] = 1.0
                            cond_seg_kind[slot] = 2
                            if reference_timeline:
                                cond_seg_sparse_mappings.append({
                                    "slot": slot,
                                    "seg_id": -1,
                                    "kind": 2,
                                    "source_t": 1,
                                    "physical_dst_range": (t_star, t_star + 1),
                                    "visible_dst_range": (t_star, t_star + 1),
                                    "physical_dst_t": 1,
                                    "relation": "equal",
                                    "source_indices": [0],
                                    "target_indices": [t_star],
                                    "visible_source_indices": [0],
                                    "visible_target_indices": [t_star],
                                })
                            cond_debug_anchor_slots += 1
                            cond_seg_count_int += 1
        # ============================================================================

        anchor_frames_for_prompt = []
        for slot_i in range(cond_seg_count_int):
            if int(cond_seg_kind[slot_i].item()) != 2:
                continue
            ds_i = int(cond_seg_dst_ranges[slot_i, 0].item())
            if ds_i >= 0:
                anchor_frames_for_prompt.append(ds_i)

        cond_seg_native_ranges = torch.full((max_segs, 2), -1, dtype=torch.long)
        if native_t_concat and cond_seg_count_int > 0:
            native_chunks = []
            native_cursor = 0
            for slot_i in range(cond_seg_count_int):
                native_chunk = cond_seg_native_latents_by_slot[slot_i]
                if native_chunk is None:
                    raise RuntimeError(
                        f"native T concat slot {slot_i} is active but has no native latent; "
                        f"videoid={videoid} dst={cond_seg_dst_ranges[slot_i].tolist()}"
                    )
                native_len = int(native_chunk.shape[1])
                cond_seg_native_ranges[slot_i] = torch.tensor(
                    [native_cursor, native_cursor + native_len], dtype=torch.long
                )
                native_chunks.append(native_chunk)
                native_cursor += native_len
            cond_seg_native_latents = torch.cat(native_chunks, dim=1)
        else:
            cond_seg_native_latents = torch.zeros(
                (main_latent_c, 1, main_latent_h, main_latent_w),
                dtype=main_latent_dtype,
            )

        prompt_rewrite_changed = False
        if segment_infos_for_physrope is not None:
            segment_infos_for_physrope, prompt_rewrite_changed = rewrite_segment_infos_for_reference_conditions(
                segment_infos_for_physrope,
                video_seg_ids=active_seg_ids,
                video_ranges=active_video_ranges,
                anchor_frames=anchor_frames_for_prompt,
            )
            for seg in segment_infos_for_physrope:
                if "body_text" in seg:
                    seg["body_text"] = strip_identity_tags(str(seg.get("body_text", "")))
                if "text" in seg:
                    seg["text"] = strip_identity_tags(str(seg.get("text", "")))

        prompt = strip_identity_tags(prompt)
        if self.video_caption_i2v_instruction_enable:
            prompt_i2v_instruction = strip_identity_tags(prompt_i2v_instruction)

        # ─── Debug: print every training sample, including t2v fallbacks. ───
        worker_info_dbg = get_worker_info()
        worker_dbg = worker_info_dbg.id if worker_info_dbg is not None else "main"
        levels = []
        dst_ranges_dbg = []
        src_ranges_dbg = []
        active_kinds_dbg = []
        for slot_i in range(cond_seg_count_int):
            kind_i = int(cond_seg_kind[slot_i].item())
            ds = int(cond_seg_dst_ranges[slot_i, 0].item())
            de = int(cond_seg_dst_ranges[slot_i, 1].item())
            ss = int(cond_seg_src_ranges[slot_i, 0].item())
            se = int(cond_seg_src_ranges[slot_i, 1].item())
            dst_ranges_dbg.append((ds, de))
            src_ranges_dbg.append((ss, se))
            active_kinds_dbg.append(kind_i)
            seg_len = de - ds
            src_len = se - ss
            if kind_i == 2:
                levels.append(f"L1/L2-anchor@{ds}")
            elif src_len == seg_len:
                levels.append(f"L3-video[{ds}-{de}]")
            else:
                levels.append(f"L4-video[{ds}-{de}],src={src_len}->dst={seg_len}")
        has_video_dbg = any(kind == 1 for kind in active_kinds_dbg)
        has_anchor_dbg = any(kind == 2 for kind in active_kinds_dbg)
        if has_video_dbg and has_anchor_dbg:
            cond_debug_task = "tiv2v"
        elif has_video_dbg:
            cond_debug_task = "v2v"
        elif has_anchor_dbg:
            cond_debug_task = "ti2v"
        else:
            cond_debug_task = "t2v"
        cond_debug_status = "loaded" if cond_seg_count_int > 0 else "none_after_sampling_or_invalid"
        physrope_debug_limit = int(getattr(self.args, "physrope_prompt_debug_limit", 5))
        condition_sample_debug = (
            physrope_debug_limit > 0
            and os.environ.get("RANK", "0") == "0"
            and (worker_dbg == 0 or worker_dbg == "main")
            and self._physrope_condition_debug_count < physrope_debug_limit
        )
        # Disabled noisy per-Arrow condition debug blocks for production runs.
        condition_sample_debug = False
        if condition_sample_debug:
            self._physrope_condition_debug_count += 1
            print(
                f"[Condition SAMPLE DEBUG {self._physrope_condition_debug_count}/{physrope_debug_limit}] "
                f"rank={os.environ.get('RANK', '0')} worker={worker_dbg} "
                f"idx={idx} videoid={videoid} lang={lang} task={cond_debug_task} "
                f"range_condition={'ON' if cond_seg_count_int > 0 else 'OFF'} "
                f"range_mode={self.args.range_condition_mode} "
                f"cond_status={cond_debug_status} slots={cond_seg_count_int}/{max_segs} "
                f"video_slots={cond_debug_video_slots} anchor_slots={cond_debug_anchor_slots} "
                f"l4_slots={cond_debug_l4_slots} raw_candidates={cond_debug_raw_candidates} "
                f"prob_dropped={cond_debug_prob_dropped} bad_skipped={cond_debug_bad_skipped} "
                f"dst_ranges={dst_ranges_dbg} src_ranges={src_ranges_dbg} levels={levels} "
                f"has_segment_conditions={'segment_conditions' in self.table.schema.names} "
                f"active_seg_ids={active_seg_ids} anchor_frames={anchor_frames_for_prompt} "
                f"native_t_concat={native_t_concat} "
                f"temporal_pack_mode={temporal_pack_mode} "
                f"native_ranges={cond_seg_native_ranges[:cond_seg_count_int].tolist()} "
                f"physical_dst_ranges={cond_seg_physical_dst_ranges[:cond_seg_count_int].tolist()} "
                f"sparse_mappings={cond_seg_sparse_mappings} "
                f"reference_T={cond_seg_native_latents.shape[1] if native_t_concat and cond_seg_count_int > 0 else 0} "
                f"target_T={main_latent_t} "
                f"packed_T={(cond_seg_native_latents.shape[1] + main_latent_t) if native_t_concat and cond_seg_count_int > 0 else main_latent_t} "
                f"prompt_rewrite_changed={prompt_rewrite_changed}",
                flush=True,
            )

        def _format_segment_prompt(raw_segment_infos):
            prompt_parts = []
            for seg in raw_segment_infos:
                if int(seg["segment_id"]) <= 0:
                    continue
                time_range = seg["time_range"]
                if time_range is None:
                    time_prefix = ""
                else:
                    start, end = time_range
                    if physrope_time_range_mode == "latent":
                        time_prefix = f"[{int(round(start))}-{int(round(end))}] "
                    elif lang == "zh":
                        time_prefix = f"{start:.1f}秒-{end:.1f}秒 "
                    else:
                        time_prefix = f"{start:.1f}s-{end:.1f}s "
                seg["text"] = f"{time_prefix}{seg['body_text']}"
                if seg["text"].strip():
                    prompt_parts.append(seg["text"])
            return " ".join(prompt_parts)

        if physrope_time_range_mode == "latent" and segment_infos_for_physrope is not None:
            prompt = _format_segment_prompt(segment_infos_for_physrope)
            if self.video_caption_i2v_instruction_enable and valid_i2v_instruction == "false":
                prompt_i2v_instruction = prompt

        # ============= Non-integer RoPE jitter (future_plan §1.4(d)) =============
        # After the prompt string is built (uses int-rounded values, stable text),
        # perturb the floats that flow into text_segment_time_ranges so PhysRoPE
        # receives non-integer positions. This trains the model to be robust to
        # the L4 linear-rescale case where φ produces fractional coords.
        # Only perturb event segs (segment_id > 0); seg_id=0 is global span.
        rope_jitter_prob = float(getattr(self.args, 'rope_noninteger_jitter_prob', 0.0))
        rope_jitter_range = float(getattr(self.args, 'rope_noninteger_jitter_range', 0.0))
        if (
            rope_jitter_prob > 0.0
            and rope_jitter_range > 0.0
            and segment_infos_for_physrope is not None
            and physrope_time_range_mode == "latent"
            and random.random() < rope_jitter_prob
        ):
            for seg in segment_infos_for_physrope:
                if int(seg.get('segment_id', -1)) <= 0:
                    continue
                tr = seg.get('time_range')
                if tr is None:
                    continue
                s, e = float(tr[0]), float(tr[1])
                ds_ofs = (random.random() * 2.0 - 1.0) * rope_jitter_range
                de_ofs = (random.random() * 2.0 - 1.0) * rope_jitter_range
                seg['time_range'] = (max(0.0, s + ds_ofs), max(0.0, e + de_ofs))

        if random.random() < self.uncond_p:
            is_uncond = True
            prompt = ''
            if self.video_caption_i2v_instruction_enable:
                prompt_i2v_instruction = ''

        # add fps to the head of prompt
        if fps_prefix is not None:
            if len(prompt) > 0: # ommit uncond prompt
                prompt = f"{fps_prefix}{prompt}"
            if self.video_caption_i2v_instruction_enable and len(prompt_i2v_instruction) > 0:
                prompt_i2v_instruction = f"{fps_prefix}{prompt_i2v_instruction}"
            if len(prompt) > 0:
                segment_infos_for_physrope = _prefix_segment_prompt(segment_infos_for_physrope, fps_prefix)

        worker_info = get_worker_info()
        # Disabled noisy per-Arrow prompt/range debug blocks for production runs.
        physrope_data_debug = False
        if physrope_data_debug and (
            physrope_time_range_mode == "latent"
            and segment_infos_for_physrope is not None
            and len(prompt) > 0
            and os.environ.get("RANK", "0") == "0"
            and (worker_info is None or worker_info.id == 0)
            and self._physrope_prompt_debug_count < physrope_debug_limit
        ):
            self._physrope_prompt_debug_count += 1
            debug_ranges = [
                {
                    "segment_id": int(seg["segment_id"]),
                    "time_range": seg["time_range"],
                }
                for seg in segment_infos_for_physrope
                if seg["time_range"] is not None
            ]
            print(
                f"[PhysRoPE DATA DEBUG {self._physrope_prompt_debug_count}/{physrope_debug_limit}] "
                f"idx={idx} videoid={videoid} fps={clip_fps} "
                f"sample_n_frames={sample_n_frames} sample_n_latent={sample_n_latent} "
                f"time_range_mode={physrope_time_range_mode} task={cond_debug_task} "
                f"dst_ranges={dst_ranges_dbg} src_ranges={src_ranges_dbg} "
                f"anchor_frames={anchor_frames_for_prompt} prompt_rewrite_changed={prompt_rewrite_changed}\n"
                f"  prompt: {prompt}\n"
                f"  text_segment_time_ranges: {debug_ranges}",
                flush=True,
            )

        # get text token id
        text_segment_ids = None
        text_segment_time_ranges = None
        if segment_infos_for_physrope is not None and len(segment_infos_for_physrope) > 0:
            if is_uncond:
                text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt, self.text_len)
                text_segment_time_ranges = [
                    {
                        "segment_id": int(seg.get("segment_id", -1)),
                        "time_range": seg.get("time_range"),
                    }
                    for seg in segment_infos_for_physrope
                    if seg.get("time_range") is not None
                ]
            else:
                text_ids, text_mask, text_segment_ids, text_segment_time_ranges = \
                    self.caption_aug.tokenize_segments(
                        self.text_encoder, self.text_len, segment_infos_for_physrope, data_type="video"
                    )
        else:
            text_ids, text_mask = self.get_text_tokens(self.text_encoder, prompt, self.text_len)

        # get byt5 text token id
        if self.args.glyph_byT5_v2:
            byt5_text_ids = torch.zeros((self.byt5_max_length), dtype=torch.int64)
            byt5_text_mask = torch.zeros((self.byt5_max_length), dtype=torch.int64)
            byt5_text_valid = torch.tensor(0, dtype=torch.int64)

            pattern = r'\"(.*?)\"|“(.*?)”'
            matches = re.findall(pattern, prompt)
            glyph_byT5_text_list = [match[0] or match[1] for match in matches]
            glyph_byT5_text_list = list(dict.fromkeys(glyph_byT5_text_list)) if len(glyph_byT5_text_list) > 1 else glyph_byT5_text_list

            if len(glyph_byT5_text_list) > 0:
                # if self.video_caption_i2v_instruction_enable:
                #     prompt_i2v_instruction = prompt
                #     valid_i2v_instruction = "false"
                if random.random() < self.byt5_uncond_p:
                    glyph_byT5_text_list = ["" for _ in range(len(glyph_byT5_text_list))]

                text_prompt_style_list = [{'color': None, 'font-family': None} for _ in range(len(glyph_byT5_text_list))]
                glyph_byT5_text_formatted = self.prompt_format.format_prompt(glyph_byT5_text_list, text_prompt_style_list)

                byt5_text_ids, byt5_text_mask = self.get_byt5_text_tokens(self.byt5_tokenizer, self.byt5_max_length,
                                                                          glyph_byT5_text_formatted)
                byt5_text_ids = byt5_text_ids.squeeze(0)
                byt5_text_mask = byt5_text_mask.squeeze(0)
                byt5_text_valid = torch.tensor(1, dtype=torch.int64)

        kwargs = {
            "text": prompt,
            "index": idx,
            "type": 'video',
            "dataset_type": 'video',
            'bucket': [num_frames, height, width],
            "videoid": videoid,
            "part_version": part_version if part_version is not None else '',
        }

        # Keep PhysRoPE keys stable across samples so default_collate does not
        # fail when a batch mixes segment-aware and plain prompts.
        if text_segment_ids is None:
            kwargs["text_segment_ids"] = torch.full((self.text_len,), -1, dtype=torch.long)
        else:
            kwargs["text_segment_ids"] = torch.tensor(text_segment_ids, dtype=torch.long)
        if text_segment_time_ranges is None:
            kwargs["text_segment_time_ranges"] = json.dumps([])
        else:
            kwargs["text_segment_time_ranges"] = json.dumps(text_segment_time_ranges)

        if self.text_encoder_2 is not None:
            assert not self.video_caption_i2v_instruction_enable, f"Not supported!"
            text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, prompt, self.text_len)
            kwargs["text_ids_2"] = text_ids_2.clone()
            kwargs["text_mask_2"] = text_mask_2.clone()

        if self.args.glyph_byT5_v2:
            kwargs["byt5_text_ids"] = byt5_text_ids.clone()
            kwargs["byt5_text_mask"] = byt5_text_mask.clone()
            kwargs["byt5_text_valid"] = byt5_text_valid

        if self.video_caption_i2v_instruction_enable:
            assert self.text_encoder_2 is None, f"Not supported!"
            text_ids_i2v_instruction, text_mask_i2v_instruction = self.get_text_tokens(self.text_encoder, prompt_i2v_instruction, self.text_len)
            kwargs["text_ids_i2v_instruction"] = text_ids_i2v_instruction.clone()
            kwargs["text_mask_i2v_instruction"] = text_mask_i2v_instruction.clone()
            kwargs["text_i2v_instruction"] = prompt_i2v_instruction
            kwargs["valid_i2v_instruction"] = valid_i2v_instruction

        if recreation_mask is not None:
            kwargs["recreation_mask"] = recreation_mask
        else:
            kwargs['recreation_mask'] = torch.empty((0), dtype=torch.bool)

        if self.fl2v_enable:
            kwargs['last_frame_tensor'] = self.get_last_frame_tensor(idx=idx, latent_size=latents.size())

        # Per-segment condition tensors (always emitted; zero/-1 padding when inactive
        # or when the source arrow lacks the column). cond_seg_count == 0 means t2v.
        kwargs["cond_seg_count"] = torch.tensor(cond_seg_count_int, dtype=torch.long)
        kwargs["cond_seg_dst_ranges"] = cond_seg_dst_ranges
        kwargs["cond_seg_physical_dst_ranges"] = cond_seg_physical_dst_ranges
        kwargs["cond_seg_src_ranges"] = cond_seg_src_ranges
        kwargs["cond_seg_latents_packed"] = cond_seg_latents_packed
        kwargs["cond_seg_native_latents"] = cond_seg_native_latents
        kwargs["cond_seg_native_ranges"] = cond_seg_native_ranges
        kwargs["cond_seg_mask"] = cond_seg_mask
        kwargs["cond_seg_kind"] = cond_seg_kind
        kwargs["cond_seg_prompt_ids"] = cond_seg_prompt_ids
        kwargs["cond_seg_sparse_mappings"] = json.dumps(
            cond_seg_sparse_mappings, ensure_ascii=False, separators=(",", ":")
        )
        kwargs["cond_debug_raw_candidates"] = torch.tensor(cond_debug_raw_candidates, dtype=torch.long)
        kwargs["cond_debug_prob_dropped"] = torch.tensor(cond_debug_prob_dropped, dtype=torch.long)
        kwargs["cond_debug_bad_skipped"] = torch.tensor(cond_debug_bad_skipped, dtype=torch.long)
        kwargs["cond_debug_video_slots"] = torch.tensor(cond_debug_video_slots, dtype=torch.long)
        kwargs["cond_debug_anchor_slots"] = torch.tensor(cond_debug_anchor_slots, dtype=torch.long)
        kwargs["cond_debug_l4_slots"] = torch.tensor(cond_debug_l4_slots, dtype=torch.long)
        kwargs["cond_debug_task"] = cond_debug_task

        # 只保留固定存在的变量
        result = [
            pixel_values,
            latents,
            text_ids.clone(),
            text_mask.clone(),
            {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()}
        ]

        return tuple(result)

    def __getitem__(self, idx):
        try_times = 10000
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        def _next_idx():
            if self.multitemp and self.multireso:
                if len(self.aspect_ratio_index[self.closest_temp][self.closest_ratio]) == 1:
                    print(f'RANK: {rank}, current bucket has only one sample: {self.aspect_ratio_index[self.closest_temp][self.closest_ratio]}')
                return random.choice(self.aspect_ratio_index[self.closest_temp][self.closest_ratio])
            if self.multireso:
                return random.choice(self.aspect_ratio_index[self.closest_ratio])
            return np.random.randint(len(self))

        for i in range(try_times):
            try:
                return self.get_batch(idx)
            except Exception as e:
                print(f"RANK: {rank}, idx: {idx}, table size: {len(self.table)}, key: {self.video_key}")
                print(f"RANK: {rank}, Try {i} times - Error details: {str(e)}-{self.table[self.video_key][idx]}-{self.arrow_file}-{self.cache_folder}-{traceback.format_exc()}\n")
                idx = _next_idx()
                print(f"RANK: {rank}, get new item: idx={idx}, md5={self.table[self.video_key][idx]}")
        raise RuntimeError(f'RANK: {rank}, Too many bad data.')


class VideoArrowConcatDataset:
    def __init__(self,
                args,                       # 其他参数
                index_file,                 # arrow info存放路径json
                index_file_weights: list = None,
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
                uncond_p=0.0,               # 使用原始空字符串的概率
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
                recreation_mask_col: str='recreation_mask_data', #  recreation mask 列名
                skip_sources: list=None,    # 要跳过的source列表
                index_file_sample_seed=42,  # 对index file 采样时的seed
                video_caption_i2v_instruction_enable=None,
                video_caption_i2v_instruction_col_probs=None,
                video_caption_i2v_instruction_col=None,
                video_caption_i2v_instruction_col_2=None,
                video_caption_i2v_instruction_sub_col_probs=None,
                video_caption_i2v_instruction_sub_col=None,
                video_caption_i2v_instruction_sub_col_2=None,
                video_caption_i2v_instruction_sub_col_3=None,
                local_seed=None,
                ) -> None:
        self.args = args
        self.use_latent_shape = use_latent_shape
        self.index_file_sample_seed = index_file_sample_seed
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
        all_aspect_ratio_index_lists = []
        all_aspect_ratios_nums_lists = []
        # 读取所有index文件内容
        for idx, (index_file, weight) in enumerate(zip(self.index_files, self.index_weights)):
            if not index_file.endswith('.json'):
                raise ValueError(f'index_file {index_file} must be json file')

            with open(index_file) as f:
                d = json.load(f)
                assert 'arrow_list' in d, f'index_file {index_file} must contain arrow_list'
                arrow_list = d['arrow_list']
                cache_folder_list = d.get('cache_folder_list', [""] * len(arrow_list))
                video_folder_list = d.get('video_folder_list', [""] * len(arrow_list))

                if multireso:
                    aspect_ratio_index_list = d.get('aspect_ratio_index_list', [None] * len(arrow_list))
                    aspect_ratios_nums_list = d.get('aspect_ratios_nums_list', [None] * len(arrow_list))
                else:
                    aspect_ratio_index_list = [None] * len(arrow_list)
                    aspect_ratios_nums_list = [None] * len(arrow_list)

                if weight == 1:
                    choose_indices = list(range(len(arrow_list)))
                else:
                    # 按权重采样
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
                all_aspect_ratio_index_lists.extend([aspect_ratio_index_list[i] for i in choose_indices])
                all_aspect_ratios_nums_lists.extend([aspect_ratios_nums_list[i] for i in choose_indices])

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

        # 按相同顺序shuffle all_arrow_lists, all_video_folder_lists, all_cache_folder_lists, all_aspect_ratio_index_lists, all_aspect_ratios_nums_lists
        # 固定随机种子，保证每次shuffle结果相同
        if len(self.index_files) > 1:
            rnd = random.Random(self.index_file_sample_seed)
            random_indexs = list(range(len(all_arrow_lists)))
            rnd.shuffle(random_indexs)
            all_arrow_lists = [all_arrow_lists[i] for i in random_indexs]
            all_video_folder_lists = [all_video_folder_lists[i] for i in random_indexs]
            all_cache_folder_lists = [all_cache_folder_lists[i] for i in random_indexs]
            all_aspect_ratio_index_lists = [all_aspect_ratio_index_lists[i] for i in random_indexs]
            all_aspect_ratios_nums_lists = [all_aspect_ratios_nums_lists[i] for i in random_indexs]

        # 准备有效的数据项（过滤掉被忽略的 cache_folder）
        valid_data_items = []
        for i, (arrow_file, video_folder, cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict) in enumerate(zip(all_arrow_lists, all_video_folder_lists, all_cache_folder_lists, all_aspect_ratio_index_lists, all_aspect_ratios_nums_lists)):
            if cache_folder not in ignored_cache_folders:
                valid_data_items.append((i, arrow_file, video_folder, cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict))

        def create_dataset(data_item):
            """创建单个dataset的函数，用于并发执行"""
            original_index, arrow_file, video_folder, cache_folder, aspect_ratio_index_dict, aspect_ratios_nums_dict = data_item

            dataset = VideoArrowDataset(
                arrow_file                  = arrow_file,
                video_folder                = video_folder,
                cache_folder                = cache_folder,
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
                uncond_p                    = uncond_p,
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
                recreation_mask_col         = recreation_mask_col,
                skip_sources                = skip_sources,
                video_caption_i2v_instruction_enable        = video_caption_i2v_instruction_enable,
                video_caption_i2v_instruction_col_probs     = video_caption_i2v_instruction_col_probs,
                video_caption_i2v_instruction_col           = video_caption_i2v_instruction_col,
                video_caption_i2v_instruction_col_2         = video_caption_i2v_instruction_col_2,
                video_caption_i2v_instruction_sub_col_probs = video_caption_i2v_instruction_sub_col_probs,
                video_caption_i2v_instruction_sub_col       = video_caption_i2v_instruction_sub_col,
                video_caption_i2v_instruction_sub_col_2     = video_caption_i2v_instruction_sub_col_2,
                video_caption_i2v_instruction_sub_col_3     = video_caption_i2v_instruction_sub_col_3,
                local_seed                                  = local_seed,
            )
            return original_index, dataset

        # 使用线程池并发创建datasets
        max_workers = min(len(valid_data_items), os.cpu_count() or 4)  # 限制线程数
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
        rank0_print(logger, f"Starting post-processing of {len(valid_data_items)} datasets...")

        for dataset in dataset_results:
            if dataset is not None:
                self.datasets.append(dataset)
                self.table.append(dataset.table)
                total_length += len(dataset)

                if multitemp and multireso:
                    for temp, aspect_ratios2num in dataset.aspect_ratios_nums.items():
                        mapped_temp = _canonicalize_bucket_key(
                            temp, self.temporal_aspect_ratios_nums.keys(), int
                        )
                        for aspect_ratio, nums in aspect_ratios2num.items():
                            mapped_ratio = _canonicalize_bucket_key(
                                aspect_ratio,
                                self.temporal_aspect_ratios_nums[mapped_temp].keys(),
                                float,
                            )
                            self.temporal_aspect_ratios_nums[mapped_temp][mapped_ratio] += nums

                elif multireso:
                    for k, v in dataset.aspect_ratios_nums.items():
                        mapped_ratio = _canonicalize_bucket_key(
                            k, self.aspect_ratios_nums.keys(), float
                        )
                        self.aspect_ratios_nums[mapped_ratio] += v

        post_process_time = time.time() - post_process_start_time
        rank0_print(logger, f"Post-processing completed in {post_process_time:.3f}s")

        # PyArrow表合并耗时
        table_concat_start_time = time.time()
        self.table = pa.concat_tables(self.table, promote_options="default")
        table_concat_time = time.time() - table_concat_start_time

        # 总耗时统计
        total_cost_time = time.time() - s_time
        rank0_print(logger, f'=== Dataset Loading Summary ===')
        rank0_print(logger, f'Total datasets: {len(valid_data_items)}')
        rank0_print(logger, f'Total data length: {total_length}')
        rank0_print(logger, f'Concurrent workers: {max_workers}')
        rank0_print(logger, f'Dataset creation time: {concurrent_total_time:.3f}s')
        rank0_print(logger, f'Post-processing time: {post_process_time:.3f}s')
        rank0_print(logger, f'Table concat time: {table_concat_time:.3f}s')
        rank0_print(logger, f'Total loading time: {total_cost_time:.3f}s')
        rank0_print(logger, f'Average time per dataset: {concurrent_total_time/len(valid_data_items):.3f}s')
        rank0_print(logger, f'================================')
        self.total_length = total_length
        self.concat_dataset = ConcatDataset(self.datasets)
        self.dataset_type = 'video' # 预留字段

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, item):
        return self.concat_dataset[item]

    def get_data_info(self, index):
        if self.use_latent_shape:
            latent_shape = eval(self.table['latent_shape'][index].as_py())
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

    #print(dataset.__getitem__(0))
