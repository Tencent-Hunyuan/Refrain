import random
import os
import sys
import re
import time
import json
import requests
import tempfile
import warnings

import nltk
import numpy as np
import pyarrow as pa
from functools import partial
from typing import TypeVar, Optional, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import omegaconf
from omegaconf.listconfig import ListConfig

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from torch.utils.data import ConcatDataset, ChainDataset
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

from hymm.dataset.arrow_index import ArrowIndexV2, MultiIndexV2, arrow_mapper
from hymm.dataset.samplers import DistributedSamplerWithStartIndex

import hymm.models.audio_encoders.dac_bigvae.audio as audio_op
from hymm.utils.torch_utils import set_worker_seed_builder
from hymm.utils.distributed_utils import rank0_print

from hymm.models.t2a_modules.t5 import T5EncoderModel as UMT5EncoderModel
from hymm.constants import TEXT_ENCODER_PATH

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

NLTK_PRE_DATA_PATH = "assets/local/resource_a6f8bebf01"
MEANINGLESS_WORDS = ["is", "are", "being", "am", "be", "was", "were", "with", "object", "objects", "something"]

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    #  logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    #  kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents

class AudioDiffusionFixedLenIndexDataset(Dataset):
    """
    专门用于纯音频扩散模型的数据集，移除所有视频相关功能。
    """
    def __init__(
        self,
        index_file,
        index_kwargs=None,
        logger=None,
        max_text_length=181,
        audio_vae_latent_max_length=400,  # 8s
        debug=False,
        drop_text=0.0,
        feat_in_shadow=False,
        shadow_file_cfg={},
        audio_vae_latent_dim=64,
        keywords_training_ratio=0,
        enable_quality_tag_for_pureauio=True,  # 纯音频默认启用质量标签
        vae_sample=False,
        bandwidth_tag=False,
        vae_type='dac_vae',
        use_repa_with_audiossl=False,
        use_repa_with_eat=False,
        audiossl_dim=768,
        eat_dim=1024,
        max_audiossl_length=200,
        max_eat_length=409,
        use_transcript_text_feat=False,
        extract_asr_from_caption=False,
        online_vae_extraction=False,
        download_from_cos=False,
        max_audio_waveform_length=384000,
        wav_sampling_rate=48000,
        text_encoder=None,
    ):
        if logger is None:
            from loguru import logger

            self.logger = logger

        # Prepare index manager
        self.index_file = index_file
        index_load_kwargs = dict(
            batch_size=index_kwargs.get("batch_size", 1),
            world_size=index_kwargs.get("world_size", 1),
            sample_strategy=index_kwargs.get("index_strategy", "uniform"),
            probability=index_kwargs.get("index_probability", None),
        )

        # 纯音频数据集参数
        self.feat_in_shadow = feat_in_shadow
        self.keywords_training_ratio = keywords_training_ratio
        self.enable_quality_tag_for_pureauio = enable_quality_tag_for_pureauio
        self.drop_text = drop_text
        self.audio_vae_latent_max_length = audio_vae_latent_max_length
        self.max_text_length = max_text_length
        self.audio_vae_latent_dim = audio_vae_latent_dim
        self.extract_asr_from_caption = extract_asr_from_caption
        logger.info(f"Extract ASR from caption: {self.extract_asr_from_caption}")

        assert self.audio_vae_latent_dim is not None, "audio_vae_latent_dim must be specified"
        
        # 在线VAE特征提取相关参数
        self.online_vae_extraction = online_vae_extraction
        self.max_audio_waveform_length = max_audio_waveform_length
        self.wav_sampling_rate = wav_sampling_rate
        self.download_from_cos = download_from_cos
        
        # 在线VAE提取模式
        if self.online_vae_extraction:
            self.logger.info("启用在线VAE提取模式")
        
        # Text encoder
        self.text_encoder = text_encoder

        self.debug = debug
        self.vae_sample = vae_sample
        self.vae_type = vae_type
        self.logger.info(f"Using vae_type: {vae_type}")
        self.bandwidth_tag = bandwidth_tag
        if self.vae_sample:
            self.logger.info(f" ***********Using VAE Sample ***************")
        if self.bandwidth_tag:
            self.logger.info(f" ***********Using Bandwidth Tag ***************")

        if self.keywords_training_ratio > 0:
            nltk.data.path.append(NLTK_PRE_DATA_PATH)
        
        self.use_repa_with_audiossl = use_repa_with_audiossl
        self.use_repa_with_eat = use_repa_with_eat
        self.audiossl_dim = audiossl_dim
        self.eat_dim = eat_dim
        self.max_audiossl_length = max_audiossl_length
        self.max_eat_length = max_eat_length
        self.use_transcript_text_feat = use_transcript_text_feat

        # 仅音频相关的shadow文件配置
        shadow_file_fn = {}
        if feat_in_shadow:
            self.logger.info(shadow_file_cfg)
            # 音频VAE特征
            shadow_file_fn.update({"vae": partial(arrow_mapper, suffix=shadow_file_cfg["vae_suffix"])})
            self.vae_col = shadow_file_cfg["vae_col"]

            # 带宽标签（用于质量控制）
            if self.bandwidth_tag:
                shadow_file_fn.update({"bandwidth" : partial(arrow_mapper, suffix=shadow_file_cfg["bandwidth_suffix"])})
                self.bandwidth_col = shadow_file_cfg["bandwidth_col"]

            # 文本标题
            shadow_file_fn.update({"caption": partial(arrow_mapper, suffix=shadow_file_cfg["structured_caption_suffix"])})
            self.caption_col = shadow_file_cfg["structured_caption_col"]

            # ASR文本
            if self.use_transcript_text_feat:
                shadow_file_fn.update({"asr_text": partial(arrow_mapper, suffix=shadow_file_cfg["asr_suffix"])})
                self.asr_col = "asr_text"
                self.lang_prob_col = "language_prob" 

            # 音频SSL特征
            if self.use_repa_with_audiossl:
                shadow_file_fn.update({"audiossl_feat": partial(arrow_mapper, suffix=shadow_file_cfg["audiossl_suffix"])})
            if self.use_repa_with_eat:
                shadow_file_fn.update({"eat_feat": partial(arrow_mapper, suffix=shadow_file_cfg["eat_suffix"])})


            if self.online_vae_extraction:
                if self.download_from_cos:
                    self.cos_col = "cos_signed_url"
                else:
                    shadow_file_fn.update({"audio_dir" : partial(arrow_mapper, suffix="_audio_dir_nj")})
        
        index_load_kwargs["shadow_file_fn"] = shadow_file_fn
        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"Using {self.index_manager}")
    
    def _find_audio_file(self, index, source_id):
        """
        通过COS链接下载音频文件或从本地路径获取音频文件
        """
        if self.download_from_cos:
            cos_url = self.index_manager.get_attribute(index, column="cos_signed_url")
            
            if cos_url:
                return self._download_audio_from_cos(cos_url, source_id)
            
            self.logger.warning(f"无法获取COS链接，source_id: {source_id}")
            return None
        else:
            # 从本地读取
            audio_path = self.index_manager.get_attribute(index, column="audio_dir", shadow="audio_dir")
            if not os.path.exists(audio_path):
                self.logger.warning(f"本地音频文件不存在: {audio_path}, source_id: {source_id}")
                return None
                
            return audio_path

    def _download_audio_from_cos(self, cos_url, source_id):
        """
        从COS链接下载音频文件到临时目录
        """
        try:
            self.logger.debug(f"开始下载音频文件: {source_id}, URL: {cos_url}")
            
            # 添加请求头
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'audio/*,*/*;q=0.9',
            }
            
            response = requests.get(cos_url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # 详细检查响应
            content_type = response.headers.get('content-type', '')
            content_length = response.headers.get('content-length', '0')
            self.logger.debug(f"响应头 - Content-Type: {content_type}, Content-Length: {content_length}")
            
            # 检查响应内容
            if len(response.content) == 0:
                self.logger.error(f"下载的音频文件为空，source_id: {source_id}, URL: {cos_url}")
                return None
            
            # 检查内容是否是错误页面
            content_preview = response.content[:200]
            if content_preview.startswith(b'<!DOCTYPE') or content_preview.startswith(b'<html'):
                self.logger.error(f"下载的是HTML页面而不是音频文件，source_id: {source_id}, URL: {cos_url}")
                return None
                
            if content_preview.startswith(b'{') or content_preview.startswith(b'['):
                self.logger.error(f"下载的是JSON响应而不是音频文件，source_id: {source_id}, URL: {cos_url}")
                return None
            
            # 根据文件头确定格式
            if content_preview[4:8] == b'ftyp':
                suffix = '.m4a'
                self.logger.debug(f"检测到M4A格式文件头")
            elif content_preview[:4] == b'RIFF' and b'WAVE' in content_preview[:20]:
                suffix = '.wav'
                self.logger.debug(f"检测到WAV格式文件头")
            elif content_preview[:3] == b'ID3' or content_preview[:2] == b'\xff\xfb':
                suffix = '.mp3'
                self.logger.debug(f"检测到MP3格式文件头")
            else:
                suffix = '.audio'
                # 打印文件头用于调试
                hex_header = ' '.join([f'{b:02x}' for b in content_preview[:16]])
                self.logger.warning(f"未知格式，文件头: {hex_header}")
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            self.logger.debug(f"文件下载完成: {len(response.content)} bytes -> {temp_path}")
            
            # 简单验证文件存在和大小
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                self.logger.error(f"下载的音频文件无效，source_id: {source_id}, URL: {cos_url}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
            
            self.logger.debug(f"从COS下载音频文件成功: {source_id} -> {temp_path}")
            return temp_path
        except Exception as e:
            self.logger.error(f"从COS下载音频文件失败，source_id: {source_id}, URL: {cos_url}, error: {e}")
            return None


    def extract_nouns(self, caption, meaningless_words=MEANINGLESS_WORDS):
        words = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(words)
        nouns = [word for word, pos in pos_tags if (pos.startswith("NN") and word not in meaningless_words)]
        return nouns

    def extract_verbs(self, caption, meaningless_words=MEANINGLESS_WORDS):
        words = nltk.word_tokenize(caption)
        pos_tags = nltk.pos_tag(words)
        verbs = [word for word, pos in pos_tags if pos.startswith("VB") and word not in meaningless_words]
        return verbs

    def load_index(
        self,
        batch_size,
        world_size,
        shadow_file_fn={},
        sample_strategy="uniform",
        probability=None,
    ):
        if isinstance(self.index_file, str):
            self.index_file = [self.index_file]
        assert isinstance(self.index_file, (list, tuple)), (
            f"`index_file` should be a str or a list of str, got {type(self.index_file)}"
        )
        self.logger.info(f"Loading dataset index: {self.index_file}")
        self.logger.info(f"shadow_file_fn: {shadow_file_fn}")
        index_file = self.index_file
        if len(index_file) == 1:
            index_manager = ArrowIndexV2(
                index_file[0],
                shadow_file_fn=shadow_file_fn,
            )
        else:
            index_manager = MultiIndexV2(
                index_file,
                shadow_file_fn=shadow_file_fn,
                sample_strategy=sample_strategy,
                probability=probability,
            )
        return index_manager

    def _get_language_tag(self, index):
        """
        获取语言标签，尝试多个可能的列名
        
        Args:
            index: 数据索引
            
        Returns:
            str: 语言标签，失败时返回"en"
        """
        # 按优先级尝试不同的列名
        possible_columns = ["language", "asr_language"]
        
        for column in possible_columns:
            try:
                language_tag = self.index_manager.get_attribute(index, column=column, shadow="asr_text")
                if language_tag:
                    return language_tag
            except Exception:
                continue
        
        # 所有尝试都失败，返回默认值
        return "en"

    def _get_asr_data(self, index, drop_text=False):
        """
        获取ASR文本和语言标签，统一处理异常情况
        
        Args:
            index: 数据索引
            drop_text: 是否丢弃文本
            
        Returns:
            tuple: (asr_text, language_tag)
        """
        # 默认值
        asr_text = ""
        language_tag = "en"
        lang_prob = 1.0
        
        # 1. 获取ASR文本和置信度
        try:
            lang_prob = self.index_manager.get_attribute(index, column=self.lang_prob_col, shadow="asr_text") or 1.0
            asr_text = self.index_manager.get_attribute(index, column=self.asr_col, shadow="asr_text") or ""
        except Exception:
            # 获取失败时使用默认值
            pass
        
        # 2. 获取语言标签（仅在有ASR文本时）
        if asr_text and asr_text.strip():
            language_tag = self._get_language_tag(index)
        
        # 3. 语言标签有效性检查
        if language_tag not in ["zh", "en", "ja", "hi", "ar", "de", "es", "fr", "hu", "it", "nl", "pl", "pt", "tr"]:
            asr_text = ""
            language_tag = "en"
        
        # 4. 置信度过滤
        if lang_prob < 0.7:
            asr_text = ""

        # 5. 丢弃文本标记处理
        if drop_text:
            asr_text = ""
        
        return asr_text, language_tag

    @staticmethod
    def get_text_tokens(text_encoder, text):
        if isinstance(text_encoder, UMT5EncoderModel):
            text_encoder_outputs = text_encoder.text2tokens(text, return_mask=True, add_special_tokens=True)
            text_token_ids = text_encoder_outputs['input_ids']
            mask = text_encoder_outputs['attention_mask']
        else:
            #  Add prefix for audio caption, due to special token.
            text = "Audio capition: " + text
            text_encoder_outputs = text_encoder.text2tokens(
                text, data_type="audio", max_length=text_encoder.max_length
            )
            text_token_ids = text_encoder_outputs['input_ids']
            mask = text_encoder_outputs['attention_mask']
        
        return text_token_ids, mask

    def get_data_from_shadow(self, index):
        # 仅获取音频相关的特征
        caption = ""
        caption = self.index_manager.get_attribute(index, column=self.caption_col, shadow="caption")
        if caption is None:
            caption = ""

        audio_path = None
        vae_mean = None
        vae_std = None
        # 获取音频VAE特征
        if self.online_vae_extraction:
            # 在线提取VAE特征 - 改为获取音频文件路径
            source_id = self.index_manager.get_attribute(index, column="source_id")
            if source_id is None:
                # 如果没有source_id列，尝试其他可能的ID列
                for id_col in ['data_id', 'id', 'video_source_id']:
                    source_id = self.index_manager.get_attribute(index, column=id_col)
                    if source_id is not None:
                        break
            
            if source_id is not None:
                audio_path = self._find_audio_file(index, source_id)
            else:
                self.logger.warning(f"无法获取样本ID，index: {index}")
        else:
            # 获取音频VAE特征
            vae_mean = self.index_manager.get_attribute(index, column="audio_vae_mean", shadow="vae")
            if self.vae_type == 'ttvae':
                vae_std = self.index_manager.get_attribute(index, column="audio_vae_scale", shadow="vae")
            else:
                vae_std = self.index_manager.get_attribute(index, column="audio_vae_std", shadow="vae")
        
        return caption, audio_path, vae_mean, vae_std

    def detect_language(self, text: str) -> str:
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        if chinese_pattern.search(text):
            return "zh"
        else:
            return "en"

    def _get_asr_from_caption(self, caption):
        pattern = r'Spoken content: "([^"]*)"'
        match = re.search(pattern, caption)

        if match:
            return match.group(1)
        else:
            return ""

    def __getitem__(self, index):
        try:
            # ======================================= 获取纯音频特征 =======================================
            drop_text = random.random() < self.drop_text
            if self.feat_in_shadow:
                caption, audio_path, vae_mean, vae_std = self.get_data_from_shadow(index)

                # 带宽标签处理
                if self.bandwidth_tag:
                    try:
                        bandwidth = self.index_manager.get_attribute(index, column=self.bandwidth_col, shadow="bandwidth")
                    except Exception:
                        # NOTE(qiunnli): 有的数据集没有打上bandwith标签，默认12000
                        bandwidth = 12000
                    assert bandwidth is not None, f"bandwidth latents must be prepared, not be None"
                
                # 音频SSL特征处理
                if self.use_repa_with_audiossl:
                    audiossl_feat = self.index_manager.get_attribute(index, column="audio_feat", shadow="audiossl_feat")
                    audiossl_feat = torch.from_numpy(np.frombuffer(audiossl_feat, dtype=np.float32))
                    audiossl_feat = audiossl_feat.reshape(-1, self.audiossl_dim)    
                    if audiossl_feat.shape[0] > self.max_audiossl_length:
                        audiossl_feat = audiossl_feat[:self.max_audiossl_length]
                
                if self.use_repa_with_eat:
                    eat_feat = self.index_manager.get_attribute(index, column="audio_feat", shadow="eat_feat")
                    eat_feat = torch.from_numpy(np.frombuffer(eat_feat, dtype=np.float32))
                    eat_feat = eat_feat.reshape(-1, self.eat_dim)
                    if eat_feat.shape[0] > self.max_eat_length:
                        eat_feat = eat_feat[:self.max_eat_length]

                # ASR文本处理
                if self.use_transcript_text_feat:
                    if self.extract_asr_from_caption:
                        asr_text = self._get_asr_from_caption(caption)
                        language_tag = self.detect_language(asr_text) if asr_text else "en"
                    else:
                        asr_text, language_tag = self._get_asr_data(index, drop_text)

            # 音频标题特殊处理
            if self.keywords_training_ratio > 0 and random.random() < self.keywords_training_ratio:
                nouns = self.extract_nouns(caption)
                verbs = self.extract_verbs(caption)
                unique_words = set(nouns) | set(verbs)
                caption = ", ".join(unique_words)
            
            # 纯音频质量标签
            if self.enable_quality_tag_for_pureauio:
                caption = caption.strip(".") + ", high-quality"

            # 带宽质量标签
            if self.bandwidth_tag and bandwidth >= 16000:
                caption = caption.strip(". ") + ", high-quality"

            # ======================================= 音频VAE特征处理 =======================================
            audio_waveform = None
            if self.online_vae_extraction:
                processed_waveform = None
                if audio_path and os.path.exists(audio_path):
                    try:                    
                        waveform, _, _ = audio_op.load_and_resample_if_necessary(audio_path, target_sr=self.wav_sampling_rate)
                        waveform = torch.clamp(waveform, -1.0, 1.0)
                        processed_waveform = waveform.reshape(1, 1, -1)

                    except Exception as e:
                        # 设置为None，让后续处理跳过这个样本
                        # processed_waveform = None
                        raise e
                    finally:
                        # Clean up the temporary file downloaded from COS
                        if self.download_from_cos and audio_path.startswith(tempfile.gettempdir()):
                            try:
                                os.unlink(audio_path)
                            except Exception as cleanup_err:
                                self.logger.warning(f"Failed to cleanup temp file {audio_path}: {cleanup_err}")

                if processed_waveform is not None:
                    # Truncate or pad to max_audio_waveform_length
                    current_len = processed_waveform.shape[-1]
                    if current_len >= self.max_audio_waveform_length:
                        # Truncate
                        audio_waveform = processed_waveform[..., :self.max_audio_waveform_length]
                    else:
                        # Pad
                        pad_len = self.max_audio_waveform_length - current_len
                        audio_waveform = F.pad(processed_waveform, (0, pad_len))
                else:
                    # If processing failed, return None to have the DataLoader skip this sample.
                    raise Exception(f"Failed to process audio waveform, audio_path: {audio_path}")
                # Use a placeholder for audio_tokens, it will be ignored in the trainer
                audio_tokens = torch.zeros((self.audio_vae_latent_dim, self.audio_vae_latent_max_length))
            else:
                assert vae_mean is not None, f"vae mean latents must be prepared, not be None"
                assert vae_std is not None, f"vae std latents must be prepared, not be None"

                if "dac" in self.vae_type or self.vae_type == 'bigvae':
                    if not self.vae_sample:
                        vae_mean = torch.from_numpy(np.frombuffer(vae_mean, dtype=np.float32))
                        vae_mean = vae_mean.reshape(self.audio_vae_latent_dim, -1)
                        audio_tokens = vae_mean
                    else:
                        vae_mean = torch.from_numpy(np.frombuffer(vae_mean, dtype=np.float32))
                        vae_mean = vae_mean.reshape(self.audio_vae_latent_dim, -1)
                        vae_std = torch.from_numpy(np.frombuffer(vae_std, dtype=np.float32))
                        vae_std = vae_std.reshape(self.audio_vae_latent_dim, -1)
                        # 从VAE采样
                        audio_tokens = vae_mean + vae_std * torch.randn(vae_std.shape, dtype=torch.float32)
                elif self.vae_type == 'ttvae':
                    if not self.vae_sample:
                        vae_mean = torch.from_numpy(np.frombuffer(vae_mean, dtype=np.float32))
                        vae_mean = vae_mean.reshape(self.audio_vae_latent_dim, -1)
                        audio_tokens = vae_mean
                    else:
                        vae_mean = torch.from_numpy(np.frombuffer(vae_mean, dtype=np.float32))
                        vae_mean = vae_mean.reshape(self.audio_vae_latent_dim, -1)
                        vae_scale = torch.from_numpy(np.frombuffer(vae_std, dtype=np.float32))
                        vae_scale = vae_scale.reshape(self.audio_vae_latent_dim, -1)
                        # 从VAE采样
                        audio_tokens = vae_sample(vae_mean, vae_scale)

                # 音频序列长度处理和填充
                if audio_tokens.shape[1] >= self.audio_vae_latent_max_length:
                    audio_tokens = audio_tokens[:, : self.audio_vae_latent_max_length]
                elif self.audio_vae_latent_max_length - audio_tokens.shape[1] <= 5:
                    # 处理略微小于目标长度的音频
                    pad_audio_tokens = audio_tokens[:, -1:].repeat(1, self.audio_vae_latent_max_length - audio_tokens.shape[1])
                    audio_tokens = torch.cat((audio_tokens, pad_audio_tokens), dim=-1)
                assert (
                    audio_tokens.shape[1] == self.audio_vae_latent_max_length
                ), f"audio_tokens len is {audio_tokens.shape[1]}, which less than the expected {self.audio_vae_latent_max_length}"

            # ======================================= 纯音频训练的文本处理 =======================================
            padded_caption = "" if drop_text else caption

            # 获取tokens
            padded_caption_ids, padded_caption_mask = self.get_text_tokens(self.text_encoder, padded_caption)

            # 获取数据标识信息用于记录
            source_id = self.index_manager.get_attribute(index, column="source_id")
            if source_id is None:
                # 如果没有source_id列，尝试其他可能的ID列
                for id_col in ['data_id', 'id', 'video_source_id']:
                    source_id = self.index_manager.get_attribute(index, column=id_col)
                    if source_id is not None:
                        break
            
            # 构建数据字典（仅音频相关）
            data_dict = {
                "caption": padded_caption,
                # [1, seq_len] -> [seq_len]
                "padded_caption_ids": padded_caption_ids.squeeze(0).clone(),
                # [1, seq_len] -> [seq_len]
                "padded_caption_mask": padded_caption_mask.squeeze(0).clone(),
                "audio_token": audio_tokens,
                "drop_text": drop_text,
                "drop_visual": True,  # 纯音频模式，始终丢弃视觉特征
                "source_id": source_id if source_id is not None else f"index_{index}",  # 记录数据来源
                "data_index": index,  # 记录数据索引
                'audio_path': audio_path,
            }
            if self.online_vae_extraction:
                data_dict['audio_waveform'] = audio_waveform
            
            # 添加可选的音频SSL特征
            if self.use_repa_with_audiossl:
                data_dict["audiossl_feat"] = audiossl_feat
            if self.use_repa_with_eat:
                data_dict["eat_feat"] = eat_feat
            if self.use_transcript_text_feat:
                data_dict['transcript_text'] = asr_text
                data_dict["language_tag"] = language_tag

            return data_dict

        except Exception as e:
            # 出现任何异常时，记录错误并返回None
            if hasattr(self, 'logger'):
                self.logger.warning(f"跳过损坏数据 index={index}: {str(e)}")
            else:
                import warnings
                warnings.warn(f"跳过损坏数据 index={index}: {str(e)}")
            return None

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 4096)
        return len(self.index_manager)

def collate_fn_filter_none(batch):
    """
    自定义 collate 函数，过滤掉 None 值
    当数据加载失败时，__getitem__ 会返回 None，这个函数会过滤掉这些 None 值
    如果整个 batch 都是 None，则返回 None
    """
    # 过滤掉 None 值
    batch = [item for item in batch if item is not None]
    
    # 如果过滤后 batch 为空，返回 None
    if len(batch) == 0:
        return None
    
    # 使用默认的 collate 函数处理剩余的数据
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

# ================= Variable length dataset ==============

def sum_aspect_ratio_nums(aspect_ratios_nums):
    return sum(num for ratio_dict in aspect_ratios_nums.values() for num in ratio_dict.values())

def get_temporal_closest_ratio(audio_length: float, temporal_buckets: list):
    """
    找到最接近音频时长且比音频时长要短的桶

    Args:
        audio_length: 音频时长（秒）
        temporal_buckets: 时长桶列表

    Returns:
        最适合的桶时长
    """
    if not temporal_buckets:
        return audio_length

    # 过滤出所有小于等于音频时长的桶
    valid_buckets = [bucket for bucket in temporal_buckets if bucket <= audio_length]

    if not valid_buckets:
        # 如果没有小于等于音频时长的桶，返回最小的桶
        return min(temporal_buckets)

    # 在有效桶中找到最大的那个（最接近且不超过音频时长）
    return max(valid_buckets)

class AudioArrowDataset(Dataset):
    def __init__(self,
                arrow_file,
                logger=None,
                max_text_length=181,
                wav_sampling_rate=48000,
                audio_vae_latent_max_length=400,  # 8s
                debug=False,
                drop_text=0.0,
                audio_vae_latent_dim=64,
                keywords_training_ratio=0,
                enable_quality_tag_for_pureauio=True,  # 纯音频默认启用质量标签
                vae_sample=False,
                bandwidth_tag=False,
                vae_type='dac_vae',
                vae_rate=50,
                use_repa_with_audiossl=False,
                bucket_temporal_min_length=2.0,
                bucket_temporal_max_length=16.0,
                bucket_temporal_interval=0.5,
                duration_ratio_index=None, # duration ratio index for temporal bucketing
                duration_ratio_nums=None, # duration ratio nums for temporal bucketing
                audiossl_dim=768,
                max_audiossl_length=200,
                use_transcript_text_feat=False,
                transcript_encoder_type=None,
                online_vae_extraction=False,
                download_from_cos=False,
                max_audio_length=15.0,
                multitemp=True,
                audio_caption_col="audio_caption",
                uncond_p=0.1,
                use_audio_vae_cache=False,
                text_encoder=None,
                **kwargs,
                ) -> None:

        if logger is None:
            from loguru import logger
        self.logger = logger

        # Prepare index manager
        self.arrow_file = arrow_file
        # 纯音频数据集参数
        self.keywords_training_ratio = keywords_training_ratio
        self.enable_quality_tag_for_pureauio = enable_quality_tag_for_pureauio
        self.drop_text = drop_text
        self.audio_vae_latent_max_length = audio_vae_latent_max_length
        self.max_text_length = max_text_length
        self.audio_vae_latent_dim = audio_vae_latent_dim

        # 在线VAE特征提取相关参数
        self.online_vae_extraction = online_vae_extraction
        self.download_from_cos = download_from_cos
        
        # 在线VAE提取模式
        if self.online_vae_extraction:
            self.logger.info("启用在线VAE提取模式")

        self.debug = debug
        self.vae_sample = vae_sample
        self.vae_type = vae_type
        self.bandwidth_tag = bandwidth_tag
        if self.vae_sample:
            self.logger.info(f" ***********Using VAE Sample ***************")
        if self.bandwidth_tag:
            self.logger.info(f" ***********Using Bandwidth Tag ***************")

        self.use_repa_with_audiossl = use_repa_with_audiossl
        self.audiossl_dim = audiossl_dim
        self.max_audiossl_length = max_audiossl_length
        self.use_transcript_text_feat = use_transcript_text_feat
        self.transcript_encoder_type = transcript_encoder_type
        if self.use_transcript_text_feat:
            assert self.transcript_encoder_type is not None, "transcript_encoder_type must be set if use_transcript_text_feat is True"
            if self.transcript_encoder_type == "byt5":
                raise NotImplementedError("Not supported transcript_encoder_type: {}".format(self.transcript_encoder_type))
            elif self.transcript_encoder_type == "voice_bpe_with_convnext_v2":
                from hymm.models.tokenizers.voice_bpe.lyrics_tokenizer import VoiceBpeTokenizer
                self.transcript_tokenizer = VoiceBpeTokenizer()

        self.sample_rate = wav_sampling_rate
        self.bucket_temporal_min_length = bucket_temporal_min_length
        self.bucket_temporal_max_length = bucket_temporal_max_length
        self.bucket_temporal_interval = bucket_temporal_interval
        self.multitemp = multitemp
        self.duration_ratio_index = duration_ratio_index
        self.duration_ratio_nums = duration_ratio_nums
        self.max_audio_length = max_audio_length
        self.audio_caption_col = audio_caption_col
        self.vae_rate = vae_rate
        self.uncond_p = uncond_p
        self.use_audio_vae_cache = use_audio_vae_cache
        self.audio_cache_folder = kwargs.get('audio_cache_folder', '')
        self.audio_vae_suffix = kwargs.get('audio_vae_suffix', '.npy')

        # load arrow
        self.table = self._load_table()
        s_time = time.time()
        rank0_print(logger, f"load {self.arrow_file} \t cost {time.time() - s_time} s \t total length {len(self.table)}")
        
        if self.multitemp:
            # 多时长桶
            # generate temporal bucket
            assert bucket_temporal_max_length > bucket_temporal_min_length, f"temporal_max_length must be larger than temporal_min_length, but got {bucket_temporal_max_length}, {bucket_temporal_min_length}"
            if bucket_temporal_max_length > max_audio_length: bucket_temporal_max_length = max_audio_length
            
            audio_length_list = [bucket_temporal_min_length]

            while audio_length_list[-1] < bucket_temporal_max_length:
                next_length = audio_length_list[-1] + bucket_temporal_interval
                if next_length > bucket_temporal_max_length:
                    next_length = bucket_temporal_max_length
                audio_length_list.append(next_length)
            self.audio_length_list = audio_length_list


        self.logic2physical_index_map, self.physical2logic_index_map = self.generate_index_mapping()
        if self.use_transcript_text_feat:
            self.asr_col = "asr_text"
            self.lang_prob_col = "language_prob" 
            self.lang_col = "asr_language"

        # 音频SSL特征
        if self.use_repa_with_audiossl:
            self.audiossl_col = "audiossl_feat"

        if self.online_vae_extraction:
            if self.download_from_cos:
                self.cos_col = "cos_signed_url"
        
        self.text_encoder = text_encoder

            
    def _load_table(self):
        table = pa.memory_map(self.arrow_file, "r")
        table = pa.ipc.RecordBatchFileReader(table).read_all()  # 打印第一条数据，0.0013秒
        return table

    def generate_index_mapping(self):
        logic2physical_index_mapping = {}
        physical2logic_index_mapping = {}
        count = 0
        
        if self.multitemp:
            for _, indices in self.duration_ratio_index.items():  # 秒数到所有物理索引
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
    
    @staticmethod
    def get_text_tokens(text_encoder, text):
        if isinstance(text_encoder, UMT5EncoderModel):
            text_encoder_outputs = text_encoder.text2tokens(text, return_mask=True, add_special_tokens=True)
            text_token_ids = text_encoder_outputs['input_ids']
            mask = text_encoder_outputs['attention_mask']
        else:
            #  Add prefix for audio caption, due to special token.
            text = "Audio capition: " + text
            text_encoder_outputs = text_encoder.text2tokens(
                text, data_type="audio", max_length=text_encoder.max_length
            )
            text_token_ids = text_encoder_outputs['input_ids']
            mask = text_encoder_outputs['attention_mask']
        
        return text_token_ids, mask
    
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

        duration = self.table['duration'][index].as_py()

        return {'duration': duration}


    def prompt_valid_check(self, prompt, another_prompt):
        if prompt == "\\N" or prompt == "":
            prompt = another_prompt
            if prompt == "\\N" or prompt == "":
                self.logger.info(f'Empty caption: {prompt}')
                raise Exception('Wrong Caption')
        return prompt

    def _get_asr_data(self, index, drop_text=False):
        """
        获取ASR文本和语言标签，统一处理异常情况
        
        Args:
            index: 数据索引
            drop_text: 是否丢弃文本
            
        Returns:
            tuple: (asr_text, language_tag)
        """
        # 默认值
        asr_text = ""
        language_tag = "en"
        lang_prob = 1.0
        
        # 1. 获取ASR文本和置信度
        try:
            lang_prob = self.table[self.lang_prob_col][index].as_py()
            asr_text = self.table[self.asr_col][index].as_py()
        except Exception:
            # 获取失败时使用默认值
            lang_prob = 0.0
            asr_text = ""
        
        # 2. 获取语言标签（仅在有ASR文本时）
        if asr_text and asr_text.strip():
            language_tag = self.table[self.lang_col][index].as_py()
        
        # 3. 语言标签有效性检查
        #  if language_tag not in ["zh", "en", "ja", "hi", "ar", "cs", "de", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "ko"]:
        if language_tag not in ["zh", "en", "ja", "fr", "it", "ko"]:
            asr_text = ""
            language_tag = "en"
        
        # 4. 置信度过滤
        if lang_prob < 0.7:
            asr_text = ""

        # 5. 丢弃文本标记处理
        if drop_text:
            asr_text = ""
        
        return asr_text, language_tag

    
    def get_batch(self, idx, index_type="logic"):
        '''
        inputs:
            idx: index
            index_type: logic or physical
        '''
        if index_type == "logic":
            idx = self.logic2physical_index_map[idx]

        # 根据索引获取记录
        if 'audio_id' in self.table.schema.names:
            audio_key = 'audio_id'
        elif 'source_id' in self.table.schema.names:
            audio_key = 'source_id'
        else:
            raise Exception(f'videoid or source_id not in self.table.schema.names')

        self.audio_key = audio_key
        audioid = self.table[audio_key][idx].as_py()
        cos_url = ''

        max_audio_length = self.max_audio_length
        if self.multitemp:
            data_info = self.get_data_info(idx, index_type="physical")
            # seconds
            duration = data_info['duration']
            closest_temp = get_temporal_closest_ratio(duration, self.audio_length_list)
            self.closest_temp = closest_temp
            max_audio_length = closest_temp
            
        audio_file_path = self.table["audio_path"][idx].as_py()

        audio_caption = self.table[self.audio_caption_col][idx].as_py()
        audio_caption = "" if audio_caption is None else audio_caption

        if random.random() < self.uncond_p:
            audio_caption = ""

        padded_caption_ids, padded_caption_mask = self.get_text_tokens(self.text_encoder, audio_caption)

        # asr_text = ""

        # asr_text, lang_tag = self._get_asr_data(idx, drop_text=self.drop_text)

        if self.use_audio_vae_cache: 
            audio = None 
            audio_latent_len = int(self.vae_rate * max_audio_length)
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
                waveform, sample_rate = torchaudio.load(audio_file_path)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                if sample_rate != self.sample_rate:
                    resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                
                # 转换为一维tensor
                audio = waveform.squeeze(0)  # max_audio_length 应该是比audio短但最接近audio长度的桶长度
                if float(audio.shape[-1]) / self.sample_rate - max_audio_length > self.bucket_temporal_interval:
                    print(f'Audio duration is {float(audio.shape[-1]) / self.sample_rate} and bucket is {max_audio_length=}')
                    return None
                # 转换为一维tensor
                # audio = waveform.squeeze(0)  # max_audio_length 应该是比audio短但最接近audio长度的桶长度
                # # audio_start_frame = 0
                # # audio_end_frame = round(audio_start_frame + max_audio_length * self.sample_rate)
                # # audio = audio[audio_start_frame:audio_end_frame]

                # if len(audio) != (audio_end_frame - audio_start_frame):  # 选出 0.5s - duration+1 的音频，
                #     if (audio_end_frame - audio_start_frame) - len(audio) < self.sample_rate * 1.0 and len(audio) > self.sample_rate * 0.5:
                #         # pad to target length
                #         audio = torch.cat([audio, torch.zeros(audio_end_frame - audio_start_frame - len(audio))], dim=-1)
                #     else:
                #         self.logger.warning(f"Audio length mismatch: {len(audio)} != {audio_end_frame - audio_start_frame}")
                #         self.logger.warning(f"Index info: {idx}, {self.table[self.audio_key][idx]}, {self.arrow_file}, {self.audio_cache_folder}")
                #         raise ValueError(f"Audio length mismatch: {len(audio)} != {audio_end_frame - audio_start_frame}")

            except Exception as e:
                raise Exception(f'Loading audio error: {e}.')
            
            results = {
                "audio_waveform": audio,
                "audio_caption": audio_caption,
                 # [1, seq_len] -> [seq_len]
                "padded_caption_ids": padded_caption_ids.squeeze(0).clone(),
                "padded_caption_mask": padded_caption_mask.squeeze(0).clone(),
            }
            return results
            
    def __getitem__(self, idx): # logic index
        return self.get_batch(idx, index_type="logic")
        

class AudioArrowConcatDataset:
    def __init__(self,
                index_file,                 # arrow info存放路径json或json列表
                index_file_weights: list = None,  # index_file对应的权重列表
                index_file_sample_seed=42,  # 对index file采样时的seed
                index_kwargs=None,
                logger=None,
                multitemp=True,
                max_text_length=181,
                wav_sampling_rate=48000,
                audio_vae_latent_max_length=400,  # 8s
                debug=False,
                drop_text=0.0,
                audio_vae_latent_dim=64,
                keywords_training_ratio=0,
                enable_quality_tag_for_pureauio=False,  # 纯音频默认启用质量标签
                vae_sample=True,
                bandwidth_tag=False,
                vae_type='dac_vae',
                vae_rate=50,
                use_repa_with_audiossl=False,
                bucket_temporal_min_length=2.0,
                bucket_temporal_max_length=16.0,
                bucket_temporal_interval=0.5,
                audiossl_dim=768,
                max_audiossl_length=200,
                use_transcript_text_feat=False,
                transcript_encoder_type=None,
                online_vae_extraction=True,
                download_from_cos=False,
                max_audio_length=15.0,
                audio_caption_col="caption_new",
                uncond_p=0.1,
                use_audio_vae_cache=False,
                text_encoder=None,
                **kwargs,
                ) -> None:

        self.index_file_sample_seed = index_file_sample_seed
        # 处理多个index_file输入
        if isinstance(index_file, str):
            self.index_files = [index_file]
            self.index_weights = [1.0] if index_file_weights is None else [index_file_weights]
        elif isinstance(index_file, list) or isinstance(index_file, omegaconf.listconfig.ListConfig):
            self.index_files = index_file
            if index_file_weights is None:
                self.index_weights = [1.0] * len(index_file)
            elif isinstance(index_file_weights, list):
                self.index_weights = index_file_weights
            else:
                raise TypeError("index_file_weights must be a list when index_file is a list")
        else:
            raise TypeError("index_file must be a string or list")

        self.multitemp = multitemp
        self.text_encoder = text_encoder

        all_arrow_lists = []
        all_cache_folder_lists = []
        all_duration_ratio_index_lists = []  # list of dict 
        all_duration_ratio_nums_lists = [] # list of dict 

        # 读取所有index文件内容
        for idx, (index_file_path, weight) in enumerate(zip(self.index_files, self.index_weights)):
            if not index_file_path.endswith('.json'):
                raise ValueError(f'index_file {index_file_path} must be json file')

            with open(index_file_path) as f:
                d = json.load(f)
                assert 'arrow_list' in d, f'index_file {index_file_path} must contain arrow_list'
                arrow_list = d['arrow_list']  # arrow文件路径列表
                cache_folder_list = d.get('cache_folder_list', ["" for _ in arrow_list])

                if self.multitemp:
                    duration_ratio_index_list = d['duration_ratio_index_list']  # duration2index dict {duration: [index1, index2, ...]}
                    duration_ratio_nums_list = d['duration_ratio_nums_list']  # duration2nums dict {duration: nums}

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
                all_duration_ratio_index_lists.extend([duration_ratio_index_list[i] for i in choose_indices])
                all_duration_ratio_nums_lists.extend([duration_ratio_nums_list[i] for i in choose_indices])
            
        self.arrow_file = all_arrow_lists

        if logger is None: from loguru import logger
        self.logger = logger
        
        if multitemp:
            # 多时长桶
            # generate temporal bucket
            assert bucket_temporal_max_length > bucket_temporal_min_length, f"temporal_max_length must be larger than temporal_min_length, but got {bucket_temporal_max_length}, {bucket_temporal_min_length}"
            if bucket_temporal_max_length > max_audio_length: bucket_temporal_max_length = max_audio_length
            
            audio_length_list = [bucket_temporal_min_length]

            while audio_length_list[-1] < bucket_temporal_max_length:
                next_length = audio_length_list[-1] + bucket_temporal_interval
                if next_length > bucket_temporal_max_length:
                    next_length = bucket_temporal_max_length
                audio_length_list.append(next_length)
            self.audio_length_list = audio_length_list
            rank0_print(self.logger, f"List of temporal length buckets: {self.audio_length_list}")

        s_time = time.time()
        self.datasets = []
        self.table = []
        total_length = 0
        
        # 如果有多个index文件，对所有列表进行shuffle以混合数据源
        if len(self.index_files) > 1:
            rnd = random.Random(self.index_file_sample_seed)
            random_indexs = list(range(len(all_arrow_lists)))
            rnd.shuffle(random_indexs)
            all_arrow_lists = [all_arrow_lists[i] for i in random_indexs]
            all_cache_folder_lists = [all_cache_folder_lists[i] for i in random_indexs]
            all_duration_ratio_index_lists = [all_duration_ratio_index_lists[i] for i in random_indexs]
            all_duration_ratio_nums_lists = [all_duration_ratio_nums_lists[i] for i in random_indexs]

        # 准备有效的数据项（过滤掉被忽略的 cache_folder）
        valid_data_items = []
        for i, (arrow_file, cache_folder, duration_ratio_index_dict, duration_ratio_nums_dict) in enumerate(zip(all_arrow_lists, all_cache_folder_lists, all_duration_ratio_index_lists, all_duration_ratio_nums_lists)):
            valid_data_items.append((i, arrow_file, cache_folder, duration_ratio_index_dict, duration_ratio_nums_dict))
        
        def create_dataset(data_item):
            """创建单个dataset的函数，用于并发执行"""
            original_index, arrow_file, cache_folder, duration_ratio_index_dict, duration_ratio_nums_dict = data_item
            
            dataset = AudioArrowDataset(
                arrow_file=arrow_file,
                logger=logger,
                max_text_length=max_text_length,
                wav_sampling_rate=wav_sampling_rate,
                audio_vae_latent_max_length=audio_vae_latent_max_length,
                debug=debug,
                drop_text=drop_text,
                audio_vae_latent_dim=audio_vae_latent_dim,
                keywords_training_ratio=keywords_training_ratio,
                enable_quality_tag_for_pureauio=enable_quality_tag_for_pureauio,
                vae_sample=vae_sample,
                bandwidth_tag=bandwidth_tag,
                vae_type=vae_type,
                vae_rate=vae_rate,
                use_repa_with_audiossl=use_repa_with_audiossl,
                bucket_temporal_min_length=bucket_temporal_min_length,
                bucket_temporal_max_length=bucket_temporal_max_length,
                bucket_temporal_interval=bucket_temporal_interval,
                duration_ratio_index=duration_ratio_index_dict,
                duration_ratio_nums=duration_ratio_nums_dict,
                audiossl_dim=audiossl_dim,
                max_audiossl_length=max_audiossl_length,
                use_transcript_text_feat=use_transcript_text_feat,
                transcript_encoder_type=transcript_encoder_type,
                online_vae_extraction=online_vae_extraction,
                download_from_cos=download_from_cos,
                max_audio_length=max_audio_length,
                multitemp=multitemp,
                audio_caption_col=audio_caption_col,
                uncond_p=uncond_p,
                use_audio_vae_cache=use_audio_vae_cache,
                text_encoder=self.text_encoder,
            )
            return original_index, dataset
        
        # 根据debug模式选择创建方式
        # if True:
        #     # 单进程模式 - 用于调试
        #     # rank0_print(logger, f"Debug mode: Creating {len(valid_data_items)} datasets sequentially (single process)...")
            
        #     # 存储结果，保证顺序
        #     dataset_results = [None] * len(valid_data_items)
            
        #     # 记录单进程创建开始时间
        #     sequential_start_time = time.time()
            
        #     for idx, item in enumerate(valid_data_items):
        #         try:
        #             start_time = time.time()
        #             original_index, dataset = create_dataset(item)
        #             dataset_results[idx] = dataset
        #             elapsed_time = time.time() - start_time
        #             total_elapsed = time.time() - sequential_start_time
        #             # rank0_print(logger, f"Dataset {idx+1}/{len(valid_data_items)} created successfully (single: {elapsed_time:.3f}s, total: {total_elapsed:.3f}s)")
        #             break
        #         except Exception as exc:
        #             rank0_print(logger, f"Dataset creation failed for index {idx}: {exc}")
        #             # 在debug模式下，打印详细的错误信息和堆栈跟踪
        #             import traceback
        #             rank0_print(logger, f"Full traceback:\n{traceback.format_exc()}")
        #             raise exc
            
        #     concurrent_total_time = time.time() - sequential_start_time
        #     # rank0_print(logger, f"All {len(valid_data_items)} datasets created sequentially in {concurrent_total_time:.3f}s")
            
        # else:
        # 多线程并发模式 - 生产环境
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
        
        # 初始化全局的duration_ratios_index
        if multitemp:
            self.duration_ratio_index = {}
            self.duration_ratio_nums = {}
            for temp_bucket in self.audio_length_list:
                if float(temp_bucket) not in self.duration_ratio_index:
                    self.duration_ratio_index[float(temp_bucket)] = []
                    self.duration_ratio_nums[float(temp_bucket)] = 0
        
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
                
                if multitemp:
                    # 构建全局duration_ratios_index, bucket中存储全局逻辑索引
                    for temp, local_physical_indices in dataset.duration_ratio_index.items():
                        #  here are physical index, need to convert it to logic index
                        local_logic_indices = [dataset.physical2logic_index_map[local_physical_idx] for local_physical_idx in local_physical_indices] 
                        # 将本地索引转换为全局索引
                        global_indices = [cumulative_length - dataset_length + local_idx for local_idx in local_logic_indices]
                        self.duration_ratio_index[float(temp)].extend(global_indices)
                    
                    # 更新数量统计
                    for temp, nums in dataset.duration_ratio_nums.items():
                        self.duration_ratio_nums[float(temp)] += nums
                        
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
        if hasattr(self, 'duration_ratio_index'):
            bucket_indices_total = sum(len(indices) for indices in self.duration_ratio_index.values())
        
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
        rank0_print(logger, f"uncond_p for audio caption: {uncond_p}")
        rank0_print(logger, f'================================')
        self.total_length = total_length
        self.concat_dataset = ConcatDataset(self.datasets)
        self.dataset_type = 'audio' # 预留字段
        
        # 初始化错误处理时需要的属性
        self.closest_temp = None
        
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
                if self.multitemp and self.closest_temp is not None:
                    bucket_list = self.duration_ratio_index.get(self.closest_temp, [])
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


    # def shuffle(self, seed: int, fast: bool = False):
    #     """
    #     全局打乱数据集索引，保持多时长桶同步更新。
    #     注意：仅影响 index_to_dataset 映射，不改变底层 ConcatDataset 顺序。
    #     """
    #     import numpy as np
    #     rng = np.random.default_rng(seed)

    #     # 全局逻辑索引 [0, total_length)
    #     all_indices = np.arange(self.total_length)
    #     shuffled_indices = all_indices.copy()
    #     rng.shuffle(shuffled_indices)

    #     # 构建旧→新索引映射（O(N)，但用数组内存更高效）
    #     old_to_new = np.empty_like(shuffled_indices)
    #     old_to_new[shuffled_indices] = np.arange(self.total_length)

    #     # 更新全局映射表
    #     self.index_to_dataset = {
    #         new_idx: self.index_to_dataset[int(old_idx)]
    #         for new_idx, old_idx in enumerate(shuffled_indices)
    #     }

    #     # 更新多时长桶（如启用）
    #     if self.multitemp and hasattr(self, "duration_ratio_index"):
    #         new_duration_ratio_index = {}
    #         for temp, old_indices in self.duration_ratio_index.items():
    #             # 只保留合法范围索引
    #             old_indices = np.array(old_indices, dtype=np.int64)
    #             new_indices = old_to_new[old_indices]
    #             new_duration_ratio_index[float(temp)] = new_indices.tolist()

    #         self.duration_ratio_index = new_duration_ratio_index
    #         self.duration_ratio_nums = {
    #             t: len(v) for t, v in new_duration_ratio_index.items()
    #         }

    #     rank0_print(self.logger, f"[Shuffle] Dataset shuffled with seed={seed} (total {self.total_length} samples)")

    
    def get_batch(self, idx): 
        '''
        inputs:
            idx: logic index
        '''
        # 根据全局逻辑索引找到对应的子数据集和局部逻辑索引
        dataset_idx, local_idx = self.index_to_dataset[idx]

        try:
            item = self.datasets[dataset_idx].get_batch(local_idx)
            return item
        finally:
            # 更新closest_temp
            if hasattr(self.datasets[dataset_idx], 'closest_temp'):
                self.closest_temp = self.datasets[dataset_idx].closest_temp
        
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
            raise NotImplementedError("Physical index type not supported in concatenated dataset")


def variable_length_collate_fn(batch):
    """
    自定义 collate_fn，用于处理变长音频序列
    使用 pad_sequence 来填充变长的 waveform
    """
    from torch.nn.utils.rnn import pad_sequence
    
    # 过滤掉 None 值（如果有的话）
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # 获取所有键
    keys = batch[0].keys()
    collated_batch = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # 对于 waveform，使用 pad_sequence 进行填充
        if key == "audio_waveform" and isinstance(values[0], torch.Tensor):
            # 计算每个 waveform 的原始长度
            lengths = torch.tensor([len(wav) for wav in values], dtype=torch.long)
            # 使用 pad_sequence 进行填充（batch_first=True 表示输出为 [B, T]）
            padded_waveforms = pad_sequence(values, batch_first=True, padding_value=0.0)
            # [B, T] -> [B, 1, T]
            collated_batch[key] = padded_waveforms.unsqueeze(1)
            collated_batch["waveform_lengths"] = lengths  # 保存原始长度，可能后续有用
            
        elif isinstance(values[0], torch.Tensor):
            try:
                collated_batch[key] = torch.stack(values)
            except RuntimeError as e:
                # 如果堆叠失败（例如尺寸不匹配），尝试 pad_sequence
                if "stack expects each tensor to be equal size" in str(e):
                    if len(values[0].shape) == 2:
                        transposed = [v.T for v in values]
                        padded = pad_sequence(transposed, batch_first=True, padding_value=0.0)
                        collated_batch[key] = padded.transpose(1, 2)  # [B, D, T]
                    else:
                        raise e
                else:
                    raise e
        else:
            collated_batch[key] = values
    
    return collated_batch

