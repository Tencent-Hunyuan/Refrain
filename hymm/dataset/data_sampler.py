import os
from typing import Sequence
from torch.utils.data import BatchSampler, Sampler, Dataset
from random import shuffle, choice
from copy import deepcopy
import loguru
import numpy as np
import math 
from hymm.utils.distributed_utils import rank0_print

def get_closest_length(length, buckets: list):
    closest_length = buckets[0]
    for b in buckets: 
        if length-b>=0: closest_length = b
    return closest_length

def check_list(numbers):
    return all(n >= 0 for n in numbers)

class MutliVideoLengthBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 buckets: dict,
                 drop_last: bool = False,
                 valid_num=10,
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = buckets
        self.drop_last = drop_last
        self.bucket_nums_gt = kwargs.get('bucket_nums', None)
        assert self.bucket_nums_gt
        
        # buckets for each aspect ratio
        self._video_length_buckets = {bucket: [] for bucket in buckets}
        self.current_available_bucket_keys =  [k for k, v in self.bucket_nums_gt.items() if v > valid_num]
        loguru.logger.info(f"Using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}")

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width, num_frames =  data_info['height'], data_info['width'], data_info['num_frames']
            closest_length = get_closest_length(num_frames, self.buckets)
            
            if closest_length not in self.current_available_bucket_keys:
                continue
            bucket = self._video_length_buckets[closest_length]
            bucket.append(idx)
            # print(f'{idx}-{height}-{width}-{num_frames}-{closest_length}-{bucket}')
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                # print(f'**********{height}-{width}-{num_frames}-{closest_length}-{bucket}')
                # print(f'**********{num_frames}-{closest_length}-{bucket}')
                # print(bucket)
                yield bucket[:]
                del bucket[:]
        # print(self._aspect_ratio_buckets)
        # yield the rest data and reset the buckets
        for video_length, bucket in self._video_length_buckets.items():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]
            self._video_length_buckets[video_length] = []
                    
class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler, 
                 dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict,
                 drop_last: bool = False,
                 valid_num=10,
                 logger=None,
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        assert self.ratio_nums_gt
        
        if logger is None: from loguru import logger
        self.logger = logger
        
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        self.current_available_bucket_keys =  [k for k, v in self.ratio_nums_gt.items() if v > valid_num]
        logger.info(f"Using valid_num={valid_num} in config file. \
            Available {len(self.current_available_bucket_keys)} \
            aspect_ratios: {self.current_available_bucket_keys}")

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width =  data_info['height'], data_info['width']
            ratio = float(height) / float(width)
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios, key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            # print(f'{idx}-{height}-{width}-{closest_ratio}-{bucket}')
            
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                # print(bucket)
                yield bucket[:]
                del bucket[:]
        # print(self._aspect_ratio_buckets)
        # yield the rest data and reset the buckets
        for closest_ratio, bucket in self._aspect_ratio_buckets.items():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]
            # "bucket = []" can't clear the value of self._aspect_ratio_buckets
            self._aspect_ratio_buckets[closest_ratio] = []

class TemporalAspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler, 
                 dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict,
                 video_lengths: dict,
                 drop_last: bool = False,
                 valid_num=10,
                 logger=None,
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if (not isinstance(batch_size, int) and not isinstance(batch_size, list)):
            raise ValueError('batch_size should be a positive integer value or list, '
                             f'but got batch_size={batch_size}')
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        elif isinstance(batch_size, tuple) and not check_list(batch_size):
            raise ValueError('batch_size in the tuple should be positive integer value, '
                             f"but got batch_size={batch_size}")
        elif isinstance(batch_size, list):
            if len(batch_size)==1: batch_size = batch_size * len(video_lengths)
            elif len(batch_size)!=len(video_lengths):
                raise ValueError('batch_size in the list should be same as video_lengths, '
                             f"but got batch_size={batch_size}, video_lengths={video_lengths}")
            
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.video_lengths = video_lengths
        self.drop_last = drop_last
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        assert self.ratio_nums_gt
        self.dynamic_batch_size = False
        if isinstance(batch_size, list):
            self.dynamic_batch_size = True
            self.video_lengths2batch_size = dict(zip(video_lengths, batch_size))
        
        if logger is None: from loguru import logger
        self.logger = logger
        
        # buckets for each aspect ratio
        self._temporal_aspect_ratio_buckets = {video_length: {ratio: [] for ratio in aspect_ratios} for video_length in video_lengths}
        rank0_print(self.logger, f"Temporal aspect ratio buckets: {self._temporal_aspect_ratio_buckets}")
        self.current_available_bucket_keys = {temp: [k for k, v in aspect_ratios2num.items() if v>valid_num ] for temp, aspect_ratios2num in self.ratio_nums_gt.items()}
        
        for k, v in self.current_available_bucket_keys.items():
            rank0_print(self.logger, f"For video length: {k}, using valid_num={valid_num} in config file. \
                Available {len(v)} \
                aspect_ratios: {v}")

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            num_frames, height, width = data_info['num_frames'], data_info['height'], data_info['width']
            # find the closest video length
            closest_temp = min(self.video_lengths, key=lambda t: abs(int(t)-num_frames))
            if closest_temp not in self.current_available_bucket_keys:
                continue
                
            # find the closest aspect ratio
            ratio = float(height) / float(width)
            closest_ratio = min(self.aspect_ratios, key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys[closest_temp]:
                continue
            bucket = self._temporal_aspect_ratio_buckets[closest_temp][closest_ratio]
            # print(f'{idx}-{height}-{width}-{closest_ratio}-{bucket}')
            
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if self.dynamic_batch_size:
                if len(bucket) == self.video_lengths2batch_size[closest_temp]:
                    yield bucket[:]
                    del bucket[:]
            else:
                if len(bucket) == self.batch_size:
                    # print(bucket)
                    yield bucket[:]
                    del bucket[:]
        # yield the rest data and reset the buckets
        for temp, aspect_ratio_buckets in self._temporal_aspect_ratio_buckets.items():
            for closest_ratio, bucket in aspect_ratio_buckets.items():
                while len(bucket) > 0:
                    if self.dynamic_batch_size:
                        batch_size = self.video_lengths2batch_size[temp]
                    else:
                        batch_size = self.batch_size
                    if len(bucket) <= batch_size:
                        if not self.drop_last:
                            yield bucket[:]
                        bucket = []
                    else:
                        yield bucket[:batch_size]
                        bucket = bucket[batch_size:]
                # "bucket = []" can't clear the value of aspect_ratio_buckets 
                self._temporal_aspect_ratio_buckets[temp][closest_ratio] = []

class MultiAudioLengthBatchSampler(BatchSampler):
    """A sampler wrapper for grouping audio with similar length into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: list,  # 如果传入list，就会开启dynamic batch size
                 buckets: list,
                 drop_last: bool = False,
                 valid_num=10,
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if (not isinstance(batch_size, int) and not isinstance(batch_size, list)):
            raise ValueError('batch_size should be a positive integer value or list, '
                             f'but got batch_size={batch_size}')
        if (not isinstance(batch_size, int) and not isinstance(batch_size, list)):
            raise ValueError('batch_size should be a positive integer value or list, '
                             f'but got batch_size={batch_size}')
        elif isinstance(batch_size, tuple) and not check_list(batch_size):
            raise ValueError('batch_size in the tuple should be positive integer value, '
                             f"but got batch_size={batch_size}")
        elif isinstance(batch_size, list):
            if len(batch_size)==1: batch_size = batch_size * len(buckets)
            elif len(batch_size)!=len(buckets):
                raise ValueError('batch_size in the list should be same as buckets, '
                             f"but got batch_size={batch_size}, buckets={buckets}")
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = buckets
        self.drop_last = drop_last
        self.bucket_nums_gt = kwargs.get('bucket_nums', None)
        assert self.bucket_nums_gt
        self.dynamic_batch_size = False
        if isinstance(batch_size, list):
            self.dynamic_batch_size = True
            self.audio_lengths2batch_size = dict(zip(buckets, batch_size))
        
        # buckets for each aspect ratio
        self._audio_length_buckets = {bucket: [] for bucket in buckets}
        self.current_available_bucket_keys =  [k for k, v in self.bucket_nums_gt.items() if v > valid_num]
        loguru.logger.info(f"Using valid_num={valid_num} in config file. Available {len(self.current_available_bucket_keys)} aspect_ratios: {self.current_available_bucket_keys}")


    def __iter__(self):
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            duration =  data_info['duration']
            closest_length = get_closest_length(duration, self.buckets)
            
            if closest_length not in self.current_available_bucket_keys:
                continue
            bucket = self._audio_length_buckets[closest_length]
            bucket.append(idx)

            # yield a batch of indices in the same duration group
            if self.dynamic_batch_size:
                if len(bucket) == self.audio_lengths2batch_size[closest_length]:
                    yield bucket[:]
                    del bucket[:]
            else:
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]

        # yield the rest data and reset the buckets
        for audio_length, bucket in self._audio_length_buckets.items():
            while len(bucket) > 0:
                if self.dynamic_batch_size:
                    batch_size = self.audio_lengths2batch_size[audio_length]
                else:
                    batch_size = self.batch_size

                if len(bucket) <= batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:batch_size]
                    bucket = bucket[batch_size:]
            self._audio_length_buckets[audio_length] = []