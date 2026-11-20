import numpy as np
import time
import torch.distributed as dist
from hymm.dataset.video_loader import VideoArrowDataset, VideoArrowConcatDataset
from hymm.dataset.audio_loader import AudioDiffusionFixedLenIndexDataset, AudioArrowConcatDataset
from hymm.dataset.videoaudio_loader import VideoAudioArrowConcatDataset
from hymm.dataset.samplers import BlockDistributedSampler, DistributedSamplerWithStartIndex

class VideoImageBatchIterator:
    def __init__(self,
                 dataloaders=None,           # dataloaders is a dict with keys: 'video', 'image', 'll_cap_image'
                 samplers=None,
                 scalar_states=None,
                 video_sampling_prob = 0.5,
                 ll_cap_sample_ratio = 0.0,
                 video_ll_cap_sample_ratio = 0.0,
                 video_multireso = False,    # Unused now
                 rank_assign_mode='original',
                 force_sync_shuffle=False,
                 dp_rank=None,
                 logger=None,
                 args=None):
        assert dataloaders is not None and ('audio' in dataloaders or 'video' in dataloaders or 'image' in dataloaders or 'll_cap_image' in dataloaders  or 't2va' in dataloaders)
        self.args = args
        self.dataloaders = dataloaders
        self.all_keys_with_none = list(self.dataloaders.keys())
        self.dataloaders = {key: dataloader for key, dataloader in self.dataloaders.items() if dataloader is not None}
        self.all_keys = list(self.dataloaders.keys())
        self.iterators = {key: self.get_iterator(key) for key in self.all_keys}
        self.samplers = samplers
        self.keys2id = {key: i for i, key in enumerate(self.all_keys_with_none)} # 用于控制不同dataloader不同seed
        
        
        self.scalar_states = scalar_states
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.force_sync_shuffle = force_sync_shuffle
        self.fast_shuffle = self.args.fast_shuffle

        for key in self.all_keys_with_none:
            if key not in self.scalar_states.epoch:
                self.scalar_states.epoch[key] = 0
            if key not in self.scalar_states.consumed_samples_per_dp:
                self.scalar_states.consumed_samples_per_dp[key] = 0
        
        self.weights = []
        for key in self.all_keys:
            if key == "video":
                cur_weight = video_sampling_prob if dataloaders[key] is not None else 0
                self.weights.append(cur_weight)
            elif key == "ll_cap_image":
                cur_weight = ll_cap_sample_ratio if dataloaders[key] is not None else 0
                self.weights.append(cur_weight)
            elif key == "image":
                cur_weight = 1.0-(video_sampling_prob+ll_cap_sample_ratio+video_ll_cap_sample_ratio) if dataloaders[key] is not None else 0
                self.weights.append(cur_weight)
            elif key == "ll_cap_video":
                cur_weight = video_ll_cap_sample_ratio if dataloaders[key] is not None else 0
                self.weights.append(cur_weight)
            elif key == 'audio':
                cur_weight = 1.0
                self.weights.append(cur_weight)
            elif key == 't2va':
                cur_weight = 1.0
                self.weights.append(cur_weight)
            else:
                raise ValueError(f"key {key} not in dataloaders keys: {self.dataloaders.keys()}")
        print(f"[Rank{dp_rank}] all_keys={self.all_keys} weights: {self.weights}")

        if logger is None: from loguru import logger
        self.logger = logger
        self.dp_rank = dp_rank

        self.dp_rank_assign_mode = rank_assign_mode
        self.video_multireso = video_multireso
        self.first_flag = True
        self.first_cache = {}

        self.shuffle_and_initialize_index()
        

    def get_batch(self, data_type):
        # emit_global_shuffle is used to indicate whether to shuffle the dataset
        try:
            batch = next(self.iterators[data_type])
            if isinstance(batch, list):
                bsz = batch[0].shape[0]
            elif isinstance(batch, dict):
                if 'audio_token' in batch:
                    bsz = batch["audio_token"].shape[0]
                elif 'padded_caption_ids' in batch:
                    bsz = batch['audio_waveform'].shape[0]
                elif 'waveform' in batch:
                    bsz = batch['waveform'].shape[0]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.scalar_states.consumed_samples_per_dp[data_type] += bsz
            emit_global_shuffle = False
        except StopIteration:
            batch = None
            emit_global_shuffle = True
        return batch, emit_global_shuffle
        
    def get_iterator(self, data_type=None):
        if data_type is None or data_type not in self.dataloaders or self.dataloaders[data_type] is None:
            print(f"[Rank{self.dp_rank}] {data_type} dataloader is None")
            return None
        for idx, batch in enumerate(self.dataloaders[data_type]):
            yield batch

    def __iter__(self):
        return self

    def __next__(self):            
        emit_shuffle = {key: False for key in self.all_keys_with_none}
        if self.dp_rank_assign_mode == 'fixed':
            assert len(self.all_keys) == 1
            cur_key = self.all_keys[0]
        else:
            if len(self.weights) == 1:
                cur_key = self.all_keys[0]
            else:
                cur_key = np.random.choice(self.all_keys, size=1, p=self.weights)[0]
        if self.first_flag:
            for key in self.all_keys:
                self.first_cache[key] = self.get_batch(key)
                self.first_flag = False

        if cur_key in self.first_cache:
            batch, cur_emit_shuffle = self.first_cache.pop(cur_key)
        else:
            batch, cur_emit_shuffle = self.get_batch(cur_key)
            
        if not self.force_sync_shuffle:
            if cur_emit_shuffle:
                self.shuffle(cur_key)
                batch, _ = self.get_batch(cur_key)
            return batch
        emit_shuffle[cur_key] = cur_emit_shuffle

        if self.is_distributed:
            shuffle_flags = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(shuffle_flags, emit_shuffle)
        else:
            shuffle_flags = [emit_shuffle]

        # Emit shuffle if any worker emits shuffle.
        for key in self.all_keys:
            #if any([flag[key] for flag in shuffle_flags if key in flag]):
            if any([v for flag in shuffle_flags for k, v in flag.items()]):
                self.shuffle(key)
                if key == cur_key:
                    batch, _ = self.get_batch(key)
        return batch
                
    def shuffle(self, key):
        cur_epoch = self.scalar_states.epoch[key] + 1
        seed = self.args.global_seed + self.scalar_states.epoch[key] + self.keys2id[key] * 10000
        print(f"[Rank{self.dp_rank}] {key} dataset epoch: {cur_epoch}. "
              f"Shuffle with seed={seed}, fast={self.fast_shuffle}")
        try:
            self.shuffle_and_reset_index(self.dataloaders[key].dataset, 
                                         self.samplers[key], 
                                         key=key,
                                         seed=seed, 
                                         start_index=0,
                                         use_cache=False)
        except Exception as e:
            self.logger.error(f"[Rank{self.dp_rank}] Error in shuffling {key}: {e}")
            raise e
        self.iterators[key] = self.get_iterator(key)

        # After shuffle, reset the consumed samples and epoch for the current key.
        self.scalar_states.consumed_samples_per_dp[key] = 0
        self.scalar_states.epoch_consumed_samples[key] = 0
        self.scalar_states.epoch[key] += 1

    def shuffle_and_reset_index(self, dataset, sampler, key, seed, start_index, use_cache=False):
        if isinstance(dataset, (VideoArrowDataset, VideoArrowConcatDataset, AudioDiffusionFixedLenIndexDataset, AudioArrowConcatDataset, VideoAudioArrowConcatDataset)):
            sampler.set_epoch(int(self.scalar_states.epoch[key]))
        else:
            dataset.shuffle(seed, fast=self.fast_shuffle, use_cache=use_cache)

        # `start_index` for distributed samplers is per-rank progress inside the current
        # epoch. Resume states may contain stale/global counters from a different run.
        sampler.start_index = 0
        sampler_len = len(sampler)
        if start_index < 0 or start_index >= sampler_len:
            self.logger.warning(
                f"[Rank{self.dp_rank}] Reset invalid sampler start_index for {key}: "
                f"requested={start_index}, sampler_len={sampler_len}. Falling back to 0."
            )
            start_index = 0
            self.scalar_states.consumed_samples_per_dp[key] = 0
            self.scalar_states.epoch_consumed_samples[key] = 0
        sampler.start_index = start_index
        remaining = max(0, sampler_len - start_index)
        print(
            f"Update {sampler.__class__.__name__} states: "
            f"start_index={start_index}, remaining={remaining}, sampler_len={sampler_len}"
        )

    def shuffle_and_initialize_index(self):
        print(f"[Rank{self.dp_rank}] Start random shuffle")
        start = time.time()
        shuffle_times = {}
        for key in self.all_keys:
            if self.dataloaders[key] is None:
                continue
            # 不同dataloader的shuffle seed不同
            seed = self.args.global_seed + self.scalar_states.epoch[key] + self.keys2id[key] * 10000
            if isinstance(self.samplers[key], BlockDistributedSampler):
                consumed_samples = int(self.scalar_states.consumed_samples_per_dp[key])
            elif isinstance(self.samplers[key], DistributedSamplerWithStartIndex):
                consumed_samples = int(self.scalar_states.consumed_samples_per_dp[key])
            else:
                consumed_samples = 0
                self.logger.warning(f"{self.samplers[key].__class__.__name__} doesn't support set start_index."
                                    f"The sampler will be reset to the beginning of the dataset.")

            cur_epoch = self.scalar_states.epoch[key]
            print(
                f"[Rank{self.dp_rank}] {key} dataset epoch: {cur_epoch}. "
                f"Shuffle with seed={seed}, fast={self.fast_shuffle}"
                f"Initialize with consumed_samples={consumed_samples:,}"
            )
            try:
                self.shuffle_and_reset_index(self.dataloaders[key].dataset, 
                                             self.samplers[key], 
                                             key=key,
                                             seed=seed, 
                                             start_index=consumed_samples)
            except Exception as e:
                self.logger.error(f"[Rank{self.dp_rank}] Error in shuffling {key}: {e}")
                raise e
            shuffle_times[key] = time.time() - start
            start = time.time()
        for key, time_cost in shuffle_times.items():
            print(f"[Rank{self.dp_rank}] Shuffle {key} time cost: {time_cost: .2f} sec")
        print(f"[Rank{self.dp_rank}] End of random shuffle")
