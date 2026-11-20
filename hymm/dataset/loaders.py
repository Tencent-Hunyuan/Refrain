from transformers import AutoTokenizer
from hymm.dataset.samplers import BlockDistributedSampler, DistributedSamplerWithStartIndex, IndexBatchSampler, DistributedSampler
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda
from .video_loader import VideoArrowConcatDataset
from. audio_loader import AudioDiffusionFixedLenIndexDataset, collate_fn_filter_none, AudioArrowConcatDataset, variable_length_collate_fn
from .videoaudio_loader import VideoAudioArrowConcatDataset
from .image_loader import TextImageArrowStream
from .data_sampler import AspectRatioBatchSampler, TemporalAspectRatioBatchSampler, MultiAudioLengthBatchSampler
from .transform import (
    Normalize255,
    TemporalRandomCrop,
    CenterCropResizeVideo,
)
from hymm.utils.helpers import (
    as_tuple,
    dynamic_values_wrapper,
)
from hymm.utils.torch_utils import set_worker_seed_builder, set_manual_seed

def get_image_dataloader(args, text_encoder, text_encoder_2, dp_degree, dp_rank, logger, byt5_kwarg, is_ll_cap=False, dist_sampler_state=None):
    image_dataset = TextImageArrowStream(
            args,
            image_size=args.image_size,                      # 分辨率
            uncond_p=args.uncond_p,                          # 删除图像描述概率
            text_len=args.text_len if not is_ll_cap else args.text_len_ll_cap,                          # 文本长度,如果使用llCap，则使用1000长度的文本
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            index_file=args.index_file,                      # dataset 的索引文件
            multireso=args.multireso,                        # 多分辨率
            index_kwargs=dict(
                batch_size=1 if args.mix_scale else args.image_micro_batch_size[0] // 4 if is_ll_cap else args.image_micro_batch_size[0], # llCap使用1/4的batch_size
                world_size=1 if args.mix_scale else dp_degree,
                # world_size=1,
                image_text_arrow_suffix=args.image_text_arrow_suffix,
                image_text_col=args.image_text_col,
                image_text_col_2=args.image_text_col_2,
                image_caption_arrow_suffix=args.image_caption_arrow_suffix,
                image_caption_rate=args.image_caption_rate,
                image_caption_col=args.image_caption_col,
                image_caption_col_2=args.image_caption_col_2,
                index_probability=args.index_probability,
                index_strategy=args.index_strategy,
                ceph_base=args.ceph_base,
                ceph_base_inv=args.ceph_base_inv,
                caption_sample_ratio=args.caption_sample_ratio if (not is_ll_cap or args.ll_cap_caption_sample_ratio=="") else args.ll_cap_caption_sample_ratio,
                caption_processor=args.image_caption_processor,
                image_caption_col_probs=args.image_caption_col_probs,
                text_remove_rate=args.text_remove_rate,
                ocr_only_long_caption=args.ocr_only_long_caption,
                num_replace_rate=args.num_replace_rate
            ),
            byt5_tokenizer=byt5_kwarg["byt5_tokenizer"] if byt5_kwarg is not None else None,
            byt5_max_length=byt5_kwarg["byt5_max_length"] if byt5_kwarg is not None else None,
            byt5_uncond_p=args.uncond_p_byt5,
            debug=False,
            dataset_type = "image" if not is_ll_cap else "ll_cap_image",  # dataloader类型
            logger=logger,                                    # logger
        )
    # Dynamic anchor size for mix-scale training.
    anchor_sizes = as_tuple(args.anchor_size)
    dynamic_anchor_size = dynamic_values_wrapper(anchor_sizes, anchor_sizes)
    # Build sampler and data loader
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                            prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
                            worker_init_fn=set_worker_seed_builder(dp_rank), persistent_workers=False if args.num_workers == 0 else True)
    if args.mix_scale:
        image_sampler = BlockDistributedSampler(image_dataset,
                                                    num_replicas=dist_sampler_state.dataset_num_replicas[dist_sampler_state.cur_key],
                                                    rank=dist_sampler_state.dataset_rank[dist_sampler_state.cur_key],
                                                    shuffle=True,
                                                    seed=args.global_seed, drop_last=True)
        mix_batch_sizes = [bs // 4 for bs in args.mix_micro_batch_size] if is_ll_cap else args.mix_micro_batch_size
        dynamic_batch_size = dynamic_values_wrapper(anchor_sizes, mix_batch_sizes)
        batch_sampler = IndexBatchSampler(image_dataset.index_manager, image_sampler, batch_size=dynamic_batch_size, drop_last=True)
        image_loader = DataLoader(image_dataset, batch_sampler=batch_sampler, **loader_kwargs)
    else:
        batch_size = args.image_micro_batch_size[0]//4 if is_ll_cap else args.image_micro_batch_size[0]
        image_sampler = DistributedSamplerWithStartIndex(image_dataset,
                                                         num_replicas=dist_sampler_state.dataset_num_replicas[dist_sampler_state.cur_key],
                                                         rank=dist_sampler_state.dataset_rank[dist_sampler_state.cur_key],
                                                         shuffle=False,
                                                         seed=args.global_seed,
                                                         drop_last=True,
                                                         use_numpy_indices=True,
                                                         batch_size=batch_size if args.multireso else 1)
        if args.multireso:
            image_loader = DataLoader(image_dataset, batch_size=batch_size, sampler=image_sampler, shuffle=False, drop_last=True, **loader_kwargs)
        else:
            batch_sampler = IndexBatchSampler(image_dataset.index_manager, image_sampler, batch_size=batch_size, drop_last=True, multireso=True)
            image_loader = DataLoader(image_dataset, batch_sampler=batch_sampler, **loader_kwargs)

    return image_dataset, image_sampler, image_loader

def get_video_dataloader(args, text_encoder, text_encoder_2, dp_degree, dp_rank, logger, byt5_kwarg, is_ll_cap=False, local_seed=None, dist_sampler_state=None):
    # 根据是否使用long long caption选择文本长度
    text_len = args.video_text_len_ll_cap if is_ll_cap else args.text_len

    # 根据是否使用long long caption选择caption采样比例
    caption_sample_ratio = args.video_ll_cap_caption_sample_ratio if is_ll_cap else args.video_caption_sample_ratio

    video_dataset = VideoArrowConcatDataset(
            args,
            index_file=args.video_index_file,
            index_file_weights=args.video_index_file_weights,
            video_size=args.video_size,
            sample_n_frames=args.sample_n_frames,
            sample_stride=args.sample_stride,
            is_image=args.is_image,
            multireso=args.video_multireso,
            multitemp=args.video_multitemp,
            bucket_hw_base_size = args.video_bucket_hw_base_size,
            bucket_hw_bucket_stride = args.video_bucket_hw_bucket_stride,
            bucket_temporal_min_length = args.video_bucket_temporal_min_length,
            bucket_temporal_max_length = args.video_bucket_temporal_max_length,
            bucket_temporal_interval = args.video_bucket_temporal_interval,
            text_len=text_len,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            uncond_p=args.video_uncond_p,
            use_caption_prob=args.video_caption_rate,
            caption_sample_ratio=caption_sample_ratio,
            ocr_only_long_caption=args.ocr_only_long_caption,
            text_remove_rate=args.text_remove_rate,
            use_latent_shape=args.use_latent_shape,
            vae_spatial_ratio=args.vae_spatial_ratio,
            byt5_tokenizer=byt5_kwarg["byt5_tokenizer"] if byt5_kwarg is not None else None,
            byt5_max_length=byt5_kwarg["byt5_max_length"] if byt5_kwarg is not None else None,
            byt5_uncond_p=args.uncond_p_byt5,
            debug=False,
            num_replace_rate=args.video_num_replace_rate,
            video_caption_col_probs=args.video_caption_col_probs,
            video_caption_col=args.video_caption_col,
            video_caption_col_2=args.video_caption_col_2,
            recreation_mask_col=args.recreation_mask_col,
            skip_sources=getattr(args, 'skip_sources', None),
            logger=logger,

            video_caption_i2v_instruction_enable=args.video_caption_i2v_instruction_enable,
            video_caption_i2v_instruction_col_probs=args.video_caption_i2v_instruction_col_probs,
            video_caption_i2v_instruction_col=args.video_caption_i2v_instruction_col,
            video_caption_i2v_instruction_col_2=args.video_caption_i2v_instruction_col_2,
            video_caption_i2v_instruction_sub_col_probs=args.video_caption_i2v_instruction_sub_col_probs,
            video_caption_i2v_instruction_sub_col=args.video_caption_i2v_instruction_sub_col,
            video_caption_i2v_instruction_sub_col_2=args.video_caption_i2v_instruction_sub_col_2,
            video_caption_i2v_instruction_sub_col_3=args.video_caption_i2v_instruction_sub_col_3,
            local_seed=local_seed,
        )

    if video_dataset is not None:
        video_sampler = DistributedSamplerWithStartIndex(video_dataset,
                                                         num_replicas=dist_sampler_state.dataset_num_replicas[dist_sampler_state.cur_key],
                                                         rank=dist_sampler_state.dataset_rank[dist_sampler_state.cur_key],
                                                         shuffle=True,
                                                         seed=args.global_seed,
                                                         drop_last=True)
        if is_ll_cap:
            batch_size = [max(1, int(b) // 4) if str(b).isdigit() else 1 for b in args.video_micro_batch_size]
        else:
            batch_size = args.video_micro_batch_size

        video_batch_sampler = None
        if args.video_multitemp and args.video_multireso:
            # 多时长+多分辨率
            video_batch_sampler = TemporalAspectRatioBatchSampler(sampler=video_sampler,
                                                    dataset=video_dataset,
                                                    batch_size=batch_size,
                                                    aspect_ratios=video_dataset.aspect_ratios_buckets,
                                                    video_lengths=video_dataset.video_length_buckets,
                                                    drop_last=True,
                                                    ratio_nums=video_dataset.temporal_aspect_ratios_nums)
            video_loader = DataLoader(video_dataset,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory=True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        elif args.video_multireso:
            # 多分辨率
            video_batch_sampler = AspectRatioBatchSampler(sampler = video_sampler,
                                                          dataset = video_dataset,
                                                          batch_size = batch_size[0],
                                                          aspect_ratios = video_dataset.aspect_ratios_buckets,
                                                          drop_last = True,
                                                          ratio_nums = video_dataset.aspect_ratios_nums)

            video_loader = DataLoader(video_dataset,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory=True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        else:
            # 单分辨率
            video_loader = DataLoader(video_dataset,
                                      batch_size = batch_size[0],
                                      shuffle = False,
                                      sampler = video_sampler,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory = True,
                                      drop_last = True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        return video_dataset, video_sampler, video_loader

def get_audio_dataloader(args, cfg, text_encoder, dp_degree, dp_rank, use_var_len_dataloader=False, logger=None):
    """
    专门为text2audio训练构建的数据加载器，使用纯音频数据集
    """
    if cfg.get('use_var_dataloader', False):
        audio_dataset = AudioArrowConcatDataset(
                index_file=cfg.get('index_file'), # arrow info存放路径json或json列表
                index_file_weights=cfg.get('index_file_weight'),  # index_file对应的权重列表
                index_file_sample_seed=42,  # 对index file采样时的seed
                index_kwargs=None,
                logger=logger,
                multitemp=cfg.get('multitemp'),
                max_text_length=cfg.get('max_text_length'),
                wav_sampling_rate=cfg.get('wav_sampling_rate'),
                audio_vae_latent_max_length=cfg.get('audio_vae_latent_max_length'),  # 8s
                audio_vae_latent_dim=cfg.get('audio_vae_latent_dim'),
                vae_sample=cfg.get('vae_sample'),
                vae_type=cfg.get("vae_type"),
                vae_rate=cfg.get('vae_rate'),
                bucket_temporal_min_length=cfg.get('bucket_temporal_min_length'),
                bucket_temporal_max_length=cfg.get('bucket_temporal_max_length'),
                bucket_temporal_interval=cfg.get('bucket_temporal_interval'),
                online_vae_extraction=cfg.get("online_vae_extraction"),
                download_from_cos=cfg.get("download_from_cos"),
                max_audio_length=cfg.get("max_audio_length"),
                audio_caption_col=cfg.get('audio_caption_col'),
                uncond_p=cfg.get('uncond_p'),
                text_encoder=text_encoder)

        sampler = DistributedSamplerWithStartIndex(
            audio_dataset, shuffle=True, num_replicas=dp_degree, rank=dp_rank, seed=args.global_seed, drop_last=True
        )

        # Simple version for dynamic batch. Hardcode for 8s * 40Hz VAE and 512 textlength.
        if cfg.get('multitemp') and len(audio_dataset.audio_length_list) > 1:
            batch_scale = 0.9 # dynamic batch size potential cuda memory peak.
            micro_batch_size = [int((8.0 * 40 + 512)**2 * 16 / ((item + cfg.get('bucket_temporal_interval', 1.0)) * 40 + 512)**2 * batch_scale) for item in audio_dataset.audio_length_list]
        else:
            micro_batch_size = args.micro_batch_size[0] if isinstance(args.micro_batch_size, list) else args.micro_batch_size

        audio_batch_sampler = MultiAudioLengthBatchSampler(sampler=sampler,  # 需要定义sampler或使用None
                                                           dataset=audio_dataset,
                                                           batch_size=micro_batch_size,
                                                           buckets=audio_dataset.audio_length_list,
                                                           drop_last=True,
                                                           bucket_nums=audio_dataset.duration_ratio_nums)
        dataloader = DataLoader(audio_dataset,
                                batch_sampler=audio_batch_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=variable_length_collate_fn,
                                prefetch_factor=2,
                                worker_init_fn=None,
                                persistent_workers=True)

        return audio_dataset, sampler, dataloader

    dataset = AudioDiffusionFixedLenIndexDataset(
        index_file=list(cfg.index_file),
        drop_text=cfg.get("drop_text", 0.1),
        index_kwargs=dict(
            batch_size=args.micro_batch_size,
            world_size=dp_degree,
            **cfg.get("index_kwargs", {}),
        ),
        shadow_file_cfg=cfg.get("shadow_file", {}),
        feat_in_shadow=cfg.get("feat_in_shadow", False),
        audio_vae_latent_max_length=cfg.get("audio_vae_latent_max_length", 400),
        max_text_length=cfg.max_text_length,
        audio_vae_latent_dim=cfg.get("audio_dim", None),
        keywords_training_ratio=cfg.get("keywords_training_ratio", 0),
        enable_quality_tag_for_pureauio=cfg.get("enable_quality_tag_for_pureauio", True),
        debug=False,
        vae_sample=cfg.get("vae_sample", False),
        bandwidth_tag=cfg.get("bandwidth_tag", False),
        vae_type=cfg.get("vae_type", "dac_vae"),
        use_repa_with_audiossl=cfg.get("use_repa_with_audiossl", False),
        use_repa_with_eat=cfg.get("use_repa_with_eat", False),
        audiossl_dim=cfg.get("audiossl_dim", 768),
        eat_dim=cfg.get("eat_dim", 1024),
        max_audiossl_length=cfg.get("max_audiossl_length", 200),
        max_eat_length=cfg.get("max_eat_length", 409),
        use_transcript_text_feat=cfg.get("use_transcript_text_feat", False),
        extract_asr_from_caption=cfg.get("extract_asr_from_caption", False),
        online_vae_extraction=cfg.get("online_vae_extraction", False),
        download_from_cos=cfg.get("download_from_cos", False),
        max_audio_waveform_length=cfg.get("max_audio_waveform_length", 384000),
        wav_sampling_rate=cfg.get("wav_sampling_rate", 48000),
        text_encoder=text_encoder
    )

    # TODO: shuffle true or false?
    sampler = DistributedSamplerWithStartIndex(
        dataset, shuffle=True, num_replicas=dp_degree, rank=dp_rank, seed=args.global_seed, drop_last=True
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch_size[0] if isinstance(args.micro_batch_size, list) else args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        worker_init_fn=set_worker_seed_builder(dp_rank),
        collate_fn=collate_fn_filter_none,  # 使用自定义 collate 函数过滤 None 值
    )

    return dataset, sampler, dataloader

def get_t2va_dataloader(ori_args, args, dp_degree, dp_rank, logger, is_ll_cap=False):
    # 根据是否使用long long caption选择文本长度
    text_len = args.video_text_len_ll_cap if is_ll_cap else args.text_len

    # 根据是否使用long long caption选择caption采样比例
    caption_sample_ratio = args.video_ll_cap_caption_sample_ratio if is_ll_cap else args.video_caption_sample_ratio

    # before build datasets, set_manual_seed only used for dataloader set_worker_seed_builder
    set_manual_seed(ori_args.seed + dp_rank)

    video_dataset = VideoAudioArrowConcatDataset(
            args,
            index_file=args.video_index_file,
            video_size=args.video_size,
            sample_n_frames=args.sample_n_frames,
            sample_stride=args.sample_stride,
            is_image=args.is_image,
            multireso=args.video_multireso,
            multitemp=args.video_multitemp,
            bucket_hw_base_size = args.video_bucket_hw_base_size,
            bucket_hw_bucket_stride = args.video_bucket_hw_bucket_stride,
            bucket_temporal_min_length = args.video_bucket_temporal_min_length,
            bucket_temporal_max_length = args.video_bucket_temporal_max_length,
            bucket_temporal_interval = args.video_bucket_temporal_interval,
            text_len=text_len,
            video_uncond_p=args.video_uncond_p,
            audio_uncond_p=args.audio_uncond_p,
            use_caption_prob=args.video_caption_rate,
            caption_sample_ratio=caption_sample_ratio,
            ocr_only_long_caption=args.ocr_only_long_caption,
            text_remove_rate=args.text_remove_rate,
            use_latent_shape=args.use_latent_shape,
            vae_spatial_ratio=args.vae_spatial_ratio,
            debug=False,
            num_replace_rate=args.video_num_replace_rate,
            video_caption_col_probs=args.video_caption_col_probs,
            video_caption_col=args.video_caption_col,
            video_caption_col_2=args.video_caption_col_2,
            skip_sources=getattr(args, 'skip_sources', None),
            logger=logger,
            audio_latent_dim=args.audio_dim,
            audio_sr=args.audio_sr,
            use_audio_vae_cache=args.use_audio_vae_cache,
            audio_wave_suffix=args.audio_wave_suffix,
            audio_frame_rate=args.audio_frame_rate,
            audio_caption_col=args.audio_caption_col,
            use_merged_caption=getattr(args, 'use_merged_caption', True),
        )

    if video_dataset is not None:
        video_sampler = DistributedSamplerWithStartIndex(video_dataset, num_replicas=dp_degree, rank=dp_rank, shuffle=True,
                                                        seed=ori_args.global_seed, drop_last=True)
        if is_ll_cap:
            batch_size = [max(1, int(b) // 4) if str(b).isdigit() else 1 for b in args.video_micro_batch_size]
        else:
            batch_size = args.video_micro_batch_size

        video_batch_sampler = None
        if args.video_multitemp and args.video_multireso:
            # 多时长+多分辨率
            video_batch_sampler = TemporalAspectRatioBatchSampler(sampler=video_sampler,
                                                    dataset=video_dataset,
                                                    batch_size=batch_size,
                                                    aspect_ratios=video_dataset.aspect_ratios_buckets,
                                                    video_lengths=video_dataset.video_length_buckets,
                                                    drop_last=True,
                                                    ratio_nums=video_dataset.temporal_aspect_ratios_nums)
            video_loader = DataLoader(video_dataset,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory=True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        elif args.video_multireso:
            # 多分辨率
            video_batch_sampler = AspectRatioBatchSampler(sampler = video_sampler,
                                                          dataset = video_dataset,
                                                          batch_size = batch_size[0],
                                                          aspect_ratios = video_dataset.aspect_ratios_buckets,
                                                          drop_last = True,
                                                          ratio_nums = video_dataset.aspect_ratios_nums)

            video_loader = DataLoader(video_dataset,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory=True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        else:
            # 单分辨率
            video_loader = DataLoader(video_dataset,
                                      batch_size = batch_size[0],
                                      shuffle = False,
                                      sampler = video_sampler,
                                      batch_sampler = video_batch_sampler,
                                      num_workers = args.num_workers,
                                      pin_memory = True,
                                      drop_last = True,
                                      prefetch_factor = None if args.num_workers == 0 else args.prefetch_factor,
                                      worker_init_fn = set_worker_seed_builder(dp_rank),
                                      persistent_workers = True if args.num_workers > 0 else False)
        return video_dataset, video_sampler, video_loader
    else:
        logger.warning("T2VA dataset is None!")
        return None, None, None
