import json
import math
import time
import tqdm
import warnings
from collections import defaultdict

# Disable torchvision warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from hymm.parallelism.utils import sync_object_for_parallel_training, sync_random_states
from hymm.dataset.samplers import IndexBatchSampler
from torch.utils.data import DataLoader

from hymm.config import parse_args
from hymm.dataset.image_loader import TextImageArrowStream
from hymm.diffusion import load_denoiser
from hymm.utils.helpers import prepare_model_inputs
from hymm.utils.file_utils import safe_file, rank0_logger
from hymm.utils.torch_utils import set_worker_seed_builder
from hymm.inference import Inference
from hymm.dataset.video_loader import VideoArrowConcatDataset
#from hymm.dataset.video_image_iterator import VideoImageBatchIterator
from hymm.utils.parallel_states import (
    ParallelDims,
    initialize_sequence_parallel_state_v2,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
import os
import datetime
from hymm.utils.load import load_glyph_byT5_v2
from hymm.utils.torch_utils import all_gather_sum


def setup_distributed_evaluation(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.fsdp_sharding_strategy == 'full':
        dp_replicate = 1
    else:
        assert args.fsdp_sharding_strategy == 'hybrid'
        dp_replicate = max(1, world_size // 8)
    
    if world_size <= 8 and args.fsdp_sharding_strategy == 'hybrid':
        args.fsdp_sharding_strategy = "full"
        print("node=1 fall back hybrid to full-shard")
        
    parallel_dims = ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        sp=args.sp_size,
        tp=1,
        pp=1,
        world_size=world_size,
    )
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)  # 默认30秒，可适当延长
    )
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
    

    initialize_sequence_parallel_state_v2(parallel_dims, world_mesh, args.use_dynamic_ring_attention)

    return rank, device, world_size, parallel_dims, world_mesh, dp_degree, dp_rank


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


class LossEvaluator(Inference):
    def __init__(self, args, vae, text_encoder, model, text_encoder_2=None, pipeline=None, ckpt=None,
                 rank=0, world_size=1, device=0, logger=None, parallel_dims=None, 
                 dp_degree=None, dp_rank=None):
        super().__init__(args, vae, text_encoder, model, text_encoder_2=text_encoder_2,
                         pipeline=pipeline, device=device, logger=logger)

        # Store additional parameters that are specific to LossEvaluator
        self.rank = rank
        self.world_size = world_size
        self.ckpt = ckpt

        # Extract tokenizer from text_encoder if it has one
        if hasattr(text_encoder, 'tokenizer'):
            self.tokenizer = text_encoder.tokenizer
        else:
            self.tokenizer = None

        # ====================== Build denoise scheduler ========================
        args = self.args
        self.logger.info("Building denoise scheduler...")
        self.denoiser = load_denoiser(args)
        self.parallel_dims = parallel_dims
        self.dp_degree = dp_degree if dp_degree is not None else dist.get_world_size()
        self.dp_rank = dp_rank if dp_rank is not None else dist.get_rank()

    @torch.no_grad()
    def eval(self, index_file, multireso, image_size, save_path, batch_size=1, timesteps=8, 
             index_kwargs=None, byt5_kwarg=None, extra_model=None):
        """
        Evaluate the model on the validation dataset with `num_timesteps` equally spaced timesteps.

        Args:
            index_file (str): The validation dataset.
            multireso (bool): Whether to use the multi-resolution index.
            image_size (int): The image size of the validation dataset.
            save_path (str): The path to save the evaluation results.
            batch_size (int): Batch size of each validation step. Default is 1.
            timesteps (int or list): If an int, the number of equally spaced timesteps. If a list,
                the list of timesteps.
            index_kwargs (dict): The index kwargs for the validation dataset.
        """
        if os.path.exists(save_path):
            self.logger.info(f"Save path {save_path} already exists, skip evaluation.")
            results = json.load(open(save_path, 'r'))
            return results
        args = self.args
        if index_kwargs is None:
            index_kwargs = {}

        if isinstance(batch_size, (list, tuple)):
            assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
            batch_size = batch_size[0]

        if isinstance(timesteps, int):
            if args.denoise_type == "flow":
                timesteps = np.linspace(0, 1, timesteps + 2).tolist()[1:-1]
            elif args.denoise_type == "ddpm":
                timesteps = np.linspace(0, self.denoiser.num_timesteps - 1, timesteps + 2)[1:-1]
                # Round to the nearest timestep
                timesteps = [int(round(t)) for t in timesteps]
        assert isinstance(timesteps, (list, tuple)), f"Invalid timesteps: {timesteps}"
        # We log the first, middle, and last timesteps
        t1, t2, t3 = timesteps[0], timesteps[len(timesteps) // 2], timesteps[-1]
        self.logger.info(f"Evaluating at {len(timesteps)} timesteps: {[round(t, 4) for t in timesteps]}")

        if 'video' in args.data_type:
            val_video_dataset = VideoArrowConcatDataset(
                args,
                index_file=args.val_index_file,
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
                text_len=args.text_len,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                uncond_p=args.video_uncond_p,
                use_caption_prob=args.video_caption_rate,
                caption_sample_ratio=args.video_caption_sample_ratio,
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
                logger=self.logger,
                video_caption_i2v_instruction_enable=args.video_caption_i2v_instruction_enable,
                video_caption_i2v_instruction_col_probs=args.video_caption_i2v_instruction_col_probs,
                video_caption_i2v_instruction_col=args.video_caption_i2v_instruction_col,
                video_caption_i2v_instruction_col_2=args.video_caption_i2v_instruction_col_2,
                video_caption_i2v_instruction_sub_col_probs=args.video_caption_i2v_instruction_sub_col_probs,
                video_caption_i2v_instruction_sub_col=args.video_caption_i2v_instruction_sub_col,
                video_caption_i2v_instruction_sub_col_2=args.video_caption_i2v_instruction_sub_col_2,
                video_caption_i2v_instruction_sub_col_3=args.video_caption_i2v_instruction_sub_col_3,
                local_seed=42,
            )
            sampler = DistributedSampler(val_video_dataset, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=True,
                                            seed=args.global_seed, drop_last=False)
            
            val_video_loader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
                                worker_init_fn=set_worker_seed_builder(self.dp_rank), persistent_workers=True)
            
            num_video_samples = val_video_dataset.total_length
            num_video_steps = len(val_video_loader)
            self.logger.info(f"Validation video dataset loaded with {num_video_samples} samples.")
        else:
            val_video_dataset = None
            val_video_loader = None
            num_video_samples = 0
            num_video_steps = 0
        
        if 'image'== args.data_type:
            val_dataset = TextImageArrowStream(args,
                                            image_size=image_size,
                                           crop_type=args.image_crop_type,
                                           uncond_p=0,
                                           text_len=args.text_len,
                                           text_encoder=self.text_encoder,
                                           text_encoder_2=self.text_encoder_2,
                                           index_file=index_file,
                                           multireso=multireso,
                                           index_kwargs=index_kwargs,
                                           logger=self.logger,
                                           )
            sampler = DistributedSampler(val_dataset, num_replicas=self.dp_degree, rank=self.dp_rank, shuffle=False,
                                        drop_last=False)
            batch_sampler = IndexBatchSampler(val_dataset.index_manager, sampler=sampler, batch_size=batch_size,
                                            drop_last=False)
            val_loader = DataLoader(val_dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                                    prefetch_factor=None if args.num_workers == 0 else args.prefetch_factor,
                                    worker_init_fn=set_worker_seed_builder(self.dp_rank), persistent_workers=True)
            self.logger.info(f"Validation dataset (multireso={multireso}) loaded from {index_file}")
            self.logger.info(f"Index kwargs: {index_kwargs}")
            self.logger.info(f"Found {len(val_dataset)} samples.")
        else:
            val_dataset = None
            val_loader = None
            num_image_samples = 0
            num_image_steps = 0
        
        if 'video' not in args.data_type: loader = val_loader 
        else: loader = val_video_loader # if load video, use video_loader as the main loader
    
        target_dtype = None
        autocast_enabled = False
        if args.precision == "fp16":
            target_dtype = torch.float16
            autocast_enabled = True
        elif args.precision == "bf16":
            target_dtype = torch.bfloat16
            autocast_enabled = True

        # estimated_steps = math.ceil(len(sampler) / batch_size)
        estimated_steps = len(loader)
        self.logger.info(f"Start evaluating the model with estimated {estimated_steps} steps...")

        proc_loss = defaultdict(float)
        proc_count = defaultdict(float)
        for i, batch in enumerate(loader, start=1):
            latents, model_kwargs, data_type, n_tokens, multitask_mask, cond_latents, last_frame_cond_latents, mask_type, use_i2v_instruction, videoid, repa_feats, index, text = prepare_model_inputs(
                batch, self.device, self.model, self.vae, self.text_encoder, self.text_encoder_2, args,
                byt5_kwarg, 
                extra_model,
                seed=args.local_seed
            )
            start_time = time.time()
            for t in tqdm.tqdm(timesteps):
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    sync_random_states(self.parallel_dims)
                    self.logger.info(f"Sync random states for {self.model.__class__.__name__}")
                    latents, model_kwargs, t, data_type, multitask_mask, cond_latents, last_frame_cond_latents, args.multitask_mask_training_type, args.model_type, mask_type = \
                    sync_object_for_parallel_training([latents, model_kwargs, t, data_type, multitask_mask, cond_latents, last_frame_cond_latents, args.multitask_mask_training_type, args.model_type, mask_type ], parallel_dims=self.parallel_dims)


                    _, loss_dict = self.denoiser.training_losses(
                        model=self.model, 
                        x1=latents, 
                        args=args,
                        model_kwargs=model_kwargs,
                        timestep=t,
                        data_type=data_type,
                        multitask_mask=multitask_mask,
                        cond_latents=cond_latents,
                        last_frame_cond_latents=last_frame_cond_latents,
                        multitask_mask_training_type=args.multitask_mask_training_type,
                        reorg_token='888' in args.model_type or '16168' in args.model_type,
                        mask_type=mask_type)

                loss = loss_dict["loss"].mean().item()
                count = latents.size(0)
                proc_loss[t] += loss
                proc_count[t] += count
            elapsed_time = time.time() - start_time
            if i % 10 == 0 or latents.size(0) != batch_size:
                self.logger.info(f"[rank={self.dp_rank}] (step={i}/{estimated_steps}) "
                                 f"loss(t={t1:.4f}): {proc_loss[t1] / proc_count[t1]:.4f}, "
                                 f"loss(t={t2:.4f}): {proc_loss[t2] / proc_count[t2]:.4f}, "
                                 f"loss(t={t3:.4f}): {proc_loss[t3] / proc_count[t3]:.4f}, "
                                 f"Steps/Sec: {1 / elapsed_time:.2f} s/step, "
                                 f"Batch size: {latents.size(0)}"
                                 )

        t_losses = {}
        t_counts = {}
        results = []
        total_losses = []
        for t in timesteps:
            # Sync the total loss and count across all processes
            t_losses[t] = torch.tensor(proc_loss[t], device=self.device)
            t_counts[t] = torch.tensor(proc_count[t], device=self.device)
            dist.all_reduce(t_losses[t], op=dist.ReduceOp.SUM)
            dist.all_reduce(t_counts[t], op=dist.ReduceOp.SUM)
            t_loss = t_losses[t].item() / t_counts[t].item()
            t_count = t_counts[t].item()
            results.append({
                "timestep": float(t) if isinstance(t, float) else int(t),
                "loss": float(t_loss),
                "count": int(t_count),
            })
            total_losses.append(t_loss)
            self.logger.info(f"[rank={self.dp_rank}] Average loss at timestep {round(t, 4)}: {t_loss:.4f} ")
        results.append({
            "timesteps": -1,
            "loss": float(sum(total_losses) / len(total_losses)),
        })

        if self.dp_rank == 0:
            save_path = safe_file(save_path)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)
            self.logger.info(f"Evaluation results saved to {save_path}")

        # All ranks return the same results
        return results


def main(mode):
    args = parse_args(mode)
    rank, device, world_size, parallel_dims, world_mesh, dp_degree, dp_rank = setup_distributed_evaluation(args)
    logger = rank0_logger(rank)
    
    evaluator = LossEvaluator.from_pretrained(args.ckpt, pretrained_model_path=None,
                                              args=args,
                                              device=device,
                                              logger=logger,
                                              parallel_dims=parallel_dims,
                                              world_mesh=world_mesh)
    evaluator.dp_degree = dp_degree
    evaluator.dp_rank = dp_rank
    evaluator.world_size=world_size
    evaluator.parallel_dims = parallel_dims
    evaluator.device = device
    
    # Get the updated args
    args = evaluator.args
    
    

    if args.val_index_file is None and args.val_video_index_file is None:
        raise ValueError("Please specify the evaluation index file.")
    val_save_path = evaluator.get_metric_save_path("val_loss", logger)

    # 初始化byt5_kwarg和extra_model
    byt5_kwarg = None
    extra_model = {}
    
    assert args.vae == "16164-32c-v2", "Only 16164-32c-v2 is supported."
    args.vae_spatial_ratio = 16
    args.vae_temporal_ratio = 4
    
    byt5_kwarg = None
    if args.glyph_byT5_v2:
        byt5_kwarg = load_glyph_byT5_v2(args, device)

    # 按照train.py的方式初始化extra_model
    if hasattr(args, 'multitask_semantic_mode') and args.multitask_semantic_mode == "siglip":
        try:
            from hymm.models.vision_encoder import VisionEncoder
            vision_encoder = VisionEncoder(
                vision_encoder_type=args.vision_encoder,
                vision_encoder_precision=args.vision_encoder_precision,
                vision_encoder_path=args.vision_encoder_path,
                processor_type=args.processor_type,
                processor_path=args.processor_path,
                output_key=args.vision_output_key,
                logger=logger,
                device=device
            )
            extra_model["vision_encoder"] = vision_encoder
        except Exception as e:
            logger.warning(f"Failed to load VisionEncoder: {e}")

    logger.info(f"Start evaluating the model ckpt {args.ckpt} with args={args} ...")
    evaluator.eval(index_file=args.val_index_file,
                   multireso=args.val_multireso,
                   image_size=args.image_size,
                   save_path=val_save_path,
                   batch_size=args.val_batch_size,
                   timesteps=args.val_timesteps,
                   byt5_kwarg=byt5_kwarg,
                   extra_model=extra_model,
                   index_kwargs=dict(
                       image_caption_rate=args.val_image_caption_rate,
                       image_text_arrow_suffix=args.val_image_text_arrow_suffix,
                       image_text_col=args.val_image_text_col,
                       image_caption_arrow_suffix=args.val_image_caption_arrow_suffix,
                       image_caption_col=args.val_image_caption_col,
                       index_strategy=args.val_index_strategy,
                       index_probability=args.val_index_probability,
                       caption_sample_ratio=args.val_caption_sample_ratio
                   ))

