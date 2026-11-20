import json
import time
import os
import warnings
from pathlib import Path
from loguru import logger
import numpy as np

# Disable torchvision warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

from hymm.dataset.samplers import IndexBatchSampler

from hymm.config import parse_args
from hymm.constants import (
    FID_INCEPTION_PATH, COCO30K_FID_TARGET_PATH, VAL30K_FID_TARGET_PATH, ICS_INCEPTION_PATH, CLIP_MODEL_PATH,
    HPSV2_MODEL_PATH, COCO30K_PATH, COCO6K_PATH, VAL30K_PATH, T2I_COMPBENCH_PATH, VQA_MODEL_PATH,
)
from hymm.dataset.data_kits.coco import COCO30kDataset
from hymm.dataset.data_kits.val_30k_dataset import VAL30kDataset
from hymm.dataset.data_kits.t2i_compbench import T2ICompBenchDataset
from hymm.metrics import FIDMetric, InceptionScoreMetric, CLIPScoreMetric, HPSv2Metric, T2ICompBenchMetric, \
    GeneralGenerationBenchMetric
from hymm.utils.file_utils import safe_file, safe_save_file
from hymm.utils.helpers import as_tuple
from hymm.metrics_video.utils import str2bool, str2null, int2null
from hymm.utils.data_utils import generate_crop_size_list, get_closest_ratio

from hymm.inference import HunyuanVideoSampler
from hymm.utils.parallel_states import (
    ParallelDims,
    initialize_sequence_parallel_state_v2,
    initialize_sequence_parallel_state,
    nccl_info,
)
import datetime

import sys
import math
from typing import TypeVar, Optional, Iterator

from torch.utils.data import Dataset, Sampler, DataLoader

T_co = TypeVar('T_co', covariant=True)

DEBUG = False
MAX_SIZE = 512


def initialize_distributed(args):
    # ==================================================================================================================
    # new - ref training codes
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if args.fsdp_sharding_strategy == 'full':
        dp_replicate = 1
    else:
        assert args.fsdp_sharding_strategy == 'hybrid'
        dp_replicate = max(1, world_size // 8)

    if world_size <= 8 and args.fsdp_sharding_strategy == 'hybrid':
        args.fsdp_sharding_strategy = "full"
        print("node=1 fall back hybrid to full-shard")

    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)  # 默认30秒，可适当延长
    )
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    from hymm.parallelism.parallel_states import init_parallel_state

    parallel_dims = init_parallel_state(
        dp_replicate=dp_replicate,
        dp_shard=-1,
        sp=args.sp_size,
        tp=1,
        pp=args.pp_size,
        ep=args.ep_size,
        world_size=world_size,
    )

    initialize_sequence_parallel_state_v2(parallel_dims, parallel_dims.device_mesh_for_pp_dp_sp_tp, args.use_dynamic_ring_attention)

    return parallel_dims, device


def save_results(save_path, results):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    return save_path


class ScoreEvaluator(HunyuanVideoSampler):
    def __init__(self, args, vae, text_encoder,
                 model, text_encoder_2=None, pipeline=None, use_cpu_offload=False, rank=0, world_size=1, device=0,
                 logger=None,
                 ):
        super().__init__(args, vae, text_encoder, model, text_encoder_2=text_encoder_2, pipeline=pipeline,
                         use_cpu_offload=use_cpu_offload, device=device, logger=logger,
                         )
        self.metric_dict = {}
        self.valid_metrics = set()
        self.score_models_loaded = False
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.sp_size = args.sp_size
        # rank =  nccl_info.rank_within_group
        # print(f"self.rank(global)={self.rank}, rank(local)={rank}, world_size={self.world_size}, sp_size={self.sp_size}")

    def initialize_scores(self, scores):
        args = self.args
        metric_dict = {}
        assert len(scores) > 0, "Must specify at least one metric to evaluate."

        for score in scores:
            if score == "fid" or score == "fid@coco":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=COCO30K_FID_TARGET_PATH,
                    dataset_name="COCO30K",
                    max_size=MAX_SIZE,
                )
            elif score == "fid@coco6k":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=COCO30K_FID_TARGET_PATH,
                    dataset_name="COCO6K",
                    max_size=MAX_SIZE,
                )
            elif score == "fid@coco_qwen":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=COCO30K_FID_TARGET_PATH,
                    dataset_name="COCO30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "fid@val_trans":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=VAL30K_FID_TARGET_PATH,
                    dataset_name="VAL30K_TRANS",
                    max_size=MAX_SIZE,
                )
            elif score == "fid@val_llava":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=VAL30K_FID_TARGET_PATH,
                    dataset_name="VAL30K_LLAVA",
                    max_size=MAX_SIZE,
                )
            elif score == "fid@val_qwen":
                metric_dict[score] = FIDMetric(
                    dims=2048,
                    inception_path=FID_INCEPTION_PATH,
                    target_path=VAL30K_FID_TARGET_PATH,
                    dataset_name="VAL30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score" or score == "clip_score@coco":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="COCO30K",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score@coco6k":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="COCO6K",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score@coco_qwen":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="COCO30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score@val_trans":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="VAL30K_TRANS",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score@val_llava":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="VAL30K_LLAVA",
                    max_size=MAX_SIZE,
                )
            elif score == "clip_score@val_qwen":
                metric_dict[score] = CLIPScoreMetric(
                    clip_model_path=CLIP_MODEL_PATH,
                    dataset_name="VAL30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "hpsv2" or score == "hpsv2@coco":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="COCO30K",
                    max_size=MAX_SIZE,
                )

            elif score == "hpsv2@coco6k":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="COCO6K",
                    max_size=MAX_SIZE,
                )
            elif score == "hpsv2@coco_qwen":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="COCO30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "hpsv2@val_trans":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="VAL30K_TRANS",
                    max_size=MAX_SIZE,
                )
            elif score == "hpsv2@val_llava":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="VAL30K_LLAVA",
                    max_size=MAX_SIZE,
                )
            elif score == "hpsv2@val_qwen":
                metric_dict[score] = HPSv2Metric(
                    hpsv2_model_path=HPSV2_MODEL_PATH,
                    dataset_name="VAL30K_QWEN",
                    max_size=MAX_SIZE,
                )
            elif score == "t2i_compbench":
                metric_dict[score] = T2ICompBenchMetric(
                    vqa_model_path=VQA_MODEL_PATH,
                    dataset_name="T2ICompBench",
                    max_size=MAX_SIZE,
                )
            elif score == "ics":
                metric_dict[score] = InceptionScoreMetric(
                    dims=2048,
                    inception_path=ICS_INCEPTION_PATH,
                    dataset_name="COCO",
                    max_size=MAX_SIZE,
                )
            elif score == "dpg_bench":
                from hymm.constants import MPLUG_MODEL_PATH
                from hymm.metrics.dpg_bench.metric import DPGBenchMetric

                metric_dict[score] = DPGBenchMetric(
                    vqa_model_path=MPLUG_MODEL_PATH,
                    dataset_name="DPGBench",
                    max_size=MAX_SIZE,
                )
            elif score == "geneval":
                from hymm.metrics.geneval.metric import GenEvalMetric
                from hymm.constants import GENEVAL_MODEL_PATH, GENEVAL_DATA

                metric_dict[score] = GenEvalMetric(
                    model_path=GENEVAL_MODEL_PATH,
                    dataset_name="GenEval",
                    max_size=MAX_SIZE,
                )
            elif "general_generation" in score:
                metric_dict[score] = GeneralGenerationBenchMetric(
                    dataset_name=score,
                    max_size=MAX_SIZE,
                )
            else:
                raise NotImplementedError(f"Metric {score} not implemented.")
        self.metric_dict = metric_dict

    def load_score_models(self, size=256):
        if not self.metric_dict:
            raise ValueError("No metric is initialized. Call `initialize_scores(scores)` first.")
        for metric_name, metric in self.metric_dict.items():
            if "GeneralGeneration" in metric.dataset_name:
                continue
            if size <= metric.max_size:
                metric.load_model()
        self.score_models_loaded = True

    def release_score_models(self, size=256):
        """ Release score models to save memory."""
        if not self.metric_dict:
            raise ValueError("No metric is initialized. Call `initialize_scores(scores)` first.")
        for metric_name, metric in self.metric_dict.items():
            if "GeneralGeneration" in metric.dataset_name:
                continue
            if size <= metric.max_size:
                metric.release_model()
        self.score_models_loaded = False
        # Clear cuda cache
        torch.cuda.empty_cache()

    def get_dataloader(self, dataset, batch_size):
        dataset_sampler = DistributedSamplerFix(
            dataset=dataset,
            num_replicas=self.world_size if self.sp_size is None else self.world_size // self.sp_size,
            rank=self.rank if self.sp_size is None else self.rank // self.sp_size,
            shuffle=False,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            pin_memory=False,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
            batch_size=batch_size,
            num_workers=0,
        )
        return dataloader

    @torch.no_grad()
    def eval(self, image_size, batch_size, save_path, image_save_base=None, ignore_max_size=False):
        args = self.args
        # TODO(contributor): Find a better choice
        size = as_tuple(image_size)[0] if not ignore_max_size else 0
        if not self.score_models_loaded:
            self.load_score_models(size)

        if isinstance(batch_size, (list, tuple)):
            assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
            batch_size = batch_size[0]

        dataloaders = []

        use_COCO30K = any([(metric.dataset_name == "COCO30K") and size <= metric.max_size
                           for metric in self.metric_dict.values()])
        use_COCO6K = any([(metric.dataset_name == "COCO6K") and size <= metric.max_size
                          for metric in self.metric_dict.values()])
        use_COCO30K_QWEN = any([(metric.dataset_name == "COCO30K_QWEN") and size <= metric.max_size
                                for metric in self.metric_dict.values()])
        use_VAL30K_TRANS = any([(metric.dataset_name == "VAL30K_TRANS") and size <= metric.max_size
                                for metric in self.metric_dict.values()])
        use_VAL30K_LLAVA = any([(metric.dataset_name == "VAL30K_LLAVA") and size <= metric.max_size
                                for metric in self.metric_dict.values()])
        use_VAL30K_QWEN = any([(metric.dataset_name == "VAL30K_QWEN") and size <= metric.max_size
                               for metric in self.metric_dict.values()])
        use_T2ICompBench = any([(metric.dataset_name == "T2ICompBench") and size <= metric.max_size
                                for metric in self.metric_dict.values()])
        use_DPGBench = any([(metric.dataset_name == "DPGBench") and size <= metric.max_size
                            for metric in self.metric_dict.values()])
        use_GenEval = any([(metric.dataset_name == "GenEval") and size <= metric.max_size
                           for metric in self.metric_dict.values()])
        use_GeneralGeneration = any([("GeneralGeneration" in metric.dataset_name) and size <= metric.max_size
                                     for metric in self.metric_dict.values()])

        if use_COCO30K:
            dataset = COCO30kDataset(
                path=COCO30K_PATH,
                debug=DEBUG,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"COCO30K dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "COCO30K"))

        if use_COCO6K:
            dataset = COCO30kDataset(
                path=COCO6K_PATH,
                debug=DEBUG,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"COCO6K dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "COCO6K"))

        if use_COCO30K_QWEN:
            assert args.val_caption_sample_ratio is not None, \
                "When evaluating using QWEN caption, val-caption-sample-ratio must be specified to parse json to caption."
            dataset = COCO30kDataset(
                path=COCO30K_PATH,
                use_qwen_caption=True,
                caption_sample_ratio=args.val_caption_sample_ratio,
                debug=DEBUG,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"COCO30K_QWEN dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "COCO30K_QWEN"))

        if use_VAL30K_TRANS:
            dataset = VAL30kDataset(
                index_file=VAL30K_PATH,
                index_kwargs=dict(
                    image_text_col="trans",
                    image_text_arrow_suffix="_trans",
                    image_caption_rate=0.0,
                ),
                debug=DEBUG,
                logger=self.logger,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"VAL30K_TRANS dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "VAL30K_TRANS"))

        if use_VAL30K_LLAVA:
            dataset = VAL30kDataset(
                index_file=VAL30K_PATH,
                index_kwargs=dict(
                    image_caption_col="llava_caption",
                    image_caption_arrow_suffix="_caption",
                    image_caption_rate=1.0,
                ),
                debug=DEBUG,
                logger=self.logger,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"VAL30K_LLAVA dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "VAL30K_LLAVA"))

        if use_VAL30K_QWEN:
            assert args.val_caption_sample_ratio is not None, \
                "When evaluating using QWEN caption, val-caption-sample-ratio must be specified to parse json to caption."
            dataset = VAL30kDataset(
                index_file=VAL30K_PATH,
                index_kwargs=dict(
                    image_caption_col="caption_qwen_v02",
                    image_caption_arrow_suffix="_qwen_caption",
                    image_caption_rate=1.0,
                    caption_sample_ratio=args.val_caption_sample_ratio,
                ),
                debug=DEBUG,
                logger=self.logger,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"VAL30K_QWEN dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "VAL30K_QWEN"))

        if use_T2ICompBench:
            dataset = T2ICompBenchDataset(
                csvs=T2I_COMPBENCH_PATH,
                debug=DEBUG,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"T2ICompBench dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "T2ICompBench"))

        if use_DPGBench:
            from hymm.constants import DPG_BENCH_DATA
            from hymm.dataset.data_kits.dpg_bench import DPGBenchDataset

            dataset = DPGBenchDataset(
                csvs=DPG_BENCH_DATA,
                debug=DEBUG,
            )
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"DPGBench dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "DPGBench"))

        if use_GenEval:
            from hymm.constants import GENEVAL_DATA
            from hymm.dataset.data_kits.geneval import GenEvalDataset

            dataset = GenEvalDataset(jsonl=GENEVAL_DATA)
            dataloader = self.get_dataloader(dataset, batch_size)
            self.logger.info(f"GenEval dataset loaded. Total samples: {len(dataset)}")
            dataloaders.append((dataloader, dataset, "GenEval"))

        if use_GeneralGeneration:
            from hymm.constants import General_Generation_BENCH_DATA
            from hymm.dataset.data_kits.general_generation_bench import GeneralGenerationBenchDataset

            for metric in self.metric_dict.values():
                if "GeneralGeneration" in metric.dataset_name:
                    dataset = GeneralGenerationBenchDataset(
                        csvs=General_Generation_BENCH_DATA[metric.score_name],
                        debug=DEBUG,
                    )
                    dataloader = self.get_dataloader(dataset, batch_size)
                    self.logger.info(f"{metric.dataset_name} dataset loaded. Total samples: {len(dataset)}")
                    dataloaders.append((dataloader, dataset, metric.dataset_name))

        for dataloader, dataset, dataset_name in dataloaders:
            for metric_name, metric in self.metric_dict.items():
                if "GeneralGeneration" in metric.dataset_name:
                    continue
                if metric.dataset_name == dataset_name:
                    self.valid_metrics.add(metric_name)

        # Do prediction
        for dataloader, dataset, dataset_name in dataloaders:
            total_batches = len(dataloader)
            self.logger.info(
                f"********************************** "
                f"Evaluation on {dataset_name}. Total batches: {total_batches} "
                f"**********************************"
            )

            if image_save_base is not None:
                if '{}' in str(image_save_base):
                    save_dir = Path(str(image_save_base).format(dataset_name))
                else:
                    save_dir = Path(str(image_save_base) + f"_{dataset_name}")
                self.logger.info(f"{dataset_name} Images will be saved to {save_dir}")
            else:
                save_dir = None

            for batch_index, batch in enumerate(dataloader, start=1):
                self.logger.info(f"Batch {batch_index}/{total_batches}")

                if "prompt_embeds" in batch.keys():
                    prompts = None
                    prompt_embeds = batch["prompt_embeds"]
                    attention_mask = batch["attention_mask"]
                    negative_prompt_embeds = batch["negative_prompt_embeds"]
                    negative_attention_mask = batch["negative_attention_mask"]
                else:
                    prompts = batch["prompts"]
                    prompt_embeds = None
                    attention_mask = None
                    negative_prompt_embeds = None
                    negative_attention_mask = None
                print(batch.keys())
                if "heights" in batch and "widths" in batch:
                    if batch["heights"][0] is None or batch["widths"][0] is None:
                        height, width = as_tuple(image_size)[0], as_tuple(image_size)[0]
                    else:
                        print(batch["heights"])
                        print(batch["widths"])
                        height, width = batch["heights"][0], batch["widths"][0]
                        crop_size_list = generate_crop_size_list(image_size[0], 16)
                        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
                        closest_size, closest_ratio = get_closest_ratio(height, width, aspect_ratios, crop_size_list)
                        height, width = closest_size
                else:
                    height, width = as_tuple(image_size)[0], as_tuple(image_size)[0]

                out_dict = self.predict(
                    prompt=prompts,
                    height=height,  # 只能处理图片
                    width=width,  # 只能处理图片
                    video_length=args.num_frames,
                    batch_size=len(prompts) if isinstance(prompts, list) else 1,
                    seed=batch["seeds"],
                    flow_shift=args.flow_shift_image,
                    num_images_per_prompt=args.num_images,
                    infer_steps=args.infer_steps,
                    negative_prompt=args.neg_prompt,
                    prompt_embeds=prompt_embeds,
                    attention_mask=attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask,
                    guidance_scale=args.cfg_scale,
                    output_type="pt",  # If "pt", return torch tensor with range [0, 1] and shape (B, 3, T, H, W)
                    verbose=1,
                )
                samples = out_dict['samples'].float().squeeze(2)  # 去除时间维度
                proc_inputs = batch
                proc_inputs["images"] = samples.cuda()  # pipeline 输出结果为cpu tensor
                self.logger.info(f"images shape: {samples.shape}")

                start_time = time.time()
                for metric_name, metric in self.metric_dict.items():
                    if "GeneralGeneration" in metric.dataset_name:
                        continue
                    if metric.dataset_name == dataset_name:
                        metric.process(**proc_inputs)
                gen_time = time.time() - start_time
                self.logger.info(f"Process time: {gen_time}")

                if save_dir is not None:
                    save_names = [save_dir / f'{id_}.png' for id_ in batch['ids']]
                    self.save_batch_image(out_dict['samples'].squeeze(2), save_names)

        metric_gather_results = {}
        for metric_name, metric in self.metric_dict.items():
            if metric_name in self.valid_metrics:
                metric_gather_results[metric_name] = metric.all_gather_results()

        results = []
        if self.rank == 0:
            for metric_name, metric in self.metric_dict.items():
                if metric_name in self.valid_metrics:
                    output = metric.compute_metrics(metric_gather_results[metric_name])
                    if isinstance(output, dict):
                        for key, (value, count) in output.items():
                            count = count if self.sp_size == 1 else count // self.sp_size
                            results.append({"metric": f"{metric_name}_{key}", "value": value, "count": count})
                    elif isinstance(output, tuple):
                        value, count = output
                        count = count if self.sp_size == 1 else count // self.sp_size
                        results.append({"metric": metric_name, "value": value, "count": count})
                    else:
                        raise ValueError(
                            f"Output of metric {metric_name} should be a dict or a tuple, but got {output}")
            self.logger.info(results)

            accumulated_results = results[:]
            save_path = safe_file(save_path)
            if save_path.exists():
                with open(save_path, "r") as f:
                    ori_results = json.load(f)
                accumulated_results = ori_results + results

            save_to = safe_save_file(save_path, accumulated_results, save_fn=save_results)
            self.logger.info(f"Evaluation results saved to {save_to}")

        # gather finish, reset
        for metric_name, metric in self.metric_dict.items():
            metric.reset()
        self.valid_metrics.clear()

        # Only rank-0 returns the valid results
        return results


class DistributedSamplerFix(Sampler[T_co]):
    """
    The original DistributedSampler always align the num_samples among replicas when drop_last=False,
    This feature causes the total num_samples could be more than the actual dataset size, leading to
    vary total number of samples when using different number of GPUs, which affects the evaluation score
    such as FID, CLIP score, etc.

    This class implements a modified version of DistributedSampler that allows each replica to have different
    num_samples when drop_last=False.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            total_size = len(self.dataset)  # type: ignore[arg-type]
            self.num_samples = total_size // self.num_replicas + int(
                rank < total_size % self.num_replicas
            )
            self.total_size = total_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            pass
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def main(mode):
    args = parse_args(mode)

    parallel_dims, device = initialize_distributed(args)
    evaluator = ScoreEvaluator.from_pretrained(args.dit_weight, args.pretrained_model_name_or_path, args=args,
                                               parallel_dims=parallel_dims, device=device)
    evaluator.initialize_scores(args.scores)
    # Get the updated args
    args = evaluator.args

    val_save_path = evaluator.get_metric_save_path('score', logger)
    image_save_base = evaluator.get_sample_save_dir(
        testset=None, image_size=args.image_size) if args.score_save_image else None

    evaluator.eval(image_size=args.image_size,
                   batch_size=args.val_batch_size,
                   save_path=val_save_path,
                   image_save_base=image_save_base,
                   ignore_max_size=True,  # dataset.max_size is used for val-on-train. Ignore it for pure evaluation.
                   )


