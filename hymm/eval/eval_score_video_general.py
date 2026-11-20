import json
import time
import os
import warnings
from pathlib import Path
from loguru import logger

# Disable torchvision warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


from hymm.config import parse_args
from hymm.constants import (
    FID_INCEPTION_PATH, COCO30K_FID_TARGET_PATH, VAL30K_FID_TARGET_PATH, ICS_INCEPTION_PATH, CLIP_MODEL_PATH,
    HPSV2_MODEL_PATH, COCO30K_PATH, COCO6K_PATH, VAL30K_PATH, T2I_COMPBENCH_PATH, VQA_MODEL_PATH,
)
from hymm.dataset.data_kits.coco import COCO30kDataset
from hymm.dataset.data_kits.val_30k_dataset import VAL30kDataset
from hymm.dataset.data_kits.t2i_compbench import T2ICompBenchDataset
from hymm.metrics import GeneralGenerationBenchMetric
from hymm.utils.file_utils import safe_file, safe_save_file
from hymm.utils.helpers import as_tuple

from hymm.inference import HunyuanVideoSampler
from hymm.utils.parallel_states import (
    initialize_sequence_parallel_state_v2,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from hymm.utils.multitask_utils import get_multitask_mask, get_cond_latents, encode_frames_to_latents, get_multitask_mask_with_ranges
from hymm.utils.data_utils import get_closest_ratio, generate_crop_size_list
from hymm.utils.temporal_resample import resample_latent_temporal
from hymm.utils.native_reference import as_bool
from hymm.utils.reference_temporal_pack import (
    REFERENCE_TIMELINE,
    normalize_temporal_pack_mode,
    scatter_reference_timeline_latent,
)
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from hymm.utils.file_utils import save_videos_grid
import shutil
import datetime

DEBUG = False
MAX_SIZE = 512

def is_valid_csv(csv_path):
    if csv_path.endswith(".csv") and os.path.exists(csv_path):
        return True
    return False

def is_valid_json(json_path):
    # Reference-editing eval set (per-case `conditions`), driven through the
    # general-generation range_condition path.
    if json_path.endswith(".json") and os.path.exists(json_path):
        return True
    return False

def initialize_distributed(args):
    # ==================================================================================================================
    # # old
    # global_rank = int(os.getenv("RANK", 0))
    # local_rank = int(os.getenv("LOCAL_RANK", 0))
    # world_size = int(os.getenv("WORLD_SIZE", 1))
    # # print("local_rank: {}, world_size: {}".format(local_rank, world_size))
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(
    #     backend="nccl", init_method="env://", world_size=world_size, rank=global_rank
    # )
    # initialize_sequence_parallel_state(args.sp_size)
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
    from hymm.parallelism.parallel_states import init_parallel_state
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=3600)  # 默认30秒，可适当延长
    )
    torch.cuda.set_device(local_rank)
    parallel_dims = init_parallel_state(
         dp_replicate=dp_replicate,
         dp_shard=-1,
         sp=args.sp_size,
         tp=1,
         pp=1,
         world_size=world_size,
    )
    device = torch.cuda.current_device()
    world_mesh = parallel_dims.device_mesh_for_pp_dp_sp_tp
    initialize_sequence_parallel_state_v2(parallel_dims, parallel_dims.device_mesh_for_pp_dp_sp_tp, args.use_dynamic_ring_attention)

    return parallel_dims, world_mesh, device

    # ==================================================================================================================


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
        from hymm.constants import General_Generation_BENCH_DATA
        metric_dict = {}
        assert len(scores) > 0, "Must specify at least one metric to evaluate."
        for score in scores:
            if score.startswith('fid'):
                from hymm.metrics import FIDMetric
            elif score.startswith('clip_score'):
                from hymm.metrics import CLIPScoreMetric
            elif score == 'ics':
                from hymm.metrics import InceptionScoreMetric
            elif score.startswith('hps'):
                from hymm.metrics import HPSv2Metric
            elif score.startswith('t2i'):
                from hymm.metrics import T2ICompBenchMetric
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
                from hymm.constants import GENEVAL_MODEL_PATH

                metric_dict[score] = GenEvalMetric(
                    model_path=GENEVAL_MODEL_PATH,
                    dataset_name="GenEval",
                    max_size=MAX_SIZE,
                )
            elif "general_generation" in score or score in General_Generation_BENCH_DATA:
                metric_dict[score] = GeneralGenerationBenchMetric(
                    dataset_name=score,
                    max_size=MAX_SIZE,
                )
            elif is_valid_csv(score) or is_valid_json(score):
                try:
                    metric_dict[score] = GeneralGenerationBenchMetric(
                        dataset_name=score,
                        max_size=MAX_SIZE,
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error loading metric {score}: {e}, will skip this metric,")
                    continue
            else:
                raise NotImplementedError(f"Metric {score} not implemented.")
        self.metric_dict = metric_dict

    def load_score_models(self, size=256):
        if not self.metric_dict:
            raise ValueError("No metric is initialized. Call `initialize_scores(scores)` first.")
        for metric_name, metric in self.metric_dict.items():
            if getattr(metric, 'is_general_generation', False) or is_valid_csv(metric.score_name):
                continue
            if size <= metric.max_size:
                metric.load_model()
        self.score_models_loaded = True

    def release_score_models(self, size=256):
        """ Release score models to save memory."""
        if not self.metric_dict:
            raise ValueError("No metric is initialized. Call `initialize_scores(scores)` first.")
        for metric_name, metric in self.metric_dict.items():
            if getattr(metric, 'is_general_generation', False) or is_valid_csv(metric.score_name):
                continue
            if size <= metric.max_size:
                metric.release_model()
        self.score_models_loaded = False
        # Clear cuda cache
        torch.cuda.empty_cache()

    def get_dataloader(self, dataset, batch_size):
        dataset_sampler = DistributedSampler(
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
        use_GeneralGeneration = any([((getattr(metric, 'is_general_generation', False)) 
                                       or is_valid_csv(metric.score_name))
                                        and size <= metric.max_size
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
                if getattr(metric, 'is_general_generation', False):
                    if os.path.exists(metric.score_name):
                        csv_path = [metric.score_name]
                    else:
                        csv_path = General_Generation_BENCH_DATA[metric.score_name]
                    dataset = GeneralGenerationBenchDataset(
                        csvs=csv_path,
                        debug=DEBUG,
                    )
                    dataloader = self.get_dataloader(dataset, batch_size)
                    self.logger.info(f"{metric.dataset_name} dataset loaded. Total samples: {len(dataset)}")
                    dataloaders.append((dataloader, dataset, metric.dataset_name, metric.task_type))

        for dataloader, dataset, dataset_name in dataloaders:
            for metric_name, metric in self.metric_dict.items():
                if getattr(metric, 'is_general_generation', False):
                    continue
                if metric.dataset_name == dataset_name:
                    self.valid_metrics.add(metric_name)

        # Do prediction
        for dataloader, dataset, dataset_name, task_type in dataloaders:
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

                start_time = time.time()
                for metric_name, metric in self.metric_dict.items():
                    if getattr(metric, 'is_general_generation', False):
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

    @torch.no_grad()
    def eval_video(self, video_size, batch_size, save_path, save_base=None, ignore_max_size=False):

        i2v_target_size = {
            "256p": {"bucket_hw_base_size": 256, "bucket_hw_bucket_stride": 16},
            "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
            "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
            "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
        }

        args = self.args

        size = as_tuple(video_size)[0] if not ignore_max_size else 0
        if not self.score_models_loaded:
            self.load_score_models(size)

        if isinstance(batch_size, (list, tuple)):
            assert len(batch_size) == 1, f"Expected a single batch size, got {batch_size}"
            batch_size = batch_size[0]

        dataloaders = []

        use_GeneralGeneration = any([((getattr(metric, 'is_general_generation', False)) or is_valid_csv(metric.score_name)) and size <= metric.max_size
                                     for metric in self.metric_dict.values()])

        if use_GeneralGeneration:
            from hymm.constants import General_Generation_BENCH_DATA
            from hymm.dataset.data_kits.general_generation_bench import GeneralGenerationBenchDataset

            for metric in self.metric_dict.values():
                if getattr(metric, 'is_general_generation', False) or is_valid_csv(metric.score_name):
                    if os.path.exists(metric.score_name):
                        csv_path = [metric.score_name]
                    else:
                        csv_path = General_Generation_BENCH_DATA[metric.score_name]
                    dataset = GeneralGenerationBenchDataset(
                        csvs=csv_path,
                        debug=DEBUG,
                    )
                    dataloader = self.get_dataloader(dataset, batch_size)
                    self.logger.info(f"{metric.dataset_name} dataset loaded. Total samples: {len(dataset)}")
                    dataloaders.append((dataloader, dataset, metric.dataset_name, metric.task_type))

        for dataloader, dataset, dataset_name, task_type in dataloaders:
            for metric_name, metric in self.metric_dict.items():
                if getattr(metric, 'is_general_generation', False) or is_valid_csv(metric.score_name):
                    continue
                if metric.dataset_name == dataset_name:
                    self.valid_metrics.add(metric_name)

        # Do prediction
        for dataloader, dataset, dataset_name, task_type in dataloaders:
            total_batches = len(dataloader)
            self.logger.info(
                f"********************************** "
                f"Evaluation on {dataset_name}. Total batches: {total_batches} "
                f"**********************************"
            )

            if save_base is not None:
                if getattr(args, "flat_video_output", False):
                    save_dir = Path(save_base)
                elif '{}' in str(save_base):
                    save_dir = Path(str(save_base).format(dataset_name))
                else:
                    save_dir = Path(str(save_base) + f"/{dataset_name}")
                self.logger.info(f"{dataset_name} Images will be saved to {save_dir}")
            else:
                save_dir = None

            for batch_index, batch in enumerate(dataloader, start=1):
                self.logger.info(f"Batch {batch_index}/{total_batches}")
                if self.rank == 0:
                    self.logger.info(
                        f"[generation START] case={batch['ids'][0]} seed={batch['seeds'][0]} "
                        f"target_latent_T={batch.get('latent_target_lengths')} "
                        f"reference_spec={batch.get('condition_spec')}"
                    )

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

                if "ref_image_paths" in batch.keys():
                    ref_image_paths = batch["ref_image_paths"]
                else:
                    ref_image_paths = None

                # The sampler adds FPS after parsing shot metadata, matching
                # fixed-prompt validation. Adding it here creates a spurious
                # global text segment before the shot ranges are parsed.

                height, width = video_size[0], video_size[1]
                reorg_token = '888' in args.model_type or '16168' in args.model_type
                video_length = args.num_frames

                def _optional_int(values, name):
                    if values is None:
                        return None
                    assert isinstance(values, list) and len(values) == 1, (
                        f"{name} is per-sample metadata and requires val_batch_size_video=1, "
                        f"got {values}"
                    )
                    value = values[0]
                    if value is None or str(value).strip() == "":
                        return None
                    try:
                        return int(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{name} must be an integer, got {value}") from exc

                row_height = _optional_int(batch.get("heights"), "height")
                row_width = _optional_int(batch.get("widths"), "width")
                if row_height is not None or row_width is not None:
                    assert row_height is not None and row_width is not None, (
                        f"height/width must be provided together, got height={row_height}, width={row_width}"
                    )
                    height, width = row_height, row_width

                row_latent_target_length = _optional_int(
                    batch.get("latent_target_lengths"), "latent_target_length"
                )
                if row_latent_target_length is not None:
                    assert row_latent_target_length > 0, (
                        f"latent_target_length must be positive, got {row_latent_target_length}"
                    )
                    latent_target_length = row_latent_target_length
                    temporal_stride = 8 if reorg_token else 4
                    video_length = (latent_target_length - 1) * temporal_stride + 1
                else:
                    latent_target_length = (
                        (args.num_frames - 1) // 8 + 1
                        if reorg_token
                        else (args.num_frames - 1) // 4 + 1
                    )

                generator = torch.Generator(device=self.device).manual_seed(batch["seeds"][0])
                if task_type == "i2v":
                    assert len(ref_image_paths) == 1
                    multitask_mask, mask_type = get_multitask_mask(latent_target_length,
                                                                   {"i2v": {"prob": 1.0}, "t2v": {"prob": 0.0}})
                    bucket_hw_base_size = i2v_target_size[args.val_i2v_target_size]["bucket_hw_base_size"]
                    bucket_hw_bucket_stride = i2v_target_size[args.val_i2v_target_size]["bucket_hw_bucket_stride"]
                    ref_images = [Image.open(image_path).convert('RGB') for image_path in ref_image_paths if
                                  image_path is not None]
                    assert len(ref_images) == 1, "Only support one reference image for i2v task."
                    origin_size = ref_images[0].size
                    assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
                        f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"

                    crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
                    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
                    closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios,
                                                                    crop_size_list)

                    # 修正resize策略，避免黑边
                    target_height, target_width = closest_size
                    original_width, original_height = origin_size

                    # 计算保持宽高比的resize尺寸，确保能完全覆盖目标尺寸
                    scale_factor = max(target_width / original_width, target_height / original_height)
                    resize_width = int(round(original_width * scale_factor))
                    resize_height = int(round(original_height * scale_factor))

                    ref_image_transform = transforms.Compose([
                        transforms.Resize((resize_height, resize_width),
                                          interpolation=transforms.InterpolationMode.LANCZOS),
                        transforms.CenterCrop((target_height, target_width)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])

                    ref_images_pixel_values = [ref_image_transform(ref_image) for ref_image in ref_images]
                    ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        cond_latents = self.pipeline.vae.encode(
                            ref_images_pixel_values).latent_dist.mode()  # B, C, F, H, W
                        cond_latents.mul_(self.pipeline.vae.config.scaling_factor)

                    latents = cond_latents.repeat(1, 1, latent_target_length, 1, 1)
                    height = closest_size[0]
                    width = closest_size[1]

                    cond_latents, _, semantic_images_np, semantic_images_pil = get_cond_latents(args, latents, self.pipeline.vae,
                                                                                             is_uncond=False, is_mode=True)

                    out_dict = self.predict_video(
                        prompt=prompts,
                        height=height,  # 只能处理图片
                        width=width,  # 只能处理图片
                        video_length=video_length,
                        batch_size=len(prompts) if isinstance(prompts, list) else 1,
                        generator=generator,
                        flow_shift=args.infer_flow_shift_video,
                        num_images_per_prompt=args.num_images,
                        infer_steps=args.infer_steps_video,
                        negative_prompt=args.neg_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        guidance_scale=args.cfg_scale_video,
                        output_type="pt",  # If "pt", return torch tensor with range [0, 1] and shape (B, 3, T, H, W)
                        verbose=1,
                        multitask_mask_training_type=args.multitask_mask_training_type,
                        multitask_type=mask_type,
                        cond_latents=cond_latents,
                        multitask_mask=multitask_mask,
                        semantic_images=semantic_images_pil,
                        reference_image=semantic_images_np,
                        vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                        vision_states_dim=args.vision_states_dim,
                        bucket_hw_base_size=bucket_hw_base_size,
                        bucket_hw_bucket_stride=bucket_hw_bucket_stride,
                    )
                elif task_type == "fl2v":
                    multitask_mask, mask_type = get_multitask_mask(latent_target_length,
                                                                   {"i2v": {"prob": 0.0}, "t2v": {"prob": 0.0}, "fl2v": {"prob": 1.0}}) 
                    # encode first and last frame
                    first_frames = batch["firsts"]
                    last_frames = batch["lasts"]
                    cond_latents, target_height, target_width = encode_frames_to_latents(first_frames, self.pipeline.vae, args.val_i2v_target_size)
                    last_frame_cond_latents, _, _ = encode_frames_to_latents(last_frames, self.pipeline.vae, args.val_i2v_target_size,
                                                                             target_h=target_height, target_w=target_width)
                    
                    if cond_latents.size() != last_frame_cond_latents.size():
                        raise ValueError(f"cond_latents.size() != last_frame_cond_latents.size(): {cond_latents.size()} != {last_frame_cond_latents.size()}, \
                                          prompt: {prompts}, \
                                          first_frames: {first_frames}, \
                                          last_frames: {last_frames}")
                    out_dict = self.predict_video(
                        prompt=prompts,
                        height=target_height,  # 只能处理图片
                        width=target_width,  # 只能处理图片
                        video_length=video_length,
                        batch_size=len(prompts) if isinstance(prompts, list) else 1,
                        generator=generator,
                        flow_shift=args.infer_flow_shift_video,
                        num_images_per_prompt=args.num_images,
                        infer_steps=args.infer_steps_video,
                        negative_prompt=args.neg_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        guidance_scale=args.cfg_scale_video,
                        output_type="pt",  # If "pt", return torch tensor with range [0, 1] and shape (B, 3, T, H, W)
                        verbose=1,
                        multitask_mask_training_type=args.multitask_mask_training_type,
                        multitask_type=mask_type,
                        cond_latents=cond_latents,
                        last_frame_cond_latents=last_frame_cond_latents,
                        multitask_mask=multitask_mask,
                        semantic_images=None,
                        reference_image=None,
                        vision_num_semantic_tokens=None,
                        vision_states_dim=None,
                    )
                elif task_type == "t2v":
                    multitask_mask, mask_type = get_multitask_mask(latent_target_length,
                                                                   {"i2v": {"prob": 0.0}, "t2v": {"prob": 1.0}})
                    cond_latents = None
                    semantic_images_np = None
                    semantic_images_pil = None
                    out_dict = self.predict_video(
                        prompt=prompts,
                        height=height,  # 只能处理图片
                        width=width,  # 只能处理图片
                        video_length=video_length,
                        batch_size=len(prompts) if isinstance(prompts, list) else 1,
                        generator=generator,
                        flow_shift=args.infer_flow_shift_video,
                        num_images_per_prompt=args.num_images,
                        infer_steps=args.infer_steps_video,
                        negative_prompt=args.neg_prompt,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_attention_mask=negative_attention_mask,
                        guidance_scale=args.cfg_scale_video,
                        output_type="pt",  # If "pt", return torch tensor with range [0, 1] and shape (B, 3, T, H, W)
                        verbose=1,
                        multitask_mask_training_type=args.multitask_mask_training_type,
                        multitask_type=mask_type,
                        cond_latents=cond_latents,
                        multitask_mask=multitask_mask,
                        semantic_images=semantic_images_pil,
                        reference_image=semantic_images_np,
                        vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                        vision_states_dim=args.vision_states_dim,
                    )
                elif task_type == "range_condition":
                    # Unified PhysRoPE inference across L1/L2/L3/L4/L5.
                    # Per-row condition spec sources, in priority order:
                    #   (1) batch["condition_spec"] — JSON string with explicit
                    #       image_anchors + video_anchors lists. Highest priority.
                    #   (2) batch["ref_image_paths"] + args.range_condition_anchor_frames
                    #       (CSV string of int frame positions, e.g. "0" or "0,8,16").
                    #       Treats each frame position as an L1/L2 anchor, encoding the
                    #       same ref_image at each requested time.
                    #   (3) auto self-cond: t2v generate first, then re-condition on
                    #       the encoded sample at args.range_condition_anchor_frames.
                    #       Mirror of train.py fixed-prompt Mode B.
                    cond_spec_raw = batch.get("condition_spec", [None])[0] if isinstance(batch.get("condition_spec"), list) else batch.get("condition_spec")
                    cond_spec = None
                    if cond_spec_raw and isinstance(cond_spec_raw, str):
                        try:
                            cond_spec = json.loads(cond_spec_raw)
                        except Exception as e:
                            self.logger.warning(f"[range_condition] failed to parse condition_spec JSON: {e}")
                            cond_spec = None

                    # Use the per-case target resolution (from the eval JSON/CSV
                    # height/width), not the fixed --video-size, so the condition
                    # canvas matches the reference latents' native resolution.
                    cond_target_h, cond_target_w = height, width
                    native_t_concat = as_bool(
                        getattr(args, "range_condition_native_t_concat", False)
                    )
                    temporal_pack_mode = normalize_temporal_pack_mode(
                        getattr(args, "range_condition_temporal_pack_mode", "legacy_trilinear")
                    )
                    reference_timeline = temporal_pack_mode == REFERENCE_TIMELINE
                    model_in_channels = int(self.pipeline.transformer.config.in_channels)
                    reorg_token = '888' in args.model_type or '16168' in args.model_type
                    if reorg_token:
                        assert model_in_channels % 2 == 0, (
                            f"reorg range_condition expects even transformer input channels, got {model_in_channels}"
                        )
                        cond_dim_C = model_in_channels // 2
                    else:
                        cond_dim_C = model_in_channels
                    cond_lat_h = cond_target_h // self.pipeline.vae_scale_factor
                    cond_lat_w = cond_target_w // self.pipeline.vae_scale_factor

                    image_anchors = []
                    video_anchors = []
                    if cond_spec is not None:
                        image_anchors = list(cond_spec.get("image_anchors", []) or [])
                        video_anchors = list(cond_spec.get("video_anchors", []) or [])
                    else:
                        # Build from ref_image_paths + args.range_condition_anchor_frames
                        anchor_frames_str = getattr(args, "range_condition_anchor_frames", "0") or "0"
                        try:
                            anchor_frames = [int(s) for s in str(anchor_frames_str).split(",") if str(s).strip() != ""]
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"range_condition_anchor_frames must be comma-separated ints, got {anchor_frames_str}"
                            ) from exc
                        if ref_image_paths:
                            for path, t_anchor in zip(ref_image_paths, anchor_frames):
                                if path is None: continue
                                image_anchors.append({"path": path, "anchor_frame": int(t_anchor)})

                    # If still empty → auto-fallback to t2v + self-cond two-pass
                    do_two_pass = (not image_anchors) and (not video_anchors) and \
                        bool(getattr(args, "range_condition_self_cond_two_pass", False))
                    explicit_condition_spec = cond_spec is not None
                    requested_image_anchor_count = len(image_anchors)
                    requested_video_anchor_count = len(video_anchors)

                    if not image_anchors and not video_anchors and not do_two_pass:
                        # No anchors and not in two-pass mode → degenerate to t2v.
                        self.logger.warning("[range_condition] no anchors provided; falling back to t2v")
                        multitask_mask, mask_type = get_multitask_mask(latent_target_length,
                                                                       {"i2v": {"prob": 0.0}, "t2v": {"prob": 1.0}})
                        out_dict = self.predict_video(
                            prompt=prompts, height=height, width=width,
                            video_length=video_length,
                            batch_size=len(prompts) if isinstance(prompts, list) else 1,
                            generator=generator, flow_shift=args.infer_flow_shift_video,
                            num_images_per_prompt=args.num_images,
                            infer_steps=args.infer_steps_video, negative_prompt=args.neg_prompt,
                            prompt_embeds=prompt_embeds, attention_mask=attention_mask,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_attention_mask=negative_attention_mask,
                            guidance_scale=args.cfg_scale_video, output_type="pt", verbose=1,
                            multitask_mask_training_type=args.multitask_mask_training_type,
                            multitask_type=mask_type, cond_latents=None, multitask_mask=multitask_mask,
                            semantic_images=None, reference_image=None,
                            vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                            vision_states_dim=args.vision_states_dim,
                        )
                    else:
                        # Optional pass 1: t2v → encode → use as cond source for self-cond
                        self_cond_lat = None
                        if do_two_pass:
                            multitask_mask_p1, mask_type_p1 = get_multitask_mask(latent_target_length,
                                                                                  {"i2v": {"prob": 0.0}, "t2v": {"prob": 1.0}})
                            out_p1 = self.predict_video(
                                prompt=prompts, height=cond_target_h, width=cond_target_w,
                                video_length=video_length,
                                batch_size=len(prompts) if isinstance(prompts, list) else 1,
                                generator=generator, flow_shift=args.infer_flow_shift_video,
                                num_images_per_prompt=args.num_images,
                                infer_steps=args.infer_steps_video, negative_prompt=args.neg_prompt,
                                prompt_embeds=prompt_embeds, attention_mask=attention_mask,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_attention_mask=negative_attention_mask,
                                guidance_scale=args.cfg_scale_video, output_type="pt", verbose=1,
                                multitask_mask_training_type=args.multitask_mask_training_type,
                                multitask_type=mask_type_p1, cond_latents=None,
                                multitask_mask=multitask_mask_p1,
                                semantic_images=None, reference_image=None,
                                vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                                vision_states_dim=args.vision_states_dim,
                            )
                            try:
                                _gen = (out_p1["samples"].to(self.device, dtype=torch.float32) * 2.0 - 1.0).clamp(-1, 1)
                                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                    _gen_lat = self.pipeline.vae.encode(_gen).latent_dist.mode()
                                    if hasattr(self.pipeline.vae.config, 'shift_factor') and self.pipeline.vae.config.shift_factor:
                                        _gen_lat.sub_(self.pipeline.vae.config.shift_factor).mul_(self.pipeline.vae.config.scaling_factor)
                                    else:
                                        _gen_lat.mul_(self.pipeline.vae.config.scaling_factor)
                                self_cond_lat = _gen_lat
                                # Auto-emit single-image-anchor at center if no spec
                                if not image_anchors and not video_anchors:
                                    image_anchors.append({
                                        "self_cond_frame": int(latent_target_length // 2),
                                        "anchor_frame": int(latent_target_length // 2),
                                    })
                            except Exception as e:
                                self.logger.warning(f"[range_condition] two-pass cond encode failed: {e}")
                                self_cond_lat = None

                        # Build cond_latents tensor: [1, C, T, h, w] zero-padded
                        cond_latents = torch.zeros(
                            1, cond_dim_C, latent_target_length, cond_lat_h, cond_lat_w,
                            device=self.device, dtype=torch.float32,
                        )
                        condition_frame_ranges = []
                        condition_physical_dst_frame_ranges = []
                        condition_src_frame_ranges = []
                        condition_kinds = []
                        condition_seg_ids = []
                        native_condition_chunks = []
                        native_reference_packed_ranges = []
                        sparse_condition_mask = torch.zeros(
                            latent_target_length, device=self.device, dtype=torch.float32
                        )
                        sparse_mapping_debug = []
                        loaded_image_anchor_count = 0
                        loaded_video_anchor_count = 0

                        def _infer_range_overlaps(existing_ranges, start, end):
                            return any(max(int(start), int(rs)) < min(int(end), int(re))
                                       for rs, re in existing_ranges)

                        def _resolve_gt_latent_path(anchor):
                            gt_path = anchor.get("gt_latent_path", "") or anchor.get("gt_latent_cos_path", "")
                            if not gt_path:
                                return ""
                            if os.path.isabs(gt_path):
                                return gt_path
                            gt_cache = anchor.get("gt_latent_cache_folder", "")
                            if not gt_cache:
                                return gt_path
                            return os.path.join(gt_cache, gt_path)

                        # --- Encode image anchors at requested frames ---
                        for anc in image_anchors:
                            try:
                                t_anchor = int(anc.get("anchor_frame", 0))
                            except (TypeError, ValueError) as exc:
                                self.logger.warning(
                                    f"[range_condition SKIP] invalid image anchor frame "
                                    f"anchor={anc} exc={repr(exc)}"
                                )
                                continue
                            if t_anchor < 0 or t_anchor >= latent_target_length:
                                self.logger.warning(
                                    f"[range_condition SKIP] image anchor out of bounds "
                                    f"frame={t_anchor} latent_target_length={latent_target_length}"
                                )
                                continue
                            if _infer_range_overlaps(condition_frame_ranges, t_anchor, t_anchor + 1):
                                self.logger.warning(
                                    f"[range_condition SKIP] image anchor overlap "
                                    f"frame={t_anchor} existing={condition_frame_ranges}"
                                )
                                continue
                            native_anchor_latent = None
                            if anc.get('latent_path'):
                                # Stored unscaled slice of the original full-video
                                # encode, never a fresh single-PNG encoding.
                                try:
                                    if anc.get('latent_scale', 'unscaled_vae_mode') != 'unscaled_vae_mode':
                                        raise ValueError('Native anchor must use unscaled_vae_mode')
                                    _anchor = np.load(anc['latent_path'], allow_pickle=False)
                                    if _anchor.ndim == 5 and _anchor.shape[0] == 1:
                                        _anchor = _anchor[0]
                                    if _anchor.shape != (cond_dim_C, 1, cond_lat_h, cond_lat_w):
                                        raise ValueError(f'Native anchor shape {_anchor.shape} is not {(cond_dim_C, 1, cond_lat_h, cond_lat_w)}')
                                    native_anchor_latent = torch.from_numpy(_anchor).to(
                                        device=self.device, dtype=cond_latents.dtype).unsqueeze(0)
                                    native_anchor_latent = native_anchor_latent * float(self.pipeline.vae.config.scaling_factor)
                                    cond_latents[:, :, t_anchor:t_anchor+1] = native_anchor_latent
                                    self.logger.info(
                                        f"[reference NATIVE ANCHOR] reference_frame={4*int(anc.get('source_frame', t_anchor))} "
                                        f"reference_latent_frame={int(anc.get('source_frame', t_anchor))} "
                                        f"target_anchor_latent={t_anchor} source_control_frames=1 source_control_latents=1 "
                                        f"target_pixel_frames={video_length} generated_latent_T={latent_target_length} "
                                        f"path={anc['latent_path']}"
                                    )
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    self.logger.warning(f"[range_condition SKIP] native image latent failed path={anc['latent_path']} error={exc!r}")
                                    continue
                            elif "path" in anc and anc["path"]:
                                if not os.path.exists(anc["path"]):
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor path missing path={anc['path']}"
                                    )
                                    continue
                                try:
                                    _ri = Image.open(anc["path"]).convert('RGB')
                                    _ow, _oh = _ri.size
                                    _sf = max(cond_target_w / _ow, cond_target_h / _oh)
                                    _rw, _rh = int(round(_ow * _sf)), int(round(_oh * _sf))
                                    _tf = transforms.Compose([
                                        transforms.Resize((_rh, _rw), interpolation=transforms.InterpolationMode.LANCZOS),
                                        transforms.CenterCrop((cond_target_h, cond_target_w)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5]),
                                    ])
                                    _px = _tf(_ri).unsqueeze(0).unsqueeze(2).to(self.device)
                                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                                        _ila = self.pipeline.vae.encode(_px).latent_dist.mode().float()
                                        if hasattr(self.pipeline.vae.config, 'shift_factor') and self.pipeline.vae.config.shift_factor:
                                            _ila.sub_(self.pipeline.vae.config.shift_factor).mul_(self.pipeline.vae.config.scaling_factor)
                                        else:
                                            _ila.mul_(self.pipeline.vae.config.scaling_factor)
                                    if _ila.shape != (1, cond_dim_C, 1, cond_lat_h, cond_lat_w):
                                        self.logger.warning(
                                            f"[range_condition SKIP] image anchor latent shape mismatch "
                                            f"got={tuple(_ila.shape)} expected={(1, cond_dim_C, 1, cond_lat_h, cond_lat_w)} "
                                            f"path={anc['path']}"
                                        )
                                        continue
                                    native_anchor_latent = _ila.to(cond_latents.dtype)
                                    cond_latents[:, :, t_anchor:t_anchor+1] = native_anchor_latent
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor encode failed "
                                        f"path={anc['path']} exc={repr(exc)}"
                                    )
                                    continue
                            elif "self_cond_frame" in anc and self_cond_lat is not None:
                                try:
                                    src_t = int(anc["self_cond_frame"])
                                except (TypeError, ValueError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] invalid self_cond image frame "
                                        f"anchor={anc} exc={repr(exc)}"
                                    )
                                    continue
                                if src_t < 0 or src_t >= self_cond_lat.shape[2]:
                                    self.logger.warning(
                                        f"[range_condition SKIP] self_cond image frame out of bounds "
                                        f"src_t={src_t} self_cond_T={self_cond_lat.shape[2]}"
                                    )
                                    continue
                                native_anchor_latent = self_cond_lat[:, :, src_t:src_t+1].to(
                                    cond_latents.dtype
                                )
                                cond_latents[:, :, t_anchor:t_anchor+1] = native_anchor_latent
                            elif anc.get("gt_latent_path", "") or anc.get("gt_latent_cos_path", ""):
                                gt_path = _resolve_gt_latent_path(anc)
                                if not os.path.exists(gt_path):
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor gt latent missing path={gt_path}"
                                    )
                                    continue
                                try:
                                    src_t = int(anc.get("source_frame", anc.get("frame_idx", t_anchor)))
                                except (TypeError, ValueError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] invalid gt image source frame "
                                        f"anchor={anc} exc={repr(exc)}"
                                    )
                                    continue
                                try:
                                    _gt = np.load(gt_path)
                                except (OSError, ValueError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor gt latent load failed "
                                        f"path={gt_path} exc={repr(exc)}"
                                    )
                                    continue
                                if _gt.ndim == 5:
                                    if _gt.shape[0] != 1:
                                        self.logger.warning(
                                            f"[range_condition SKIP] image anchor gt latent batch dim must be 1, "
                                            f"got shape={tuple(_gt.shape)} path={gt_path}"
                                        )
                                        continue
                                    _gt = _gt[0]
                                if _gt.ndim != 4:
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor gt latent ndim must be 4 after squeeze, "
                                        f"got shape={tuple(_gt.shape)} path={gt_path}"
                                    )
                                    continue
                                if src_t < 0 or src_t >= _gt.shape[1]:
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor gt source frame out of bounds "
                                        f"src_t={src_t} gt_T={_gt.shape[1]} path={gt_path}"
                                    )
                                    continue
                                if _gt.shape[0] != cond_dim_C or _gt.shape[2] != cond_lat_h or _gt.shape[3] != cond_lat_w:
                                    self.logger.warning(
                                        f"[range_condition SKIP] image anchor gt latent shape mismatch "
                                        f"got={tuple(_gt.shape)} expected_C/H/W=({cond_dim_C},{cond_lat_h},{cond_lat_w}) "
                                        f"path={gt_path}"
                                    )
                                    continue
                                _gt_t = torch.from_numpy(_gt[:, src_t:src_t+1]).to(
                                    device=self.device, dtype=cond_latents.dtype
                                )
                                _gt_t = _gt_t * float(self.pipeline.vae.config.scaling_factor)
                                native_anchor_latent = _gt_t.unsqueeze(0)
                                cond_latents[:, :, t_anchor:t_anchor+1] = native_anchor_latent
                            else:
                                continue
                            condition_frame_ranges.append((t_anchor, t_anchor + 1))
                            condition_physical_dst_frame_ranges.append((t_anchor, t_anchor + 1))
                            # L1/L2 are zero-width so src == dst in latent units
                            condition_src_frame_ranges.append((t_anchor, t_anchor + 1))
                            condition_kinds.append(2)
                            condition_seg_ids.append(-1)
                            sparse_condition_mask[t_anchor] = 1.0
                            if native_t_concat:
                                native_condition_chunks.append(native_anchor_latent)
                            loaded_image_anchor_count += 1

                        # --- Encode video anchors (L3/L4) ---
                        def _parse_infer_src_range(value):
                            if isinstance(value, str):
                                text = value.strip()
                                if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
                                    text = text[1:-1]
                                sep = "," if "," in text else "-"
                                value = [item.strip() for item in text.split(sep) if item.strip()]
                            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                                return None
                            try:
                                ss_val = int(round(float(value[0])))
                                se_val = int(round(float(value[1])))
                            except (TypeError, ValueError):
                                return None
                            if se_val <= ss_val:
                                return None
                            return ss_val, se_val

                        for van in video_anchors:
                            try:
                                ds = int(van.get("dst_start", 0))
                                de = int(van.get("dst_end", 0))
                            except (TypeError, ValueError) as exc:
                                self.logger.warning(
                                    f"[range_condition SKIP] invalid video anchor dst "
                                    f"anchor={van} exc={repr(exc)}"
                                )
                                continue
                            if de <= ds or ds < 0 or ds >= latent_target_length:
                                self.logger.warning(
                                    f"[range_condition SKIP] invalid video anchor dst=({ds},{de}) "
                                    f"latent_target_length={latent_target_length}"
                                )
                                continue
                            dst_end = min(de, latent_target_length)
                            if _infer_range_overlaps(condition_frame_ranges, ds, dst_end):
                                self.logger.warning(
                                    f"[range_condition SKIP] video anchor overlap "
                                    f"physical_dst=({ds},{de}) visible_dst=({ds},{dst_end}) "
                                    f"existing={condition_frame_ranges}"
                                )
                                continue
                            lp = van.get("latent_path", "")
                            video_path = van.get("video_path", "") or van.get("path", "")
                            if van.get('require_native_latent') and (not lp or not os.path.isfile(lp)):
                                self.logger.warning(f"[range_condition SKIP] required native video latent missing path={lp}")
                                continue
                            if lp and os.path.exists(lp):
                                try:
                                    _seg = np.load(lp)
                                except (OSError, ValueError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor latent load failed "
                                        f"path={lp} exc={repr(exc)}"
                                    )
                                    continue
                                if _seg.ndim == 5: _seg = _seg[0]
                                if _seg.ndim != 4:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor latent ndim must be 4 after squeeze, "
                                        f"got shape={tuple(_seg.shape)} path={lp}"
                                    )
                                    continue
                                # Scale just like dataset path (segments stored unscaled)
                                try:
                                    _seg_t = torch.from_numpy(_seg).to(device=self.device, dtype=cond_latents.dtype)
                                except (TypeError, ValueError, RuntimeError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor latent tensor convert failed "
                                        f"shape={tuple(_seg.shape)} path={lp} exc={repr(exc)}"
                                    )
                                    continue
                                t_src = int(_seg_t.shape[1])
                                self.logger.info(
                                    f"[reference NATIVE VIDEO] reference_frame=0 source_control_frames={4*(t_src-1)+1} "
                                    f"source_control_latents={t_src} target_condition_latents={de-ds} "
                                    f"target_pixel_frames={video_length} generated_latent_T={latent_target_length} "
                                    f"dst_range=({ds},{de}) path={lp}"
                                )
                            elif video_path:
                                if not os.path.exists(video_path):
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor video path missing path={video_path}"
                                    )
                                    continue
                                dst_len_for_online = de - ds
                                explicit_src_for_online = van.get("src_range", van.get("src_latent_range", None))
                                parsed_src_for_online = _parse_infer_src_range(explicit_src_for_online)
                                if explicit_src_for_online is not None and parsed_src_for_online is None:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor src_range parse failed "
                                        f"src_range={explicit_src_for_online} path={video_path}"
                                    )
                                    continue
                                if parsed_src_for_online is not None:
                                    ss_for_online, se_for_online = parsed_src_for_online
                                    t_src_for_online = se_for_online - ss_for_online
                                else:
                                    t_src_for_online = int(van.get("src_latent_length", dst_len_for_online))
                                if t_src_for_online <= 0:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor invalid online source length "
                                        f"src_T={t_src_for_online} path={video_path}"
                                    )
                                    continue
                                pixel_frame_count = 4 * (t_src_for_online - 1) + 1
                                try:
                                    import cv2

                                    cap = cv2.VideoCapture(video_path)
                                    if not cap.isOpened():
                                        raise RuntimeError(f"cv2 failed to open video_path={video_path}")
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    if frame_count <= 0:
                                        raise RuntimeError(f"empty video_path={video_path}")
                                    # The released MP4 is the selected reference timeline.
                                    # Encode its native frames without stretching to the target.
                                    if (frame_count - 1) % 4:
                                        raise ValueError(f"Reference must have 4k+1 native frames, got {frame_count}")
                                    native_t = (frame_count - 1) // 4 + 1
                                    if parsed_src_for_online is not None and native_t != t_src_for_online:
                                        self.logger.warning(
                                            f"[reference METADATA] declared_src_T={t_src_for_online} "
                                            f"actual_native_T={native_t}; preserve every native frame and original start; "
                                            f"source endpoint follows actual media path={video_path}"
                                        )
                                    t_src_for_online = native_t
                                    pixel_frame_count = frame_count
                                    indices = range(frame_count)
                                    self.logger.info(
                                        f"[reference RGB] reference_frame=0 source_control_frames={frame_count} "
                                        f"source_control_latents={native_t} target_condition_latents={dst_len_for_online} "
                                        f"target_pixel_frames={video_length} generated_latent_T={latent_target_length} "
                                        f"dst_range=({ds},{de}) src_range={parsed_src_for_online} path={video_path}"
                                    )
                                    frames = []
                                    for frame_idx in indices:
                                        ok, frame_bgr = cap.read()
                                        if not ok or frame_bgr is None:
                                            raise RuntimeError(
                                                f"failed to read frame={int(frame_idx)} "
                                                f"from video_path={video_path}"
                                            )
                                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                        src_h, src_w = frame_rgb.shape[:2]
                                        scale = max(cond_target_w / src_w, cond_target_h / src_h)
                                        resize_w = int(round(src_w * scale))
                                        resize_h = int(round(src_h * scale))
                                        frame_rgb = cv2.resize(
                                            frame_rgb,
                                            (resize_w, resize_h),
                                            interpolation=cv2.INTER_AREA,
                                        )
                                        x0 = max((resize_w - cond_target_w) // 2, 0)
                                        y0 = max((resize_h - cond_target_h) // 2, 0)
                                        frames.append(frame_rgb[y0:y0 + cond_target_h, x0:x0 + cond_target_w])
                                    cap.release()
                                    frames_np = np.stack(frames, axis=0)
                                    pixel = torch.from_numpy(frames_np).permute(3, 0, 1, 2).unsqueeze(0)
                                    pixel = pixel.to(device=self.device, dtype=self.pipeline.vae.dtype)
                                    pixel = pixel / 127.5 - 1.0
                                    with torch.no_grad(), torch.autocast(
                                        device_type="cuda",
                                        dtype=self.pipeline.vae.dtype,
                                        enabled=(self.device.type == "cuda"),
                                    ):
                                        _encoded = self.pipeline.vae.encode(pixel).latent_dist.mode()
                                    if _encoded.ndim != 5 or _encoded.shape[0] != 1:
                                        raise RuntimeError(
                                            f"online encoded latent must be [1,C,T,H,W], "
                                            f"got={tuple(_encoded.shape)} path={video_path}"
                                        )
                                    _seg_t = _encoded[0].to(device=self.device, dtype=cond_latents.dtype)
                                    t_src = int(_seg_t.shape[1])
                                    if t_src != t_src_for_online:
                                        raise RuntimeError(
                                            f"online encoded latent T mismatch: got={t_src}, "
                                            f"requested={t_src_for_online}, "
                                            f"pixel_frames={pixel_frame_count}, path={video_path}"
                                        )
                                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor online VAE encode failed "
                                        f"path={video_path} src_T={t_src_for_online} "
                                        f"pixel_frames={pixel_frame_count} exc={repr(exc)}"
                                    )
                                    continue
                            elif lp:
                                self.logger.warning(
                                    f"[range_condition SKIP] video anchor latent path missing path={lp}"
                                )
                                continue
                            if lp or video_path:
                                if _seg_t.shape[0] != cond_dim_C or _seg_t.shape[2] != cond_lat_h or _seg_t.shape[3] != cond_lat_w or t_src <= 0:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor latent C/H/W/T mismatch "
                                        f"got={tuple(_seg_t.shape)} expected_C/H/W=({cond_dim_C},{cond_lat_h},{cond_lat_w}) "
                                        f"path={lp or video_path}"
                                    )
                                    continue
                                dst_len = de - ds
                                visible_len = dst_end - ds
                                explicit_src = van.get("src_range", van.get("src_latent_range", None))
                                parsed_src = _parse_infer_src_range(explicit_src)
                                if explicit_src is not None and parsed_src is None:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor src_range parse failed "
                                        f"src_range={explicit_src} path={lp or video_path}"
                                    )
                                    continue
                                if parsed_src is not None:
                                    ss, se = parsed_src
                                else:
                                    ss = int(van.get("src_start", ds))
                                    se = int(van.get("src_end", ds + t_src))
                                if se <= ss:
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor invalid src range src=({ss},{se}) path={lp or video_path}"
                                    )
                                    continue
                                if se - ss != t_src:
                                    # Trust the actual reference latent over the (systematically
                                    # off-by-one for the unique96 split) src_range metadata in the
                                    # eval JSON. The full [C,t_src,H,W] latent is always copied
                                    # verbatim and both the sparse scatter and the legacy resample
                                    # key off t_src, not off se-ss; (ss,se) is only bookkeeping fed
                                    # to the PhysRoPE source mapping. Realign se to the real frame
                                    # count instead of dropping the segment: a dropped segment can
                                    # zero out an explicit-condition_spec case's anchors and trip the
                                    # all-anchors-loaded assertion below, crashing that rank and
                                    # dead-locking every other rank on the next FSDP all-gather.
                                    self.logger.warning(
                                        f"[range_condition FIX] video anchor src_range metadata "
                                        f"src=({ss},{se}) len={se - ss} != actual latent T={t_src}; "
                                        f"realigning se={ss + t_src} path={lp or video_path}"
                                    )
                                    se = ss + t_src
                                if (not native_t_concat) and (not reference_timeline) and t_src != dst_len:
                                    try:
                                        _seg_t = resample_latent_temporal(
                                            _seg_t,
                                            src_frames=t_src,
                                            dst_frames=dst_len,
                                            align_corners=False,
                                            mode="trilinear",
                                        )
                                    except (ValueError, RuntimeError) as exc:
                                        self.logger.warning(
                                            f"[range_condition SKIP] video anchor temporal resample failed "
                                            f"src_T={t_src} dst_T={dst_len} path={lp or video_path} exc={repr(exc)}"
                                        )
                                        continue
                                expected_t = t_src if (native_t_concat or reference_timeline) else dst_len
                                if _seg_t.shape != (cond_dim_C, expected_t, cond_lat_h, cond_lat_w):
                                    self.logger.warning(
                                        f"[range_condition SKIP] video anchor packed shape mismatch "
                                        f"got={tuple(_seg_t.shape)} expected={(cond_dim_C, expected_t, cond_lat_h, cond_lat_w)} "
                                        f"path={lp or video_path}"
                                    )
                                    continue
                                _seg_t = _seg_t * float(self.pipeline.vae.config.scaling_factor)
                                if native_t_concat:
                                    native_condition_chunks.append(_seg_t.unsqueeze(0))
                                elif reference_timeline:
                                    canvas, sparse_mask, sparse_mapping = scatter_reference_timeline_latent(
                                        _seg_t,
                                        target_t=latent_target_length,
                                        physical_dst_range=(ds, de),
                                        visible_dst_range=(ds, dst_end),
                                    )
                                    cond_latents[0] = cond_latents[0] + canvas
                                    sparse_condition_mask = torch.maximum(
                                        sparse_condition_mask, sparse_mask
                                    )
                                    sparse_mapping_debug.append(sparse_mapping)
                                else:
                                    _seg_t = _seg_t[:, :visible_len]
                                    cond_latents[:, :, ds:dst_end] = _seg_t
                            elif self_cond_lat is not None:
                                try:
                                    ss = int(van.get("src_start", ds))
                                    se = int(van.get("src_end", de))
                                except (TypeError, ValueError) as exc:
                                    self.logger.warning(
                                        f"[range_condition SKIP] invalid self_cond video src "
                                        f"anchor={van} exc={repr(exc)}"
                                    )
                                    continue
                                if ss < 0 or se <= ss or se > self_cond_lat.shape[2]:
                                    self.logger.warning(
                                        f"[range_condition SKIP] self_cond video src out of bounds "
                                        f"src=({ss},{se}) self_cond_T={self_cond_lat.shape[2]}"
                                    )
                                    continue
                                dst_len = de - ds
                                visible_len = dst_end - ds
                                src_len = se - ss
                                _self_seg = self_cond_lat[:, :, ss:se].to(cond_latents.dtype)
                                if (not native_t_concat) and (not reference_timeline) and src_len != dst_len:
                                    try:
                                        _self_seg = resample_latent_temporal(
                                            _self_seg,
                                            src_frames=src_len,
                                            dst_frames=dst_len,
                                            align_corners=False,
                                            mode="trilinear",
                                        )
                                    except (ValueError, RuntimeError) as exc:
                                        self.logger.warning(
                                            f"[range_condition SKIP] self_cond video temporal resample failed "
                                            f"src_T={src_len} dst_T={dst_len} exc={repr(exc)}"
                                        )
                                        continue
                                if native_t_concat:
                                    native_condition_chunks.append(_self_seg)
                                elif reference_timeline:
                                    canvas, sparse_mask, sparse_mapping = scatter_reference_timeline_latent(
                                        _self_seg[0],
                                        target_t=latent_target_length,
                                        physical_dst_range=(ds, de),
                                        visible_dst_range=(ds, dst_end),
                                    )
                                    cond_latents[0] = cond_latents[0] + canvas
                                    sparse_condition_mask = torch.maximum(
                                        sparse_condition_mask, sparse_mask
                                    )
                                    sparse_mapping_debug.append(sparse_mapping)
                                else:
                                    _self_seg = _self_seg[:, :, :visible_len]
                                    if _self_seg.shape != cond_latents[:, :, ds:dst_end].shape:
                                        self.logger.warning(
                                            f"[range_condition SKIP] self_cond video packed shape mismatch "
                                            f"got={tuple(_self_seg.shape)} expected={tuple(cond_latents[:, :, ds:dst_end].shape)}"
                                        )
                                        continue
                                    cond_latents[:, :, ds:dst_end] = _self_seg
                            else:
                                self.logger.warning(
                                    f"[range_condition SKIP] video anchor missing latent_path and self_cond_lat dst=({ds},{de})"
                                )
                                continue
                            condition_frame_ranges.append(
                                (ds, dst_end) if native_t_concat else (ds, de)
                            )
                            condition_physical_dst_frame_ranges.append((ds, de))
                            condition_src_frame_ranges.append((ss, se))
                            condition_kinds.append(1)
                            condition_seg_ids.append(int(van.get("seg_id", -1)))
                            loaded_video_anchor_count += 1

                        if explicit_condition_spec and getattr(args, 'strict_reference_inputs', False):
                            complete = (
                                loaded_image_anchor_count == requested_image_anchor_count
                                and loaded_video_anchor_count == requested_video_anchor_count
                            )
                            valid_on_all_ranks = torch.tensor(int(complete), device=self.device)
                            dist.all_reduce(valid_on_all_ranks, op=dist.ReduceOp.MIN)
                            if not valid_on_all_ranks.item():
                                reason = {
                                    'case': batch['ids'][0],
                                    'reason': 'reference_load_failed_on_one_or_more_ranks; see reference SKIP logs',
                                    'loaded_images': loaded_image_anchor_count,
                                    'requested_images': requested_image_anchor_count,
                                    'loaded_videos': loaded_video_anchor_count,
                                    'requested_videos': requested_video_anchor_count,
                                    'target_pixel_frames': video_length,
                                    'target_latent_T': latent_target_length,
                                    'reference_spec': cond_spec,
                                }
                                self.logger.error(f"[generation SKIP] {reason}")
                                if self.rank == 0:
                                    Path(save_base).mkdir(parents=True, exist_ok=True)
                                    with (Path(save_base) / 'bad.jsonl').open('a') as record:
                                        record.write(json.dumps(reason) + '\n')
                                continue
                        if explicit_condition_spec:
                            # Do NOT hard-assert here. A single case whose anchors fail
                            # to load must not raise on one rank while the other ranks
                            # proceed into predict_video's per-layer FSDP all-gathers —
                            # that desyncs the collective count and hangs the whole job
                            # until the NCCL watchdog fires. Instead warn and degrade:
                            # whatever loaded is used, and a fully-empty case falls through
                            # to the t2v branch below (same single predict_video call, so
                            # the all-gather count stays identical across ranks).
                            if loaded_image_anchor_count != requested_image_anchor_count:
                                self.logger.warning(
                                    f"[range_condition DEGRADE] image anchor load mismatch: "
                                    f"loaded={loaded_image_anchor_count} "
                                    f"requested={requested_image_anchor_count} spec={cond_spec}"
                                )
                            if loaded_video_anchor_count != requested_video_anchor_count:
                                self.logger.warning(
                                    f"[range_condition DEGRADE] video anchor load mismatch: "
                                    f"loaded={loaded_video_anchor_count} "
                                    f"requested={requested_video_anchor_count} spec={cond_spec}"
                                )

                        # Sort by dst start, keep src in lock-step
                        if condition_frame_ranges:
                            order = sorted(range(len(condition_frame_ranges)),
                                           key=lambda i: condition_frame_ranges[i][0])
                            condition_frame_ranges = [condition_frame_ranges[i] for i in order]
                            condition_physical_dst_frame_ranges = [
                                condition_physical_dst_frame_ranges[i] for i in order
                            ]
                            condition_src_frame_ranges = [condition_src_frame_ranges[i] for i in order]
                            condition_kinds = [condition_kinds[i] for i in order]
                            condition_seg_ids = [condition_seg_ids[i] for i in order]
                            if native_t_concat:
                                if len(native_condition_chunks) != len(order):
                                    raise RuntimeError(
                                        "native reference chunks must match loaded condition ranges: "
                                        f"chunks={len(native_condition_chunks)} ranges={len(order)}"
                                    )
                                native_condition_chunks = [native_condition_chunks[i] for i in order]
                                native_cursor = 0
                                for chunk in native_condition_chunks:
                                    chunk_t = int(chunk.shape[2])
                                    native_reference_packed_ranges.append(
                                        (native_cursor, native_cursor + chunk_t)
                                    )
                                    native_cursor += chunk_t
                                cond_latents = torch.cat(native_condition_chunks, dim=2)
                            multitask_mask = (
                                sparse_condition_mask
                                if reference_timeline
                                else get_multitask_mask_with_ranges(
                                    latent_target_length,
                                    condition_frame_ranges,
                                    device=self.device,
                                )
                            )
                            cond_frame_count = int(multitask_mask.sum().item())
                            if (
                                cond_frame_count >= latent_target_length
                                and not native_t_concat
                                and not reference_timeline
                            ):
                                self.logger.warning(
                                    f"[range_condition] condition union covers all frames; "
                                    f"degenerating to t2v. cond_frames={cond_frame_count} "
                                    f"latent_target_length={latent_target_length} ranges={condition_frame_ranges}"
                                )
                                condition_frame_ranges = []
                                condition_src_frame_ranges = []

                        if condition_frame_ranges:
                            mask_type = "range_condition"
                            l4_active = any(d != s for d, s in zip(condition_frame_ranges, condition_src_frame_ranges))
                            source_lengths = [
                                int(end) - int(start)
                                for start, end in condition_src_frame_ranges
                            ]
                            target_lengths = [
                                int(end) - int(start)
                                for start, end in condition_physical_dst_frame_ranges
                            ]
                            self.logger.info(
                                f"[range_condition INFER] dst_ranges={condition_frame_ranges} "
                                f"src_ranges={condition_src_frame_ranges} l4={'ON' if l4_active else 'OFF'} "
                                f"source_control_lengths={source_lengths} "
                                f"target_condition_lengths={target_lengths} "
                                f"mask_sum={int(multitask_mask.sum().item())} "
                                f"native_t_concat={native_t_concat} "
                                f"temporal_pack_mode={temporal_pack_mode} "
                                f"sparse_fixed_target_clock={reference_timeline} "
                                f"sparse_mappings={sparse_mapping_debug} "
                                f"reference_T={int(cond_latents.shape[2])} "
                                f"target_T={latent_target_length} "
                                f"native_ranges={native_reference_packed_ranges} "
                                f"two_pass={do_two_pass}"
                            )
                            out_dict = self.predict_video(
                                prompt=prompts, height=cond_target_h, width=cond_target_w,
                                video_length=video_length,
                                batch_size=len(prompts) if isinstance(prompts, list) else 1,
                                generator=generator, flow_shift=args.infer_flow_shift_video,
                                num_images_per_prompt=args.num_images,
                                infer_steps=args.infer_steps_video, negative_prompt=args.neg_prompt,
                                prompt_embeds=prompt_embeds, attention_mask=attention_mask,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_attention_mask=negative_attention_mask,
                                guidance_scale=args.cfg_scale_video, output_type="pt", verbose=1,
                                multitask_mask_training_type=args.multitask_mask_training_type,
                                multitask_type="range_condition",
                                cond_latents=cond_latents,
                                multitask_mask=multitask_mask,
                                condition_frame_ranges=condition_frame_ranges,
                                condition_physical_dst_frame_ranges=condition_physical_dst_frame_ranges,
                                condition_src_frame_ranges=condition_src_frame_ranges,
                                condition_kinds=condition_kinds,
                                condition_seg_ids=condition_seg_ids,
                                range_condition_temporal_pack_mode=temporal_pack_mode,
                                native_reference_packed_ranges=native_reference_packed_ranges,
                                semantic_images=None, reference_image=None,
                                vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                                vision_states_dim=args.vision_states_dim,
                            )
                        else:
                            self.logger.warning("[range_condition] all anchors invalid; degenerating to t2v")
                            multitask_mask, mask_type = get_multitask_mask(latent_target_length,
                                                                           {"i2v": {"prob": 0.0}, "t2v": {"prob": 1.0}})
                            out_dict = self.predict_video(
                                prompt=prompts, height=height, width=width,
                                video_length=video_length,
                                batch_size=len(prompts) if isinstance(prompts, list) else 1,
                                generator=generator, flow_shift=args.infer_flow_shift_video,
                                num_images_per_prompt=args.num_images,
                                infer_steps=args.infer_steps_video, negative_prompt=args.neg_prompt,
                                prompt_embeds=prompt_embeds, attention_mask=attention_mask,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_attention_mask=negative_attention_mask,
                                guidance_scale=args.cfg_scale_video, output_type="pt", verbose=1,
                                multitask_mask_training_type=args.multitask_mask_training_type,
                                multitask_type=mask_type, cond_latents=None, multitask_mask=multitask_mask,
                                semantic_images=None, reference_image=None,
                                vision_num_semantic_tokens=args.vision_num_semantic_tokens,
                                vision_states_dim=args.vision_states_dim,
                            )

                samples = out_dict['samples'].float().squeeze(2)  # 去除时间维度
                proc_inputs = batch
                proc_inputs["images"] = samples.cuda()  # pipeline 输出结果为cpu tensor

                start_time = time.time()
                for metric_name, metric in self.metric_dict.items():
                    if getattr(metric, 'is_general_generation', False) or is_valid_csv(metric.score_name):
                        continue
                    if metric.dataset_name == dataset_name:
                        metric.process(**proc_inputs)
                gen_time = time.time() - start_time
                self.logger.info(f"Process time: {gen_time}")

                if save_dir is not None and self.rank % self.sp_size == 0:
                    save_names = [save_dir / f'{id_}.mp4' for id_ in batch['ids']]
                    save_videos_grid(out_dict['samples'], save_names[0], n_rows=1, fps=args.video_fps)
                    self.logger.info(
                        f"[generation COMPLETE] case={batch['ids'][0]} seed={batch['seeds'][0]} "
                        f"reference_spec={batch.get('condition_spec')} "
                        f"target_pixel_frames={video_length} generated_pixel_frames={out_dict['samples'].shape[2]} "
                        f"generated_latent_T={latent_target_length} height={height} width={width}"
                    )
                    if getattr(args, "inference_save_tensors", False):
                        torch.save(out_dict['samples'].cpu(), save_dir / f"{batch['ids'][0]}.pt")
                    with (Path(save_base) / 'completed.jsonl').open('a') as record:
                        record.write(json.dumps({
                            'case': batch['ids'][0], 'seed': batch['seeds'][0],
                            'video': str(save_names[0]),
                            'target_pixel_frames': video_length,
                            'generated_pixel_frames': int(out_dict['samples'].shape[2]),
                            'generated_latent_shape': getattr(self.pipeline, 'last_generated_latent_shape', None),
                            'reference_spec': batch.get('condition_spec'),
                        }) + '\n')
            # add barrier to wait for all rank finish
            dist.barrier()
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

            os.makedirs(save_base, exist_ok=True)
            if not os.path.exists(f"{save_base}/eval_exp_info.json"):
                eval_dict = {"val_yaml_path": args.val_yaml_path, "val_exp_name": args.val_exp_name}
                with open(f"{save_base}/eval_exp_info.json", 'w') as f:
                    json.dump(eval_dict, f, indent=2)

            if not os.path.exists(f"{save_base}/eval.yaml") and os.path.exists(args.val_yaml_path):
                shutil.copy2(args.val_yaml_path, f"{save_base}/eval.yaml")

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

def main(mode):
    args = parse_args(mode)

    parallel_dims, world_mesh, device = initialize_distributed(args)
    evaluator = ScoreEvaluator.from_pretrained(args.dit_weight, args.pretrained_model_name_or_path, args=args,
                                               device=device, parallel_dims=parallel_dims, world_mesh=world_mesh)
    evaluator.initialize_scores(args.scores)
    # Get the updated args
    args = evaluator.args

    val_save_path = evaluator.get_metric_save_path('score', logger)

    save_base = evaluator.get_sample_save_dir(
        testset=None, image_size=args.video_size) if args.score_save_image else None
    save_base = save_base.parent.parent / f"offline_eval_{args.num_frames}f" / save_base.name

    evaluator.eval_video(video_size=args.video_size,
                         batch_size=args.val_batch_size_video,
                         save_path=val_save_path,
                         save_base=save_base,
                         ignore_max_size=True,  # dataset.max_size is used for val-on-train. Ignore it for pure evaluation.
                        )
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
