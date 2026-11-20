from pathlib import Path
import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import argparse
import torch
import numpy as np
import os
import shutil
import json

from hymm.metrics_video.fvd.videogpt.fvd import frechet_distance
from hymm.metrics_video.fid.fid_score import calculate_frechet_distance
from scipy.stats import entropy
import pickle

import logging
import time
import copy

try:
    from rich.console import Console
    from rich.logging import RichHandler
except:
    import pip
    pip.main(["install", "rich"])
    from rich.console import Console
    from rich.logging import RichHandler

import torch.distributed as dist
from tqdm.auto import tqdm
import hymm.metrics_video.clip_score.clip as clip
from hymm.metrics_video.dataset import VideoDataset
from hymm.metrics_video.video_evaluator import VideoEvaluator
from hymm.utils.file_utils import dump_codes

GREEN = '\033[32m'
RESET = '\033[0m'

category_dict = {"human":
                     ['human_walk', 'human_jump', 'human_swim', 'human_photographing', 'human_bike', 'human_horse',
                      'human_ball', 'human_fight', 'human_gym', 'human_dance', 'human_clean', 'human_rest',
                      'human_gymnastics', 'human_building', 'human_hug', 'human_instruments', 'human_handshake',
                      'human_watering', 'human_typing', 'human_planting', 'human_cooking', 'human_reading',
                      'human_writing', 'human_clapping', 'human_cutting', 'human_repairing', 'human_digging',
                      'human_arrow', 'human_shooting', 'human_speaking', 'human_eating', 'human_nodding',
                      'human_kissing', 'human_singing', 'human_expression'],
                 "animal":
                     ['animal_bird', 'animal_reptile', 'animal_insect', 'animal_mammal', 'animal_fish'],
                 "botany":
                     ['botany_grass', 'botany_flower', 'botany_tree', 'botany_vegetable', 'botany_fruit',
                      'botany_agri', 'botany_garden', 'botany_forest'],
                 "landscape":
                     ['landscape_landscape', 'landscape_weather'],
                 "event":
                     ['event_outdoor', 'event_indoor', 'event_sports', 'event_concert', 'event_festival',
                      'event_museum'],
                 "vehicle":
                     ['vehicle_fp', 'vehicle_car_motion', 'vehicle_flight_motion', 'vehicle_car', 'vehicle_motorbike',
                      'vehicle_bike', 'vehicle_train', 'vehicle_ship', 'vehicle_uav'],
                 "goods":
                     ['goods_makeup', 'goods_equipment', 'goods_baby', 'goods_luxury', 'goods_art',
                      'goods_electronics', 'goods_public', 'goods_daily', 'goods_furniture', 'goods_office',
                      'goods_sports', 'goods_military', 'goods_manufacturing', 'goods_energy'],
                 "acg":
                     ['acg_comic']
                 }

dataset_dict = {"CollectName32_256p256p65f": {
                   "arrow": "assets/local/resource_60e5fd3e7b.arrow",
                   "caption": "name",
                   "gt_fvd": "assets/local/resource_d577dfd6b6.npy",
                   "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "CollectName2500_256p256p65f": {
                    "arrow": "assets/local/resource_d86dd92f85.arrow",
                    "caption": "name",
                    "gt_fvd": "assets/local/resource_4f0addb884.npy",
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "CollectShort2500_256p256p65f": {
                    "arrow": "assets/local/resource_d86dd92f85.arrow",
                    "caption": "assets/local/resource_5715051ba8.arrow",
                    "gt_fvd": "assets/local/resource_4f0addb884.npy",
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "CollectLong2500_256p256p65f": {
                    "arrow": "assets/local/resource_d86dd92f85.arrow",
                    "caption": "assets/local/resource_4e4b53c38a.arrow",
                    "gt_fvd": "assets/local/resource_4f0addb884.npy",
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "CollectShort101_256p256p65f": {
                    "arrow": "assets/local/resource_23b7dbaf9b.arrow",
                    "caption": "assets/local/resource_a4485280eb.arrow",
                    "gt_fvd": "assets/local/resource_3fa781d2fb.npy",
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "CollectLong101_256p256p65f": {
                    "arrow": "assets/local/resource_23b7dbaf9b.arrow",
                    "caption": "assets/local/resource_1c5265f888.arrow",
                    "gt_fvd": "assets/local/resource_3fa781d2fb.npy",
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "Vbench1_256p256p65f": {
                    "json": "assets/local/resource_0f325f2c77.json",
                    "num": 1,
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "Vbench5_256p256p65f": {
                    "json": "assets/local/resource_0f325f2c77.json",
                    "num": 5,
                    "resolution_h": 256, "resolution_w": 256, "video_length": 65},
                "std1000_192p336p65f": {
                    "csv": "assets/local/resource_48b29cf373.csv",
                    "resolution_h": 192, "resolution_w": 336, "video_length": 65, "col_name": "prompt"},
                 "std100_192p336p65f": {
                    "csv": "assets/local/resource_b349da1e20.csv",
                    "resolution_h": 192, "resolution_w": 336, "video_length": 65, "col_name": "prompt"},
                }

def str2bool(s):
    if isinstance(s, bool):
        return s
    elif s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected !')

def str2null(s):
    if s.lower() == 'none' or s.lower() == 'None' or s.lower() == 'null':
        return None
    else:
        return s

def int2null(s):

    if s.lower() == 'none' or s.lower() == 'None' or s.lower() == 'null':
        return None
    else:
        return int(s)

def get_save_list(args):
    if args.evaluator_type == "fvd" or args.evaluator_type == "fid":
        if args.is_gt:
            save_list = [[], [], []]
        else:
            save_list = [[], []]
    elif args.evaluator_type == "clip":
        save_list = [[], []]
    elif args.evaluator_type == "inception":
        save_list = [[], []]

    return save_list

def video_evaluator_forward(args, video_evaluator, batch, save_list):

    idx = batch["idx"]
    frames = batch["frames"]

    if args.evaluator_type == "fvd" or args.evaluator_type == "fid":
        if args.evaluator_type == "fid":
            video_evaluator.module.fid_model.eval()
        output = video_evaluator(frames)
        save_list[0].append(idx.cpu())
        save_list[1].append(output.cpu())
        if args.is_gt:
            save_list[2].append(batch["sample_index"].cpu())
    elif args.evaluator_type == "clip":
        text_emb = batch["text_emb"]
        output = video_evaluator(frames, text_emb=text_emb)
        save_list[0].append(idx.cpu())
        save_list[1].append(output.cpu())
    elif args.evaluator_type == "inception":
        output = video_evaluator(frames)
        save_list[0].append(idx.cpu())
        save_list[1].append(output.cpu())

    return save_list

def save_func(args, save_list, global_rank, local_rank):

    if args.evaluator_type == "fvd" or args.evaluator_type == "fid":
        save_np(save_list[0], save_list[1], args.save_dir, global_rank, local_rank)
        if args.is_gt:
            save_np(save_list[0], save_list[2], args.save_sample_index_dir, global_rank, local_rank)
    elif args.evaluator_type == "clip":
        save_np(save_list[0], save_list[1], args.save_dir, global_rank, local_rank)
    elif args.evaluator_type == "inception":
        save_np(save_list[0], save_list[1], args.save_dir, global_rank, local_rank)

def save_np(keys_list, values_list, save_dir, global_rank, local_rank):
    save_dict = {"keys": torch.concat(keys_list, dim=0).numpy(),
                 "values": torch.concat(values_list, dim=0).numpy()}
    np.save(f"{save_dir}/{global_rank}_{local_rank}.npy", save_dict)

def reorganize_np(base_dir, keys_name, values_name, dataset_num):
    names = os.listdir(base_dir)

    keys_list = []
    values_list = []
    for name in names:
        load_path = f"{base_dir}/{name}"
        dictionary = np.load(load_path, allow_pickle=True).item()
        keys_list.append(dictionary["keys"])
        values_list.append(dictionary["values"])

    keys_np = np.concatenate(keys_list, axis=0)
    values_np = np.concatenate(values_list, axis=0)
    sorted_rank = np.argsort(keys_np)
    sorted_keys_np = keys_np[sorted_rank]
    sorted_values_np = values_np[sorted_rank]
    if dataset_num is not None:
        sorted_keys_np = sorted_keys_np[:dataset_num]
        sorted_values_np = sorted_values_np[:dataset_num]

    save_dict = {keys_name: sorted_keys_np,
                 values_name: sorted_values_np}
    save_path = f"{'/'.join(base_dir.split('/')[:-1])}/total_info.npy"
    np.save(save_path, save_dict)

    shutil.rmtree(f"{base_dir}")
    return save_path

def reorganize_np_split(base_dir, keys_name, values_name, dataset_num, split_per_num=2500):
    names = os.listdir(base_dir)

    keys_list = []
    values_list = []
    for name in names:
        load_path = f"{base_dir}/{name}"
        dictionary = np.load(load_path, allow_pickle=True).item()
        keys_list.append(dictionary["keys"])
        values_list.append(dictionary["values"])

    keys_np = np.concatenate(keys_list, axis=0)
    values_np = np.concatenate(values_list, axis=0)
    sorted_rank = np.argsort(keys_np)
    sorted_keys_np = keys_np[sorted_rank]
    sorted_values_np = values_np[sorted_rank]
    if dataset_num is not None:
        sorted_keys_np = sorted_keys_np[:dataset_num]
        sorted_values_np = sorted_values_np[:dataset_num]

    nums = sorted_keys_np.shape[0]//split_per_num if sorted_keys_np.shape[0]%split_per_num==0 else sorted_keys_np.shape[0]//split_per_num+1
    for num in range(nums):
        start_num = num * split_per_num
        end_num = min((num + 1) * split_per_num, sorted_keys_np.shape[0])

        sorted_keys_np_i = sorted_keys_np[start_num:end_num]
        sorted_values_np_i = sorted_values_np[start_num:end_num]

        save_dict = {keys_name: sorted_keys_np_i,
                     values_name: sorted_values_np_i}
        save_path = f"{'/'.join(base_dir.split('/')[:-1])}/total_info_{num}.npy"
        np.save(save_path, save_dict)

    save_path = f"{'/'.join(base_dir.split('/')[:-1])}"
    shutil.rmtree(f"{base_dir}")
    return save_path

def get_IdxToCategory_mapping(args):
    if args.is_category1:
        with open(args.idx_to_category1_mapping_path, 'rb') as f:
            OrigIdxToCategory1_mapping = pickle.load(f)
    else:
        OrigIdxToCategory1_mapping = None
    if args.is_category2:
        with open(args.idx_to_category2_mapping_path, 'rb') as f:
            OrigIdxToCategory2_mapping = pickle.load(f)
    else:
        OrigIdxToCategory2_mapping = None
    return OrigIdxToCategory1_mapping, OrigIdxToCategory2_mapping

def cal_func(args, device, logger):

    args.OrigIdxToCategory1_mapping, args.OrigIdxToCategory2_mapping = get_IdxToCategory_mapping(args)

    if args.evaluator_type == "fvd":
        gen_fvd_path = reorganize_np(args.save_dir, "idx", "fvd_feats", args.dataset_num)
        if args.is_gt:
            reorganize_np(args.save_sample_index_dir, "idx", "sample_index", args.dataset_num)
        else:
            get_fvd_score(args, gen_fvd_path, logger, device=device)
    elif args.evaluator_type == "fid":
        gen_fid_path = reorganize_np_split(args.save_dir, "idx", "fid_feats", args.dataset_num)
        if args.is_gt:
            reorganize_np(args.save_sample_index_dir, "idx", "sample_index", args.dataset_num)
        else:
            get_fid_score(args, gen_fid_path, logger)
    elif args.evaluator_type == "clip":
        clip_scores_path = reorganize_np(args.save_dir, "idx", "clip_scores", args.dataset_num)
        get_clip_score(args, clip_scores_path, logger)
    elif args.evaluator_type == "inception":
        inception_logits_path = reorganize_np(args.save_dir, "idx", "inception_logits", args.dataset_num)
        get_inception_score(args, inception_logits_path, logger)

def get_mapping_split(path, keys_name, values_name, is_return_ZeroIdxToOrigIdx_mapping=False):
    OrigIdxToValues_mapping = {}

    names = os.listdir(path)
    sorted_names = sorted(names, key=lambda s: int(s.split('_')[-1].split('.')[0]))
    keys_np_list = []
    values_np_list = []
    for name in sorted_names:
        dictionary = np.load(f"{path}/{name}", allow_pickle=True).item()
        keys_np = dictionary[keys_name]
        values_np = dictionary[values_name]
        keys_np_list.append(keys_np)
        values_np_list.append(values_np)
    keys_np = np.concatenate(keys_np_list, axis=0)
    values_np = np.concatenate(values_np_list, axis=0)

    if is_return_ZeroIdxToOrigIdx_mapping:
        ZeroIdxToOrigIdx_mapping = {}

    for i in range(keys_np.shape[0]):
        OrigIdxToValues_mapping[int(keys_np[i])] = values_np[i]
        if is_return_ZeroIdxToOrigIdx_mapping:
            ZeroIdxToOrigIdx_mapping[i] = int(keys_np[i])

    if is_return_ZeroIdxToOrigIdx_mapping:
        return OrigIdxToValues_mapping, ZeroIdxToOrigIdx_mapping
    else:
        return OrigIdxToValues_mapping

def get_mapping(path, keys_name, values_name, is_return_ZeroIdxToOrigIdx_mapping=False):
    OrigIdxToValues_mapping = {}
    dictionary = np.load(path, allow_pickle=True).item()
    keys_np = dictionary[keys_name]
    values_np = dictionary[values_name]

    if is_return_ZeroIdxToOrigIdx_mapping:
        ZeroIdxToOrigIdx_mapping = {}

    for i in range(keys_np.shape[0]):
        OrigIdxToValues_mapping[int(keys_np[i])] = values_np[i]
        if is_return_ZeroIdxToOrigIdx_mapping:
            ZeroIdxToOrigIdx_mapping[i] = int(keys_np[i])

    if is_return_ZeroIdxToOrigIdx_mapping:
        return OrigIdxToValues_mapping, ZeroIdxToOrigIdx_mapping
    else:
        return OrigIdxToValues_mapping

def get_category_fvd_score(results, zero_idxs, category_num, total_values_A, total_values_B, category1_name,
                           category2_name=None):
    if category2_name is not None:
        category_name = f"{category1_name} - {category2_name}"
    else:
        category_name = category1_name

    if len(zero_idxs) != 0:
        results["info"][category_num]["score"][category_name] = frechet_distance(total_values_A[zero_idxs],
                                                                                 total_values_B[zero_idxs])
    else:
        results["info"][category_num]["score"][category_name] = None
    results["info"][category_num]["num"][category_name] = len(zero_idxs)

    return results


def get_category_fid_score(results, zero_idxs, category_num, total_values_A, total_values_B, category1_name,
                           category2_name=None):
    if category2_name is not None:
        category_name = f"{category1_name} - {category2_name}"
    else:
        category_name = category1_name

    if len(zero_idxs) != 0:
        total_values_A_flatten = total_values_A[zero_idxs].reshape(-1, total_values_A[zero_idxs].shape[2])
        total_values_B_flatten = total_values_B[zero_idxs].reshape(-1, total_values_B[zero_idxs].shape[2])

        A_mu = np.mean(total_values_A_flatten, axis=0)
        A_sigma = np.cov(total_values_A_flatten, rowvar=False)

        B_mu = np.mean(total_values_B_flatten, axis=0)
        B_sigma = np.cov(total_values_B_flatten, rowvar=False)

        results["info"][category_num]["score"][category_name] = calculate_frechet_distance(A_mu, A_sigma,
                                                                                           B_mu, B_sigma)  # 很慢
    else:
        results["info"][category_num]["score"][category_name] = None
    results["info"][category_num]["num"][category_name] = len(zero_idxs)

    return results

def get_category_clip_score(results, zero_idxs, category_num, total_values, category1_name, category2_name=None):
    if category2_name is not None:
        category_name = f"{category1_name} - {category2_name}"
    else:
        category_name = category1_name

    if len(zero_idxs) != 0:
        score = total_values[zero_idxs].mean().astype(np.float64)
        results["info"][category_num]["score"][category_name] = score
    else:
        results["info"][category_num]["score"][category_name] = None
    results["info"][category_num]["num"][category_name] = len(zero_idxs)

    return results

def get_category_inception_score(results, zero_idxs, category_num, total_values, category1_name, category2_name=None,
                                 args=None):
    if category2_name is not None:
        category_name = f"{category1_name} - {category2_name}"
    else:
        category_name = category1_name

    if len(zero_idxs) != 0:
        category_InceptionLogits = total_values[zero_idxs]

        N = len(category_InceptionLogits)
        split_scores = []
        for k in range(args.inception_splits):
            part = category_InceptionLogits[k * (N // args.inception_splits): (k + 1) * (N // args.inception_splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        category_score_mean = np.mean(split_scores).astype(np.float64)
        category_score_std = np.std(split_scores).astype(np.float64)


        results["info"][category_num]["score"][category_name] = category_score_mean
        results["info"][category_num]["score_std"][category_name] = category_score_std
    else:
        results["info"][category_num]["score"][category_name] = None
        results["info"][category_num]["score_std"][category_name] = None
    results["info"][category_num]["num"][category_name] = len(zero_idxs)

    return results

def get_category1_score(args, results, ZeroIdxToOrigIdx_mapping, score_type, total_values_A, total_values_B=None):

    Category1ToZeroIdx = {}
    for category1_name in category_dict:
        Category1ToZeroIdx[category1_name] = []

    for zero_idx, orig_idx in ZeroIdxToOrigIdx_mapping.items():
        category_list = args.OrigIdxToCategory1_mapping[orig_idx]
        for category_i in category_list:
            if category_i in Category1ToZeroIdx.keys():
                Category1ToZeroIdx[category_i].append(zero_idx)

    results["info"]["category 1"] = {"score": {}, "num": {}}

    if score_type == "inception":
        results["info"]["category 1"]["score_std"] = {}
    for category1_name, zero_idxs in Category1ToZeroIdx.items():
        if score_type == "fvd":
            results = get_category_fvd_score(results, zero_idxs, "category 1", total_values_A, total_values_B, category1_name)
        elif score_type == "fid":
            results = get_category_fid_score(results, zero_idxs, "category 1", total_values_A, total_values_B, category1_name)
        elif score_type == "clip":
            results = get_category_clip_score(results, zero_idxs, "category 1", total_values_A, category1_name)
        elif score_type == "inception":
            results = get_category_inception_score(results, zero_idxs, "category 1", total_values_A, category1_name,
                                                   args=args)

    return results

def get_category2_score(args, results, ZeroIdxToOrigIdx_mapping, score_type, total_values_A, total_values_B=None):

    results["info"]["category 2"] = {"score": {}, "num": {}}
    if score_type == "inception":
        results["info"]["category 2"]["score_std"] = {}

    for category1_name, category2_name_list in category_dict.items():
        Category2ToZeroIdx = {}
        for category2_name in category2_name_list:
            Category2ToZeroIdx[category2_name] = []

        for zero_idx, orig_idx in ZeroIdxToOrigIdx_mapping.items():
            category_list = args.OrigIdxToCategory2_mapping[orig_idx]

            for category_i in category_list:
                if category_i in Category2ToZeroIdx.keys():
                    Category2ToZeroIdx[category_i].append(zero_idx)

            for category2_name, zero_idxs in Category2ToZeroIdx.items():
                if score_type == "fvd":
                    results = get_category_fvd_score(results, zero_idxs, "category 2", total_values_A, total_values_B,
                                                     category1_name, category2_name=category2_name)
                elif score_type == "fid":
                    results = get_category_fid_score(results, zero_idxs, "category 2", total_values_A, total_values_B,
                                                     category1_name, category2_name=category2_name)
                elif score_type == "clip":
                    results = get_category_clip_score(results, zero_idxs, "category 2", total_values_A,
                                                      category1_name, category2_name=category2_name)
                elif score_type == "inception":
                    results = get_category_inception_score(results, zero_idxs, "category 2", total_values_A,
                                                           category1_name, category2_name=category2_name, args=args)

    return results

# scipy==1.7.3/1.9.3, if you use 1.11.3, you will calculate a WRONG FVD VALUE!!!
def get_fvd_score(args, gen_path, logger, device):

    gt_OrigIdxToFvdFeats_mapping, ZeroIdxToOrigIdx_mapping = get_mapping(args.gt_fvd_path, "idx", "fvd_feats", is_return_ZeroIdxToOrigIdx_mapping=True)
    gen_OrigIdxToFvdFeats_mapping = get_mapping(gen_path, "idx", "fvd_feats")

    total_gt_FvdFeats = []
    total_gen_FvdFeats = []
    for i in range(len(gt_OrigIdxToFvdFeats_mapping)):
        total_gt_FvdFeats.append(torch.from_numpy(gt_OrigIdxToFvdFeats_mapping[ZeroIdxToOrigIdx_mapping[i]]))
        total_gen_FvdFeats.append(torch.from_numpy(gen_OrigIdxToFvdFeats_mapping[ZeroIdxToOrigIdx_mapping[i]]))

    total_gt_FvdFeats = torch.stack(total_gt_FvdFeats, dim=0).to(device=device)
    total_gen_FvdFeats = torch.stack(total_gen_FvdFeats, dim=0).to(device=device)

    score = frechet_distance(total_gt_FvdFeats, total_gen_FvdFeats)

    logger.info(f"fvd - video num: {len(gen_OrigIdxToFvdFeats_mapping)}, video length: {args.video_length}, "
                f"score: {score}")

    results = {"evaluator_type": "fvd",
               "info": {"video_num": len(gen_OrigIdxToFvdFeats_mapping),
                        "video_length": args.video_length,
                        "score": score}}

    if args.is_category1:
        results = get_category1_score(args, results, ZeroIdxToOrigIdx_mapping, "fvd", total_gt_FvdFeats, total_gen_FvdFeats)
    if args.is_category2:
        results = get_category2_score(args, results, ZeroIdxToOrigIdx_mapping, "fvd", total_gt_FvdFeats, total_gen_FvdFeats)

    save_json_path = f"{'/'.join(gen_path.split('/')[:-1])}/info_{args.evaluator_type}.json"
    with open(save_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

    save_total_info(args, logger)

# pip install scipy==1.5.4
# export MKL_NUM_THREADS=1
# https://github.com/mseitzer/pytorch-fid/issues/67
def get_fid_score(args, gen_path, logger):

    # gt_OrigIdxToFidFeats_mapping, ZeroIdxToOrigIdx_mapping = get_mapping(args.gt_fid_path, "idx", "fid_feats", is_return_ZeroIdxToOrigIdx_mapping=True)
    # gen_OrigIdxToFidFeats_mapping = get_mapping(gen_path, "idx", "fid_feats")
    gt_OrigIdxToFidFeats_mapping, ZeroIdxToOrigIdx_mapping = get_mapping_split(args.gt_fid_path, "idx", "fid_feats", is_return_ZeroIdxToOrigIdx_mapping=True)
    gen_OrigIdxToFidFeats_mapping = get_mapping_split(gen_path, "idx", "fid_feats")

    total_gt_FidFeats = []
    total_gen_FidFeats = []
    for i in range(len(gt_OrigIdxToFidFeats_mapping)):
        total_gt_FidFeats.append(gt_OrigIdxToFidFeats_mapping[ZeroIdxToOrigIdx_mapping[i]][np.newaxis, :])
        total_gen_FidFeats.append(gen_OrigIdxToFidFeats_mapping[ZeroIdxToOrigIdx_mapping[i]][np.newaxis, :])

    total_gt_FidFeats = np.concatenate(total_gt_FidFeats, axis=0)
    total_gen_FidFeats = np.concatenate(total_gen_FidFeats, axis=0)

    total_gt_FidFeats_flatten = total_gt_FidFeats.reshape(-1, total_gt_FidFeats.shape[2])
    total_gen_FidFeats_flatten = total_gen_FidFeats.reshape(-1, total_gen_FidFeats.shape[2])

    gt_mu = np.mean(total_gt_FidFeats_flatten, axis=0)
    gt_sigma = np.cov(total_gt_FidFeats_flatten, rowvar=False)

    gen_mu = np.mean(total_gen_FidFeats_flatten, axis=0)
    gen_sigma = np.cov(total_gen_FidFeats_flatten, rowvar=False)

    score = calculate_frechet_distance(gt_mu, gt_sigma, gen_mu, gen_sigma)  # 很慢

    logger.info(f"fid - video num: {len(gen_OrigIdxToFidFeats_mapping)}, video length: {args.video_length}, "
                f"batch size: {args.fid_batch_size}, score: {score}")

    results = {"evaluator_type": "fid",
               "info": {"video_num": len(gen_OrigIdxToFidFeats_mapping),
                        "video_length": args.video_length,
                        "batch_size": args.fid_batch_size,
                        "score": score}}

    if args.is_category1:
        results = get_category1_score(args, results, ZeroIdxToOrigIdx_mapping, "fid", total_gt_FidFeats, total_gen_FidFeats)
    if args.is_category2:
        results = get_category2_score(args, results, ZeroIdxToOrigIdx_mapping, "fid", total_gt_FidFeats, total_gen_FidFeats)

    save_json_path = f"{'/'.join(gen_path.split('/')[:-1])}/info_{args.evaluator_type}.json"
    with open(save_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

    save_total_info(args, logger)

def get_clip_score(args, scores_path, logger):
    OrigIdxToClipScores_mapping, ZeroIdxToOrigIdx_mapping = get_mapping(scores_path, "idx", "clip_scores", is_return_ZeroIdxToOrigIdx_mapping=True)

    total_ClipScores = []
    for i in range(len(OrigIdxToClipScores_mapping)):
        total_ClipScores.append(OrigIdxToClipScores_mapping[ZeroIdxToOrigIdx_mapping[i]])
    total_ClipScores = np.concatenate(total_ClipScores, axis=0)
    score = total_ClipScores.mean().astype(np.float64)

    logger.info(f"clip - video num: {len(OrigIdxToClipScores_mapping)}, video length: {args.video_length}, "
                f"score: {score}")

    results = {"evaluator_type": "clip",
               "info": {"video_num": len(OrigIdxToClipScores_mapping),
                        "video_length": args.video_length,
                        "score": score}}

    if args.is_category1:
        results = get_category1_score(args, results, ZeroIdxToOrigIdx_mapping, "clip", total_ClipScores)
    if args.is_category2:
        results = get_category2_score(args, results, ZeroIdxToOrigIdx_mapping, "clip", total_ClipScores)

    save_json_path = f"{'/'.join(scores_path.split('/')[:-1])}/info_{args.evaluator_type}.json"
    with open(save_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

    save_total_info(args, logger)

def get_inception_score(args, logits_path, logger):
    OrigIdxToInceptionLogits_mapping, ZeroIdxToOrigIdx_mapping = get_mapping(logits_path, "idx", "inception_logits", is_return_ZeroIdxToOrigIdx_mapping=True)

    total_InceptionLogits = []
    for i in range(len(OrigIdxToInceptionLogits_mapping)):
        total_InceptionLogits.append(OrigIdxToInceptionLogits_mapping[ZeroIdxToOrigIdx_mapping[i]])
    total_InceptionLogits = np.concatenate(total_InceptionLogits, axis=0)

    # Now compute the mean kl-div
    N = len(total_InceptionLogits)
    split_scores = []
    for k in range(args.inception_splits):
        part = total_InceptionLogits[k * (N // args.inception_splits): (k+1) * (N // args.inception_splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    score_mean = np.mean(split_scores).astype(np.float64)
    score_std = np.std(split_scores).astype(np.float64)

    logger.info(f"inception - video num: {len(OrigIdxToInceptionLogits_mapping)}, video length: {args.video_length}, "
                f"score_mean: {score_mean}, score_std: {score_std}")

    results = {"evaluator_type": "inception",
               "info": {"video_num": len(OrigIdxToInceptionLogits_mapping),
                        "video_length": args.video_length,
                        "score": score_mean,
                        "score_std": score_std}}

    if args.is_category1:
        results = get_category1_score(args, results, ZeroIdxToOrigIdx_mapping, "inception", total_InceptionLogits)
    if args.is_category2:
        results = get_category2_score(args, results, ZeroIdxToOrigIdx_mapping, "inception", total_InceptionLogits)

    save_json_path = f"{'/'.join(logits_path.split('/')[:-1])}/info_{args.evaluator_type}.json"
    with open(save_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

    save_total_info(args, logger)

def save_total_info(args, logger):

    target_dir = "/".join(args.save_dir.split("/")[:-2])

    total_info_dict = {}
    json_dir_names = os.listdir(f"{target_dir}")
    for json_dir_name in json_dir_names:
        json_path = f"{target_dir}/{json_dir_name}/info_{args.evaluator_type}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_obj = json.load(json_file)
                total_info_dict[f"{json_dir_name}"] = json_obj['info']
            json_file.close()

    if len(total_info_dict) != 0:
        save_json_path = f"{target_dir}/info_{args.evaluator_type}.json"
        with open(save_json_path, 'w') as json_file:
            json.dump(total_info_dict, json_file, indent=4, sort_keys=True)

    target_dir = "/".join(args.save_dir.split("/")[:-3])
    total_info_dict = {}
    dir_names = os.listdir(f"{target_dir}")
    for dir_name in dir_names:
        json_path = f"{target_dir}/{dir_name}/info_{args.evaluator_type}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_obj = json.load(json_file)
                total_info_dict[f"{dir_name}"] = json_obj
            json_file.close()

    if len(total_info_dict) != 0:
        save_json_path = f"{target_dir}/info_{args.evaluator_type}.json"
        with open(save_json_path, 'w') as json_file:
            json.dump(total_info_dict, json_file, indent=4, sort_keys=True)

    target_dir = "/".join(args.save_dir.split("/")[:-4])
    total_info_dict = {}
    dir_names = os.listdir(f"{target_dir}")
    for dir_name in dir_names:
        json_path = f"{target_dir}/{dir_name}/info_{args.evaluator_type}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_obj = json.load(json_file)
                total_info_dict[f"{dir_name}"] = json_obj
            json_file.close()

    if len(total_info_dict) != 0:
        save_json_path = f"{target_dir}/info_{args.evaluator_type}.json"
        with open(save_json_path, 'w') as json_file:
            json.dump(total_info_dict, json_file, indent=4, sort_keys=True)
        logger.info(f"save_json_path: {save_json_path}")

def args_check(args):
    if args.is_gt:
        assert (args.evaluator_type == "fvd" or args.evaluator_type == "fid"), "is_gt must match fvd or fid mode"
    else:
        if args.evaluator_type == "fvd" and args.gt_fvd_path is None:
            raise ValueError("please provide gt_fvd_path when is_gt=False")
        if args.evaluator_type == "fid" and args.gt_fid_path is None:
            raise ValueError("please provide gt_fid_path when is_gt=False")

def preprocess_args(args, ckpt_num):

    if args.is_gt and (args.evaluator_type == "fid" or args.evaluator_type == "fvd"):
        args.save_dir = f"{args.save_base_dir}/{args.save_name}/{args.evaluator_type}_gt/tmp"
    else:
        if args.dataset_num is not None and args.suffix is not None:
            args.save_dir = f"{'/'.join(args.ckpt_path.split('/')[:-1])}/evaluation_metrics_{args.suffix}_{args.dataset_num}/{ckpt_num}/{args.video_name}/{args.evaluator_type}/tmp"
        elif args.dataset_num is not None and args.suffix is None:
            args.save_dir = f"{'/'.join(args.ckpt_path.split('/')[:-1])}/evaluation_metrics_{args.dataset_num}/{ckpt_num}/{args.video_name}/{args.evaluator_type}/tmp"
        elif args.dataset_num is None and args.suffix is not None:
            args.save_dir = f"{'/'.join(args.ckpt_path.split('/')[:-1])}/evaluation_metrics_{args.suffix}/{ckpt_num}/{args.video_name}/{args.evaluator_type}/tmp"
        else:
            args.save_dir = f"{'/'.join(args.ckpt_path.split('/')[:-1])}/evaluation_metrics/{ckpt_num}/{args.video_name}/{args.evaluator_type}/tmp"

    if not args.is_gt and args.evaluator_type == "fid" and args.gt_fid_path is None:
        args.gt_fid_path = f"{args.save_base_dir}/{args.save_name}/{args.evaluator_type}_gt/total_info.npy"

    if not args.is_gt and args.evaluator_type == "fvd" and args.gt_fvd_path is None:
        args.gt_fvd_path = f"{args.save_base_dir}/{args.save_name}/{args.evaluator_type}_gt/total_info.npy"

    if args.is_automatic_sample_index:
        if os.path.exists(f"{args.save_base_dir}/{args.save_name}/index/total_info.npy"):
            args.gt_sample_index_path = f"{args.save_base_dir}/{args.save_name}/index/total_info.npy"

    return args

def get_logger(level, name):
    """
    Sets up a logger with the specified log level.
    This logger uses RichHandler for `rich` formatted logging output to the console.

    Args:
        level (int | str): Logging level, e.g., 20 or logging.INFO, 30 or logging.WARNING.
        name (str, optional): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Configure the logger for this library
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(stderr=True), markup=True)],
        force=True,
    )
    # Hook warnings to logger
    logging.captureWarnings(True)
    return logging.getLogger(name)

def parse_eval_type_and_ckpt(args):
    args.evaluator_types = args.evaluator_types.split(',')

    if not args.is_gt:
        if args.ckpt_nums is None:
            args.ckpt_nums = [ckpt_num for ckpt_num in os.listdir(args.ckpt_path) if ckpt_num.isnumeric()]
        else:
            args.ckpt_nums = args.ckpt_nums.split(',')
    else:
        args.ckpt_nums = ["gt"]

    return args

def get_video_metrics_args(args, evaluator_type, clip_caption_type, save_dir, ckpt_num):
    video_metrics_args = copy.deepcopy(args)

    video_metrics_args.is_gt = args.val_video_metrics_is_gt
    video_metrics_args.pretrained_model_base_path = args.val_video_metrics_pretrained_model_base_path
    video_metrics_args.gt_sample_index_path = args.val_video_metrics_gt_sample_index_path
    video_metrics_args.gt_fid_path = args.val_video_metrics_gt_fid_path
    video_metrics_args.fid_batch_size = args.val_video_metrics_fid_batch_size
    video_metrics_args.inception_batch_size = args.val_video_metrics_inception_batch_size
    video_metrics_args.inception_splits = args.val_video_metrics_inception_splits
    video_metrics_args.is_category1 = args.val_video_metrics_is_category1
    video_metrics_args.is_category2 = args.val_video_metrics_is_category2
    video_metrics_args.idx_to_category1_mapping_path = args.val_video_metrics_idx_to_category1_mapping_path
    video_metrics_args.idx_to_category2_mapping_path = args.val_video_metrics_idx_to_category2_mapping_path
    video_metrics_args.is_automatic_sample_index = args.val_video_metrics_is_automatic_sample_index

    video_metrics_args.clip_model = args.val_video_metrics_clip_model
    video_metrics_args.clip_context_length = args.val_video_metrics_clip_context_length
    video_metrics_args.inception_resize = args.val_video_metrics_inception_resize
    video_metrics_args.fid_dims = args.val_video_metrics_fid_dims

    video_metrics_args.gt_arrow_file = args.val_video_index_file
    video_metrics_args.resolution_h = args.video_size[0]
    video_metrics_args.resolution_w = args.video_size[1]
    video_metrics_args.video_length = args.sample_n_frames

    video_metrics_args.dataset_num = args.val_video_dataset_num
    video_metrics_args.batch_size = args.val_video_batch_size
    video_metrics_args.num_workers = args.val_video_num_workers

    video_metrics_args.gt_fvd_path = args.val_video_metrics_gt_fvd_path

    video_metrics_args.evaluator_type = evaluator_type
    video_metrics_args.clip_caption_type = clip_caption_type
    video_metrics_args.save_dir = save_dir
    video_metrics_args.ckpt_num = ckpt_num

    return video_metrics_args

def video_metrics_func(args, global_rank, world_size, device, gen_video_path):
    local_rank = global_rank % 8

    # args = preprocess_args(args, args.ckpt_num)
    caption_name = f"{args.save_dir.split('/')[-2]}"
    args.save_dir = f"{args.save_dir}/tmp"
    device = f"cuda:{device}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if global_rank == 0:
        start_time = time.time()
        logger = get_logger(level=20, name="multi_evaluator")
        logger.info(f"===============================================")
        logger.info(f"evaluator_type: {args.evaluator_type}, ckpt_num: {args.ckpt_num}, is_gt: {args.is_gt}")

    args_check(args)

    video_evaluator = VideoEvaluator(evaluator_type=args.evaluator_type,
                                     device=device,
                                     pretrained_model_base_path=args.pretrained_model_base_path,
                                     clip_model=args.clip_model,
                                     inception_resize=args.inception_resize,
                                     inception_batch_size=args.inception_batch_size,
                                     fid_dims=args.fid_dims,
                                     fid_batch_size=args.fid_batch_size)

    dataset = VideoDataset(gt_arrow_file=args.gt_arrow_file,
                           resolution_h=args.resolution_h,
                           resolution_w=args.resolution_w,
                           evaluator_type=args.evaluator_type,
                           video_length=args.video_length,
                           is_gt=args.is_gt,
                           gen_video_path=gen_video_path,
                           gt_sample_index_path=args.gt_sample_index_path,
                           clip_tokenizer=clip.tokenize if args.evaluator_type == "clip" else None,
                           clip_caption_type=args.clip_caption_type,
                           clip_context_length=args.clip_context_length if args.evaluator_type == "clip" else None,
                           logger=logger if global_rank == 0 else None,
                           dataset_num=args.dataset_num)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle=False,
                                                 sampler=sampler)

    progress_bar = tqdm(range(len(dataset)//world_size), disable=global_rank != 0)
    progress_bar.set_description(f"{caption_name} ({len(dataset)}//{world_size}={len(dataset)//world_size}) - samples")

    save_list = get_save_list(args)

    for i, batch in enumerate(dataset_loader):

        with torch.no_grad():
            save_list = video_evaluator_forward(args, video_evaluator, batch, save_list)

        progress_bar.update(1)

    save_func(args, save_list, global_rank, local_rank)
    dist.barrier()

    if global_rank == 0:
        cal_func(args, device, logger)
        end_time = time.time()
        logger.info(f"caption_name: {caption_name}, {args.evaluator_type}: {args.ckpt_num}, time: {end_time-start_time}")

    del video_evaluator
    del dataset_loader
    torch.cuda.empty_cache()

def get_val_video_summary_events(metrics_output_base_dir, update_steps):
    assert int(metrics_output_base_dir.split('/')[-1]) == update_steps, \
        f"{int(metrics_output_base_dir.split('/')[-1])} != {update_steps}"
    json_path = f"{metrics_output_base_dir}/info_{args.evaluator_type}.json"
    with open(json_path, 'r') as json_file:
        json_obj = json.load(json_file)

    val_video_summary_events = []
    for k1, v1 in json_obj.items():
        for k2, v2 in v1.items():
            val_video_summary_events.append(
                (f"Val/Steps/{k1}/{k2}", v2["score"], update_steps)
            )
    return val_video_summary_events

def logger_filter(name):
    def filter_(record):
        return record["extra"].get("name") == name
    return filter_

def empty_logger():
    logger = logging.getLogger("hymm_empty_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger

def gt_fvd_id_check(gt_fvd_id):
    for i in range(gt_fvd_id.shape[0]):
        assert i == gt_fvd_id[i], f"i: {i} != gt_fvd_id[i]: {gt_fvd_id[i]}, please check gt_fvd_id np !"

def check_dir(save_path, global_ckpt_num, offset):

    rm_dirs = [f"{'/'.join(save_path.split('/')[:-1])}/{rm_dir}" for rm_dir in
               os.listdir("/".join(save_path.split("/")[:-1])) if "ExpMetrics" in rm_dir]
    if len(rm_dirs) > 0:
        for rm_dir in rm_dirs:
            shutil.rmtree(rm_dir)

    video_dirs = [f"{video_dir}" for video_dir in
                  os.listdir("/".join(save_path.split("/")[:-1])) if "metrics" in video_dir]
    if len(video_dirs) > 0:
        sorted_video_dirs = sorted(video_dirs, key=lambda s: int(s.split('-')[1]))
        cur_ckpt_num = int(sorted_video_dirs[-1].split('-')[1])

        # =是兼容script评估多指标的情况
        assert global_ckpt_num > cur_ckpt_num, f"global_ckpt_num: {global_ckpt_num} must greater than " \
                                                f"cur_ckpt_num: {cur_ckpt_num}, please check offset: {offset}"

def preprocess_args_v2(args, ckpt_num, global_rank):

    if "Collect" in args.dataset_name:
        args.gt_arrow_file = dataset_dict[args.dataset_name]["arrow"]
        args.clip_caption_type = dataset_dict[args.dataset_name]["caption"]
        args.gt_fvd_path = dataset_dict[args.dataset_name]["gt_fvd"]
        args.resolution_h = dataset_dict[args.dataset_name]["resolution_h"]
        args.resolution_w = dataset_dict[args.dataset_name]["resolution_w"]
        args.video_length = dataset_dict[args.dataset_name]["video_length"]

    global_ckpt_num = int(ckpt_num) + args.offset

    if args.suffix is not None:
        save_name = f"ExpMetrics-{global_ckpt_num:07d}-{args.timestamp}-{args.suffix}"
    else:
        save_name = f"ExpMetrics-{global_ckpt_num:07d}-{args.timestamp}"

    args.save_dir = f"{args.base_metrics_path}/{args.dataset_name}/{save_name}"

    if not os.path.exists(f"{args.base_metrics_path}/{args.dataset_name}"):
        os.makedirs(f"{args.base_metrics_path}/{args.dataset_name}", exist_ok=True)

    if global_rank == 0 and args.is_first_metric:
        check_dir(args.save_dir, global_ckpt_num, args.offset)
    dist.barrier()

    video_dirs = [video_dir for video_dir in
                  os.listdir(f"{args.base_video_path}/{args.dataset_name}") if "videos" in video_dir]
    args.gen_video_path = None
    count = 0
    for video_dir in video_dirs:
        if f"{int(global_ckpt_num):07d}" in video_dir.split("-")[1] and "videos" in video_dir:
            args.gen_video_path = f"{args.base_video_path}/{args.dataset_name}/{video_dir}"
            count += 1
    assert count == 1, f"number videos-{int(global_ckpt_num):07d} must be 1, but found {count}, " \
                       f"please check {args.base_video_path}/{args.dataset_name}."

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if global_rank == 0:
        from loguru import logger
        logger.add(f"{args.save_dir}/{args.evaluator_type}/val.log", level="DEBUG", colorize=False, backtrace=True,
                   diagnose=True, encoding="utf-8", filter=logger_filter("val"))
        logger = logger.bind(name="val")
    else:
        logger = empty_logger()

    logger.info(f"gen_video_path: {args.gen_video_path}")
    logger.info(f"base_video_path: {args.base_video_path}")
    logger.info(f"dataset_name: {args.dataset_name}")
    logger.info(f"ckpt_num: {args.ckpt_num}")

    return args, logger

# def preprocess_args_vbench_v2(args, ckpt_num):
#
#     sub_name = f"{args.dataset_name}-{args.resolution_h}p{args.resolution_w}p{args.video_length}f"
#
#     args.save_dir = f"{args.base_metrics_path}/{sub_name}/{int(ckpt_num):07d}-{args.timestamp}-metrics"
#
#     all_video_path = [video_path for video_path in os.listdir(f"{args.base_video_path}/{sub_name}") if "videos" in video_path]
#     args.gen_video_path = None
#     count = 0
#     for video_path in all_video_path:
#         if f"{int(ckpt_num):07d}" in video_path and "videos" in video_path:
#             args.gen_video_path = f"{args.base_video_path}/{sub_name}/{video_path}"
#             count += 1
#     assert count == 1, f"num {int(ckpt_num):07d}_videos must be 1, but found {count}, please check {args.base_video_path}."
#
#     from loguru import logger
#     logger.add(f"{args.save_dir}/val.log", level="DEBUG", colorize=False, backtrace=True,
#                diagnose=True, encoding="utf-8", filter=logger_filter("val"))
#     logger = logger.bind(name="val")
#
#     print('args.save_dir = ', args.save_dir)
#     print('args.gen_video_path = ', args.gen_video_path)
#
#     return args, logger

def save_total_info_v2(args, logger):

    target_dir = f"{args.save_dir}"
    total_info_dict = {}
    json_dir_names = [json_dir_name for json_dir_name in os.listdir(f"{target_dir}") if
                      ("val.log" not in json_dir_name and "codes.tar.gz" not in json_dir_name)]

    for json_dir_name in json_dir_names:
        json_path = f"{target_dir}/{json_dir_name}/info_{args.evaluator_type}.json"
        if os.path.exists(json_path):
            with open(json_path, 'r') as json_file:
                json_obj = json.load(json_file)
                total_info_dict[f"{json_dir_name}"] = json_obj['info']
            json_file.close()

    if len(total_info_dict) != 0:
        save_json_path = f"{target_dir}/info_{args.evaluator_type}.json"
        with open(save_json_path, 'w') as json_file:
            json.dump(total_info_dict, json_file, indent=4, sort_keys=True)

    if args.is_last_metric:
        old_folder_name = args.save_dir
        new_folder_name = f"{'/'.join(args.save_dir.split('/')[:-1])}/metrics-{(int(args.ckpt_num) + args.offset):07d}-{args.timestamp}"
        os.rename(old_folder_name, new_folder_name)
        dump_codes(f"{new_folder_name}/{args.evaluator_type}/codes.tar.gz",
                   root=Path(__file__).parent.parent.parent,
                   sub_dirs=["hymm", "jobs"])

    if args.is_aggregate_results and args.is_last_metric:
        print('args.save_dir = ', args.save_dir)
        target_dir = '/'.join(args.save_dir.split('/')[:-1])
        print('target_dir = ', target_dir)
        total_info_dict = {}
        json_dir_names = [json_dir_name for json_dir_name in os.listdir(f"{target_dir}") if "metrics" in json_dir_name]
        check_metric_files = [int(json_dir_name.split('-')[1]) for json_dir_name in json_dir_names]
        assert len(check_metric_files)==len(set(check_metric_files)), \
            f"len(check_metric_files): {len(check_metric_files)} must be equal to len(set(check_metric_files)): {len(set(check_metric_files))}"

        for json_dir_name in json_dir_names:
            json_path = f"{target_dir}/{json_dir_name}/info_{args.evaluator_type}.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    json_obj = json.load(json_file)
                    total_info_dict[f"{int(json_dir_name.split('-')[1])}"] = json_obj
                json_file.close()

        if len(total_info_dict) != 0:
            save_json_path = f"{target_dir}/info_{args.evaluator_type}.json"
            with open(save_json_path, 'w') as json_file:
                json.dump(total_info_dict, json_file, indent=4, sort_keys=True)
            logger.info(f"save_json_path: {save_json_path}")

def cal_func_v2(args, logger, global_rank, dataset_loader, video_evaluator, device, world_size, data_num):

    progress_bar = tqdm(range(data_num//world_size), disable=global_rank != 0)
    progress_bar.set_description(f"{args.dataset_name}-{args.evaluator_type} ({data_num}//{world_size}={data_num//world_size}) - samples")

    id_list = []
    output_list = []
    for i, batch in enumerate(dataset_loader):
        with torch.no_grad():
            if args.evaluator_type in ["fvd", "clip", "inception"]:
                idx = batch["idx"]
                frames = batch["frames"]

                if args.evaluator_type == "fvd":
                    output = video_evaluator(frames)
                elif args.evaluator_type == "clip":
                    text_emb = batch["text_emb"]
                    output = video_evaluator(frames, text_emb=text_emb)
                elif args.evaluator_type == "inception":
                    output = video_evaluator(frames)

                id_list.append(idx.item())
                output_list.append(output.cpu())

                progress_bar.update(1)

    if args.evaluator_type == "fvd":
        results = {"id": id_list, "output": output_list}
        gather_results_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(gather_results_list, results)
        if global_rank == 0:
            gather_id = []
            gather_output = []
            for gather_result in gather_results_list:
                gather_id.extend(gather_result["id"])
                gather_output.extend(gather_result["output"])

            gt_fvd = np.load(args.gt_fvd_path, allow_pickle=True).item()
            gt_fvd_id = gt_fvd["idx"]
            gt_fvd_feats = gt_fvd["fvd_feats"]
            new_gt_fvd_id = []
            new_gt_fvd_feats = []
            for i in range(gt_fvd_id.shape[0]):
                if int(gt_fvd_id[i]) not in new_gt_fvd_id:
                    new_gt_fvd_id.append(int(gt_fvd_id[i]))
                    new_gt_fvd_feats.append(gt_fvd_feats[i][None, :])
            gt_fvd_id = np.array(new_gt_fvd_id)
            gt_fvd_feats = np.concatenate(new_gt_fvd_feats, axis=0)
            gt_fvd_id_check(gt_fvd_id)

            gather_id = np.array(gather_id)

            gt_fvd_feats = torch.from_numpy(gt_fvd_feats[gather_id])
            gen_fvd_feats = torch.concat(gather_output, dim=0)

            gt_fvd_feats = gt_fvd_feats.to(device=device)
            gen_fvd_feats = gen_fvd_feats.to(device=device)

            from hymm.metrics_video.fvd.videogpt.fvd import frechet_distance
            score = frechet_distance(gt_fvd_feats, gen_fvd_feats)

            results = {"evaluator_type": "fvd",
                       "info": {"video_num": gen_fvd_feats.shape[0],
                                "video_length": args.video_length,
                                "score": score}}
            logger.info(results)

            save_json_path = f"{args.save_dir}/{args.evaluator_type}/info_{args.evaluator_type}.json"
            logger.info(f"{args.evaluator_type}_save_json_path: {save_json_path}")
            with open(save_json_path, 'w') as json_file:
                json.dump(results, json_file, indent=4, sort_keys=True)
            save_total_info_v2(args, logger)

    elif args.evaluator_type == "clip":
        results = {"id": id_list, "output": output_list}
        gather_results_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(gather_results_list, results)
        if global_rank == 0:
            gather_id = []
            gather_output = []
            for gather_result in gather_results_list:
                gather_id.extend(gather_result["id"])
                gather_output.extend(gather_result["output"])

            gather_output_ = torch.concat(gather_output, dim=0)
            video_num = gather_output_.shape[0]
            score = gather_output_.view(-1).mean().item()

            results = {"evaluator_type": "clip",
                       "info": {"video_num": video_num,
                                "video_length": args.video_length,
                                "score": score}}

            save_json_path = f"{args.save_dir}/{args.evaluator_type}/info_{args.evaluator_type}.json"
            with open(save_json_path, 'w') as json_file:
                json.dump(results, json_file, indent=4, sort_keys=True)
            save_total_info_v2(args, logger)
    elif args.evaluator_type == "inception":
        results = {"id": id_list, "output": output_list}
        gather_results_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(gather_results_list, results)
        if global_rank == 0:
            gather_id = []
            gather_output = []
            for gather_result in gather_results_list:
                gather_id.extend(gather_result["id"])
                gather_output.extend(gather_result["output"])
            gather_output_ = torch.concat(gather_output, dim=0)
            video_num = gather_output_.shape[0]
            inception_logits = gather_output_.view(-1, 1000).numpy()

            # Now compute the mean kl-div
            N = len(inception_logits)
            split_scores = []
            for k in range(args.inception_splits):
                part = inception_logits[k * (N // args.inception_splits): (k+1) * (N // args.inception_splits), :]
                py = np.mean(part, axis=0)
                scores = []
                for i in range(part.shape[0]):
                    pyx = part[i, :]
                    scores.append(entropy(pyx, py))
                split_scores.append(np.exp(np.mean(scores)))

            score_mean = np.mean(split_scores).astype(np.float64)
            score_std = np.std(split_scores).astype(np.float64)

            results = {"evaluator_type": "inception",
                       "info": {"video_num": video_num,
                                "video_length": args.video_length,
                                "score": score_mean,
                                "score_std": score_std}}

            save_json_path = f"{args.save_dir}/{args.evaluator_type}/info_{args.evaluator_type}.json"
            with open(save_json_path, 'w') as json_file:
                json.dump(results, json_file, indent=4, sort_keys=True)
            save_total_info_v2(args, logger)