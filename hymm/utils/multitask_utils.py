import torch
import math
import random
import copy
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
from hymm.utils.torch_utils import PRECISION_TO_TYPE
from hymm.constants import TARGET_SIZE
from hymm.utils.data_utils import get_closest_ratio, generate_crop_size_list
from einops import rearrange
from typing import List
import numpy as np
import PIL.Image
from PIL import Image

def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    r"""
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (`np.ndarray`):
            The image array to convert to PIL format.

    Returns:
        `List[PIL.Image.Image]`:
            A list of PIL images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def black_image(width, height):
    # 创建一张黑色图像
    black_image = Image.new('RGB', (width, height), (0, 0, 0))
    return black_image

def get_mask_from_mask_type(video_length, mask_type, mask_args, seed=None):

    if mask_type == "t2v":
        # 文生视频
        mask = torch.zeros(video_length)
    elif mask_type == "frame_interpolation":
        # 传统插帧
        frame_interpolation_rate = mask_args["frame_interpolation_rate"]
        a = video_length // frame_interpolation_rate + 1
        mask = [torch.ones(a)]
        for i in range(frame_interpolation_rate - 1):
            mask.append(torch.zeros(a))
        mask = torch.stack(mask, dim=-1).reshape(-1)[:video_length]
    elif mask_type == "fl2v": # named as "fl2v" or "image_interpolation"
        # 两图片插帧
        mask = torch.zeros(video_length)
        mask[0] = 1.0
        mask[-1] = 1.0
    elif mask_type == "video_connect":
        # 两视频插帧
        mask = torch.cat((torch.ones(video_length//4), torch.zeros(video_length-((video_length//4)*2)), torch.ones(video_length//4)))
    elif mask_type == "random":
        # 随机mask
        mask = torch.randint(0, 2, (video_length,)).float()
    elif mask_type == "expansion":
        # 视频扩展
        expansion_save_rate = mask_args["expansion_save_rate"]
        expansion_reverse_rate = mask_args["expansion_reverse_rate"]
        if seed is not None:
            rng = random.Random(seed + 1)
            random_num = rng.random()
        else:
            random_num = random.random() # [0, 1)
        if random_num < expansion_reverse_rate:
            expansion_reverse = True
        else:
            expansion_reverse = False

        video_length_save = math.floor(video_length * expansion_save_rate)
        video_length_drop = video_length - video_length_save
        assert video_length_drop != 0, f"video_length_drop: {video_length_drop} must be non-zero."
        mask_save = torch.ones((video_length_save))
        mask_drop = torch.zeros((video_length_drop))
        if expansion_reverse:
            mask = torch.concat([mask_drop, mask_save], dim=-1)
        else:
            mask = torch.concat([mask_save, mask_drop], dim=-1)
    elif mask_type == "i2v":
        # 图生视频
        mask = torch.zeros(video_length)
        mask[0] = 1.0
    elif mask_type == "range_condition":
        # 任意帧范围条件 — unified PhysRoPE condition control
        condition_ranges = mask_args.get("condition_ranges", [])
        mask = get_multitask_mask_with_ranges(video_length, condition_ranges)
    else:
        raise ValueError(f"{mask_type} is not supported !")

    return mask

def multitask_check(multi_task_args):
    count_prob = 0
    for k, v in multi_task_args.items():
        assert k in ["t2v", "frame_interpolation", "fl2v", "expansion", "i2v", "video_connect", "random", "range_condition"], \
            f"{k} must be within [t2v, frame_interpolation, fl2v, expansion, i2v, video_connect, random, range_condition]"
        count_prob += v["prob"]
        if k == "frame_interpolation":
            assert "mask_args" in v, f"mask_args must be provided for frame_interpolation"
            assert "frame_interpolation_rate" in v["mask_args"], f"frame_interpolation_rate must be provided for frame_interpolation"
        elif k == "expansion":
            assert "mask_args" in v, f"mask_args must be provided for expansion"
            assert "expansion_save_rate" in v["mask_args"], f"expansion_save_rate must be provided for expansion"
            assert "expansion_reverse_rate" in v["mask_args"], f"expansion_reverse_rate must be provided for expansion"
    assert abs(1 - count_prob) < 0.000001, f"sum(prob): {count_prob} must be 1"

def get_multitask_mask(video_length, multi_task_args, seed=None):
    cumsum_threshold = 0
    thresholds = [0]
    mask_types = []
    for k, v in multi_task_args.items():
        # 兼容两种格式：直接是数值或字典包含prob字段（refrain:dict格式, t2va:数值格式）
        if isinstance(v, dict):
            cumsum_threshold += v["prob"]
        else:
            cumsum_threshold += v
        thresholds.append(cumsum_threshold)
        mask_types.append(k)

    random_num = random.random()
    
    for i in range(0, len(thresholds) - 1):
        if thresholds[i] <= random_num < thresholds[i + 1]:
            mask_type = mask_types[i]
            if mask_type in ("frame_interpolation", "expansion"):
                mask_args = multi_task_args[mask_type]["mask_args"]
            elif mask_type == "range_condition":
                # range_condition frame ranges are dynamically generated per-sample
                # in prepare_model_inputs (helpers.py), not from static config.
                # If mask_args provided in config, use them; otherwise return
                # a placeholder all-zeros mask (prepare_model_inputs will overwrite).
                cfg = multi_task_args[mask_type]
                if isinstance(cfg, dict) and "mask_args" in cfg:
                    mask_args = cfg["mask_args"]
                else:
                    mask_args = {"condition_ranges": []}
            else:
                mask_args = None
            break
    if video_length == 1:
        mask_type = "t2v"
        mask_args = None
    multi_task_mask = get_mask_from_mask_type(video_length, mask_type=mask_type, mask_args=mask_args, seed=seed)
    return multi_task_mask, mask_type

def get_multitask_mask_with_ranges(video_length, condition_ranges, device=None):
    """Create a multitask mask with arbitrary condition frame ranges.

    Args:
        video_length (int): Total number of (latent) frames.
        condition_ranges (list[tuple[int, int]]): Each tuple (start, end)
            marks frames [start, end) as condition frames (end exclusive).
        device: Target device for the returned tensor.

    Returns:
        torch.Tensor: Mask [video_length], 1.0=condition, 0.0=generation.
    """
    mask = torch.zeros(video_length)
    for start, end in condition_ranges:
        s = max(0, int(start))
        e = min(video_length, int(end))
        if s < e:
            mask[s:e] = 1.0
    if device is not None:
        mask = mask.to(device)
    return mask


def ensure_condition_segments_have_generation_frame(
    cond_seg_latents_packed,
    cond_seg_dst_ranges,
    cond_seg_src_ranges,
    cond_seg_mask,
    *,
    active_slot_ids=None,
    cond_seg_kind=None,
):
    """Mutate per-segment condition tensors so their union leaves a gen frame.

    range_condition training needs both reference frames and at least one
    generation frame. L4 stretch plus anchors can otherwise cover every latent
    frame, which makes the loss path invalid and can desync distributed ranks.
    """
    assert cond_seg_latents_packed.ndim in (5, 6), (
        f"cond_seg_latents_packed must be [S,C,T,H,W] or [B,S,C,T,H,W], "
        f"got {tuple(cond_seg_latents_packed.shape)}"
    )
    has_batch_dim = cond_seg_latents_packed.ndim == 6
    if has_batch_dim:
        batch_size, max_segs, _, t_full, _, _ = cond_seg_latents_packed.shape
        assert cond_seg_dst_ranges.shape == (batch_size, max_segs, 2), (
            f"batched cond_seg_dst_ranges must be {(batch_size, max_segs, 2)}, "
            f"got {tuple(cond_seg_dst_ranges.shape)}"
        )
        assert cond_seg_mask.shape == (batch_size, max_segs, t_full), (
            f"batched cond_seg_mask must be {(batch_size, max_segs, t_full)}, "
            f"got {tuple(cond_seg_mask.shape)}"
        )
        ranges_first = cond_seg_dst_ranges[0]
        mask_first = cond_seg_mask[0]
        src_first = cond_seg_src_ranges[0] if cond_seg_src_ranges is not None else None
        kind_first = cond_seg_kind[0] if cond_seg_kind is not None else None
    else:
        max_segs, _, t_full, _, _ = cond_seg_latents_packed.shape
        assert cond_seg_dst_ranges.shape == (max_segs, 2), (
            f"cond_seg_dst_ranges must be {(max_segs, 2)}, got {tuple(cond_seg_dst_ranges.shape)}"
        )
        assert cond_seg_mask.shape == (max_segs, t_full), (
            f"cond_seg_mask must be {(max_segs, t_full)}, got {tuple(cond_seg_mask.shape)}"
        )
        ranges_first = cond_seg_dst_ranges
        mask_first = cond_seg_mask
        src_first = cond_seg_src_ranges
        kind_first = cond_seg_kind

    if cond_seg_src_ranges is not None:
        assert cond_seg_src_ranges.shape == cond_seg_dst_ranges.shape, (
            f"cond_seg_src_ranges shape {tuple(cond_seg_src_ranges.shape)} must match "
            f"cond_seg_dst_ranges {tuple(cond_seg_dst_ranges.shape)}"
        )
    if cond_seg_kind is not None:
        expected_kind_shape = (batch_size, max_segs) if has_batch_dim else (max_segs,)
        assert cond_seg_kind.shape == expected_kind_shape, (
            f"cond_seg_kind must be {expected_kind_shape}, got {tuple(cond_seg_kind.shape)}"
        )

    if active_slot_ids is None:
        slot_ids = list(range(max_segs))
    else:
        slot_ids = [int(slot_id) for slot_id in active_slot_ids]
        assert all(0 <= slot_id < max_segs for slot_id in slot_ids), (
            f"active_slot_ids must be within [0,{max_segs}), got {slot_ids}"
        )

    def _active_slots():
        slots = []
        for slot_idx in slot_ids:
            ds = int(ranges_first[slot_idx, 0].item())
            de = int(ranges_first[slot_idx, 1].item())
            if ds < 0 or de <= ds:
                continue
            visible_ds = max(0, ds)
            visible_de = min(de, t_full)
            if visible_ds >= visible_de:
                continue
            kind = int(kind_first[slot_idx].item()) if kind_first is not None else 1
            slots.append({
                "slot": slot_idx,
                "dst": (ds, de),
                "visible_dst": (visible_ds, visible_de),
                "kind": kind,
                "length": visible_de - visible_ds,
            })
        return slots

    def _union_count(slots):
        union = torch.zeros(t_full, dtype=torch.bool, device=mask_first.device)
        for item in slots:
            slot_idx = item["slot"]
            union |= (mask_first[slot_idx] > 0)
        return int(union.sum().item())

    def _trim_slot(item):
        slot_idx = item["slot"]
        ds, de = item["dst"]
        visible_ds, visible_de = item["visible_dst"]
        new_de = visible_de - 1
        assert new_de > ds, (
            f"internal error: cannot trim length-1 condition slot {slot_idx} range=({ds},{de})"
        )
        if has_batch_dim:
            cond_seg_dst_ranges[:, slot_idx, 1] = new_de
            cond_seg_mask[:, slot_idx, new_de:visible_de] = 0.0
            cond_seg_latents_packed[:, slot_idx, :, new_de:visible_de] = 0.0
        else:
            cond_seg_dst_ranges[slot_idx, 1] = new_de
            cond_seg_mask[slot_idx, new_de:visible_de] = 0.0
            cond_seg_latents_packed[slot_idx, :, new_de:visible_de] = 0.0

        if cond_seg_src_ranges is not None:
            ss = int(src_first[slot_idx, 0].item())
            se = int(src_first[slot_idx, 1].item())
            if ss >= 0 and se > ss and (se - ss) == (de - ds):
                if has_batch_dim:
                    cond_seg_src_ranges[:, slot_idx, 1] = se - 1
                else:
                    cond_seg_src_ranges[slot_idx, 1] = se - 1

    def _drop_slot(item):
        slot_idx = item["slot"]
        visible_ds, visible_de = item["visible_dst"]
        if has_batch_dim:
            cond_seg_dst_ranges[:, slot_idx, :] = -1
            if cond_seg_src_ranges is not None:
                cond_seg_src_ranges[:, slot_idx, :] = -1
            cond_seg_mask[:, slot_idx, :] = 0.0
            cond_seg_latents_packed[:, slot_idx, :, visible_ds:visible_de] = 0.0
        else:
            cond_seg_dst_ranges[slot_idx, :] = -1
            if cond_seg_src_ranges is not None:
                cond_seg_src_ranges[slot_idx, :] = -1
            cond_seg_mask[slot_idx, :] = 0.0
            cond_seg_latents_packed[slot_idx, :, visible_ds:visible_de] = 0.0

    slots = _active_slots()
    if not slots or _union_count(slots) < t_full:
        return [item["slot"] for item in slots]

    for _ in range(t_full + max_segs):
        trim_candidates = [item for item in slots if item["kind"] == 1 and item["length"] > 1]
        if not trim_candidates:
            trim_candidates = [item for item in slots if item["length"] > 1]

        if trim_candidates:
            chosen = sorted(
                trim_candidates,
                key=lambda item: (-item["length"], -item["dst"][0], item["slot"]),
            )[0]
            _trim_slot(chosen)
        else:
            chosen = sorted(slots, key=lambda item: (-item["dst"][0], item["slot"]))[0]
            _drop_slot(chosen)

        slots = _active_slots()
        assert slots, "range_condition adjustment removed all condition slots"
        if _union_count(slots) < t_full:
            return [item["slot"] for item in slots]

    raise AssertionError(
        f"Failed to leave a generation frame after adjusting condition slots; "
        f"T={t_full}, slots={slots}"
    )


def merge_tensor_by_mask(tensor_1, tensor_2, mask, dim):
    assert tensor_1.shape == tensor_2.shape
    # Mask is a 0/1 verctor. Choose tensor_2, when the value is 1; otherwise, tensor_1
    masked_indices = torch.nonzero(mask).squeeze(1)
    tmp = copy.deepcopy(tensor_1)
    if dim==0:
        tmp[masked_indices] = tensor_2[masked_indices]
    if dim==1:
        tmp[:, masked_indices] = tensor_2[:, masked_indices]
    if dim==2:
        tmp[:, :, masked_indices] = tensor_2[:, :, masked_indices]
    return tmp

def decode_latents_to_image(vae, latents):
    """Decode latents to image for debug"""
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = image.squeeze(2)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    image[0].save('./debug_decode_image.png')    

def get_cond_latents(args, latents, vae, reorg_token=False, is_uncond=False, is_mode=True, last_frame_tensor=None):
    """
    Get the conditional latents from the latents.
    Args:
        latents: The latents to get the conditional latents from.
        vae: The VAE to use.
        reorg_token: Whether to reorganize the tokens. For time compression 8x only
        last_frame_tensor: The last image tensor. used for fl2v
    """
    if reorg_token: 
        latents = rearrange(latents, 'b c f h w -> b f c h w')
        latents = rearrange(latents, 'b f (n c) h w -> b (f n) c h w', n=2)
        latents = rearrange(latents, 'b f c h w -> b c f h w')
        latents = latents[:, :, 1:]
    first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
    first_images = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0]    
    first_images = first_images.squeeze(2)
    first_images = (first_images / 2 + 0.5).clamp(0, 1)
    first_images = first_images.cpu().permute(0, 2, 3, 1).float().numpy()
    first_images_pil = numpy_to_pil(first_images)
    first_images_np = (first_images * 255).round().astype("uint8")
    # first_images[0].save('testxxx_1.png')
    
    vae_dtype = PRECISION_TO_TYPE[args.vae_precision]

    #image_transform = transforms.Compose([
    #        transforms.ToTensor(), 
    #        transforms.Normalize([0.5], [0.5])
    #    ])
    #first_images_pixel_values = [image_transform(image).unsqueeze(0) for image in first_images_pil]
    #first_images_pixel_values = torch.cat(first_images_pixel_values).unsqueeze(2).to(vae.device)
    #with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
    #    if is_mode:
    #        cond_latents = vae.encode(first_images_pixel_values).latent_dist.mode()  # B, C, F, H, W
    #    else:
    #        cond_latents = vae.encode(first_images_pixel_values).latent_dist.sample() # B, C, F, H, W
    #    cond_latents.mul_(vae.config.scaling_factor)

    cond_latents = latents[:, :, 0:1, ...]  # first frame cond latents
 
    if is_uncond:
        first_images = [black_image(img.size[0], img.size[1]) for img in first_images_pil]

    last_frame_cond_latents = None
    if last_frame_tensor is not None:
        last_frame_tensor = last_frame_tensor if len(last_frame_tensor.shape) == 5 else last_frame_tensor.unsqueeze(2)
        with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
            last_frame_tensor = last_frame_tensor.to(vae.device).half()
            if is_mode:
                last_frame_cond_latents = vae.encode(last_frame_tensor).latent_dist.mode()  # B, C, F, H, W
            else:
                last_frame_cond_latents = vae.encode(last_frame_tensor).latent_dist.sample() # B, C, F, H, W
            last_frame_cond_latents.mul_(vae.config.scaling_factor)
        # decode_latents_to_image(vae, last_frame_cond_latents)

    return cond_latents, last_frame_cond_latents, first_images_np, first_images_pil

def get_cond_images(args, latents, vae, is_uncond=False):
    sematic_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
    sematic_image_latents = 1 / vae.config.scaling_factor * sematic_image_latents
    semantic_images = vae.decode(sematic_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0]
    semantic_images = semantic_images.squeeze(2)
    semantic_images = (semantic_images / 2 + 0.5).clamp(0, 1)
    semantic_images = semantic_images.cpu().permute(0, 2, 3, 1).float().numpy()
    semantic_images = numpy_to_pil(semantic_images)
    if is_uncond:
        semantic_images = [black_image(img.size[0], img.size[1]) for img in semantic_images]
        
    return semantic_images



def encode_frames_to_latents(ref_image_paths, vae, target_size, target_h=None, target_w=None):
    ref_images = [Image.open(image_path).convert('RGB') for image_path in ref_image_paths if image_path is not None]
    assert len(ref_images) == 1, "Only support one reference image for i2v/fl2v task."
    origin_size = ref_images[0].size
    original_width, original_height = origin_size
    if target_h is None or target_w is None:
        bucket_hw_base_size = TARGET_SIZE[target_size]["bucket_hw_base_size"]
        bucket_hw_bucket_stride = TARGET_SIZE[target_size]["bucket_hw_bucket_stride"]
        
        assert bucket_hw_base_size in [128, 256, 480, 512, 640, 720, 960], \
            f"bucket_hw_base_size must be in [128, 256, 480, 512, 640, 720, 960], but got {bucket_hw_base_size}"
        crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
        aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios,
                                                            crop_size_list)
        # 修正resize策略，避免黑边
        target_height, target_width = closest_size
    else:
        target_height, target_width = target_h, target_w

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
    ref_images_pixel_values = torch.cat(ref_images_pixel_values).unsqueeze(0).unsqueeze(2).to(vae.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        cond_latents = vae.encode(ref_images_pixel_values).latent_dist.mode()  # B, C, F, H, W
        cond_latents.mul_(vae.config.scaling_factor)
    return cond_latents, target_height, target_width



def temporal_resample_condition_segments(
    cond_seg_latents_packed,  # [MAX_SEGS, C, T, H, W] or [B, MAX_SEGS, C, T, H, W]
    cond_seg_dst_ranges,       # [MAX_SEGS, 2] or [B, MAX_SEGS, 2]
    cond_seg_src_ranges,       # [MAX_SEGS, 2] or [B, MAX_SEGS, 2]
    cond_seg_mask,             # [MAX_SEGS, T] or [B, MAX_SEGS, T]
    resample_prob=0.1,
    resample_range=(0.8, 1.25),
    seed=None,
    device=None,
):
    """
    Resample selected condition segments to variable lengths using trilinear interpolation.
    
    This function enables L4 conditioning: variable-length clip insertion with automatic 
    temporal scaling. For each active segment, randomly decide whether to resample, then
    apply a random stretch factor to create temporal variability in training.
    
    Args:
        cond_seg_latents_packed (torch.Tensor): Packed condition latents [S, C, T, H, W] or [B, S, C, T, H, W]
        cond_seg_dst_ranges (torch.Tensor): Destination frame ranges [[ds, de], ...] [S, 2] or [B, S, 2]
        cond_seg_src_ranges (torch.Tensor): Source frame ranges [[ss, se], ...] [S, 2] or [B, S, 2]
        cond_seg_mask (torch.Tensor): Active segment mask [S, T] or [B, S, T]
        resample_prob (float): Probability to resample each segment (0.1 recommended)
        resample_range (tuple): (min_factor, max_factor) for stretch (0.8-1.25 recommended)
        seed (int): Optional seed for reproducibility (SP-safe seed recommended)
        device: Optional device override
    
    Returns:
        tuple: (cond_seg_latents_packed, cond_seg_dst_ranges) with resampled segments
    
    Raises:
        ValueError: If invalid ranges or dtype mismatches detected
    """
    import torch.nn.functional as F
    
    if cond_seg_latents_packed is None or cond_seg_dst_ranges is None:
        return cond_seg_latents_packed, cond_seg_dst_ranges
    
    assert cond_seg_latents_packed.ndim in (5, 6), (
        f"cond_seg_latents_packed must be [S,C,T,H,W] or [B,S,C,T,H,W], "
        f"got {tuple(cond_seg_latents_packed.shape)}"
    )
    has_batch_dim = cond_seg_latents_packed.ndim == 6
    if has_batch_dim:
        batch_size, max_segs, c, t_full, h, w = cond_seg_latents_packed.shape
    else:
        batch_size = None
        max_segs, c, t_full, h, w = cond_seg_latents_packed.shape
    assert cond_seg_dst_ranges.ndim in (2, 3), (
        f"cond_seg_dst_ranges must be [S,2] or [B,S,2], got {tuple(cond_seg_dst_ranges.shape)}"
    )
    assert cond_seg_src_ranges is not None, "cond_seg_src_ranges must be provided for L4 temporal resample"
    assert cond_seg_src_ranges.shape == cond_seg_dst_ranges.shape, (
        f"cond_seg_src_ranges shape {tuple(cond_seg_src_ranges.shape)} must match "
        f"cond_seg_dst_ranges {tuple(cond_seg_dst_ranges.shape)}"
    )
    assert cond_seg_mask.ndim in (2, 3), (
        f"cond_seg_mask must be [S,T] or [B,S,T], got {tuple(cond_seg_mask.shape)}"
    )
    if has_batch_dim:
        assert cond_seg_dst_ranges.shape == (batch_size, max_segs, 2), (
            f"batched cond_seg_dst_ranges must be {(batch_size, max_segs, 2)}, "
            f"got {tuple(cond_seg_dst_ranges.shape)}"
        )
        assert cond_seg_mask.shape == (batch_size, max_segs, t_full), (
            f"batched cond_seg_mask must be {(batch_size, max_segs, t_full)}, "
            f"got {tuple(cond_seg_mask.shape)}"
        )
    else:
        assert cond_seg_dst_ranges.shape == (max_segs, 2), (
            f"cond_seg_dst_ranges must be {(max_segs, 2)}, got {tuple(cond_seg_dst_ranges.shape)}"
        )
        assert cond_seg_mask.shape == (max_segs, t_full), (
            f"cond_seg_mask must be {(max_segs, t_full)}, got {tuple(cond_seg_mask.shape)}"
        )
    original_dtype = cond_seg_latents_packed.dtype
    
    # Use provided seed or fall back to random
    rng = random.Random(seed) if seed is not None else random.Random()
    
    resample_debug = False  # Enable only for short bring-up runs.
    resample_log = []
    
    dst_ranges_first = cond_seg_dst_ranges[0] if cond_seg_dst_ranges.ndim == 3 else cond_seg_dst_ranges
    mask_first = cond_seg_mask[0] if cond_seg_mask.ndim == 3 else cond_seg_mask
    active_ranges = []
    for seg_idx in range(max_segs):
        ds0 = int(dst_ranges_first[seg_idx, 0].item())
        de0 = int(dst_ranges_first[seg_idx, 1].item())
        if ds0 >= 0 and de0 > ds0:
            active_ranges.append((ds0, de0, seg_idx))
    active_ranges.sort(key=lambda item: item[0])
    next_start_by_slot = {}
    for order_idx, (ds0, de0, seg_idx) in enumerate(active_ranges):
        next_start_by_slot[seg_idx] = active_ranges[order_idx + 1][0] if order_idx + 1 < len(active_ranges) else t_full

    for seg_idx in range(max_segs):
        # Skip inactive segments
        if mask_first[seg_idx].sum() == 0:
            continue
        
        # Check if should resample this segment
        if rng.random() > resample_prob:
            continue
        
        # Get destination range from sample-0 layout; helpers.py requires all
        # batch rows to share the same mask layout before calling this function.
        ds = int(dst_ranges_first[seg_idx, 0].item())
        de = int(dst_ranges_first[seg_idx, 1].item())

        if ds < 0 or de <= ds or de > t_full:
            if resample_debug:
                print(
                    f"[L4-Resample SKIP] invalid dst_range seg_idx={seg_idx} "
                    f"dst=({ds},{de}) t_full={t_full}",
                    flush=True,
                )
            continue

        T_src = de - ds
        if T_src <= 0:
            if resample_debug:
                print(f"[L4-Resample SKIP] empty source span seg_idx={seg_idx} dst=({ds},{de})", flush=True)
            continue

        # Sample random stretch factor.
        min_factor, max_factor = resample_range
        stretch_factor = rng.uniform(float(min_factor), float(max_factor))

        # Compute target length. Cap at the next active segment start to avoid
        # overlapping condition slots. If this segment starts at 0 and is alone,
        # keep at least one generation frame.
        T_dst = max(1, int(round(T_src * stretch_factor)))
        max_end = int(next_start_by_slot.get(seg_idx, t_full))
        max_len = max_end - ds
        if ds == 0 and max_end == t_full:
            max_len = max(1, max_len - 1)
        T_dst = min(T_dst, max_len)
        if T_dst <= 0:
            if resample_debug:
                print(
                    f"[L4-Resample SKIP] no temporal room seg_idx={seg_idx} "
                    f"dst=({ds},{de}) next_start={max_end}",
                    flush=True,
                )
            continue

        # Resample only if different from source.
        if T_dst != T_src:
            if has_batch_dim:
                src_latent = cond_seg_latents_packed[:, seg_idx, :, ds:de]  # [B, C, T_src, H, W]
                resampled = F.interpolate(
                    src_latent.float(),
                    size=(T_dst, h, w),
                    mode="trilinear",
                    align_corners=False,
                ).to(original_dtype)  # [B, C, T_dst, H, W]
                cond_seg_latents_packed[:, seg_idx, :, ds:ds + T_dst] = resampled
                if ds + T_dst < de:
                    cond_seg_latents_packed[:, seg_idx, :, ds + T_dst:de] = 0.0
            else:
                src_latent = cond_seg_latents_packed[seg_idx, :, ds:de]  # [C, T_src, H, W]
                resampled = F.interpolate(
                    src_latent.unsqueeze(0).float(),
                    size=(T_dst, h, w),
                    mode="trilinear",
                    align_corners=False,
                ).to(original_dtype).squeeze(0)  # [C, T_dst, H, W]
                cond_seg_latents_packed[seg_idx, :, ds:ds + T_dst] = resampled
                if ds + T_dst < de:
                    cond_seg_latents_packed[seg_idx, :, ds + T_dst:de] = 0.0

            # Update destination range and mask. Source ranges are intentionally
            # left unchanged so downstream PhysRoPE sees src_len != dst_len.
            if cond_seg_dst_ranges.ndim == 3:
                cond_seg_dst_ranges[:, seg_idx, 1] = ds + T_dst
            else:
                cond_seg_dst_ranges[seg_idx, 1] = ds + T_dst

            if cond_seg_mask.ndim == 3:
                cond_seg_mask[:, seg_idx, :] = 0.0
                cond_seg_mask[:, seg_idx, ds:ds + T_dst] = 1.0
            else:
                cond_seg_mask[seg_idx, :] = 0.0
                cond_seg_mask[seg_idx, ds:ds + T_dst] = 1.0

            resample_log.append({
                'seg_idx': seg_idx,
                'T_src': T_src,
                'T_dst': T_dst,
                'stretch_factor': stretch_factor,
            })
    
    if resample_debug and len(resample_log) > 0 and seed is not None:
        # Log first few resamples for debugging (limit to prevent log spam)
        if len(resample_log) <= 3:
            for log_entry in resample_log:
                print(f"[L4-Resample] seg={log_entry['seg_idx']} "
                      f"{log_entry['T_src']}→{log_entry['T_dst']} "
                      f"(factor={log_entry['stretch_factor']:.3f})")
    
    return cond_seg_latents_packed, cond_seg_dst_ranges
