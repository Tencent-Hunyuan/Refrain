from typing import Tuple
import math

import torch
from torch import BoolTensor, IntTensor
from torch.nn.attention.flex_attention import create_block_mask, create_mask
import numpy as np
from typing import List
import random
import os
# Peiyuan: This is neccesay. Dont know why. see https://github.com/pytorch/pytorch/issues/135028
torch._inductor.config.realize_opcount_threshold = 100


def generate_sta_with_block_mask(canvas_thw, kernel_thw, tile_thw, text_length, 
                                 block_thw, q_to_block_list):
    """Generates a 3D NATTEN attention mask with a given kernel size.
    
    Args:
        canvas_t: The time dimension of the canvas.
        canvas_h: The height of the canvas.
        canvas_w: The width of the canvas.
        kernel_t: The time dimension of the kernel.
        kernel_h: The height of the kernel.
        kernel_w: The width of the kernel.
    """
    canvas_t, canvas_h, canvas_w = canvas_thw
    kernel_t, kernel_h, kernel_w = kernel_thw
    tile_t_size, tile_h_size, tile_w_size = tile_thw
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    total_kernel_size = kernel_t * kernel_h * kernel_w
    canvas_tile_t, canvas_tile_h, canvas_tile_w = math.ceil(canvas_t/tile_t_size), math.ceil(canvas_h/tile_h_size), math.ceil(canvas_w/tile_w_size)
    img_seq_len = canvas_thw[0] * canvas_thw[1] * canvas_thw[2]
    block_t_size, block_h_size, block_w_size = block_thw

    def get_tile_t_x_y(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        pixel_t = idx // (canvas_h * canvas_w)
        pixel_h = (idx % (canvas_h * canvas_w)) // canvas_w
        pixel_w = idx % canvas_w

        tile_t = pixel_t // tile_t_size
        tile_h = pixel_h // tile_h_size
        tile_w = pixel_w // tile_w_size

        return tile_t, tile_h, tile_w
    
    def idx_to_pixel_idx(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        pixel_t = idx // (canvas_h * canvas_w)
        pixel_h = (idx % (canvas_h * canvas_w)) // canvas_w
        pixel_w = idx % canvas_w
        return pixel_t, pixel_h, pixel_w

    def sta_mask_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        valid_seq_length = img_seq_len + text_length
        q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(q_idx)
        kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t_tile.clamp(kernel_t // 2, (canvas_tile_t - 1) - kernel_t // 2)
        kernel_center_x = q_x_tile.clamp(kernel_h // 2, (canvas_tile_h - 1) - kernel_h // 2)
        kernel_center_y = q_y_tile.clamp(kernel_w // 2, (canvas_tile_w - 1) - kernel_w // 2)
        time_mask = (kernel_center_t - kv_t_tile).abs() <= kernel_t // 2
        hori_mask = (kernel_center_x - kv_x_tile).abs() <= kernel_h // 2
        vert_mask = (kernel_center_y - kv_y_tile).abs() <= kernel_w // 2

        image_mask = (q_idx < img_seq_len) & (kv_idx < img_seq_len)
        image_to_text_mask = (q_idx < img_seq_len) & (kv_idx >= img_seq_len) & (kv_idx < valid_seq_length)
        text_to_all_mask = (q_idx >= img_seq_len) & (q_idx < valid_seq_length) & (kv_idx < valid_seq_length)
        
        # q_idx_to_block_idx
        id_t, id_h, id_w = idx_to_pixel_idx(kv_idx)
        clamp_q_idx = q_idx.clamp(0, q_to_block_list.shape[2] - 1)
        choose_index_list=q_to_block_list[b, h, clamp_q_idx] # topk * 6
        choose_topk_mask=None
        for choose_index in choose_index_list:
            cur_mask = (id_t >= choose_index[0]) &  (id_t < choose_index[1]) & \
                       (id_h >= choose_index[2]) &  (id_h < choose_index[3]) & \
                       (id_w >= choose_index[4]) &  (id_w < choose_index[5]) & (q_idx < img_seq_len) & (kv_idx < img_seq_len)
            if choose_topk_mask is None:
                choose_topk_mask = cur_mask
            else:
                choose_topk_mask = choose_topk_mask | cur_mask
        return (image_mask & time_mask & hori_mask & vert_mask) | image_to_text_mask | text_to_all_mask | choose_topk_mask 

    sta_mask_3d.__name__ = f"natten_3d_c{canvas_t}x{canvas_w}x{canvas_h}_k{kernel_t}x{kernel_w}x{kernel_h}"
    return sta_mask_3d


def get_sliding_tile_attention_with_block_mask(kernel_size, tile_size, img_size, text_length, device,
                                               batch_size=None,
                                               head_size=None,
                                               block_shape=None,
                                               text_max_len=None, is_full=False, q_to_block_list=None):
    """
    q_to_block_list: list of int, the choose block index of each query.
    结合STA local sliding tile attention 和block attention, 实现局部和跨块的attention.
    """
    if is_full:
        def image_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) | (q_idx < kv_idx)
    else:
        image_mask = generate_sta_with_block_mask(canvas_thw=img_size, 
                                                  kernel_thw=kernel_size, 
                                                  tile_thw=tile_size, 
                                                  text_length=text_length, 
                                                  block_thw=block_shape,
                                                  q_to_block_list=q_to_block_list)
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    mask = create_block_mask(image_mask,
                            B=batch_size,
                            H=head_size,
                            Q_LEN=img_seq_len + text_max_len,
                            KV_LEN=img_seq_len + text_max_len,
                            device=device,
                            _compile=True,
                            )
    return mask
    
def get_q_to_block_list(q: torch.Tensor, 
                        k: torch.Tensor,
                        image_shape: Tuple[int, int, int] = None,
                        block_shape: Tuple[int, int, int] = None,
                        topk: int = 2,
                        threshold: float = 0.0,
                        enforce_share_topk_block: bool = False,
                        enforce_share_head: bool = True,
                        layer_name: str = None) -> torch.Tensor:
    """
    根据block size 将k分为多个block，计算每个block的平均值，
    然后计算每个query和每个block的cosine similarity，选择相似度最高的block，
    最后将每个query的block index返回。
    Args:
        block_size (int): 每个块的大小。
        q (torch.Tensor): 查询张量，形状为 [b, h, s, d]。
        k (torch.Tensor): 键张量，形状为 [b, h, s, d]。
        topk (int, optional): 选择的相似度最高的块的数量。默认为 2。
        enforce_share_topk_block (bool, optional): 是否强制共享top-k块。默认为 False。
    Returns:
        torch.Tensor: 每个查询的块索引，形状为 [b,h,s,topk,6]，其中6表示每个topk块的起始和结束索引。
    """    
    batch_size, num_heads, seq_len, head_dim  = k.shape
    num_blocks_t = np.ceil(image_shape[0] / block_shape[0]).astype(int)
    num_blocks_h = np.ceil(image_shape[1] / block_shape[1]).astype(int)
    num_blocks_w = np.ceil(image_shape[2] / block_shape[2]).astype(int)
    num_blocks = num_blocks_t * num_blocks_h * num_blocks_w
    
    def get_block_avg_feat(x: torch.Tensor) -> torch.Tensor:
        # 将输入张量 x 分割为多个块，并计算每个块的平均值
        # 输入张量 x 的形状为 [b, h, s, d]
        # 输出张量 k_block_means 的形状为 [b, h, num_blocks, d]
        k_block_means_list: List[torch.Tensor] = []
        image_length=np.prod(image_shape)
        x_review = x[:,:,:image_length,:].reshape(batch_size, num_heads,
                                                  image_shape[0], image_shape[1],image_shape[2], head_dim)
        for t in range(num_blocks_t):
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    start_t = t * block_shape[0]
                    end_t = min(start_t + block_shape[0], image_shape[0])
                    start_h = h * block_shape[1]
                    end_h = min(start_h + block_shape[1], image_shape[1])
                    start_w = w * block_shape[2]
                    end_w = min(start_w + block_shape[2], image_shape[2])
                    k_block = x_review[:, :, start_t:end_t, start_h:end_h, start_w:end_w, :]
                    k_block_mean = torch.mean(k_block, dim=(2, 3, 4))
                    k_block_means_list.append(k_block_mean)
        k_block_means = torch.stack(k_block_means_list, dim=2)
        return k_block_means
    k_block_means = get_block_avg_feat(k)
    if enforce_share_topk_block:
        q = get_block_avg_feat(q)
    
    q = q.type(torch.float32)
    k_block_means = k_block_means.type(torch.float32)  # float logit for better gate logit perception
    if enforce_share_head:
        q = q.mean(dim=1, keepdim=True)
        k_block_means = k_block_means.mean(dim=1, keepdim=True)
    gate = torch.einsum("bhsd,bhkd->bhsk", q, k_block_means)
    q = q.type_as(k)
    topk_param = min(topk, gate.size(-1))  # 确保topk不超过块的数量
    if threshold > 0.0:
        gate_ = torch.softmax(gate, dim=-1)
        sorted_gate, sorted_indices = torch.sort(gate_, dim=-1, descending=True)
        cum_scores = torch.cumsum(sorted_gate, dim=-1)
        # 动态计算 k（处理全不满足阈值的情况）
        above_threshold = cum_scores >= threshold
        has_any_above = above_threshold.any(dim=-1, keepdim=True)
        topk = above_threshold.int().argmax(dim=-1, keepdim=True) + 1
        topk = torch.where(has_any_above, topk, torch.ones_like(topk) * topk_param)

        # 限制最大 k 值
        topk = torch.clamp(topk, max=topk_param)

        # 生成固定形状的索引并将top1的索引填充到未满足阈值的部分
        index_shape = gate.size()[:-1] + (topk_param,)
        indices = torch.arange(topk_param, device=gate.device).expand(index_shape)
        mask = (indices < topk).int() 
        top_block_indices = torch.gather(sorted_indices, -1, indices) * mask + (1 - mask) * sorted_indices[..., 0:1]
        # top_block_indices的形状为 [b, h, s, topk]
        # 打印选中块的相似度得分和
        layer_name_set = {"double_block_1", "double_block_6", "double_block_12", 
                             "single_block_1", "single_block_12", "single_block_24"} # FOR XL(D12 & S24)
        if int(os.environ["RANK"]) <= 0 and layer_name in layer_name_set:
            with torch.no_grad():
                top_block_scores = torch.gather(sorted_gate, -1, indices) * mask 
                top_block_scores = top_block_scores.sum(dim=-1, keepdim=True)
                print(f"""
                        {layer_name} topk_param={topk_param}
                        topk={topk[0,0,[0,topk.size(-1)//2, -1],:].detach().cpu().tolist()}
                        top_block_scores={top_block_scores[0,0,[0,top_block_scores.size(-2)//2, -1],:].detach().cpu().tolist()}
                     """)    
    else:
        _, top_block_indices = gate.topk(k=topk_param, dim=-1)  # 取索引，形状为 [b, h, s, topk]
    
    block_t_start = top_block_indices // (num_blocks_h * num_blocks_w)
    block_h_start = (top_block_indices % (num_blocks_h * num_blocks_w)) // num_blocks_w
    block_w_start = top_block_indices % num_blocks_w
    
    # idx 到每个维度上的起始和结束索引
    block_t_start = block_t_start * block_shape[0]
    block_h_start = block_h_start * block_shape[1]
    block_w_start = block_w_start * block_shape[2]
    block_t_end = torch.min(block_t_start + block_shape[0], torch.tensor(image_shape[0]))
    block_h_end = torch.min(block_h_start + block_shape[1], torch.tensor(image_shape[1]))
    block_w_end = torch.min(block_w_start + block_shape[2], torch.tensor(image_shape[2]))
    top_block_indices = torch.stack([block_t_start, block_t_end, block_h_start, block_h_end, block_w_start, block_w_end], dim=-1)
    
    if enforce_share_topk_block:
        # top_block_indices为[b,h,block_num,topk,6]时，需要将其转换为[b,h,s,topk,6]
        # 转换逻辑：对于同一个block内部的数据，将其复制block_size次，最后得到s长度的序列
        output = torch.zeros((batch_size, num_heads, seq_len, topk_param, 6), device=top_block_indices.device)
        image_length=np.prod(image_shape)
        output_review = output[:, :, :image_length, :, :].view(batch_size, num_heads,
                                                  image_shape[0], image_shape[1],image_shape[2], topk_param, 6)
        for t in range(num_blocks_t):
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    block_index = t * num_blocks_h * num_blocks_w + h * num_blocks_w + w
                    start_t = t * block_shape[0]
                    end_t = min(start_t + block_shape[0], image_shape[0])
                    start_h = h * block_shape[1]
                    end_h = min(start_h + block_shape[1], image_shape[1])
                    start_w = w * block_shape[2]
                    end_w = min(start_w + block_shape[2], image_shape[2])
                    block_choose_index = top_block_indices[:, :, block_index, :, :]
                    block_choose_index = block_choose_index[:, :, None, None, None, :, :] # [b,h,1,1,1,topk,6]
                    output_review[:, :, start_t:end_t, start_h:end_h, start_w:end_w, :, :] = block_choose_index 
        top_block_indices = output
    return top_block_indices