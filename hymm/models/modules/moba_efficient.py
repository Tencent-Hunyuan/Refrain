import torch
import os

from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from functools import lru_cache
from einops import rearrange
import random

@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size, causal=False):
    """calc chunks that needs moba attention"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    
    # 每个batch的chunk数量
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    if causal:
        chunk_to_remove = cu_num_chunk[1:] - 1
        chunk_to_remain = torch.ones(
            (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
        )
        chunk_to_remain[chunk_to_remove] = False
        filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    else:
        filtered_chunk_indices = torch.arange(
            0, num_chunk, dtype=torch.int32, device=cu_seqlen.device
        )
        
    num_filtered_chunk = len(filtered_chunk_indices)
    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    )

class MixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        causal
    ):
        ctx.max_seqlen = max_seqlen
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)
        ctx.causal = causal

        # self attn
        self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=causal,
                dropout_p=0.0,
            )
        )

        # moba attn
        moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # convert lse shape hs -> sh ( follow the legacy mix attn logic )
        self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
        moba_attn_lse = moba_attn_lse_hs.t().contiguous()

        # output buffer [S, H, D], same shape as q
        output = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # flatten vS & H for index ops
        output_2d = output.view(-1, q.shape[2])

        # calc mixed_lse
        # minus max lse to avoid exp explosion
        max_lse_1d = self_attn_lse_sh.view(-1)
        max_lse_1d = max_lse_1d.index_reduce(
            0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
        )
        self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
        moba_attn_lse = (
            moba_attn_lse.view(-1)
            .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
            .reshape_as(moba_attn_lse)
        )

        mixed_attn_se_sh = self_attn_lse_sh.exp()
        moba_attn_se = moba_attn_lse.exp()

        mixed_attn_se_sh.view(-1).index_add_(
            0, moba_q_sh_indices, moba_attn_se.view(-1)
        )
        mixed_attn_lse_sh = mixed_attn_se_sh.log()

        # add attn output
        factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()  # [ vS, H ]
        self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
        output_2d += self_attn_out_sh.reshape_as(output_2d)

        # add moba output
        mixed_attn_lse = (
            mixed_attn_lse_sh.view(-1)
            .index_select(0, moba_q_sh_indices)
            .view_as(moba_attn_lse)
        )
        factor = (moba_attn_lse - mixed_attn_lse).exp()  # [ vS, H ]
        moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
        raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
        output = output.to(q.dtype)
        # add back max lse
        mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):

        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=ctx.causal,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        moba_k = moba_kv[:, 0]
        moba_v = moba_kv[:, 1]
        dmq, dmk, dmv = torch.empty_like(moba_q), torch.empty_like(moba_k), torch.empty_like(moba_v)
        _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_k,
            v=moba_v,
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq,
            dk=dmk,
            dv=dmv,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None


def moba_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    causal: bool = False,
    padding: bool = True,
    use_mean_padding=True,
) -> torch.Tensor:
    """An efficient version of moba implementation with triton kernels and flash-attn, the core logic:
    1. Calculate the chunks and the number of chunks, n = floor(data_size / chunk_size)
       - tokens in the tail chunk are reserved for self attn
       - tokens in other chunks will be processed in later steps
    2. K in each chunk will calculate mean value as the representative k, and Q will attend to these representative
    k to get the gate logit, which will be used to select topk chunks
    3. Select the topk chunks and get the dense q for each kv chunk pair and do the varlen attention
    4. Combine the varlen attn and self attn results via online softmax to get the final result

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    
    seqlen = q.shape[0]
    # 对每个batch单独进行padding
    if padding:
        batch_sizes = cu_seqlens[1:] - cu_seqlens[:-1]
        padded_batch_sizes = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size * moba_chunk_size
        pad_lens = padded_batch_sizes - batch_sizes
        assert (pad_lens == 0).all(), \
          f"padding length should be 0, padding has been performed before calling moba_attn_varlen, \
          batch_sizes: {batch_sizes}, padded_batch_sizes: {padded_batch_sizes}, pad_lens: {pad_lens}"
        
        # 对q,k,v进行padding
        padded_q = []
        padded_k = []
        padded_v = []
        
        for i in range(len(batch_sizes)):
            start = cu_seqlens[i]
            end = cu_seqlens[i+1]
            pad_len = pad_lens[i]
            if use_mean_padding:
                q_mean = q[start:end].mean(dim=0, keepdim=True)
                k_mean = k[start:end].mean(dim=0, keepdim=True)
                v_mean = v[start:end].mean(dim=0, keepdim=True)
                padded_q.append(torch.cat([q[start:end], q_mean.expand(pad_len, *q.shape[1:])], dim=0))
                padded_k.append(torch.cat([k[start:end], k_mean.expand(pad_len, *k.shape[1:])], dim=0))
                padded_v.append(torch.cat([v[start:end], v_mean.expand(pad_len, *v.shape[1:])], dim=0))
            else:
                padded_q.append(torch.cat([q[start:end], torch.zeros(pad_len, *q.shape[1:], device=q.device, dtype=q.dtype)], dim=0))
                padded_k.append(torch.cat([k[start:end], torch.zeros(pad_len, *k.shape[1:], device=k.device, dtype=k.dtype)], dim=0))
                padded_v.append(torch.cat([v[start:end], torch.zeros(pad_len, *v.shape[1:], device=v.device, dtype=v.dtype)], dim=0))
            #print(f"batch index: {i} padding {pad_len}")
        q = torch.cat(padded_q, dim=0)
        k = torch.cat(padded_k, dim=0)
        v = torch.cat(padded_v, dim=0)

        # 更新cu_seqlens
        new_cu_seqlens = torch.zeros_like(cu_seqlens)
        new_cu_seqlens[1:] = padded_batch_sizes.cumsum(dim=0)
        cu_seqlens = new_cu_seqlens
        max_seqlen = padded_batch_sizes.max().item()
    
    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size, causal=causal)
    #filtered_chunk_indices: chunk index list that excludes the last chunk of each batch

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    # num_filtered_chunk: 除去最后一个chunk，剩下的chunk数量
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=causal
        )
    self_attn_cu_seqlen = cu_chunk # 是一个累积的偏移量张量，记录了每个chunk在序列中的起始位置

    # filtered_kv是一个密集矩阵，只包含过滤后的chunk的kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    # filtered_kv： [ CHUNK_NUM * CHUNK_SIZE, 2, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1) # shape: [ F_N_CHUNK, SEQ ]

    # 每个chunk结束的位置，chunk_end是结束位置，所以需要加1
    chunk_end = cu_chunk[filtered_chunk_indices + 1]

    # chunk所属于batch结束的位置
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    batch_start = cu_seqlens[chunk_to_batch[filtered_chunk_indices]]

    if causal:
        # pos(q) < I * B 等价于 I * B > pos(q), 选择q位置后面的块
        gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]  #过滤掉query对应的chunk的位置相关性(jef, 因为在MixedAttention中有对应chunk的计算，所以这里会mask掉)
        gate_batch_end_mask = gate_seq_idx >= batch_end[:, None] #超出当前batch范围的样本
        gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    else:
        gate_inf_mask = gate_seq_idx >= batch_end[:, None]
        # 在生成gate_inf_mask时增加当前块排除逻辑(!!!!! 避免当前块重复计算，在self-attn中), 强制排除当前块  
        gate_current_block_mask = (gate_seq_idx >= chunk_end[:, None] - moba_chunk_size) & (gate_seq_idx < chunk_end[:, None])
        
        # 来自其他batch的chunk，需要排除
        gate_avg_key_with_future_mask = gate_seq_idx < batch_start[:, None]
        gate_inf_mask = gate_inf_mask | gate_current_block_mask | gate_avg_key_with_future_mask 

    #gate 表示avg block feat与每个query的相关性
    gate_ = gate.clone().detach()
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    if int(os.environ["RANK"]) <= 0 and random.random() < 0.001: # 偶尔打印一下
        #print(f"Selected block indices({gate_top_k_idx.shape}:", 
        #      f"Query Chunk Index 0, {gate_top_k_idx[:, 0, 0:3]}",
        #      f"Query Chunk Index -1, {gate_top_k_idx[:, 0, -3-moba_chunk_size:-moba_chunk_size]}",
        #)
        with torch.no_grad():
            gate_current_block_mask = (gate_seq_idx >= chunk_end[:, None] - moba_chunk_size) & (gate_seq_idx < chunk_end[:, None])
            adjusted_gate = torch.softmax(gate_, dim=0)
            topk_weight = adjusted_gate.gather(0, gate_top_k_idx).sum()
            current_block_weight=adjusted_gate.mul(gate_current_block_mask.unsqueeze(1)).sum()
            numerator = topk_weight + current_block_weight
            denominator = adjusted_gate.sum() + 1e-6
            print(f"""
                      --Selected similarity_score: {numerator / denominator},
                      --TopK(K={moba_topk}) Weight: {topk_weight/denominator},
                      --Current Block Weight: {current_block_weight/denominator}
                   """)
    # gate_top_k_idx: [ TOPK, HEAD, SEQ]

    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)


    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[-1]
    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    # s: 序列长度, x: block的大小, h: head, d: dim
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d") 
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    # Wrapping up the flash attn call and online softmax dlse inside MixedAttention class
    output = MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        causal
    )
    if padding:
        unpadded_output = []
        for i in range(len(batch_sizes)):
            start = cu_seqlens[i]
            end = cu_seqlens[i+1] - pad_lens[i]
            unpadded_output.append(output[start:end])
        output = torch.cat(unpadded_output, dim=0)
    return output


def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype, requires_grad):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # gen qkv
    q = torch.randn(
        (seqlen * batch, num_q_head, headdim), dtype=dtype, device=device, requires_grad=requires_grad
    )
    k = torch.randn(
        (seqlen * batch, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=requires_grad
    )
    v = torch.randn(
        (seqlen * batch, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=requires_grad
    )

    if batch > 1:
        cu_seqlen =  [i * seqlen for i in range(batch + 1)]
        cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)
    else:
        cu_seqlen = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    # max_seqlen
    max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])
    
    return q, k, v, cu_seqlen, max_seqlen.item()


def test_attn_varlen_moba_speed(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk, 
                                dtype=torch.bfloat16,
                                causal=True,
                                forward_only=False):
    import time
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch, seqlen, head, head, head_dim, dtype, requires_grad=True)
    vo_grad = torch.randn_like(q)
    
    
    # Warmup
    warmup_iters = 5
    perf_test_iters = 20
    # Warmup
    for _ in range(warmup_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        if not forward_only:
            torch.autograd.backward(o, vo_grad)
    
    torch.cuda.synchronize()
    start_flash = time.perf_counter()
    for _ in range(perf_test_iters):
        o = flash_attn_varlen_func(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen, causal=True)
        if not forward_only:
            torch.autograd.backward(o, vo_grad)
        
    torch.cuda.synchronize()
    time_flash = (time.perf_counter() - start_flash) / perf_test_iters * 1000


    # Warmup
    for _ in range(warmup_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk, causal=causal)
        if not forward_only:
            torch.autograd.backward(om, vo_grad)

        
    torch.cuda.synchronize()
    start_moba = time.perf_counter()
    for _ in range(perf_test_iters):
        om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk, causal=causal)
        if not forward_only:
            torch.autograd.backward(om, vo_grad)
    
    torch.cuda.synchronize()
    time_moba = (time.perf_counter() - start_moba) / perf_test_iters * 1000
    
    print(f"\nbatch:{batch} head:{head} seqlen:{seqlen} chunk:{moba_chunk_size} topk:{moba_topk}")
    print(f"Flash: {time_flash:.2f}ms, MoBA: {time_moba:.2f}ms")
    print(f"Speedup:  {time_flash / time_moba:.2f}x")
    
    return time_flash, time_moba, time_flash / time_moba

def simple_test():
    q, k, v, cu_seqlen, max_seqlen = generate_data(batch=1, seqlen=30000, num_q_head=24, num_kv_head=24, headdim=128, dtype=torch.bfloat16)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    om = moba_attn_varlen(q, k, v, cu_seqlen, max_seqlen, moba_chunk_size=1024, moba_topk=3)
    vo_grad = torch.randn_like(q)
    res = torch.autograd.backward(om, vo_grad)
    print(res)

def speed_test(save_name="speed_test.png", causal=True, forward_only=False):
    import matplotlib.pyplot as plt
    import numpy as np
    
    results = {}
    time_results = {}  
    head = 16
    for batch in [1]:
        for seq in [10000, 30000, 60000, 100000]:
            for head_dim in [128]:
                for moba_chunk_size in [256, 512, 1024]:
                    for moba_topk in [3, 5, 7, 10]:
                        print(f"batch:{batch} head:{head} seqlen:{seq} chunk:{moba_chunk_size} topk:{moba_topk}")
                        time_flash, time_moba, speedup = test_attn_varlen_moba_speed(batch=batch, head=head, seqlen=seq, head_dim=head_dim, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk,
                                                                                     causal=causal,forward_only=forward_only)
                        key = (batch, seq)
                        if key not in results:
                            results[key] = {}
                            time_results[key] = {} 
                        sub_key = moba_chunk_size
                        if sub_key not in results[key]:
                            results[key][sub_key] = []
                            time_results[key][sub_key] = [] 
                        results[key][sub_key].append((moba_topk, speedup))
                        time_results[key][sub_key].append((moba_topk, time_flash, time_moba)) 
                        print("-"*50)

    # 计算子图布局
    num_rows = len(set([k[0] for k in results.keys()]))
    num_cols = len(set([k[1] for k in results.keys()]))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

    # 遍历每个子图
    for i, batch in enumerate(set([k[0] for k in results.keys()])):
        for j, seq in enumerate(set([k[1] for k in results.keys()])):
            key = (batch, seq)
            ax = axes[i, j] if num_rows > 1 else axes[j]
            for moba_chunk_size, values in results[key].items():
                moba_topk_values, speedup_values = zip(*values)
                ax.plot(moba_topk_values, speedup_values, label=f"chunk={moba_chunk_size}")
                for (topk, flash_time, moba_time) in time_results[key][moba_chunk_size]:
                    ax.text(topk, speedup_values[moba_topk_values.index(topk)], f"{flash_time:.1f}/{moba_time:.1f}={flash_time/moba_time:.1f}", ha='center', va='bottom')
            ax.set_xlabel('moba_topk')
            ax.set_ylabel('Speedup (Flash / MoBA)')
            ax.set_title(f'batch={batch}, seq={seq}')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()



