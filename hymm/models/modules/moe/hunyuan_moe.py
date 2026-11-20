from functools import partial
from einops import rearrange, reduce
from transformers.activations import ACT2FN

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import time
import logging

import einx
from colt5_attention import topk as maybe_differentiable_topk

# from hymm.parallelism.parallel_states import get_parallel_state
# from hymm.utils.moe_parallel import EPGroupGemm
#
# from hymm.utils.torch_utils import _Timer

MIN_EXPERT_CAPACITY = 4

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class MoeConfig():
    def __init__(
        self,
        num_experts=1,
        use_mixed_mlp_moe=False,
        num_shared_expert=1,
        moe_topk=1,
        intermediate_size=None,
        hidden_size=None,
        threshold_train=0.,
        threshold_eval=0.,
        capacity_factor_train=1.25,
        capacity_factor_eval=1.25,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        balance_loss_coef=1e-2,
        router_z_loss_coef=1e-3,
        fused_experts=False,
        use_padded_moe=False,
        moe_use_deepep=False,
        moe_pad_seqlen=0,
        enable_moe_token_mask=False
    ):
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

        self.threshold_train = threshold_train
        self.threshold_eval = threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.differentiable_topk = differentiable_topk
        self.differentiable_topk_fused = differentiable_topk_fused
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.fused_experts = fused_experts
        self.use_padded_moe = use_padded_moe
        self.moe_use_deepep = moe_use_deepep
        self.moe_pad_seqlen = moe_pad_seqlen
        self.enable_moe_token_mask = enable_moe_token_mask

    def to_dict(self):
        return self.__dict__

class HunYuanExpert(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = torch.nn.Parameter(
            torch.empty(config.num_experts, self.intermediate_size, self.hidden_size),
            requires_grad=True
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(config.num_experts, self.intermediate_size, self.hidden_size),
            requires_grad=True
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(config.num_experts, self.hidden_size, self.intermediate_size),
            requires_grad=True
        )

        def init_param(param):
            import math
            for i in range(param.shape[0]):
                nn.init.kaiming_uniform_(param[i], a=math.sqrt(5), nonlinearity='leaky_relu')

        init_param(self.up_proj)
        init_param(self.gate_proj)
        init_param(self.down_proj)

        self.act_fn = ACT2FN['silu']

    def forward(self, hidden_states, index=None, cumsum=None):
        if cumsum is not None:
            assert get_parallel_state().ep_enabled
            assert hidden_states.dtype in [torch.bfloat16, torch.float16]
            gate_proj_weight = getattr(self, f'gate_proj_{ep_rank}').bfloat16()
            up_proj_weight = getattr(self, f'up_proj_{ep_rank}').bfloat16()
            down_proj_weight = getattr(self, f'down_proj_{ep_rank}').bfloat16()

            final_permute_tokens = EPGroupGemm.apply(
                hidden_states,
                cumsum,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            )
            return final_permute_tokens
        elif index is not None:
            if get_parallel_state().ep_enabled:
                ep_rank = dist.get_rank(group=get_parallel_state().ep_group)
                gate_out = torch.matmul(hidden_states, getattr(self, f'gate_proj')[index].transpose(0, 1))
                up_out = torch.matmul(hidden_states, getattr(self, f'up_proj')[index].transpose(0, 1))
                inter_out = self.act_fn(gate_out) * up_out
                out = torch.matmul(inter_out, getattr(self, f'down_proj')[index].transpose(0, 1))
            else:
                gate_out = torch.matmul(hidden_states, self.gate_proj[index].transpose(0, 1))
                up_out = torch.matmul(hidden_states, self.up_proj[index].transpose(0, 1))
                inter_out = self.act_fn(gate_out) * up_out
                out = torch.matmul(inter_out, self.down_proj[index].transpose(0, 1))
            return out
        else:
            raise NotImplementedError("Only support index and cumsum")



class HunYuanMLP(nn.Module):
    def __init__(self, config: MoeConfig, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if is_shared_mlp:
            self.intermediate_size = config.intermediate_size * config.num_shared_expert
        else:
            self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN['silu']

    def forward(self, x, attn_out=None):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

def topkgating(
        gate_logits, # input logis
        top_n, # top_k
        num_gates, # number of experts
        threshold, # threshold for top-n
        capacity_factor, # capacity factor
        differentiable_topk, # True or False, use differential topk
        differentiable_topk_fused, # True or False, wether fused topk
        return_loss, # True or False, wether return loss
        # straight_through_dispatch_tensor, #TODO True or False, wether use straight through dispatch tensor
        noise_mult = 1.,
        noise_gates = False,
        eps = 1e-9
):
    """Implements Top1Gating on logits."""
    b, group_size, dim = gate_logits.shape

    top_n_minus_1 = top_n - 1
    threshold = cast_tuple(threshold, top_n_minus_1)
    assert len(threshold) == top_n_minus_1
    threshold = torch.tensor([eps, *threshold]).to(gate_logits.device)


    # Each sequence sends (at most?) expert_capacity positions to each expert.
    # Static expert_capacity dimension is needed for expert batch sizes
    # expert_capacity need times top_k and batch_size
    expert_capacity =  b * min(group_size, int(top_n * (group_size * capacity_factor) / num_gates))
    expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
    expert_capacity_f = float(expert_capacity)

    maybe_noised_gate_logits = gate_logits # shape: (b, group_size, num_gates)

    if noise_gates:
        noise = gumbel_noise(maybe_noised_gate_logits) # shape: (b, group_size, num_gates)
        maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult # shape: (b, group_size, num_gates)

    # raw_gates = maybe_noised_gate_logits.softmax(dim = -1) # shape: (b, group_size, num_gates)

    # kevinkhwu: float softmax
    raw_gates = maybe_noised_gate_logits.float().softmax(dim = -1) # shape: (b, group_size, num_gates)
    raw_gates = raw_gates.to(maybe_noised_gate_logits.dtype)

    # find top N experts per position
    topk_func = partial(
        maybe_differentiable_topk,
        non_differentiable = not differentiable_topk,
        fused = differentiable_topk_fused # use triton fused coordinate descent if possible by default
    )
    topk_return = topk_func(raw_gates, k = top_n)
    gate_indices = topk_return.indices # shape: (b, group_size, top_n)
    if differentiable_topk:
        # allow for differentiable topk using coordinate descent
        # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention
        gates = topk_return.coor_descent_values # shape: (b, group_size, top_n)
    else:
        gates = topk_return.values # shape: (b, group_size, top_n)
    # move the top-n dimension to be first
    gates = rearrange(gates, '... k -> k ...') # shape: (top_n, b, group_size)
    gate_indices = rearrange(gate_indices, '... k -> k ...') # shape: (top_n, b, group_size)

    # create masks
    one_hot_gate_indices = F.one_hot(gate_indices, num_gates) # shape: (top_n, b, group_size, num_gates)
    mask = one_hot_gate_indices.float() # shape: (top_n, b, group_size, num_gates)
    count_top_k_rate = torch.sum(mask, dim=[0, 1, 2]).detach().cpu()/torch.sum(mask).detach().cpu()
    
    mask_1 = mask[0] # shape: (b, group_size, num_gates), needed for balancing loss
    count_top_1_rate = torch.sum(mask_1, dim=[0, 1]).detach().cpu()/torch.sum(mask_1).detach().cpu()


    # normalize top-n gate scores
    denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min = eps) # shape: (1, b, group_size)
    gates = gates / denom # shape: (top_n, b, group_size) (normalized)

    # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
    # optimal threshold was ~ 0.2
    # generalized to more than 2 experts

    probs = torch.zeros_like(gates).uniform_(0., 1.) # shape: (top_n, b, group_size)
    should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps)) # shape: (top_n, b, group_size), boolean

    # tokens should always be routed to first expert
    # threshold for first expert already set to very small number, but just in case
    should_route[0, ...] = True # shape: (top_n, b, group_size), boolean, first slice is True

    # rearrange(should_route.float(), '... -> ... 1') shape: (top_n, b, group_size, 1)
    mask *= rearrange(should_route.float(), '... -> ... 1') # shape: (top_n, b, group_size, num_gates), updated based on should_route

    mask_cumsum = cumsum_exclusive(mask, dim = -2) # shape: (top_n, b, group_size, num_gates), cumsum along group_size dim

    # capacity
    prev_expert_count = 0.
    for n in range(top_n):
        position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n] # shape: (b, group_size, num_gates)

        # Remove the elements that don't fit. (batch, sequence, experts)
        mask[n] *= (position_in_expert < expert_capacity_f).float() # mask[n] updated, shape: (b, group_size, num_gates)

        # How many examples in this sequence go to this expert - needed for the next iteration as offset
        prev_expert_count = reduce(mask[n], '... gs e -> ... 1 e', 'sum') + prev_expert_count # prev_expert_count shape: (b, 1, num_gates)


    mask_flat = reduce(mask, 'k ... gs e -> k ... gs', 'sum') # shape: (top_n, b, group_size) (gs is group_size)
    # (k, batch, sequence) - weighted assignment
    # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
    gates = gates * mask_flat # shape: (top_n, b, group_size), these are the final combining weights for chosen experts


    # balance losses - (batch, experts)
    # We want to equalize the fraction of the batch assigned to each expert
    if return_loss:
        density_1 = reduce(mask_1, '... s e -> ... e', 'mean') # shape: (b, num_gates)
        density_1_proxy = reduce(raw_gates, '... s e -> ... e', 'mean') # shape: (b, num_gates)

        balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2) # scalar


    # calculate the router z-loss proposed in paper
    if return_loss:
        router_z_loss_tmp = torch.logsumexp(gate_logits, dim = -1) # shape: (b, group_size)
        router_z_loss_tmp = torch.square(router_z_loss_tmp) # shape: (b, group_size)
        router_z_loss = router_z_loss_tmp.mean() # scalar

    topk_idx = gate_indices.permute(1, 2, 0).contiguous()
    topk_weight = gates.permute(1, 2, 0).contiguous()
    return balance_loss, router_z_loss, topk_idx, topk_weight, count_top_1_rate, count_top_k_rate
    

class HunYuanTopKGate(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.wg = nn.Linear(config.hidden_size, config.num_experts, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.wg(hidden_states)
        gate_output = topkgating(
            logits,
            top_n = self.config.moe_topk,
            num_gates = self.config.num_experts,
            threshold = self.config.threshold_train if self.training else self.config.threshold_eval,
            capacity_factor = self.config.capacity_factor_train if self.training else self.config.capacity_factor_eval,
            differentiable_topk = self.config.differentiable_topk,
            differentiable_topk_fused = self.config.differentiable_topk_fused,
            return_loss = True
        )
        return gate_output


class HunYuanMoE(nn.Module):
    def __init__(self, config: MoeConfig, mlp_class="HunYuanMLP"):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.mlp_class = mlp_class
        
        if config.use_mixed_mlp_moe:
            self.shared_mlp = HunYuanMLP(config, is_shared_mlp=True)
        self.gate = HunYuanTopKGate(config)
        self.experts = HunYuanExpert(config)

        self.balance_loss_coef = self.config.balance_loss_coef
        self.router_z_loss_coef = self.config.router_z_loss_coef

        self.l_moe = [None, None]        
        self.count_top = [None, None]

    def get_moe_loss(self):
        return self.l_moe

    def get_count_top(self):
        return self.count_top
        
    def forward(self, hidden_states, attn_out=None, **kwargs):
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states, attn_out)

        balance_loss, router_z_loss, topk_idx, topk_weight, count_top_1_rate, count_top_k_rate = self.gate(hidden_states)
        if get_parallel_state().ep_enabled:
            from hymm.utils.moe_parallel import preprocess, token_pre_all2all, tokens_post_all2all

            selected_experts = topk_idx.view(-1, topk_idx.shape[-1])
            topk_weight = topk_weight.view(-1, topk_weight.shape[-1])

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
                preprocess(
                    expert_mask=expert_mask,
                    num_experts=self.num_experts,
                    ep_group=get_parallel_state().ep_group,
                )
            )

            permute_tokens, routing_map, local_input_permutation_mapping, org_hidden_states_shape = token_pre_all2all(
                hidden_states=hidden_states,
                expert_mask=expert_mask,
                num_experts=self.num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                ep_group=get_parallel_state().ep_group,
            )

            if self.config.fused_experts:
                cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)
                final_permute_tokens = self.experts(permute_tokens, cumsum=cumsum)
            else:
                cumsum = torch.cat([torch.tensor([0]), num_global_sum_tokens_per_local_expert.cumsum(dim=0)])

                # Loop over all available experts in the model and perform the computation on each expert
                final_permute_tokens = torch.zeros(
                    (permute_tokens.shape),
                    dtype=permute_tokens.dtype,
                    device=permute_tokens.device,
                )                    

                num_local_experts = self.num_experts // get_parallel_state().ep

                for expert_idx in range(num_local_experts):
                    start_idx = cumsum[expert_idx]
                    end_idx = cumsum[expert_idx + 1]

                    current_permute_tokens = permute_tokens[start_idx:end_idx]
                    final_permute_tokens[start_idx:end_idx] = self.experts(current_permute_tokens, index=expert_idx).to(final_permute_tokens.dtype)

            unpermute_tokens = tokens_post_all2all(
                expert_outputs=final_permute_tokens,
                selected_experts=selected_experts,
                num_experts=self.num_experts,
                input_splits=input_splits,
                output_splits=output_splits,
                num_global_tokens_per_local_expert=num_global_tokens_per_local_expert,
                routing_map=routing_map,
                local_input_permutation_mapping=local_input_permutation_mapping,
                org_hidden_states_shape=org_hidden_states_shape,
                routing_weights=topk_weight,
                ep_group=get_parallel_state().ep_group,
            )

            combined_output = unpermute_tokens.to(hidden_states.dtype).view(bsz, seq_len, hidden_size)
        else:
            flat_topk_idx = topk_idx.view(-1)
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            hidden_states = hidden_states.repeat_interleave(self.config.moe_topk, dim=0)

            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i in range(self.num_experts):
                chunk_hidden_states = hidden_states[flat_topk_idx == i]
                if chunk_hidden_states.numel() > 0:  # 只在有token时计算
                    y[flat_topk_idx == i] = self.experts(chunk_hidden_states, index=i).to(y.dtype)

            combined_output = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=2)
            combined_output = combined_output.reshape(bsz, seq_len, hidden_size)
            
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        self.l_moe[0] = weighted_balance_loss
        self.l_moe[1] = weighted_router_z_loss

        self.count_top[0] = count_top_1_rate
        self.count_top[1] = count_top_k_rate

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output

        return output
    
