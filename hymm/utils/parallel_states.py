import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from functools import cached_property


from hymm.parallelism.parallel_states import ParallelDims, get_parallel_state, init_parallel_state, set_parallel_state


class COMM_INFO:
    def __init__(self):
        self.sp_group = None
        self.tp_group = None
        self.sp_size = 1
        self.tp_size = 1
        self.global_rank = 0
        self.rank_within_spgroup = 0
        self.rank_within_tpgroup = 0
        self.parallel_dims = None
        self.device_mesh = None
        self.use_dynamic_ring_attention = False
        self.sp_stream = None
        self.sp_rank_list = None
        

nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_TEACHER_STUDENT_PARALLEL_STATE = False
_TENSOR_PARALLEL_STATE = False


def initialize_sequence_parallel_state_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    global _SEQUENCE_PARALLEL_STATE
    set_parallel_state(parallel_dims)
    _SEQUENCE_PARALLEL_STATE = parallel_dims.sp_enabled
    if parallel_dims.sp_enabled or parallel_dims.tp_enabled:
        initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention)
    else:
        nccl_info.sp_group = None
        nccl_info.tp_group = None
        nccl_info.sp_size = 1
        nccl_info.tp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_spgroup = 0
        nccl_info.rank_within_tpgroup = 0
        nccl_info.sp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.tp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.parallel_dims = parallel_dims
        nccl_info.device_mesh = device_mesh


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def set_tensor_parallel_state(state):
    global _TENSOR_PARALLEL_STATE
    _TENSOR_PARALLEL_STATE = state

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def get_tensor_parallel_state():
    return _TENSOR_PARALLEL_STATE

def initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    nccl_info.global_rank = rank
    nccl_info.sp_size = parallel_dims.sp
    nccl_info.tp_size = parallel_dims.tp
    nccl_info.global_rank = rank
    nccl_info.parallel_dims = parallel_dims
    nccl_info.device_mesh = device_mesh
    nccl_info.use_dynamic_ring_attention = use_dynamic_ring_attention
    if use_dynamic_ring_attention:
        nccl_info.sp_stream = torch.cuda.Stream()
        nccl_info.sp_rank_list = device_mesh["sp"].mesh.tolist()
    if nccl_info.sp_size > 1:
        set_sequence_parallel_state(True)
        nccl_info.sp_group = device_mesh.get_group(mesh_dim="sp")
        nccl_info.rank_within_spgroup = device_mesh.get_local_rank(mesh_dim="sp")
    if nccl_info.tp_size > 1:
        set_tensor_parallel_state(True)
        nccl_info.tp_group = device_mesh.get_group(mesh_dim="tp")
        nccl_info.rank_within_tpgroup = device_mesh.get_local_rank(mesh_dim="tp")

def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i


def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()