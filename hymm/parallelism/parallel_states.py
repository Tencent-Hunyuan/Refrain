"""SP and FSDP meshes using the public HunyuanVideo-1.5 DeviceMesh design.

The data mesh excludes SP ranks: ranks generating parts of the same sample
share inputs. FSDP shards parameters across both data and sequence ranks.
"""
import os
from dataclasses import dataclass
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelDims:
    dp_replicate: int = 1
    dp_shard: int = -1
    sp: int = 1
    tp: int = 1
    pp: int = 1
    world_size: int = -1
    ep: int = 1

    def __post_init__(self):
        if self.world_size == -1:
            self.world_size = dist.get_world_size() if dist.is_initialized() else int(os.getenv("WORLD_SIZE", "1"))
        if (self.tp, self.pp, self.ep) != (1, 1, 1):
            raise ValueError("The public dense runtime supports DP/FSDP and SP; tp, pp and ep must be 1.")
        if min(self.dp_replicate, self.sp, self.world_size) < 1:
            raise ValueError("Parallel degrees must be positive.")
        divisor = self.dp_replicate * self.sp
        if self.dp_shard == -1:
            if self.world_size % divisor:
                raise ValueError("WORLD_SIZE must be divisible by dp_replicate * sp.")
            self.dp_shard = self.world_size // divisor
        if self.dp_shard < 1 or self.dp_shard * divisor != self.world_size:
            raise ValueError("dp_replicate * dp_shard * sp must equal WORLD_SIZE.")
        self.device_mesh = None

    def build_mesh(self, device_type="cuda"):
        if self.device_mesh is None:
            mesh = init_device_mesh(device_type,
                (self.dp_replicate, self.dp_shard, self.sp),
                mesh_dim_names=("dp_replicate", "dp_shard", "sp"))
            mesh[("dp_replicate", "dp_shard")]._flatten(mesh_dim_name="dp")
            mesh[("dp_shard", "sp")]._flatten(mesh_dim_name="dp_shard_sp")
            mesh[("dp_replicate", "dp_shard", "sp")]._flatten(mesh_dim_name="dp_sp")
            self.device_mesh = mesh
            self.device_mesh_for_pp_dp_sp_tp = mesh
            self.dp_mesh, self.sp_mesh = mesh["dp"], mesh["sp"]
            self.sp_group = self.sp_mesh.get_group()
            self.sp_rank = self.sp_mesh.get_local_rank()
            self.fsdp_mesh = mesh[("dp_replicate", "dp_shard_sp")] if self.dp_replicate_enabled else mesh["dp_shard_sp"]
        set_parallel_state(self)
        return self.device_mesh

    @property
    def dp_enabled(self): return self.dp_replicate * self.dp_shard > 1
    @property
    def dp_replicate_enabled(self): return self.dp_replicate > 1
    @property
    def dp_shard_enabled(self): return self.dp_shard > 1
    @property
    def sp_enabled(self): return self.sp > 1
    @property
    def tp_enabled(self): return False
    @property
    def pp_enabled(self): return False
    @property
    def ep_enabled(self): return False
    @property
    def non_data_parallel_size(self): return self.sp


_state = None

def set_parallel_state(state):
    global _state
    _state = state


def init_parallel_state(device_type="cuda", **kwargs):
    state = ParallelDims(**kwargs)
    state.build_mesh(device_type)
    return state


def get_parallel_state():
    global _state
    if _state is None:
        _state = ParallelDims()
    return _state
