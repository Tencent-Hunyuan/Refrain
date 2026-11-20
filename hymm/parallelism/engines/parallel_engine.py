"""Dense-model training with public PyTorch FSDP2 and AdamW.

Checkpoint directories use torch.distributed.checkpoint. Model tensor keys are
kept unchanged; the engine wrapper is excluded from saved state dictionaries.
"""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
from ..parallel_states import get_parallel_state


def get_optimizer_config(optimizer_class, optimizer_kwargs):
    return {"optimizer_class": optimizer_class, "optimizer_kwargs": optimizer_kwargs}


class BaseParallelEngine(nn.Module):
    def __init__(self, model, *, optimizer_config=None, get_lr_scheduler_func=None,
                 parallel_dims=None, enable_autocast=True, gradient_accumulation_steps=1,
                 enable_gradient_checkpointing=True, auto_benchmark=False, m_microbatch=1):
        super().__init__()
        self.parallel_dims = parallel_dims or get_parallel_state()
        if self.parallel_dims.pp_enabled or self.parallel_dims.ep_enabled:
            raise ValueError("Only the released dense DP/FSDP+SP configuration is supported.")
        self.enable_pp = False
        self.enable_autocast = enable_autocast
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        self.micro_steps = 0
        self._loss_closure, self._cached = None, {}
        if enable_gradient_checkpointing and optimizer_config is not None:
            self.apply_ac(model)
        if dist.is_initialized() and dist.get_world_size() > 1:
            model = self.apply_fsdp(model)
        self.model = model
        self.optimizer = (optimizer_config["optimizer_class"](
            [p for p in model.parameters() if p.requires_grad],
            **optimizer_config["optimizer_kwargs"]) if optimizer_config is not None else None)
        self.scheduler = get_lr_scheduler_func(self.optimizer) if get_lr_scheduler_func and self.optimizer is not None else None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def apply_ac(self, model):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, CheckpointImpl)
        for name in ("double_blocks", "single_blocks"):
            blocks = getattr(model, name, ())
            for i, block in enumerate(blocks):
                if block is not None:
                    blocks[i] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        return model

    def apply_fsdp(self, model):
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-rank training uses FSDP2 and requires CUDA.")
        options = dict(mesh=self.parallel_dims.fsdp_mesh,
                       mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32))
        for name in ("double_blocks", "single_blocks"):
            for block in getattr(model, name, ()):
                if block is not None: fully_shard(block, **options)
        fully_shard(model, **options)
        return model

    def register_loss_closure(self, closure):
        self._loss_closure = closure

    def get_cached_result(self, key):
        return self._cached[key]

    def forward(self, *args, **kwargs):
        closure, self._loss_closure = self._loss_closure, None
        device_type = next(self.model.parameters()).device.type
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.enable_autocast and device_type == "cuda" else nullcontext()
        with context:
            output = self.model(*args, **kwargs)
        if closure is not None:
            loss, terms = closure(output, model_input={"args": args, "kwargs": kwargs})
            self._cached = {"loss": loss, "loss_dict": terms}
        return output

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        norm = torch.nn.utils.clip_grad_norm_(list(parameters), max_norm, foreach=False)
        return norm.full_tensor() if hasattr(norm, "full_tensor") else norm

    def zero_grad(self, set_to_none=True):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.optimizer is None:
            raise RuntimeError("This engine was constructed for inference without an optimizer.")
        self.micro_steps += 1
        if self.micro_steps % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step()
            self.zero_grad()

    def get_last_lr(self):
        return self.scheduler.get_last_lr() if self.scheduler else [g["lr"] for g in self.optimizer.param_groups]

    def save_checkpoint(self, output_dir, client_state=None, tag=None):
        import random
        import numpy as np
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        if self.optimizer is None:
            raise RuntimeError("Saving training state requires an optimizer.")
        target = Path(output_dir) / (tag or f"global_step-{self.micro_steps // self.gradient_accumulation_steps}")
        target.mkdir(parents=True, exist_ok=True)
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=target / "distributed")
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
        torch.save(rng, target / f"rng_rank_{rank}.pt")
        if rank == 0:
            torch.save({"client_state": client_state, "micro_steps": self.micro_steps,
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None}, target / "trainer.pt")
        if dist.is_initialized(): dist.barrier()
        return str(target)

    def load_checkpoint(self, checkpoint, load_optimizer_states=True):
        import random
        import numpy as np
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_state_dict, get_model_state_dict, set_state_dict, set_model_state_dict)
        path = Path(checkpoint)
        if not (path / "distributed" / ".metadata").is_file():
            raise ValueError("Expected a public Refrain checkpoint with distributed/.metadata and trainer.pt. "
                             "For a weights-only start, provide a safetensors backbone and leave resume unset.")
        load_optimizer_states = load_optimizer_states and self.optimizer is not None
        if load_optimizer_states:
            model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
            state = {"model": model_sd, "optimizer": optim_sd}
            dcp.load(state, checkpoint_id=path / "distributed")
            set_state_dict(self.model, self.optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
        else:
            state = {"model": get_model_state_dict(self.model)}
            dcp.load(state, checkpoint_id=path / "distributed")
            set_model_state_dict(self.model, state["model"])
        # Load only checkpoints from a trusted training run.
        extra = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
        self.micro_steps = int(extra.get("micro_steps", 0))
        if load_optimizer_states and self.scheduler is not None and extra.get("scheduler") is not None:
            self.scheduler.load_state_dict(extra["scheduler"])
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng_path = path / f"rng_rank_{rank}.pt"
        if load_optimizer_states and rng_path.is_file():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng["cuda"] is not None: torch.cuda.set_rng_state(rng["cuda"])
        return str(path), extra
