# import
import os
import json
import pickle as pkl
import torch
from hymm.utils.logging_ import main_print
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from safetensors.torch import save_file, load_file
import torch.distributed as dist


def save_checkpoint_without_optim(transformer, rank, output_dir, scalar_states=None):
    cpu_state = transformer.state_dict()
    cpu_state = {k: v.full_tensor().cpu().clone() for k, v in cpu_state.items()} 

    global_step = scalar_states.train_steps
    # sync scalar_state
    if dist.is_available() and dist.is_initialized():
        scalar_states_list = [None for _ in range(dist.get_world_size())]
        torch.distributed.all_gather_object(scalar_states_list, scalar_states.to_dict())

    if rank <= 0:
        save_dir = os.path.join(output_dir, 'checkpoints', f"global_step-{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        # save using safetensors
        weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
        save_file(cpu_state, weight_path)

        scalar_states_path = os.path.join(save_dir, "scalar_states.pkl")
        pkl.dump(scalar_states_list, open(scalar_states_path, "wb"))

        config_dict = dict(transformer.config)
        if "dtype" in config_dict:
            del config_dict["dtype"]  # TODO
        if "moe_config" in config_dict and hasattr(config_dict["moe_config"], "to_dict"):
            config_dict["moe_config"] = config_dict["moe_config"].to_dict()
        config_path = os.path.join(save_dir, "config.json")
        # save dict as json
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)


def resume_training(model, checkpoint_dir, convert_state_dict_16168_to_16164=False):
    weight_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
    scalar_states_path = os.path.join(checkpoint_dir, "scalar_states.pkl")

    data = load_file(weight_path)
    model_weights = {k: v for k, v in data.items() if k != "__metadata__"}

    if convert_state_dict_16168_to_16164:
        tmp_state_dict = {}
        for k in model_weights.keys():
            if k == 'img_in.proj.weight':
                #tmp_state_dict[k] = torch.cat([model_weights['img_in.proj.weight'][:, :32, :, :],
                #                               model_weights['img_in.proj.weight'][:, -33:, :, :]], dim=1)
                tmp_state_dict[k] = model_weights['img_in.proj.weight'][:, :32, :, :]
            elif k == 'final_layer.linear.weight':
                tmp_state_dict[k] = model_weights['final_layer.linear.weight'][:32]
            elif k == 'final_layer.linear.bias':
                tmp_state_dict[k] = model_weights['final_layer.linear.bias'][:32]
            else:
                tmp_state_dict[k] = model_weights[k]
        state_dict = tmp_state_dict
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(model_weights, strict=True)

    scalar_states_list = pkl.load(open(scalar_states_path, "rb"))
    return model, scalar_states_list


