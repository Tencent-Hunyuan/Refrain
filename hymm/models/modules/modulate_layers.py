from typing import Callable

import torch
import torch.nn as nn


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = act_layer()
        self.linear = nn.Linear(
            hidden_size, factor * hidden_size, bias=True, **factory_kwargs
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, is_token_replace=False, token_replace_vec=None) -> torch.Tensor:

        x_out = self.linear(self.act(x))

        if is_token_replace:
            x_token_replace_out = self.linear(self.act(token_replace_vec))
            return x_out, x_token_replace_out
        else:
            return x_out

@torch.autocast("cuda", dtype=torch.float32)
def modulate(x, shift=None, scale=None, is_token_replace=False, tr_shift=None, tr_scale=None, first_frame_mask=None):
    """modulate by shift and scale

    Args:
        x (torch.Tensor): input tensor.
        shift (torch.Tensor, optional): shift tensor. Defaults to None.
        scale (torch.Tensor, optional): scale tensor. Defaults to None.

    Returns:
        torch.Tensor: the output tensor after modulate.
    """

    if is_token_replace:
        x_list = []
        assert x.shape[0] == first_frame_mask.shape[0], f"x.shape[0]: {x.shape[0]}, first_frame_mask.shape[0]: {first_frame_mask.shape[0]}"
        for i in range(first_frame_mask.shape[0]):
            x_zero_i = x[i, :int(first_frame_mask[i].item())].unsqueeze(0) * (1 + tr_scale[i].unsqueeze(0).unsqueeze(1)) + tr_shift[i].unsqueeze(0).unsqueeze(1)
            x_orig_i = x[i, int(first_frame_mask[i].item()):].unsqueeze(0) * (1 + scale[i].unsqueeze(0).unsqueeze(1)) + shift[i].unsqueeze(0).unsqueeze(1)
            x_i = torch.concat((x_zero_i, x_orig_i), dim=1)
            x_list.append(x_i)
        x = torch.concat(x_list, dim=0)
        return x
    else:
        if scale is None and shift is None:
            return x
        elif shift is None:
            return x.float() * (1 + scale.unsqueeze(1))
        elif scale is None:
            return x.float() + shift.unsqueeze(1)
        else:
            return x.float() * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@torch.autocast("cuda", dtype=torch.float32)
def apply_gate(x, gate=None, tanh=False, is_token_replace=False, tr_gate=None, first_frame_mask=None):
    """AI is creating summary for apply_gate

    Args:
        x (torch.Tensor): input tensor.
        gate (torch.Tensor, optional): gate tensor. Defaults to None.
        tanh (bool, optional): whether to use tanh function. Defaults to False.

    Returns:
        torch.Tensor: the output tensor after apply gate.
    """

    if is_token_replace:
        if gate is None:
            return x
        if tanh:
            x_list = []
            for i in range(first_frame_mask.shape[0]):
                x_zero_i = x[i, :int(first_frame_mask[i].item())].unsqueeze(0) * tr_gate[i].unsqueeze(0).unsqueeze(1).tanh()
                x_orig_i = x[i, int(first_frame_mask[i].item()):].unsqueeze(0) * gate[i].unsqueeze(0).unsqueeze(1).tanh()
                x_i = torch.concat((x_zero_i, x_orig_i), dim=1)
                x_list.append(x_i)
            x = torch.concat(x_list, dim=0)
            return x
        else:
            x_list = []
            for i in range(first_frame_mask.shape[0]):
                x_zero_i = x[i, :int(first_frame_mask[i].item())].unsqueeze(0) * tr_gate[i].unsqueeze(0).unsqueeze(1)
                x_orig_i = x[i, int(first_frame_mask[i].item()):].unsqueeze(0) * gate[i].unsqueeze(0).unsqueeze(1)
                x_i = torch.concat((x_zero_i, x_orig_i), dim=1)
                x_list.append(x_i)
            x = torch.concat(x_list, dim=0)
            return x
    else:
        if gate is None:
            return x
        if tanh:
            return x.float() * gate.unsqueeze(1).tanh()
        else:
            return x.float() * gate.unsqueeze(1)


def ckpt_wrapper(module):
    def ckpt_forward(*inputs):
        outputs = module(*inputs)
        return outputs

    return ckpt_forward


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")
