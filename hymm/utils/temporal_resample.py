"""Temporal resampling utilities for L4 conditioning."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def resample_latent_temporal(
    latent: torch.Tensor,
    src_frames: int,
    dst_frames: int,
    align_corners: bool = False,
    mode: str = "trilinear",
) -> torch.Tensor:
    """
    Resample latent tensor along temporal dimension.
    
    Args:
        latent: (C, T, H, W) or (B, C, T, H, W) latent tensor
        src_frames: Number of source frames
        dst_frames: Number of destination frames
        align_corners: PyTorch interpolation flag
        mode: "trilinear" (default) for smooth interpolation
        
    Returns:
        Resampled latent (C, dst_frames, H, W) or (B, C, dst_frames, H, W)
    """
    
    assert isinstance(latent, torch.Tensor), f"latent must be a torch.Tensor, got {type(latent)}"
    src_frames = int(src_frames)
    dst_frames = int(dst_frames)
    assert src_frames > 0 and dst_frames > 0, (
        f"src_frames and dst_frames must be positive, got {src_frames}->{dst_frames}"
    )

    # Handle both 4D and 5D inputs. The temporal dimension is always dim=1
    # after the optional batch dimension is removed: [C, T, H, W].
    if latent.ndim == 4:
        C, T, H, W = latent.shape
        latent = latent.unsqueeze(0)  # Add batch dim
        squeeze_batch = True
    elif latent.ndim == 5:
        B, C, T, H, W = latent.shape
        squeeze_batch = False
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {latent.ndim}D")

    assert T == src_frames, (
        f"latent temporal dim T={T} must match src_frames={src_frames}; "
        f"latent shape={tuple(latent.shape)}"
    )
    assert C > 0 and H > 0 and W > 0, f"invalid latent shape after normalization: {tuple(latent.shape)}"
    
    # Identity case: no interpolation needed
    if latent.shape[2] == dst_frames:
        result = latent
    else:
        # Interpolate: (B, C, T, H, W) → (B, C, dst_frames, H, W)
        old_dtype = latent.dtype
        latent_float = latent.float()
        
        result = F.interpolate(
            latent_float,
            size=(dst_frames, latent.shape[-2], latent.shape[-1]),
            mode=mode,
            align_corners=align_corners,
        ).to(dtype=old_dtype)
    
    if squeeze_batch:
        result = result.squeeze(0)

    expected_shape = (C, dst_frames, H, W) if squeeze_batch else (latent.shape[0], C, dst_frames, H, W)
    assert tuple(result.shape) == expected_shape, (
        f"resampled latent shape {tuple(result.shape)} must equal {expected_shape}"
    )
    
    return result


def get_temporal_resample_indices(
    src_frames: int,
    dst_frames: int,
    align_corners: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get source indices for mapping destination frames to source frames.
    Useful for advanced interpolation strategies.
    
    Args:
        src_frames: Number of source frames
        dst_frames: Number of destination frames
        align_corners: If True, first/last frames perfectly align
        
    Returns:
        Tuple of (src_idx_low, src_idx_high, weight_low, weight_high)
    """
    
    if align_corners:
        # Map [0, dst_frames-1] to [0, src_frames-1]
        src_pos = torch.linspace(0, src_frames - 1, dst_frames)
    else:
        # Map [0, dst_frames-1] to [0, src_frames-1] without aligning corners
        src_pos = torch.linspace(0, src_frames - 1, dst_frames)
    
    # Decompose into integer and fractional parts
    src_idx_low = torch.floor(src_pos).long()
    src_idx_high = torch.ceil(src_pos).long()
    
    # Clamp to valid range
    src_idx_low = torch.clamp(src_idx_low, 0, src_frames - 1)
    src_idx_high = torch.clamp(src_idx_high, 0, src_frames - 1)
    
    # Fractional parts for weighting
    frac = src_pos - src_idx_low.float()
    weight_low = 1.0 - frac
    weight_high = frac
    
    return src_idx_low, src_idx_high, weight_low, weight_high


def resample_latent_temporal_manual(
    latent: torch.Tensor,
    src_frames: int,
    dst_frames: int,
) -> torch.Tensor:
    """
    Manual temporal interpolation for more control.
    Slower than PyTorch's built-in but allows custom logic.
    
    Args:
        latent: (C, T, H, W) or (B, C, T, H, W)
        src_frames: Number of source frames
        dst_frames: Number of destination frames
        
    Returns:
        Resampled latent
    """
    
    if latent.ndim == 4:
        C, T, H, W = latent.shape
        latent = latent.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Get interpolation indices
    src_idx_low, src_idx_high, w_low, w_high = get_temporal_resample_indices(
        src_frames, dst_frames, align_corners=False
    )
    
    device = latent.device
    src_idx_low = src_idx_low.to(device)
    src_idx_high = src_idx_high.to(device)
    w_low = w_low.to(device).view(1, 1, dst_frames, 1, 1)
    w_high = w_high.to(device).view(1, 1, dst_frames, 1, 1)
    
    # Interpolate
    resampled = (
        latent[:, :, src_idx_low] * w_low +
        latent[:, :, src_idx_high] * w_high
    )
    
    if squeeze:
        resampled = resampled.squeeze(0)
    
    return resampled
