"""Video tensor layout helpers shared by online pixel paths."""

from __future__ import annotations

import torch


def normalize_video_bcfhw(video: torch.Tensor) -> torch.Tensor:
    """Return a 5D video tensor in ``[B, C, F, H, W]`` layout.

    Raw ``VideoArrowDataset`` batches use ``[B, F, C, H, W]``, while the VAE,
    inference pipeline, and video saving utilities use ``[B, C, F, H, W]``.
    """

    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError(
            f"expected a 5D video tensor, got {type(video).__name__} "
            f"shape={getattr(video, 'shape', None)}"
        )

    channel_sizes = (1, 3, 4)
    dim1_is_channel = int(video.shape[1]) in channel_sizes
    dim2_is_channel = int(video.shape[2]) in channel_sizes

    if dim1_is_channel:
        return video.contiguous()
    if dim2_is_channel:
        return video.permute(0, 2, 1, 3, 4).contiguous()

    raise ValueError(
        f"cannot infer video channel dimension from shape={tuple(video.shape)}; "
        "expected [B,F,C,H,W] or [B,C,F,H,W] with C in {1,3,4}"
    )
