"""Bundled public Arrow readers from Tencent-Hunyuan/HunyuanDiT.

See LICENSE and THIRD_PARTY_NOTICES.md. No external index package is required.
"""
from pathlib import Path
from .indexer import ArrowIndexV2
from .bucket import MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2


def arrow_mapper(path, suffix):
    """Map a primary Arrow file to an adjacent sidecar with the given suffix."""
    path = Path(path)
    return str(path.with_name(path.stem + suffix + path.suffix))
