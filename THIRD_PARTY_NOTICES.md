# Upstream components

The runtime retains the HunyuanVideo-derived backbone/encoder/VAE implementation
and its existing source headers. Evaluation depends on CLIP, ViCLIP/InternVid,
DINOv2, CSD, VGG16, GroundingDINO, SAM, TransNetV2 and VideoAlign. Install their
released checkpoints and dependencies under their respective upstream terms.

This preparation does not assign a new license to upstream code or model weights.
No root license was present in the source tree supplied for this export. Project
maintainers should supply the repository's intended license before publication.
No model checkpoints are distributed in this package.

## Bundled parallelism and Arrow readers

- `hymm/parallelism/communications.py` is adapted from
  [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/hyvideo/utils/communications.py).
  The DeviceMesh layout follows its public parallel-state implementation.
  See `hymm/parallelism/LICENSE` for the upstream license. Local changes add
  uneven-length gathering and a CPU testing path.
- `hymm/dataset/arrow_index/indexer.py` and `bucket.py` are adapted from
  [HunyuanDiT's public Arrow readers](https://github.com/Tencent-Hunyuan/HunyuanDiT/tree/main/IndexKits/index_kits).
  See `hymm/dataset/arrow_index/LICENSE`. Local additions apply sidecar path mappings and expose resolution
  selection for the retained image loader. The sidecar path helper and samplers
  are provided in this repository. No external IndexKits distribution is installed.

Upstream sources were retrieved on 2026-10-01.
