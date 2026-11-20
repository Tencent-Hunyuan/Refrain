# Weights and evaluation models

## Release status

| Artifact | Status |
|---|---|
| Refrain DiT (8.49B, final 45k-step model) | **Not released** |
| Refrain 3D VAE (16× spatial, 4× temporal) | **Not released** |
| Initialization DiT used by the first training stage | **Not released** |
| Refrain Benchmark (300 requests, references, image anchors) | Released separately as `benchmark.zip` |
| SDC real-video calibration bank | Included in this repository, `evaluation/assets/sdc_bank/` (see below) |
| Qwen3-VL-8B-Instruct text encoder | Public upstream model |

The code for training and inference is complete, but generating videos requires a
DiT and VAE matching this architecture. Public HunyuanVideo-1.5 weights are a
related architecture and are **not** drop-in replacements: the Refrain VAE has 48
latent channels and the DiT was trained against it.

## Expected layout

```text
weights/
├── Qwen3-VL-8B-Instruct/               # text encoder, tokenizer and processor
├── vae_3d/refrain_vae/                 # config.json + pytorch_model.pt (your own compatible VAE)
├── initialization/model.safetensors    # stage-1 initialization DiT (training only)
└── evaluation/                         # metric models, see below
```

`MODEL_BASE` defaults to `weights/` and can point to another model root.

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir weights/Qwen3-VL-8B-Instruct
```

The VAE loader expects `config.json` and `pytorch_model.pt`, with a configuration
matching `WanVAE_` in `hymm/models/vae/autoencoder_kl_causal_rmsnorm_3d_v3.py`.
`tools/dcp_to_safetensors.py` converts a distributed training checkpoint into a
single `model.safetensors` file for inference.

The released configurations set `glyph_byT5_v2: false`, `text_encoder_2: null` and
`vision_encoder: null`, so ByT5, Glyph-SDXL-v2 and SigLIP weights are not needed.

## SDC real-video bank

The bank is the calibration data behind Shot-Debiased Consistency, fitted on
2,144,201 real videos. It ships with the code in `evaluation/assets/sdc_bank/` as
plain text (about 4.6 MB), so no download is needed. It contains only fitted
distributions: no videos, frames, embeddings or benchmark scores.

| Property | Value |
|---|---|
| Cells | 30 = 2 style heads (CSD, VGG) × 3 calibration groups × 5 source folds |
| Per cell | robust content–style slope κ, content-energy CDF (257 knots), 256 content-bin centres, residual-style CDF per bin (49 knots) |
| Files | `metadata.json` (scalars and cell list) and one `<head>_fold<k>_group<g>.txt` per cell |
| Loader | `evaluation/style_consistency/bank/compact_bank.py` |

The full fitted bank stores every real shot pair (about 5M per cell, 4.7 GiB). The
text bank keeps each one-dimensional CDF as quantile knots and evaluates it with
monotone cubic interpolation. Compared with the full bank on the Refrain Benchmark,
per-video SDC differs by at most 7e-4 and the benchmark mean is identical
(0.31538 for both). Each request's held-out source fold is stored as `sdc_fold` in
`configs/benchmark.json`.

To calibrate on your own real-video corpus, see
[EVALUATION.md](EVALUATION.md#building-an-sdc-bank).

## Evaluation models

| Metric / component | Official source | Expected local location |
|---|---|---|
| CLIP-T, OpenAI CLIP ViT-L/14 | [OpenAI CLIP](https://github.com/openai/CLIP) | `weights/evaluation/clip/ViT-L-14.pt` |
| Semantic metrics, ViCLIP-L | [OpenGVLab/ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP) | `ViCLIP-L_InternVid-FLT-10M.pth` placed as `weights/evaluation/quality_eval/viclip.pth` |
| TA, VideoReward | [VideoAlign](https://github.com/KlingAIResearch/VideoAlign), [KwaiVGI/VideoReward](https://huggingface.co/KwaiVGI/VideoReward) | `weights/evaluation/quality_eval/videoalign.pth`, `videoalign_config.json`, `qwen_config/` |
| DINOv2-small | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | `weights/evaluation/dinov2-small/` |
| GroundingDINO tiny | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | `weights/evaluation/grounding-dino/` |
| SAM ViT-base | [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) | `weights/evaluation/sam/` |
| CSD ViT-L (both heads) | [CSD](https://github.com/learn2phoenix/CSD#download-the-pretrained-weights-for-the-csd-model) | `weights/evaluation/csd/csd-vit-l.pth` |
| VGG16 | [TorchVision](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.vgg16.html) | `weights/evaluation/vgg16-397923af.pth` |
| TransNetV2 | [TransNetV2](https://github.com/soCzech/TransNetV2) `inference-pytorch` | `weights/evaluation/transnetv2/transnetv2_pytorch.py`, `model.pt` |

```bash
hf download facebook/dinov2-small --local-dir weights/evaluation/dinov2-small
hf download IDEA-Research/grounding-dino-tiny --local-dir weights/evaluation/grounding-dino
hf download facebook/sam-vit-base --local-dir weights/evaluation/sam
```
