# Runtime paths

Installation: [INSTALL.md](INSTALL.md). Model sources and download availability:
[CHECKPOINTS.md](CHECKPOINTS.md).

All locations below are relative to the repository unless supplied explicitly by
CLI or environment. Populate them with your own local files; no original machine
paths or credentials are required or distributed.

| Path / setting | Contents | Used by |
|---|---|---|
| `weights/initialization/model.safetensors` | Initialization DiT for stage 1 (not released) | Training |
| `MODEL_BASE` (default `weights/`) | Parent of the encoder/VAE directories below | Training/inference |
| `weights/Qwen3-VL-8B-Instruct/` | Complete text encoder/tokenizer snapshot | Training/inference |
| `weights/vae_3d/refrain_vae/` | 3D VAE, spatial stride 16 and temporal stride 4 (not released) | Training/inference |
| `train/t2v/index.json` | Pure T2V Arrow index | Stage 1 |
| `train/references_equal/index.json` | Equal-duration reference reconstruction Arrow index | Stage 2 |
| `train/durmix/index.json` | Variable-duration DurMix Arrow index | Stage 3 |
| `REFRAIN_LATENT_ROOT` (default `train/latents/`) | Root for relative source-latent paths in Arrow records | Reference training |
| `outputs/text_to_video/global_step-19000/` | Step-19k training checkpoint written by stage 1 | Stage-2 resume |
| `outputs/reference_training/global_step-36000/` | Step-36k training checkpoint written by stage 2 | Stage-3 resume |
| `--checkpoint <file>.safetensors` | DiT weights for inference (not released) | Inference CLI |
| `benchmark/requests.json` | Extracted benchmark manifest | Generation/evaluation |
| `weights/evaluation/clip/ViT-L-14.pt` | OpenAI CLIP ViT-L/14 | CLIP-T |
| `weights/evaluation/quality_eval/viclip.pth` | Released ViCLIP checkpoint | Both semantic metrics |
| `weights/evaluation/quality_eval/videoalign.pth` | Released VideoAlign weights | TA |
| `weights/evaluation/quality_eval/videoalign_config.json` | Matching VideoAlign model and PEFT settings | TA |
| `weights/evaluation/quality_eval/qwen_config/` | Matching VideoAlign tokenizer, processor, and model configuration | TA |
| `weights/evaluation/dinov2-small/` | Complete DINOv2-small snapshot | Subject/background and SDC |
| `weights/evaluation/grounding-dino/` | GroundingDINO processor/model snapshot | Foreground detection |
| `weights/evaluation/sam/` | SAM processor/model snapshot | Foreground masking |
| `weights/evaluation/csd/csd-vit-l.pth` | CSD ViT-L/14 checkpoint with both projection heads | SDC |
| `weights/evaluation/vgg16-397923af.pth` | VGG16 weights | SDC |
| `weights/evaluation/transnetv2/model.pt` | TransNetV2 weights | MAE and detected-shot SDC |
| `weights/evaluation/transnetv2/transnetv2_pytorch.py` | Official TransNetV2 PyTorch implementation and its dependencies | Cut detection |
| `evaluation/assets/sdc_bank/` | SDC calibration bank (in the repository, plain text) | SDC scoring |

The evaluation root can be changed with `--weights`; the SDC bank with `--bank`.
`SIGLIP_BASE` configures optional upstream vision branches, which the released
training stages do not use.

`train/images/index.json` is an inactive image-data argument retained from the
recorded video-only configuration. Training data themselves are not distributed.
Dataset indexes must reference local Arrow files, local RGB inputs where used,
and local latent files. Relative path roots inside those indexes must be updated
for your environment.

The benchmark archive includes native NPY reference tensors, reference videos and image anchors. The default inference path reads NPY tensors directly.
