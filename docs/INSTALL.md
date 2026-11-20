# Installation

The supported environment uses **Python 3.12** with the package versions below.
The installation order follows [OmniWeaving](https://github.com/Tencent-Hunyuan/OmniWeaving#-preparation)
and [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5#-dependencies-and-installation).
Refrain has additional training dependencies and uses Qwen3-VL-8B, so their
requirements and model downloads are not interchangeable with Refrain's.

## 1. Python, PyTorch, and public packages

Use Linux and Python 3.12. A fresh environment can be created with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==68.2.2 wheel==0.45.1 ninja==1.11.1.2
```

Use Torch 2.6.0, TorchVision 0.21.0, TorchAudio 2.6.0, Triton 3.2.0 and CUDA 12.4.
Install the
[official PyTorch 2.6.0 CUDA 12.4 wheels](https://pytorch.org/get-started/previous-versions/#v260):

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation --no-deps
```

Run repository commands from the Refrain root. On an existing working GPU
container, create the isolated `.venv` before installing packages. Do not install
or uninstall packages in the container's global interpreter. The inference
launcher selects `.venv/bin/python` explicitly and disables user site packages.
`pip list` identifies Python package versions, but does not establish the local
CUDA compiler version or the C++ ABI of compiled extensions.

If the official PyTorch wheel index is unreachable, the Linux x86-64 Python 3.12
PyPI wheels pinned by `requirements.txt` also provide Torch 2.6.0 with CUDA 12.4.


On a slow shared filesystem, the virtual environment can live on node-local disk
with `.venv` as a symlink to it.

Important pins include `diffusers==0.31.0`, `transformers==4.57.1`,
`accelerate==1.9.0`, `peft==0.17.0`, `timm==1.0.7`, `numpy==2.0.0`, and
`imageio-ffmpeg==0.6.0`. These follow the environment used for the paper.
OmniWeaving currently specifies Diffusers 0.32.2 and NumPy 1.26.4 instead.
`aiohttp==3.11.13` is retained from that environment; PyPI marks this
release as yanked for a regression. It remains installable by exact pin and is
not silently upgraded here. Only `opencv-python-headless` is installed: both OpenCV distributions provide
`cv2`, so a clean environment should select one.

For command-line video operations, install your distribution's `ffmpeg` package
and confirm that `ffmpeg` and `ffprobe` are on PATH. `imageio-ffmpeg` supplies the
ImageIO backend and does not establish availability of a system `ffprobe`.

## 2. FlashAttention

Install after PyTorch, with a working CUDA development toolkit and compiler:

```bash
python -m pip install --no-build-isolation --no-deps -r requirements-kernels.txt
python -c "from flash_attn import flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func; print('FlashAttention import OK')"
```

The pin is `flash-attn==2.8.3`. Build requirements
are documented by [Dao-AILab](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3).
An existing wheel built for the same Python, Torch, CUDA, and C++ ABI may be
installed into `.venv` with `python -m pip install --no-deps /path/to/the/wheel`.

The released configurations say `attn_mode: flash3`. In this code snapshot,
`hymm/utils/flash_attn_no_pad.py` imports `flash_attn_varlen_func` from
`flash_attn` under its v3 helper name. Qwen3-VL also requests
`flash_attention_2`. The active video path therefore uses the `flash-attn`
package above; the configuration label alone does not imply the separate
`flash_attn_interface` Hopper extension.

## 3. Flex-Block-Attention (optional)

The released configurations select dense FlashAttention. Flex is needed
only when enabling a compatible sparse-attention backend.

Use the [official Tencent-Hunyuan source](https://github.com/Tencent-Hunyuan/flex-block-attn),
as linked by OmniWeaving and HunyuanVideo-1.5. The upstream build requires CUDA
Toolkit **12.8** and Hopper/SM90 support. Check `nvcc --version`; Python packages
named `nvidia-*-cu12` do not supply this compiler requirement.

```bash
mkdir -p third_party
git clone https://github.com/Tencent-Hunyuan/flex-block-attn.git third_party/flex-block-attn
git -C third_party/flex-block-attn checkout bc2ec1eefcbff089c9bd4b59cfd8cb4db9e90d59
git -C third_party/flex-block-attn submodule update --init --recursive
python -m pip install --no-build-isolation --no-deps ./third_party/flex-block-attn
python -c "from flex_block_attn import flex_block_attn_func; print('Flex kernel import OK')"
```

The upstream instructions use `python setup.py install` inside the cloned
repository. The pip command above builds the same source using the installed
Torch, avoiding a legacy egg installation.
Use the pinned commit above to reproduce the documented Flex build.

## 4. Repository-local parallelism and data loading

Parallelism and data loading are included in this repository.

- `hymm/parallelism/parallel_states.py` builds PyTorch DeviceMesh groups following
  the public HunyuanVideo-1.5 design. Data-parallel ranks receive different samples;
  sequence-parallel ranks share a sample.
- `hymm/parallelism/communications.py` includes public HunyuanVideo-1.5 all-to-all
  communication and an uneven-length all-gather with gradient reduction.
- `hymm/parallelism/engines/` supplies native PyTorch FSDP2, AdamW, gradient
  accumulation, activation checkpointing, and distributed checkpoint save/load.
- `hymm/dataset/samplers.py` supplies resumable PyTorch samplers. Arrow readers
  adapted from public HunyuanDiT source are bundled under `hymm/dataset/arrow_index/`.
  Training datasets are imported lazily by the dataset package.

The released configurations use dense Refrain, AdamW, TP=PP=EP=1, and their
recorded SP sizes. MoE and pipeline-parallel variants are outside this
runtime. Training code is retained. Refer to [RUNTIME.md](RUNTIME.md) for checkpoint
format, supported paths, and validation scope. Source licenses and attribution
are included in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

`pdsh` is unnecessary for the public launchers, which use `torchrun`. Install the
same environment on participating nodes through your scheduler or container image.

## 5. ViCLIP evaluation dependency

`evaluation/encoders.py` loads the ViCLIP implementation from VBench for
Inter-Shot Semantic and Reference Semantic. Model training and generation do not
require this evaluation package. Install it only when running these metrics:

```bash
python -m pip install --no-deps vbench==0.1.5
```

VBench 0.1.5's [package metadata](https://pypi.org/pypi/vbench/0.1.5/json)
requires `transformers==4.33.2` and `numpy<2.0.0`, conflicting with the supplied
Refrain environment. A normal resolver install cannot satisfy both. This
exception is limited to the existing ViCLIP loader, which bypasses the full
VBench suite. It does not establish support for running the full VBench package.
`pip check` will still report VBench's unsatisfied full-suite requirements.
A future release can vendor the licensed ViCLIP dependency to remove this
packaging conflict.

The public requirements also contain `datasets` for VideoAlign's imports and
`open-clip-torch` for CSD. `qwen-vl-utils==0.0.8` appears in OmniWeaving's setup
but is not imported by Refrain's
active text path. The relevant VideoAlign vision helpers are bundled locally.
WandB and the optional audio/reward-training stacks are not installed by the
main requirements.

## 6. Checks before launching

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from transformers import Qwen3VLForConditionalGeneration; import diffusers, imageio_ffmpeg; print(diffusers.__version__, imageio_ffmpeg.__version__)"
python -c "from hymm.parallelism.communications import all_gather, all_to_all_4D; from hymm.parallelism.engines.parallel_engine import BaseParallelEngine; from hymm.dataset.samplers import IndexBatchSampler; print('Distributed/data APIs OK')"
python -c "from hymm.eval.eval_score_video_general import ScoreEvaluator; print('Inference imports OK')"
```

Model weights and their availability are documented in [CHECKPOINTS.md](CHECKPOINTS.md).
