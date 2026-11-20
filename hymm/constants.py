import os
import torch


__all__ = [
    "C_SCALE", "PROMPT_TEMPLATE",
    "MODEL_BASE",
    "normalize_model_region", "resolve_model_base_by_region", "resolve_siglip_base_by_region",
    "apply_model_region_override",
    "PRECISIONS", "ATTENTION_MODE", "NORMALIZATION_TYPE", "ACTIVATION_TYPE",
    "VAE_PATH", "TEXT_ENCODER_PATH", "TOKENIZER_PATH", "TEXT_PROJECTION",
    "TESTSET_TEMPLATE", "VALIDATION_METRICS", "LOSS_METRICS", "SCORE_METRICS", "SAMPLE_METRICS",
    "MODULATION_TYPE", "TEXT_POOL_TYPE",
    "DENOISE_TYPE", "DDPM_NOISE_SCHEDULES", "DDPM_PREDICT_TYPE",
    "FLOW_PATH_TYPE", "FLOW_PREDICT_TYPE", "FLOW_LOSS_WEIGHT", "FLOW_SNR_TYPE", "FLOW_SOLVER",
    "IMAGE_CROP_TYPE", "INDEX_STRATEGY", "DATA_TYPE", "BUCKET_BASE_SIZE", "BUCKET_TEMPORAL_BASE_SIZE", "VISION_ENCODER_PATH",
    "BYT5_FONT", "BYT5_COLOR", "BYT5_GOOGLE", "BYT5_CKPT", "REPA_ENCODER_PATH"
]

# =================== Constant Values =====================
# Computation scale factor, 1P = 1_000_000_000_000_000. Tensorboard will display the value in PetaFLOPS to avoid
# overflow error when tensorboard logging values.
C_SCALE = 1_000_000_000_000_000

PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

PROMPT_TEMPLATE_ENCODE_IMAGE_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere."},
    {"role": "user", "content": "{}"}
]

PROMPT_TEMPLATE_ENCODE_VIDEO_JSON = [
    {"role": "system", "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."},
    {"role": "user", "content": "{}"}
]

LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE_MULTI_MODAL = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the image by detailing the following aspects according to the reference image: "
    "color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)

LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO_MULTI_MODAL = (
    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE = (
    "<|im_start|>system\nYou are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere.<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)

QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO = (
    "<|im_start|>system\nYou are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)



QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE_V2 = (
    "<|im_start|>system\nYou are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere.<|im_end|>\n<|im_start|>user\n{}"
)
QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO_V2 = (
    "<|im_start|>system\nYou are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.<|im_end|>\n<|im_start|>user\n{}"
)

PROMPT_TEMPLATE = {
    "li-dit-encode": {"template": PROMPT_TEMPLATE_ENCODE, "crop_start": 36},
    "li-dit-encode-video": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO,"crop_start": 95},
    "li-dit-encode-image-json": {"template": PROMPT_TEMPLATE_ENCODE_IMAGE_JSON, "crop_start": -1}, # auto-calculate crop_start
    "li-dit-encode-video-json": {"template": PROMPT_TEMPLATE_ENCODE_VIDEO_JSON, "crop_start": -1}, # auto-calculate crop_start
    "li-dit-encode-image-multitask-llava": {"template": LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE_MULTI_MODAL,
                                            "crop_start": 47, "image_emb_start": 5, "image_emb_end": 581, "image_emb_len": 576,
                                            "double_return_token_id": 271},
    "li-dit-encode-video-multitask-llava": {"template": LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO_MULTI_MODAL,
                                            "crop_start": 103, "image_emb_start": 5, "image_emb_end": 581, "image_emb_len": 576,
                                            "double_return_token_id": 271},
    "li-dit-encode-image-qwen-vl-v2": {"template": QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE_V2, "crop_start": 74},
    "li-dit-encode-video-qwen-vl-v2": {"template": QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO_V2, "crop_start": 108},
    "li-dit-encode-image-qwen-vl": {"template": QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_IMAGE, "crop_start": 74},
    "li-dit-encode-video-qwen-vl": {"template": QWEN_VL_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO, "crop_start": 108},
}
NEGATIVE_PROMPT=""

# ======================= Model ======================
PRECISIONS = {"fp32", "fp16", "bf16"}
ATTENTION_MODE = {"flash", "flash3", "torch", "sta_flex", "block_attn", "ssta_flex", "flex_sparse_attn"}
NORMALIZATION_TYPE = {"layer", "rms"}
ACTIVATION_TYPE = {"relu", "silu", "gelu", "gelu_tanh"}
TEXT_PAD_MODE = {"orig", "same_pad"}

# =================== Model Path =====================
from pathlib import Path

def normalize_model_region(region):
    return "auto"
def resolve_model_base_by_region(region):
    return None
def resolve_siglip_base_by_region(region):
    return None
def _resolve_model_base():
    return os.environ.get("MODEL_BASE", str(Path(__file__).resolve().parents[1] / "weights"))
def _resolve_siglip_base():
    return os.environ.get("SIGLIP_BASE", str(Path(_resolve_model_base()) / "siglip"))
MODEL_BASE = _resolve_model_base()
SIGLIP_BASE = _resolve_siglip_base()


def _build_vae_path(model_base):
    return {
        # 2D VAE
        "88-4c-sd": f"{model_base}/sd-vae-ft-ema",
        "88-4c-sdxl": f"{model_base}/sdxl-vae-fp16-fix",
        "88-16c-hy": f"{model_base}/vae_2d/hyvae_v1_0723",
        "88-16c-sd3": f"{model_base}/sd3-vae",
        "88-16c-flux": f"{model_base}/flux-vae",
        # 3D VAE
        "884-8c-hy": f"{model_base}/vae_3d/hyvae_v1_0430",
        "884-16c-hy0801": f"{model_base}/vae_3d/hyvae_v1_0801",
        "16164-32c-v1": f"{model_base}/vae_3d/video_vae_s16_t4_c32_v1",
        "16164-32c-v2": f"{model_base}/vae_3d/video_vae_s16_t4_c32_v2",
        "16164-48c-v3": f"{model_base}/vae_3d/refrain_vae",
    }


def _build_text_encoder_path(model_base):
    return {
        "t5": f"{model_base}/PixArt-XL-2-512x512/text_encoder",
        "t5_v11_xxl": f"{model_base}/t5_v1_1_xxl",
        "clipL": f"{model_base}/openai_clip-vit-large-patch14",
        "llava-llama-3-8b": f"{model_base}/llava-llama-3-8b-v1_1-pure-llama",
        "glm-4v-9b": f"{model_base}/glm-4v-9b-pure-glm",
        "qwen-2.5vl-7b": f"{model_base}/Qwen2.5-VL-7B-Instruct",
        "qwen-2.5vl-32b": f"{model_base}/Qwen2.5-VL-32B-Instruct",
        "gemma-3-27b-it": f"{model_base}/gemma-3-27b-it",
        "llava-llama-3-8b-all": f"{model_base}/llava-llama-3-8b-v1_1-transformers",
        "qwen-3vl-8b": f"{model_base}/Qwen3-VL-8B-Instruct",
    }


def _build_tokenizer_path(model_base):
    return {
        "t5": f"{model_base}/PixArt-XL-2-512x512/tokenizer",
        "t5_v11_xxl": f"{model_base}/t5_v1_1_xxl",
        "clipL": f"{model_base}/openai_clip-vit-large-patch14",
        "llava-llama-3-8b": f"{model_base}/llava-llama-3-8b-v1_1-pure-llama",
        "glm-4v-9b": f"{model_base}/glm-4v-9b-pure-glm",
        "qwen-2.5vl-7b": f"{model_base}/Qwen2.5-VL-7B-Instruct",
        "qwen-2.5vl-32b": f"{model_base}/Qwen2.5-VL-32B-Instruct",
        "gemma-3-27b-it": f"{model_base}/gemma-3-27b-it",
        "llava-llama-3-8b-all": f"{model_base}/llava-llama-3-8b-v1_1-transformers",
        "qwen-3vl-8b": f"{model_base}/Qwen3-VL-8B-Instruct",
    }


def _build_repa_encoder_path(model_base):
    return {
        "dinov3": f"{model_base}/dinov3",
        "qwen-3vl-8b": f"{model_base}/Qwen3-VL-8B-Instruct",
    }


def apply_model_region_override(region):
    """Optionally override model-related paths after argparse has parsed region.

    Default/auto/none leaves all existing import-time resolution untouched. A
    concrete region is intentionally applied after those existing resolution
    paths so explicit config can force a region without changing default runs.
    """
    global MODEL_BASE, SIGLIP_BASE
    global BYT5_FONT, BYT5_COLOR, BYT5_GOOGLE, BYT5_CKPT

    region = normalize_model_region(region)
    model_base = resolve_model_base_by_region(region)
    if model_base is None:
        return None

    siglip_base = resolve_siglip_base_by_region(region) or SIGLIP_BASE

    MODEL_BASE = model_base
    SIGLIP_BASE = siglip_base

    VAE_PATH.clear()
    VAE_PATH.update(_build_vae_path(model_base))
    TEXT_ENCODER_PATH.clear()
    TEXT_ENCODER_PATH.update(_build_text_encoder_path(model_base))
    TOKENIZER_PATH.clear()
    TOKENIZER_PATH.update(_build_tokenizer_path(model_base))
    VISION_ENCODER_PATH["siglip"] = siglip_base
    REPA_ENCODER_PATH.clear()
    REPA_ENCODER_PATH.update(_build_repa_encoder_path(model_base))

    BYT5_FONT = f"{model_base}/google-byt5-small/multilingual_10-lang_idx.json"
    BYT5_COLOR = f"{model_base}/google-byt5-small/color_idx.json"
    BYT5_GOOGLE = f"{model_base}/google-byt5-small/snapshots/68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
    BYT5_CKPT = f"{model_base}/Glyph-SDXL-v2/checkpoints/byt5_model.pt"
    return model_base

# =================== Data =======================
DATA_TYPE = {"image", "video", "image_video"}
BUCKET_BASE_SIZE = {128, 256, 480, 512, 720, 960, 1024, 2048}
BUCKET_TEMPORAL_BASE_SIZE = {1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149, 153, 157, 161, 165, 169, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209, 213, 217, 221, 225, 229, 233, 237, 241, 245, 249, 253, 257, 261, 265, 269, 273, 277, 281, 285, 289, 293, 297, 301, 305, 309, 313, 317, 321, 325, 329, 333, 337, 341, 345, 349, 353, 357, 361, 365, 369, 373, 377, 381, 385, 389, 393, 397}

TARGET_SIZE = {
    "256p": {"bucket_hw_base_size": 256, "bucket_hw_bucket_stride": 16},
    "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
    "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
    "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
}

# VAE
VAE_PATH = _build_vae_path(MODEL_BASE)

# Glyph byT5 v2
BYT5_FONT=f"{MODEL_BASE}/google-byt5-small/multilingual_10-lang_idx.json"
BYT5_COLOR=f"{MODEL_BASE}/google-byt5-small/color_idx.json"
BYT5_GOOGLE=f"{MODEL_BASE}/google-byt5-small/snapshots/68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
BYT5_CKPT=f"{MODEL_BASE}/Glyph-SDXL-v2/checkpoints/byt5_model.pt"

# Text Encoder
TEXT_ENCODER_PATH = _build_text_encoder_path(MODEL_BASE)
TOKENIZER_PATH = _build_tokenizer_path(MODEL_BASE)

TEXT_PROJECTION = {
    "linear",                               # Default, an nn.Linear() layer
    "single_refiner",                       # Single TokenRefiner. Refer to LI-DiT
    "double_refiner",                       # Double TokenRefiner. Refer to LI-DiT
}

VISION_ENCODER_PATH = {
    "siglip": _resolve_siglip_base(),
    None: None,
}
REPA_ENCODER_PATH = _build_repa_encoder_path(MODEL_BASE)


# =================== Evaluation Path =====================
# Get the base directory (project root) - constants.py is in hymm/
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_CURRENT_FILE_DIR)  # Go up one level from hymm/ to project root

EVAL_BASE = os.getenv("EVAL_BASE", "assets/local/resource_d9ce0d8e05")
# EVAL_BASE = os.getenv("EVAL_BASE", "assets/local/resource_5bb919ae25")

COCO30K_PATH = f"{EVAL_BASE}/dataset/COCO/COCO30k.json"
COCO6K_PATH = f"{EVAL_BASE}/dataset/COCO/COCO6k.json"

VAL30K_PATH = "assets/local/resource_3e7623c65a.json"
# VAL30K_PATH = "assets/local/resource_309eb33ce7.json"

FID_INCEPTION_PATH = f"{EVAL_BASE}/FID/pt_inception-2015-12-05-6726825d.pth"
COCO30K_FID_TARGET_PATH = f"{EVAL_BASE}/FID/coco_val_fid_stats.pt"
VAL30K_FID_TARGET_PATH = f"{EVAL_BASE}/FID/val_30k_fid_stats.pt"

ICS_INCEPTION_PATH = f"{EVAL_BASE}/FID/pt_inception-2015-12-05-6726825d.pth"
# CLIP Score: CLIP model
CLIP_MODEL_PATH = f"{EVAL_BASE}/CLIP/ViT-B-32.pt"
# HPSv2: HPSv2 model
HPSV2_MODEL_PATH = f"{EVAL_BASE}/xswu-HPSv2/HPS_v2.1_compressed.pt"
# T2I-CompBench: BLIP_VQA model - Use absolute paths based on project root
T2I_COMPBENCH_PATH = [
    os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/t2i_compbench/color_val.csv"),
    os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/t2i_compbench/shape_val.csv"),
    os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/t2i_compbench/texture_val.csv"),
]
VQA_MODEL_PATH = f"{EVAL_BASE}/T2I-CompBench"
# DPG-Bench: MPLUG model
MPLUG_MODEL_PATH = f"{EVAL_BASE}/mplug"
DPG_BENCH_DATA = [
    os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/dpg_bench/data.csv"),
    os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/dpg_bench/questions.json"),
]
GENEVAL_MODEL_PATH = f"{EVAL_BASE}/GenEval"
GENEVAL_DATA = os.path.join(PROJECT_ROOT, "hymm/dataset/data_kits/geneval/geneval.jsonl")

General_Generation_BENCH_DATA = {
    # Named request files; an existing file path may also be passed directly.
    "refrain_benchmark": [os.path.join(PROJECT_ROOT, "benchmark/requests.json")],
}

# =================== Test constants =====================
TESTSET_TEMPLATE = os.environ.get("TESTSET_TEMPLATE", "data/test/{}.csv")

VALIDATION_METRICS = {
    "val_loss",     # Validation loss
    "fid",          # FID
    "clip_score",   # CLIP Score
    "hpsv2",        # HPSv2
    "ics",          # Inception Score
    "t2i_compbench",    # T2I-CompBench
    "image",        # Sample images
    "dpg_bench",    # DPG-Bench
    "geneval",      # GenEval
}
LOSS_METRICS = {"val_loss"}
#SCORE_METRICS = {"fid", "clip_score", "hpsv2", "ics", "t2i_compbench"}
SCORE_METRICS = {"fid", "fid@coco6k", "clip_score", "hpsv2", "ics", "t2i_compbench", "hpsv2@coco6k", "clip_score@coco6k", "dpg_bench", "geneval"}

SAMPLE_METRICS = {"image"}

# ================ Network Structure ================
# Timestep (and text) modulation type
MODULATION_TYPE = {
    "dit",                  # http://arxiv.org/abs/2212.09748
    "pixart",               # http://arxiv.org/abs/2310.00426
    "lumina",               # http://arxiv.org/abs/2406.18583
}

# Text pooling type
TEXT_POOL_TYPE = {
    "avg",                  # http://arxiv.org/abs/2406.18583
    "clip",                 # Flux.1: use CLIP text encoder <eos> token
     None
}

# ================ Denoise Schedule ================
# Denoise type
DENOISE_TYPE = {"ddpm", "flow", "diffusers_ddpm"}

# DDPM noise schedule
DDPM_NOISE_SCHEDULES = {
    "linear",               # DDPM (0.0001, 0.02)
    "scaled_linear",        # StableDiffusion (0.00085, 0.012)
    "squaredcos_cap_v2",
}

# DDPM predict type
DDPM_PREDICT_TYPE = {
    "epsilon",              # Predict epsilon
    "sample",               # Predict sample, x0
    "v_prediction",         # Predict velocity, sqrt_alpha_bar * epsilon - sqrt_(1 - alpha_bar) * x0
}

# Flow Matching path type
FLOW_PATH_TYPE = {
    "linear",               # Linear trajectory between noise and data
    "gvp",                  # Generalized variance-preserving SDE
    "vp",                   # Variance-preserving SDE
}

# Flow Matching predict type
FLOW_PREDICT_TYPE = {
    "velocity",             # Predict velocity
    "score",                # Predict score
    "noise",                # Predict noise
}

# Flow Matching loss weight
FLOW_LOSS_WEIGHT = {
    "velocity",             # Weight loss by velocity
    "likelihood",           # Weight loss by likelihood
}

# Flow Matching SNR type
FLOW_SNR_TYPE = {
    "lognorm",              # Log-normal SNR
    "uniform",              # Uniform SNR
    "mix",                  # Mix SNR
}

# Flow Matching solvers
FLOW_SOLVER = {
    "euler",                # Euler solver
}

# ================ Data Loader ================
IMAGE_CROP_TYPE = {"random", "center"}
INDEX_STRATEGY = {"uniform", "probability"}


# for TPDistributedEMA
PARALLELISM_WEIGHTS = [
    "self_attention.img_attn_qkv.weight", # 列切分
    "self_attention.img_attn_qkv.bias",
    "self_attention.img_mlp.fc1.weight",
    "self_attention.img_mlp.fc1.bias",
    "self_attention.txt_attn_qkv.weight",
    "self_attention.txt_attn_qkv.bias",
    "self_attention.txt_mlp.fc1.weight",
    "self_attention.txt_mlp.fc1.bias",
    "self_attention.qkv.weight",
    "self_attention.qkv.bias",
    "self_attention.linear1.weight",
    "self_attention.linear1.bias",
    "self_attention.img_attn_proj.weight", # 行切分
    "self_attention.img_mlp.fc2.weight",
    "self_attention.txt_attn_proj.weight",
    "self_attention.txt_mlp.fc2.weight",
    "self_attention.linear2.weight",
]

# for try-catch traceback replace
TRACEBACK_REPLACE = [
    ("CUDA error", "#C#U#D#A# error"),
    ("AssertionError", "#A#s#s#e#r#t#i#o#n#Error")
]
