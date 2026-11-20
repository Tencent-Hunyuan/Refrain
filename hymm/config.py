import argparse
from .metrics_video.utils import str2bool, str2null, int2null
from hymm.constants import *
import json

def parse_args(mode="train", known_args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Hunyuan Multimodal training/inference script")

    # text encoder & vae & diffusion model
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--convert-state-dict-in-out", action="store_true", help="convert the intput and final_layer of state_dict to adatpt 888-vae.")

    parser = add_common_args(parser)

    parser = add_data_args(parser)
    parser = add_extra_models_args(parser)

    parser = add_network_args(parser)
    parser = add_denoise_schedule_args(parser)

    parser = add_evaluation_args(parser)

    parser = add_training_args(parser)

    parser = add_parallelism_args(parser)
    parser = add_i2v_args(parser)
    if known_args:
        return parser.parse_args(known_args, namespace=namespace)
    args = parser.parse_args(namespace=namespace)

    return args

def add_i2v_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Image-to-video")
    group.add_argument("--multitask-mask-training-type", type=str, default=None, choices=["concat", "token_replace", "none"],
                        help="training type for multitask mask training. e.g., none, concat, merge")
    group.add_argument("--multitask-mask-training-args", type=json.loads, default=None,
                        help="json string format caption-sample-ratio for multitask mask training. e.g.{\"t2v\":{\"prob\":1}, ...}.")
    group.add_argument("--multitask-semantic-mode", type=str2null, default=None, choices=[None, "llava", "siglip"])
    group.add_argument("--vision-projection", type=str2null, default=None, choices=[None, "linear"])
    group.add_argument("--vision-states-dim", type=int, default=None)
    group.add_argument("--vision-num-semantic-tokens", type=int, default=None)
    group.add_argument("--use-multitask-co-drop-semantic-and-text", action="store_true", default=False)
    group.add_argument("--multitask-semantic-drop-independent-p", type=float, default=0.1)
    group.add_argument("--convert-state-dict-img-in-for-i2v", action="store_true", default=False)
    group.add_argument("--convert-state-dict-16168-to-16164", action="store_true", default=False)
    group.add_argument("--force-multitask-type", type=str, default="auto", choices=["auto", "i2v", "t2v"],
                        help="Force multitask type to override CSV data. 'auto' uses type from data, 'i2v' forces image-to-video, 't2v' forces text-to-video.")
    group.add_argument("--drop-first-frame-loss", action="store_true", default=True)

    # Vision Encoder
    group.add_argument("--vision-encoder", type=str2null, default=None, choices=list(VISION_ENCODER_PATH.keys()),
                       help="Name of the vision encoder model.")
    group.add_argument("--vision-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the vision encoder model.")
    group.add_argument("--vision-encoder-path", type=str, default=None,
                       help="Path to the vision encoder model. If None, use the default path from VISION_ENCODER_PATH.")
    group.add_argument("--processor-type", type=str, default=None,
                       help="Type of the image processor. If None, use the same as vision-encoder.")
    group.add_argument("--processor-path", type=str, default=None,
                       help="Path to the image processor. If None, use the same as vision-encoder-path.")
    group.add_argument("--vision-output-key", type=str, default=None,
                       help="Output key for vision encoder. If None, use default based on encoder type.")
        
    # REPA
    group.add_argument("--use-repa", action="store_true", help="Enable REPA projection loss.")
    group.add_argument("--repa-loss-weight", type=float, default=0.0, help="Weight for REPA loss.")
    group.add_argument("--repa-projector-hidden-dim", type=int, default=None,
                       help="Hidden dimension of the REPA projector MLP (maps vision feats to model hidden size).")
    group.add_argument("--repa-projector-out-dim", type=int, default=None,
                       help="Input dimension of REPA projector. If None, will try to infer from vision encoder config.")
    group.add_argument("--repa-encoder", type=str, default="dinov3", choices=list(REPA_ENCODER_PATH.keys()),
                       help="Name of the repa encoder model.")
    group.add_argument("--repa-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the repa encoder model.")
    group.add_argument("--repa-online-decode-video", action="store_true", help="Enable online decoder for video dataset.")
    return parser

def add_common_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Common")
    group.add_argument(
        "--global-seed", type=int, default=42, help="A seed for reproducible training."
    )
    group.add_argument(
        "--region",
        type=str,
        default="none",
        help="Optional model-path region override. Default 'none' keeps existing path resolution; e.g. 'gy' forces Guangyuan model paths.",
    )
    group.add_argument("--output-dir", type=str, default='./outputs', help="Directory to save logs and models")
    group.add_argument("--output-samples-dir", type=str2null, default=None)
    group.add_argument("--is-profiler", action="store_true", help="Enable profiler.")
    return parser

def add_data_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Data")

    group.add_argument("--data-type", type=str, default="image", help="Type of the dataset.")
    # image dataset
    group.add_argument("--image-size", type=int, nargs='+', default=512,
                       help="Image size for training/sampling. If a single value is provided, it will be used "
                            "for both width and height. If two values are provided, they will be used for width "
                            "and height respectively. If more than two values are provided, each pair of values "
                            "will be used for width and height respectively.")
    group.add_argument("--fixed-prompt-data-type", type=str, default='')
    group.add_argument("--val-image-size", type=int, nargs='+', default=512)
    group.add_argument("--uncond-p", type=float, default=0.1,
                       help="Probability of randomly dropping image description.")
    group.add_argument("--uncond-p-byt5", type=float, default=0.1,
                       help="Probability of randomly dropping image description.")
    group.add_argument("--uncond-token", type=str, default=None, help="Unconditional token for the text encoder.")
    group.add_argument("--mix-scale", action="store_true", help="Enable mix-scale training.")
    group.add_argument("--anchor-size", type=int, nargs='+', default=[256],
                       help="Anchor size for mix-scale training.")
    group.add_argument("--image-crop-type", type=str, default="random",
                       help="Image crop type when preprocessing images.")
    group.add_argument("--index-file", type=str, nargs='+', default=None, help="Index file for the dataset.")
    group.add_argument("--multireso", action="store_true",
                       help="Specify the index-file/val-index-file is a multi-resolution dataset.")
    group.add_argument("--multireso-step", type=int, default=16,
                       help="Specify the step for multi-resolution dataset.")
    group.add_argument("--index-strategy", type=str, default="uniform", choices=["uniform", "probability"],
                       help="Sample strategy for multiple index files.")
    group.add_argument("--index-probability", type=float, nargs='+', default=None, 
                       help="Probability for each index file when index-strategy is probability.")
    group.add_argument("--image-caption-rate", type=float, default=0.,
                       help="Set a rate to use caption instead of original description.")
    group.add_argument("--image-caption-processor", type=str, default=None, help="Set image-caption-processor for different caption types")
    group.add_argument("--image-caption-col-probs", type=str, default=None, help="Set image-caption-col-probs for different caption types")

    group.add_argument("--image-text-arrow-suffix", type=str, default='', help="Suffix name for text arrow.")
    group.add_argument("--image-text-col", type=str, default=None, help="Column name for text arrow.")
    group.add_argument("--image-text-col-2", type=str, default=None, help="Column name for text arrow.")
    group.add_argument("--image-caption-arrow-suffix", type=str, default=None, help="Suffix name for caption arrow.")
    group.add_argument("--image-caption-col", type=str, default="llava_caption", help="Column name for caption arrow.")
    group.add_argument("--image-caption-col-2", type=str, default=None, help="Column name for caption arrow.")
    group.add_argument("--ceph-base", type=str, default=None, help="Ceph base dir.")
    group.add_argument("--ceph-base-inv", type=str, default="{}", help="JSON path-prefix remapping for Arrow indexes.")
    group.add_argument("--caption-sample-ratio", type=str, default=None,
                       help="json string format caption-sample-ratio for training, e.g.{\"long_caption\": 1.0, ...}."
                            "all keys: long caption, short caption, background, shot type, style, light, atmosphere, camera movement")
    group.add_argument("--ocr-only-long-caption", action="store_true",
                       help="Only use long caption when ocr is in the long caption.")
    group.add_argument("--text-remove-rate", type=float, default=0.0,
                       help="Remove text with ocr rate.")
    group.add_argument("--raise-text-error", action="store_true",
                       help="Raise exception when catching an error when loading text.")
    group.add_argument("--save-training-samples", action="store_true", help="Save training samples for debugging.")
    group.add_argument("--fast-shuffle", action="store_true", help="Enable fast shuffle for data loading.")
    group.set_defaults(fast_shuffle=True)
    group.add_argument("--num-replace-rate", type=float, default=0.0,
                       help="The probability of replacing letters with Arabic numerals.")
    group.add_argument("--video-num-replace-rate", type=float, default=0.0,
                       help="The probability of replacing letters with Arabic numerals in video caption.")

    group.add_argument("--video-caption-i2v-instruction-enable", type=str2bool, default=False)
    group.add_argument("--video-caption-i2v-instruction-prob", type=float, default=0.0)
    group.add_argument("--video-caption-i2v-instruction-col-probs", type=json.loads, default=None,
                       help="Set the video-caption-i2v-instruction-col-probs for different caption types")
    group.add_argument("--video-caption-i2v-instruction-col", type=str2null, default="structure_caption_v2",
                       help="Set the column of i2v instruction captions")
    group.add_argument("--video-caption-i2v-instruction-col-2", type=str2null, default="structure_caption_cn_v2",
                       help="Set the column 2 of i2v instruction captions")
    group.add_argument("--video-caption-i2v-instruction-sub-col-probs", type=json.loads, default=None,
                       help="Set video-caption-i2v-instruction-sub-col-probs for different caption types")
    group.add_argument("--video-caption-i2v-instruction-sub-col", type=str2null, default="objects",
                          help="Column name for i2v instruction sub captions")
    group.add_argument("--video-caption-i2v-instruction-sub-col-2", type=str2null, default="objects_summary",
                          help="Column 2 name for i2v instruction sub captions")
    group.add_argument("--video-caption-i2v-instruction-sub-col-3", type=str2null, default="background",
                          help="Column 3 name for i2v instruction sub captions")

    group.add_argument("--use-ll-cap-dataset", action="store_true", help="Use long long caption dataset for training.")
    group.add_argument("--ll-cap-sample-ratio", type=float, default=0.2, help="Sampling ratio for ll_cap dataset.")
    group.add_argument("--ll-cap-caption-sample-ratio", type=str, default=None, help="Caption sample ratio for ll_cap dataset.")
    group.add_argument("--image-cos-root", type=str2null, default=None, help="Cos root path for image dataset.")
    group.add_argument("--rank-assign-mode", type=str, default="fixed", choices=["fixed", "random"], help="Rank assign mode for data loading.")

    # video dataset
    group.add_argument("--video-index-file", type=str, nargs='+', default=None,
                       help="Index file for the video dataset.")
    group.add_argument("--video-index-file-weights", type=float, nargs='+', default=None,
                       help="Weights for each video index file.")
    group.add_argument("--video-size", type=int, nargs='+', default=512,
                       help="Video size for training. If a single value is provided, it will be used for both width "
                            "and height. If two values are provided, they will be used for width and height "
                            "respectively.")
    group.add_argument("--sample-stride", type=int, default=1,
                       help="How many frames to skip when sampling from a video.")
    group.add_argument("--vae-sample-stride", type=int, default=1,
                       help="How many frames to skip when sampling from a video.")
    group.add_argument("--is-image", action="store_true", help="Use one image from video for training")
    group.add_argument("--video-multireso", action="store_true", help="Use multiple resolution for video training.")
    group.add_argument("--video-multitemp", action="store_true",
                       help="Use multiple temporal bukect for video training.")
    group.add_argument("--use-latent-shape", action="store_true", help="Use latent shape as datainfo for training.")
    group.add_argument("--filtered-video-ids-json-file", type=str, default=None,
                       help="Filtered video ids json file for training. Must be a json or jsonl file. ")

    group.add_argument("--video-uncond-p", type=float, default=0.1,
                       help="Probability of unconditional training for video.")
    group.add_argument("--video-caption-sample-ratio", type=str, default=None,
                       help="Set video-caption-sample-ratio for different caption types")
    group.add_argument("--video-caption-processor", type=str, default=None,
                       help="Set video-caption-processor for different caption types")
    group.add_argument("--video-caption-col-probs", type=json.loads, default=None,
                          help="Set video-caption-col-probs for different caption types")
    group.add_argument("--video-caption-col", type=str, default="structure_caption",
                          help="Column name for English captions")
    group.add_argument("--video-caption-col-2", type=str, default="structure_caption_cn",
                          help="Column name for Chinese captions")
    group.add_argument("--video-uncond-p-byt5", type=float, default=0.1,
                       help="Probability of randomly dropping image description.")
    group.add_argument("--video-caption-rate", type=float, default=0.,
                       help="Set a rate to use caption instead of original description.")
    group.add_argument("--video-sampling-prob", type=float, default=1.0,
                       help="The prob to sample a video from image and video dataset.")
    group.add_argument("--video-bucket-hw-base-size", type=int, default=256,
                       help="Video base bucket spatial size for online bucketing.")
    group.add_argument("--video-bucket-hw-bucket-stride", type=int, default=16,
                       help="Video base bucket spatial stride for online bucketing.")
    group.add_argument("--video-bucket-temporal-min-length", type=int, default=5,
                       help="Video min bucket temporal size for online bucketing.")
    group.add_argument("--video-bucket-temporal-max-length", type=int, default=257,
                       help="Video min bucket temporal size for online bucketing.")
    group.add_argument("--video-bucket-temporal-interval", type=int, default=8,
                       help="Video bucket temporal interval for online bucketing.")
    
    # Skip training data by source field
    group.add_argument("--skip-sources", type=str, nargs='*', default=None,
                       help="List of source values to skip during training. Data with these source values will be filtered out.")
    
    # Video long long caption support
    group.add_argument("--use-video-ll-cap-dataset", action="store_true", help="Use long long caption dataset for video training.")
    group.add_argument("--video-ll-cap-sample-ratio", type=float, default=0.2, help="Sampling ratio for video ll_cap dataset.")
    group.add_argument("--video-ll-cap-caption-sample-ratio", type=str, default=None, help="Caption sample ratio for video ll_cap dataset.")
    group.add_argument("--video-text-len-ll-cap", type=int, default=1000, help="Maximum length of the video text input for long long caption.")

    # Recreation mask support
    group.add_argument("--enable-recreation-mask", action="store_true", help="Enable recreation region masking in video latents.")
    group.add_argument("--recreation-mask-col", type=str, default="recreation_mask_data", help="Column name for recreation mask.")
    group.add_argument("--recreation-mask-ratio", type=float, default=0.3, help="Maximum ratio of latent to mask for recreation regions (0.0-1.0).")
    group.add_argument("--recreation-mask-position", type=str, default="bottom", choices=["bottom", "center", "top", "random"], 
                       help="Position of recreation mask in video latent.")
    group.add_argument("--recreation-mask-margin-ratio", type=float, default=0.1, help="Margin ratio around recreation mask (0.0-0.3). 0.1 means 10% margin around the mask.")
    group.add_argument("--recreation-mask-margin-ratio-w", type=float, default=None, 
                       help="Width margin ratio for recreation mask (0.0-0.3). If None, use recreation-mask-margin-ratio.")
    group.add_argument("--recreation-mask-margin-ratio-h", type=float, default=None, 
                       help="Height margin ratio for recreation mask (0.0-0.3). If None, use recreation-mask-margin-ratio.")
    group.add_argument("--recreation-min-margin-pixels", type=int, default=0, 
                       help="Minimum margin pixels to ensure visibility in latent space. Recommended: 8-16 pixels.")
    group.add_argument("--recreation-clustering-eps", type=float, default=2.0, 
                       help="DBSCAN clustering distance threshold for recreation boxes.")
    group.add_argument("--recreation-mask-lr-margin-ratio", type=float, default=0.05, 
                       help="Left-right margin ratio for fallback random mask (0.0-0.3).")
                       
    # Validation Dataset
    group.add_argument("--image-batch-multiply-ratio", type=int, default=1,
                       help="When using image_video dataset, the image batch size will be multiplied micro_batch_size by this ratio.")
    group.add_argument("--same-data-batch", action="store_true",
                       help="Use same data type for all rank in a batch for training.")
    group.add_argument("--sample-n-frames", type=int, default=129)    
    group.add_argument("--video-fps", type=float, default=24.0, help="Video fps for train/val and PhysRoPE.")

    # Image Validation Dataset
    group.add_argument("--val-index-file", type=str, nargs='*', help="Index file for the validation dataset.")
    group.add_argument("--val-multireso", action="store_true",
                       help="Specify the index-file/val-index-file is a multi-resolution dataset.")
    group.add_argument("--val-index-strategy", type=str, default="uniform", choices=INDEX_STRATEGY,
                       help="Sample strategy for multiple index files.")
    group.add_argument("--val-index-probability", type=float, nargs='+', default=None,
                       help="Probability for each index file when index-strategy is probability.")
    group.add_argument("--val-image-caption-rate", type=float, default=0.,
                       help="Set a rate to use caption instead of original description.")
    group.add_argument("--val-image-text-arrow-suffix", type=str, default=None, help="Suffix name for text arrow.")
    group.add_argument("--val-image-text-col", type=str, default=None, help="Column name for text arrow.")
    group.add_argument("--val-image-caption-arrow-suffix", type=str, default=None, help="Suffix name for caption arrow.")
    group.add_argument("--val-image-caption-col", type=str, default="llava_caption", help="Column name for caption arrow.")
    group.add_argument("--val-caption-sample-ratio", type=str, default=None, 
                        help="json string format caption-sample-ratio for validation, e.g.{\"long_caption\": 1.0, ...}."
                             "all keys: long caption, short caption, background, shot type, style,light, atmosphere, camera movement")

    # image decrypt
    group.add_argument("--image-decrypt", action="store_true", help="If true, decrypt the image.")

    return parser

def add_denoise_schedule_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Denoising")
    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        choices=["flow", "ddpm", "diffusers_ddpm"],
    )
    # Flow Matching
    group.add_argument("--flow-path-type", type=str, default="linear", choices=FLOW_PATH_TYPE,
                       help="Path type for flow matching schedulers.")
    group.add_argument("--flow-predict-type", type=str, default="velocity", choices=FLOW_PREDICT_TYPE,
                       help="Prediction type for flow matching schedulers.")
    group.add_argument("--flow-loss-weight", type=str, default=None, choices=FLOW_LOSS_WEIGHT,
                       help="Loss weight type for flow matching schedulers.")
    group.add_argument("--flow-train-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during training.")
    group.add_argument("--flow-sample-eps", type=float, default=None,
                       help="Small epsilon for avoiding instability during sampling.")
    group.add_argument("--flow-snr-type", type=str, default="lognorm", choices=FLOW_SNR_TYPE,
                       help="Type of SNR to use for flow matching schedulers.")
    group.add_argument("--flow-shift-image", type=float, default=1.0, help="Shift factor for flow matching schedulers for image.")
    group.add_argument("--flow-shift-video", type=float, default=None, help="Shift factor for flow matching schedulers for video.")
    group.add_argument("--flow-reverse", action="store_true", help="If reverse, learning/sampling from t=1 -> t=0.")
    group.add_argument("--flow-solver", type=str, default="euler", choices=FLOW_SOLVER,
                       help="Solver for flow matching.")
    return parser

def add_extra_models_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Extra Models (VAE, Text Encoder, Tokenizer)")

    # VAE
    group.add_argument("--vae", type=str, default="88-4c-sdxl", choices=list(VAE_PATH), help="Name of the VAE model.")
    group.add_argument("--vae-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the VAE model.")
    group.add_argument("--vae-tiling", action="store_true", help="Enable tiling for the VAE model.")
    group.add_argument("--vae-encode-tile-size", type=int, default=-1, help="Vae encode tile size.")

    # - Encoder: T5, Decoder: Llava/GLM
    group.add_argument("--text-encoder", type=str, default="t5", help="Name of the text encoder model.")
    group.add_argument("--text-encoder-precision", type=str, default="fp16", choices=PRECISIONS,
                       help="Precision mode for the text encoder model.")
    group.add_argument("--text-states-dim", type=int, default=4096, help="Dimension of the text encoder hidden states.")
    group.add_argument("--text-len", type=int, default=256, help="Maximum length of the text input.")
    group.add_argument("--text-len-ll-cap", type=int, default=1000, help="Maximum length of the text input.")
    group.add_argument("--tokenizer", type=str, default="t5", choices=list(TOKENIZER_PATH),
                       help="Name of the tokenizer model.")
    group.add_argument("--text-encoder-infer-mode", type=str, default="encoder", choices=["encoder", "decoder"],
                       help="Inference mode for the text encoder model. It should match the text encoder type. T5 and "
                            "CLIP can only work in 'encoder' mode, while Llava/GLM can work in both modes.")
    group.add_argument("--prompt-template", type=str, default=None, choices=PROMPT_TEMPLATE,
                       help="Prompt template for the decoder-only text encoder model.")
    group.add_argument("--prompt-template-video", type=str, default=None, choices=PROMPT_TEMPLATE,
                       help="Video prompt template for the decoder-only text encoder model.")
    group.add_argument("--hidden-state-skip-layer", type=int, default=None,
                       help="Skip layer for hidden states.")
    group.add_argument("--apply-final-norm", action="store_true",
                       help="Apply final normalization to the used text encoder hidden states.")

    # - CLIP
    group.add_argument("--text-encoder-2", type=str, default=None, choices=list(TEXT_ENCODER_PATH),
                       help="Name of the second text encoder model.")
    group.add_argument("--text-encoder-precision-2", type=str, default=None, choices=PRECISIONS,
                       help="Precision mode for the second text encoder model.")
    group.add_argument("--text-states-dim-2", type=int, default=None,
                       help="Dimension of the second text encoder hidden states.")
    group.add_argument("--tokenizer-2", type=str, default=None, choices=list(TOKENIZER_PATH),
                       help="Name of the second tokenizer model.")
    group.add_argument("--text-len-2", type=int, default=None, help="Maximum length of the second text input.")

    # Text Encoder Common
    group.add_argument("--enable-attention-mask", action="store_true", dest="use_attention_mask",
                       help="Enable attention mask for text encoders and the diffusion model.")
    group.add_argument("--disable-attention-mask", action="store_false", dest="use_attention_mask",
                       help="Disable attention mask for text encoders and the diffusion model.")
    group.set_defaults(use_attention_mask=True)
    group.add_argument("--text-projection", type=str, default="linear", choices=TEXT_PROJECTION,
                       help="A projection layer for bridging the text encoder hidden states and the diffusion model "
                            "conditions.")

    # Glyph byT5 v2
    group.add_argument("--glyph-byT5-v2", action="store_true")
    group.add_argument("--multilingual-prompt-format-font-path", type=str, default=BYT5_FONT)
    group.add_argument("--multilingual-prompt-format-color-path", type=str, default=BYT5_COLOR)
    group.add_argument("--byT5-google-path", type=str, default=BYT5_GOOGLE)
    group.add_argument("--byT5-ckpt-path", type=str,default=BYT5_CKPT)
    group.add_argument("--byt5-max-length", type=int, default=128)
    
    group.add_argument("--siglip", action="store_true")
    
    # condtion Type Embedding
    group.add_argument("--use-cond-type-embedding", action="store_true")
    
    # use scale rope
    group.add_argument("--use-scale-rope", action="store_true", help="Use scale rope for the model.")

    # condition 3D-ROPE
    group.add_argument("--use-cond-3d-rope", action="store_true")

    # PhysRoPE (Physical Rotary Position Embedding)
    group.add_argument("--use-visual-events", action="store_true",
                       help="Enable visual_events caption pipeline (segment prompt with time ranges). "
                            "Independent of PhysRoPE — can be used for baseline training.")
    group.add_argument("--physrope-single-shot-full-range", action="store_true",
                       help="Wrap non-visual-event single-shot captions in one full-length time segment.")
    group.add_argument("--physrope-transition-event-prob", type=float, default=0.0,
                       help="Probability of inserting a synthetic transition event between adjacent visual-event segments.")
    group.add_argument("--physrope-transition-event-duration", type=float, default=2.0,
                       help="Inserted transition duration in the active PhysRoPE time coordinate.")
    group.add_argument("--physrope-transition-event-max-sample-frames", type=int, default=0,
                       help="Hard cap for special-transition augmented samples. 0 disables the cap.")
    group.add_argument("--physrope-transition-event-types", type=str,
                       default="cross_dissolve,fade_black,fade_white,flash_white,wipe_left,wipe_right,wipe_up,wipe_down,push_left,push_right,push_up,push_down,iris_open,iris_close,vertical_blinds,horizontal_blinds",
                       help="Comma-separated transition event types for synthetic transition captions.")
    group.add_argument("--use-physical-rope", action="store_true",
                       help="Enable PhysRoPE for encoding physical time into RoPE.")
    group.add_argument("--physical-rope-theta", type=float, default=256.0,
                       help="Theta for PhysRoPE frequency computation.")
    group.add_argument("--physical-rope-scale", type=float, default=0.1,
                       help="Scale factor for PhysRoPE time values.")
    group.add_argument("--physical-rope-dim", type=int, default=32,
                       help="Dimension of PhysRoPE (split equally between start/end).")
    group.add_argument("--disable-text-physical-rope", action="store_true",
                       help="Disable PhysRoPE on text tokens (only apply to image tokens).")
    group.add_argument("--physical-rope-time-range-mode", type=str, default="second",
                       help="Time range mode for PhysRoPE: 'second' or 'frame'.")
    # PhysRoPE v2 parameters (passing physical_rope_alpha activates v2 instead of v1)
    group.add_argument("--physical-rope-alpha", type=float, default=None,
                       help="Segment time offset coefficient. Activates v2 (time-coord reparameterization). None=use v1.")
    group.add_argument("--physical-rope-offset-mode", type=str, default="physical",
                       help="Offset mode: 'physical' (offset=start_time*alpha) or 'fixed' (offset=seg_index*alpha, like MSM).")
    group.add_argument("--physical-rope-cut", action="store_true",
                       help="Enable RoPE Cut: reset time coords per segment.")
    group.add_argument("--physical-rope-speed-aware", action="store_true",
                       help="Enable speed-aware intra-segment time warping.")
    group.add_argument("--physical-rope-speed-mode", type=str, default="uniform",
                       help="Speed mode: uniform/accelerating/decelerating/slow/fast.")
    group.add_argument("--physical-rope-v2-dim", type=int, default=0,
                       help="Number of RoPE dims affected by v2 time offset. 0=use rope_dim_list[0] only (16 dims). "
                            "When >16, extra dims beyond temporal get Gaussian-decayed offset (soft decay into spatial dims). "
                            "Try 32 or 64.")
    group.add_argument("--physrope-route-debug-limit", type=int, default=2,
                       help="Number of PhysRoPE v2 frequency-routing calls verified and "
                            "printed on every distributed rank.")
    group.add_argument(
        "--reference-timeline-debug-limit",
        type=int,
        default=0,
        help=(
            "Number of ReferenceTimeline fixed-target clock mappings printed per "
            "rank group. The print verifies that source/target length mismatch "
            "does not change target PhysRoPE positions."
        ),
    )
    # Unified PhysRoPE: range_condition training (soft reference by default)
    group.add_argument("--unified-physrope-enable", type=str, default="false",
                       help="Enable unified PhysRoPE range_condition training. "
                            "When true, a fraction of t2v samples are converted to range_condition "
                            "(random frames become condition, rest are generated).")
    group.add_argument("--unified-physrope-condition-prob", type=float, default=0.0,
                       help="Probability of converting a t2v sample to range_condition. "
                            "E.g. 0.5 means 50%% of t2v samples become range_condition.")
    group.add_argument("--unified-physrope-cond-loss-weight", type=float, default=1.0,
                       help="Loss weight for condition frames in range_condition training. "
                            "1.0 = full-frame loss like OmniWeaving (default). "
                            "0.0 < w < 1.0 = weighted condition frame loss. "
                            "0.0 is only valid with --range-condition-mode hard_legacy.")
    group.add_argument("--range-condition-mode", type=str, default="soft_reference",
                       choices=["soft_reference", "hard_legacy"],
                       help="range_condition injection semantics. soft_reference keeps "
                            "conditions only in concat reference channels and computes "
                            "full-frame/weighted loss. hard_legacy pins preserve_mask "
                            "frames in the main latent; use only to reproduce old exact-insert runs.")
    group.add_argument("--range-condition-native-t-concat", type=str, default="false",
                       help="Keep each range-condition reference at its native temporal length, "
                            "concatenate reference streams before the noisy target along T, and "
                            "slice model output back to the target interval. Default false keeps "
                            "the existing target-length channel-concat behavior unchanged.")
    group.add_argument(
        "--range-condition-temporal-pack-mode",
        type=str,
        default="legacy_trilinear",
        choices=["legacy_trilinear", "reference_timeline"],
        help=(
            "How variable-length range-condition latents are packed into the fixed target "
            "timeline. reference_timeline copies only exact source frames and never temporally "
            "interpolates or repeats reference latents."
        ),
    )
    group.add_argument("--unified-physrope-temporal-resample-prob", type=float, default=0.0,
                       help="Optional L4 self-reference temporal resample probability "
                            "applied after per-segment conditions are loaded. External "
                            "T_src!=T_dst conditions are resampled only by the legacy "
                            "target-canvas path; native T concat forbids this augmentation.")
    group.add_argument("--unified-physrope-temporal-resample-range", type=str, default="0.8,1.25",
                       help="Length stretch range for optional L4 self-reference resample, "
                            "formatted as 'min,max'.")
    group.add_argument("--l4-temporal-resample-mode", type=str, default="trilinear",
                       choices=["trilinear"],
                       help="Temporal interpolation mode for L4 condition latent packing.")
    group.add_argument("--l4-temporal-resample-align-corners", type=str, default="false",
                       help="Whether L4 trilinear temporal interpolation uses align_corners.")

    # Per-segment condition (TIV2V) training: arrows with `segment_conditions`
    # column carry per-shot edited latents; each is independently activated as
    # a reference. Disjoint segments stay separate (cond_seg_dst_ranges list);
    # forward path merges them at prepare_model_inputs time.
    group.add_argument("--per-segment-condition-enable", type=str, default="false",
                       help="Enable per-segment condition loading from arrow's "
                            "segment_conditions column. Arrows without the column "
                            "transparently fall back to t2v.")
    group.add_argument("--per-segment-condition-prob", type=float, default=0.3,
                       help="Per-segment independent activation probability. 0 segs → t2v, "
                            "1 seg → range_condition single, 2+ segs → multi-range TIV2V.")
    group.add_argument("--per-segment-condition-max", type=int, default=4,
                       help="Hard cap on the number of active segments per sample.")
    group.add_argument("--per-segment-prompt-empty-ratio", type=float, default=0.7,
                       help="Per active seg: prob of keeping '[a-b] ' time prefix only "
                            "(empty body_text). §8.2 default.")
    group.add_argument("--per-segment-prompt-remove-ratio", type=float, default=0.2,
                       help="Per active seg: prob of removing the seg from prompt entirely "
                            "(time prefix gone too). §8.2 default.")
    group.add_argument("--per-segment-condition-require-explicit-ranges", type=str, default="true",
                       help="Require prompt_dst_range and target_dst_range in each condition. "
                            "Set false only for legacy stage1 arrows; prompt-shot identity is "
                            "still checked from source time, prepared ranges, and seg_id.")
    group.add_argument("--physrope-prompt-debug-limit", type=int, default=5,
                       help="Number of rank0 worker0 PhysRoPE training samples to print "
                            "with prompt/range/condition debug.")

    # Condition dropout (future_plan §1.4(c)) — after loading conditions,
    # randomly mask out entire condition slots to train robustness to partial
    # specification. Unlike per_segment_condition_prob (which controls loading),
    # this drops ALREADY-LOADED conditions so the model learns to degrade
    # gracefully when conditions are missing at inference.
    group.add_argument("--condition-dropout-prob", type=float, default=0.0,
                       help="Probability of dropping each loaded condition slot. "
                            "0.0=no dropout (default). 0.1-0.3 recommended. "
                            "Applied AFTER per-segment loading in helpers.py.")

    # Non-integer RoPE jitter (future_plan §1.4(d)) — perturbs PhysRoPE float
    # time coords so the model is robust to L4 linear-rescale's non-integer
    # positions. Prompt string keeps int-rounded values; only RoPE consumer sees
    # the jittered floats.
    group.add_argument("--rope-noninteger-jitter-prob", type=float, default=0.0,
                       help="Probability of applying non-integer RoPE jitter on a sample.")
    group.add_argument("--rope-noninteger-jitter-range", type=float, default=0.0,
                       help="Half-range of uniform jitter applied to each segment "
                            "time_range endpoint, in latent frames (e.g. 0.2 → ±0.2).")

    # Image-anchor (L1/L2): sample K random GT latent frames as 1-frame condition
    # anchors per future_plan §1.2 L1/L2 (DreaMontage Shared-RoPE style).
    # No external data — anchors come from the same arrow's GT.
    group.add_argument("--image-anchor-enable", type=str, default="false",
                       help="Enable in-video image-anchor sampling.")
    group.add_argument("--image-anchor-prob", type=float, default=0.0,
                       help="Per-sample probability of activating image anchors.")
    group.add_argument("--image-anchor-max", type=int, default=2,
                       help="Max number of anchors per sample (further capped by "
                            "MAX_SEGS - already-active video segments).")

    return parser


def add_training_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Training")
    if not external_engine:
        group.add_argument("--optimizer", type=str, default='adamw', 
                           help="Choose the optimizer type in adamw and muon.")
        group.add_argument(
            "--weight-decay", type=float, default=0.01, help="Weight decay to apply."
        )
        group.add_argument(
            "--num-workers", type=int, default=10, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        )
        group.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Initial learning rate (after the potential warmup period) to use.",
        )        
        group.add_argument(
            "--train-iters",
            type=int,
            default=1000000,
            help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
        )
    group.add_argument("--image-micro-batch-size", type=int, default=1, nargs='*',
                       help="Image batch size per model instance (local batch size).")
    group.add_argument("--video-micro-batch-size", type=int, default=None, nargs='*',
                       help="Video batch size per model instance (local batch size).")
    if not external_engine:
        group.add_argument(
            "--checkpointing-steps",
            type=int,
            default=500,
            help=(
                "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
                " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
                " training using `--resume_from_checkpoint`."
            ),
        )
        group.add_argument(
            "--resume",
            type=str,
            default=None,
            help=(
                "Whether training should be resumed from a previous checkpoint. Use a path saved by"
                ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
            ),
        )
        group.add_argument(
            "--logging-dir",
            type=str,
            default="logs",
            help=(
                "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
                " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
            ),
        )
        group.add_argument(
            "--num-train-epochs",
            type=int,
            default=100
        )
        group.add_argument(
            "--gradient-accumulation-steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        group.add_argument(
            "--distributed-timeout-sec",
            type=int,
            default=3600,
            help="Timeout in seconds for the default torch.distributed process group.",
        )
        group.add_argument(
            "--scale-lr",
            action="store_true",
            default=False,
            help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
        )
        group.add_argument(
            "--lr-warmup-steps",
            type=int,
            default=10,
            help="Number of steps for the warmup in the lr scheduler.",
        )
        group.add_argument(
            "--max-grad-norm", default=1.0, type=float, help="Max gradient norm."
        )
        group.add_argument("--selective-checkpointing", type=float, default=1.0)
        group.add_argument(
            "--allow-tf32",
            action="store_true",
            help=(
                "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
            ),
        )
        group.add_argument(
            "--lr-scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        group.add_argument(
            "--lr-num-cycles",
            type=int,
            default=1,
            help="Number of cycles in the learning rate scheduler.",
        )
        group.add_argument(
            "--lr-power",
            type=float,
            default=1.0,
            help="Power factor of the polynomial scheduler.",
        )
        group.add_argument(
            "--log-interval",
            type=int,
            default=20,
            help="Interval of logging.",
        )

    group.add_argument(
        "--master-weight-type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )

    group.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    group.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for data loading.")

    group.add_argument("--sample-interval", type=int, default=200, help="sample-interval")

    group.add_argument("--validation-every", type=int, default=-1, help="Evaluate every N update steps.")
    group.add_argument("--validation-at-steps", type=int, default=[0], nargs='+',
                       help="Evaluate at specific update steps during training. It is useful to evaluate the model at "
                            "early steps to check if everything is working properly.")
    group.add_argument("--validation-metrics", type=str, default=None, nargs='+', choices=SCORE_METRICS|LOSS_METRICS,
                       help="Validation types to evaluate during training.")
    group.add_argument("--fixed-prompt-validation-every", type=int, default=1000, help="fixed prompt validation every N steps.")
    group.add_argument("--try-first-iter", action="store_true", help="Try first iter validation.")
    group.add_argument(
        "--resume-fixed-prompt-step0",
        action="store_true",
        default=False,
        help="Keep try_first_iter enabled after loading a resume checkpoint so current-run step 0 runs fixed prompt validation.",
    )

    group.add_argument("--dry-run", action="store_true", help="Dry run mode.")
    group.add_argument(
        "--loss-spike-threshold",
        type=float,
        default=5.0,
        help="Skip optimizer update if loss exceeds this threshold."
    )
    group.add_argument("--abnormal-loss-rank-threshold", type=float, default=-1)
    group.add_argument("--abnormal-loss-rank-max-num", type=int, default=100)
    group.add_argument("--abnormal-loss-rank-history-num", type=int, default=10)
    group.add_argument("--abnormal-loss-text-trunc", type=int, default=50)

    return parser


def add_network_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Network")

    # Main model
    group.add_argument("--model-type", type=str, default="DiT-S/2",
                       help="Model architecture to use. It it also used to determine the experiment directory.")
    group.add_argument("--modulate-type", type=str, default="dit", choices=MODULATION_TYPE, help="Modulation type for the transformer block.")
    group.add_argument("--text-pool-type", type=str2null, default=None, choices=TEXT_POOL_TYPE,
                       help="Text pooling type for the text hidden states. Pooled vector will be added to the "
                            "timestep embedding.")
    group.add_argument("--precision", type=str, default="bf16", choices=PRECISIONS,
                       help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.")

    group.add_argument("--latent-channels", type=str, default=None,
                       help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
                            "it still needs to match the latent channels of the VAE model.")

    group.add_argument("--attn-drop-rate", type=float, default=0.0, help="Dropout rate for attention QKV layers.")
    group.add_argument("--proj-drop-rate", type=float, default=0.0, help="Dropout rate for attention project layers.")
    group.add_argument("--mlp-drop-rate", type=float, default=0.0, help="Dropout rate for MLP layers.")
    group.add_argument("--norm-type", type=str, default="layer", choices=NORMALIZATION_TYPE,
                       help="Normalization type for the transformer block.")
    group.add_argument("--act-type", type=str, default="silu", choices=ACTIVATION_TYPE,
                       help="Activation function for the transformer block.")
    group.add_argument("--qk-norm", action="store_true", help="Enable QK-Normalization in all attention layers.")
    group.add_argument("--no-qk-norm", dest="qk_norm", action="store_false", help="Disable QK-Normalization.")
    group.set_defaults(qk_norm=True)
    group.add_argument("--attn-mode", type=str, default="flash", choices=ATTENTION_MODE,
                       help="Attention mode for the transformer block.")
    # RoPE
    group.add_argument("--rope-theta", type=int, default=256, help="Theta used in RoPE.")

    # Block Attn
    group.add_argument("--ba-attn-chunk-size", type=int, default=512)
    group.add_argument("--ba-attn-topk", type=int, default=3)
    group.add_argument("--ba-attn-causal", action="store_true", help="Whether to use causal attention.")
    group.add_argument("--ba-attn-slice-method", type=str, default="3D", choices=["3D", "1D"])
    group.add_argument("--ba-attn-full-attn-block-num", type=int, default=0, help="Number of full attention blocks for ba/full hybrid.")
    group.add_argument("--ba-attn-full-attn-every-n-blocks", type=int, default=0, help="Number of blocks between full attention blocks.")
    
    # SSTA config
    group.add_argument("--ssta-topk", type=int, default=2)
    group.add_argument("--ssta-threshold", type=float, default=0.0) # minimum attention score to be considered
    group.add_argument("--ssta-enforce-share-topk-block", type=int, default=1)
    group.add_argument("--sparse-attn-sparse-type", type=str, default=None)
    group.add_argument("--sparse-attn-pad-type", type=str, default="zero") # zero pad 
    group.add_argument("--sparse-attn-use-text-mask", type=int, default=1) # 是否使用text mask传入到稀疏attn计算中
    group.add_argument("--sparse-attn-mask-share-within-head", type=int, default=1) # 



    # MOE config
    if not external_engine:
        group.add_argument("--num-experts", type=int, default=None, help="Number of experts.")
    group.add_argument("--use-mixed-mlp-moe",type=int, default=1, help="Whether to use mixed MLP MoE.")
    group.add_argument("--moe-topk", type=int, default=1, help="Top-k for MoE.")
    group.add_argument("--capacity-factor", type=float, default=1.0, help="Capacity factor for MoE.")
    group.add_argument("--moe-drop-tokens", type=int, default=0, help="Whether to drop tokens in MoE.")
    group.add_argument("--moe-random-routing-dropped-token", type=int, default=0, help="Whether to use random routing for dropped tokens in MoE.")
    group.add_argument("--moe-loss-weight", type=float, default=1e-3, help="Whether to use random routing for dropped tokens in MoE.")

    group.add_argument("--is-dense-with-moe-mlp", type=str2bool, default=False)
    group.add_argument("--dense-with-moe-mlp-intermediate-size", type=int, default=None)

    # padded-sequence MoE
    group.add_argument("--use-padded-moe", action="store_true", help="pad MoE token sequences")
    group.add_argument("--moe-pad-seqlen", type=int, default=0, help="padding seqlen")
    group.add_argument("--enable-moe-token-mask", action="store_true", help="mask pad token")



    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    group.add_argument(
        "--win-size",
        type=str, default="3_3_3",
        help="win-size.",
    )
    group.add_argument(
        "--tile-size",
        type=str, default="8_4_4",
        help="tile-size.",
    )
    group.add_argument(
        "--win-ratio",
        type=int, default="9",
        help="win-size.",
    )
    group.add_argument(
        "--win-type",
        type=str, default="fix",
        help="win type.",
    )
    return parser



def add_parallelism_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="FSDP")

    group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    # Parallelism
    group.add_argument("--sp-size", type=int, default=1, help="For sequence parallel")
    group.add_argument("--ep-size", type=int, default=1, help="Size of expert parallel.")
    group.add_argument("--pp-size", type=int, default=1, help="Size of pipeline parallel.")


    group.add_argument("--use-dynamic-ring-attention", action="store_true", help="Use dynamic Ring Attention.")
    group.add_argument(
        "--train-sp-batch-size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    group.add_argument("--fsdp-sharding-strategy", default="full", choices=['full', 'hybrid'])

    return parser

def add_evaluation_args(parser: argparse.ArgumentParser, external_engine=False):
    group = parser.add_argument_group(title="Validation Loss Evaluation")


    # ================== Inference Model config =====================
    group.add_argument("--dit-weight", type=str, default=None, help="Path to the diffusion model weight.")
    parser.add_argument("--tp-size", type=int, default=1, help="For tensor parallel")
    parser.add_argument("--reproduce", action="store_true",
                       help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    # ======================== Inference mode ========================
    group.add_argument("--num-frames", type=int, default=1)
    group.add_argument("--val-batch-size", type=int, default=1, nargs='*',
                       help="Batch size for inference and evaluation.")
    group.add_argument("--val-batch-size-video", type=int, default=1, nargs='*',
                       help="Batch size for inference and evaluation for video.")
    group.add_argument("--val-disable-autocast", action="store_true",
                       help="Disable autocast for denoising loop and vae decoding in pipeline sampling.")
    group.add_argument("--num-images", type=int, default=1, help="Number of images to generate for each prompt.")
    group.add_argument("--save-path", type=str, default=None, help="Path to save the generated samples.")
    # --- prompt ---
    group.add_argument("--prompt", type=str, default=None, help="Prompt for sampling during evaluation.")
    group.add_argument("--interactive", action="store_true", help="Enable interactive sampling.")
    # --- csv ---
    group.add_argument("--csv", type=str, default=None, help="CSV file for evaluation.")
    group.add_argument("--seed-type", type=str, default="auto", choices=["file", "random", "fixed", "auto"],
                       help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
                            "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
                            "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
                            "fixed `seed` value.")
    
    
    if not external_engine:
        group.add_argument("--seed", type=int, default=42, help="Seed for evaluation.")
    group.add_argument("--save-path-suffix", type=str, default="", help="Suffix for the directory of saved samples.")

    # ============= range_condition inference (L1/L2/L3/L4/L5) =============
    # Used when the eval dataset's task_type == "range_condition" and the
    # batch does not carry an explicit "condition_spec" JSON field. Specifies
    # which latent frames to drop the ref_image into for L1/L2 anchors.
    group.add_argument("--range-condition-anchor-frames", type=str, default="0",
                       help="Comma-separated latent frame indices for ref_image_paths anchors "
                            "(e.g. '0' = first-frame i2v-style; '0,8,16' = three-anchor L2). "
                            "Maps positionally to ref_image_paths.")
    group.add_argument("--range-condition-self-cond-two-pass", action="store_true",
                       help="When no explicit anchors and no ref_image_paths, do a two-pass "
                            "eval: pass-1 t2v generation → encode → pass-2 self-condition at "
                            "latent_target_length//2. Mirrors train.py fixed-prompt Mode B.")
    group.add_argument("--name-suffix", type=str, default="", help="Suffix for the names of saved samples.")
    group.add_argument("--skip-exist", action="store_true", help="Skip existing samples.")
    group.add_argument("--val-image-suffix", type=str2null, default=None)

    # Extra parameters
    group.add_argument("--ddp", action="store_true", help="Enable Distributed Data Parallel (DDP) during evaluation.")
    group.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for DDP.")
    group.add_argument("--node-index", type=int, default=0, help="Node index for DDP.")
    # Classifier-Free Guidance
    group.add_argument("--neg-prompt", type=str, default=None, help="Negative prompt for sampling.")
    # DDPM
    group.add_argument("--cfg-scale", type=float, default=6.0, help="Classifier free guidance scale.")
    group.add_argument("--cfg-scale-video", type=float, default=6.0, help="Classifier free guidance scale for video.")

    #FPS tag
    group.add_argument("--prompt-fps-prefix", type=str2null, default=None, help="FPS tag")
    group.add_argument("--video-fps-col-name", type=str, default=None, help="Column name for FPS in video data.")

    # ======================== Evaluation mode ========================
    group.add_argument("--offline-ckpt-base-path", type=str, default=None)
    group.add_argument("--offline-ckpt-start", type=int, default=None)
    group.add_argument("--offline-ckpt-end", type=int, default=None)
    group.add_argument("--offline-ckpt-interval", type=int, default=None)
    group.add_argument("--offline-save-path-auto", type=str2bool, default=False)

    group.add_argument("--ckpt", type=str, help="Path to the checkpoint for offline eval.")
    group.add_argument("--val-save-path", type=str, help="Path to save the evaluation results.")
    group.add_argument("--val-name-suffix", type=str, default="",
                       help="Suffix for the names of saved results.")
    # --- validation loss ---
    group.add_argument("--val-timesteps", type=int, default=8, help="Number of timesteps for evaluation.")
    # --- scores ---
    group.add_argument("--scores", type=str, nargs="+", help="Scores to be evaluated.")
    group.add_argument("--score-save-image", action="store_true", help="Save images for score evaluation.")
    group.add_argument("--use-pre-extraction", action="store_true",
                       help="Whether to use pre-extracted coco text embedding.")
    group.add_argument("--fid-inception-path", type=str, help="Path to Inception model.")
    group.add_argument("--fid-target-path", type=str, help="Path to FID target statistics.")
    group.add_argument("--ics-inception-path", type=str, help="Path to Inception model.")
    group.add_argument("--clip-model-path", type=str, help="Path to CLIP model.")
    group.add_argument("--hpsv2-model-path", type=str, help="Path to HPSv2 model.")
    group.add_argument("--t2i-compbench-path", type=str, nargs='+', help="Paths to T2I-CompBench dataset.")
    group.add_argument("--vqa-model-path", type=str,
                       help="Path to BLIP VQA model. Used for T2I-CompBench evaluation.")
    # --- video ---
    group.add_argument("--video-csv", type=str, nargs='+', default=None, help="CSV file for evaluation.")
    group.add_argument("--val-video", action="store_true", help="Evaluate validation dataset during training.")
    group.add_argument("--val-video-index-file", type=str, help="Index file for the validation dataset.")
    group.add_argument("--val-video-batch-size", type=int, default=1, help="Batch size for inference and evaluation.")
    group.add_argument("--val-video-caption", type=str, nargs="+", help="(name, rename or path) caption for video clip evalution.")
    group.add_argument("--val-video-dataset-num", type=int2null, default=None, help="Validation dataset number.")
    group.add_argument("--val-video-every", type=int, help="Validation dataset number.")
    group.add_argument("--val-video-num-workers", type=int, default=1, help="Validation dataset number.")
    group.add_argument("--val-video-suffix", type=str2null, default=None)
    group.add_argument("--val-video-offset", type=int, default=0)
    group.add_argument("--val-video-output-dir", type=str, help="Validation video output dir.")
    group.add_argument("--val-video-timestamp", type=str, help="Validation video timestamp.")
    group.add_argument("--val-video-dataset-name", type=str, help="Validation video dataset name.")

    # --- video scores ---
    group.add_argument("--val-video-metrics-evaluator-type", type=str, default=None)
    group.add_argument("--val-video-metrics-gt-arrow-file", type=str, default=None)
    group.add_argument("--val-video-metrics-is-gt", action="store_true")
    group.add_argument("--val-video-metrics-pretrained-model-base-path", type=str2null, default="weights/evaluation")
    group.add_argument("--val-video-metrics-gt-sample-index-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-gt-fvd-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-gt-fid-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-fid-batch-size", type=int2null, default=16)
    group.add_argument("--val-video-metrics-inception-batch-size", type=int2null, default=16)
    group.add_argument("--val-video-metrics-inception-splits", type=int2null, default=10)
    group.add_argument("--val-video-metrics-is-category1", type=str2bool, default=False)
    group.add_argument("--val-video-metrics-is-category2", type=str2bool, default=False)
    group.add_argument("--val-video-metrics-idx-to-category1-mapping-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-idx-to-category2-mapping-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-is-automatic-sample-index", action="store_true")
    group.add_argument("--val-video-metrics-clip-model", type=str2null, default="ViT-B/32")
    group.add_argument("--val-video-metrics-clip-caption-type", type=str2null, default=None)
    group.add_argument("--val-video-metrics-clip-context-length", type=int2null, default=77)
    group.add_argument("--val-video-metrics-inception-resize", type=str2bool, default=True)
    group.add_argument("--val-video-metrics-fid-dims", type=int2null, default=2048)
    group.add_argument("--val-video-metrics-ckpt-num", type=str2null, default=None)
    group.add_argument("--val-video-metrics-ckpt-path", type=str2null, default=None)
    group.add_argument("--val-video-metrics-dataset-num", type=int2null, default=None)
    group.add_argument("--val-video-metrics-video-name", type=str2null, default=None)
    group.add_argument("--val-video-metrics-suffix", type=int, default=0)


    group.add_argument("--infer-flow-shift", type=float, default=1.0, help="Shift factor for flow matching schedulers for image.")
    group.add_argument("--infer-flow-shift-video", type=float, default=1.0, help="Shift factor for flow matching schedulers for video.")
    group.add_argument("--infer-steps", type=int, default=100, help="Number of denoising steps for inference for image.")
    group.add_argument("--infer-steps-video", type=int, default=50, help="Number of denoising steps for inference for video.")
    group.add_argument(
        "--val-num-workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )

    group.add_argument("--val-yaml-path", type=str, default=None)
    group.add_argument("--val-exp-name", type=str, default=None)
    group.add_argument("--val-i2v-target-size", type=str, default="360p")

    return parser
