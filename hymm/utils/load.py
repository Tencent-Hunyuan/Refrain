import torch
from hymm.models.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from transformers import T5EncoderModel, AutoTokenizer
import os
from torch import nn

# Path
from pathlib import Path
import torch.nn.functional as F
from hymm.models.text_encoder import TextEncoder
from hymm.utils.logging_ import main_print
from hymm.models.modules import load_model as load_model

from hymm.constants import *
from hymm.utils.helpers import load_byt5_and_byt5_tokenizer
# audio related tokenizer
from hymm.models.t2a_modules.t5 import T5EncoderModel as UMT5EncoderModel

def load_hunyuan_state_dict(args, model, dit_model_name_or_path, convert_state_dict_in_out=True,
                            convert_state_dict_img_in_for_i2v=False,
                            convert_state_dict_16168_to_16164=False, logger=None):
    if not os.path.exists(dit_model_name_or_path):
        main_print(f"Invalid dit_model_name_or_path path: {dit_model_name_or_path}")
        return model
    load_key = "module"
    model_path = dit_model_name_or_path
    bare_model = "unknown"

    # 兼容 safetensors 格式
    if model_path.endswith('.safetensors'):
        from safetensors import safe_open
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(
            model_path, map_location="cpu", weights_only=True
        )

    if convert_state_dict_in_out:
        tmp_state_dict = {}
        for k in state_dict.keys():
            if k == 'img_in.proj.weight':
                tmp_state_dict[k] = torch.cat([state_dict['img_in.proj.weight'], state_dict['img_in.proj.weight']], dim=1)
            elif k == 'final_layer.linear.weight':
                tmp_state_dict[k] = torch.cat([state_dict['final_layer.linear.weight'], state_dict['final_layer.linear.weight']], dim=0)
            elif k == 'final_layer.linear.bias':
                tmp_state_dict[k] = torch.cat([state_dict['final_layer.linear.bias'], state_dict['final_layer.linear.bias']], dim=0)
            else:
                tmp_state_dict[k] = state_dict[k]
        state_dict = tmp_state_dict
    
    if convert_state_dict_img_in_for_i2v:
        tmp_state_dict = {}
        for k in state_dict.keys():
            if k == 'img_in.proj.weight':
                tmp_state_dict[k] = torch.cat([state_dict['img_in.proj.weight'], model.state_dict()['img_in.proj.weight'][:,model.in_channels:,:,:].cpu()], dim=1)
            else:
                tmp_state_dict[k] = state_dict[k]
        state_dict = tmp_state_dict

    if convert_state_dict_16168_to_16164:
        tmp_state_dict = {}
        for k in state_dict.keys():
            if k == 'img_in.proj.weight':
                #tmp_state_dict[k] = torch.cat([state_dict['img_in.proj.weight'][:, :32, :, :],
                #                               state_dict['img_in.proj.weight'][:, -32:, :, :]], dim=1)
                tmp_state_dict[k] = state_dict['img_in.proj.weight'][:, :32, :, :]
            elif k == 'final_layer.linear.weight':
                tmp_state_dict[k] = state_dict['final_layer.linear.weight'][:32]
            elif k == 'final_layer.linear.bias':
                tmp_state_dict[k] = state_dict['final_layer.linear.bias'][:32]
            else:
                tmp_state_dict[k] = state_dict[k]
        state_dict = tmp_state_dict

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )
    if model_path.endswith('.safetensors'): # Refrain 模型训练使用safetensors格式保存
        new_dict = state_dict
    else: # 兼容Aries老版本
        new_dict = convert_hunyuan_dict_for_tensor_parallel(state_dict)

    if args.multitask_semantic_mode == "siglip":
        result = model.load_state_dict(new_dict, strict=False)
    else:
        result = model.load_state_dict(new_dict, strict=True)

    if result.missing_keys:
        logger.info(f"[load.py] Missing keys when loading state_dict:")
        for key in result.missing_keys:
            logger.info(f"[load.py] Missing key: {key}")
    if result.unexpected_keys:
        logger.info(f"[load.py] Unexpected keys when loading state_dict:")
        for key in result.unexpected_keys:
            logger.info(f"[load.py] Unexpected key: {key}")
    return model


def convert_hunyuan_dict_for_tensor_parallel(state_dict):
    """ 将QKV 单个权重矩阵转换为三个子权重: weight_Q、weight_K、weight_V
    """
    print("converting checkpoint to tensor parallel model")
    new_dict = {}
    for k, w in state_dict.items():
        if k.startswith("double_blocks") and "attn_qkv.weight" in k:
            hidden_size = w.shape[1]
            k1 = k.replace("attn_qkv.weight", "attn_q.weight")
            w1 = w[:hidden_size, :]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.weight", "attn_k.weight")
            w2 = w[hidden_size:2 * hidden_size, :]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.weight", "attn_v.weight")
            w3 = w[-hidden_size:, :]
            new_dict[k3] = w3
        elif k.startswith("double_blocks") and "attn_qkv.bias" in k:
            hidden_size = w.shape[0] // 3
            k1 = k.replace("attn_qkv.bias", "attn_q.bias")
            w1 = w[:hidden_size]
            new_dict[k1] = w1
            k2 = k.replace("attn_qkv.bias", "attn_k.bias")
            w2 = w[hidden_size:2 * hidden_size]
            new_dict[k2] = w2
            k3 = k.replace("attn_qkv.bias", "attn_v.bias")
            w3 = w[-hidden_size:]
            new_dict[k3] = w3
        elif k.startswith("single_blocks") and "linear1" in k:
            hidden_size = state_dict[k.replace("linear1", "linear2")].shape[0]

            k1 = k.replace("linear1", "linear1_q")
            w1 = w[:hidden_size]
            new_dict[k1] = w1

            k2 = k.replace("linear1", "linear1_k")
            w2 = w[hidden_size:2*hidden_size]
            new_dict[k2] = w2

            k3 = k.replace("linear1", "linear1_v")
            w3 = w[2*hidden_size:3*hidden_size]
            new_dict[k3] = w3

            k4 = k.replace("linear1", "linear1_mlp")
            w4 = w[3*hidden_size:]
            new_dict[k4] = w4
        elif k.startswith("single_blocks") and "linear2" in k:
            k1 = k.replace("linear2", "linear2.fc")
            new_dict[k1] = w
        else:
            new_dict[k] = w
    
    return new_dict


def load_transformer(
    model_type,
    dit_model_name_or_path,
    factor_kwargs=None,
    convert_state_dict_in_out=False,
    convert_state_dict_img_in_for_i2v=False,
    convert_state_dict_16168_to_16164=False,
    args=None,
    logger=None,
):
    transformer = load_model(model_type, args, factor_kwargs)
    transformer = load_hunyuan_state_dict(args, transformer, dit_model_name_or_path, convert_state_dict_in_out,
                                          convert_state_dict_img_in_for_i2v, convert_state_dict_16168_to_16164, logger)
    return transformer


def load_vae(pretrained_model_name_or_path, device, logger, args):
    vae_precision = torch.float16

    vae_path = os.path.join(
            pretrained_model_name_or_path, "hunyuan-video-t2v-720p/vae"
    )

    config = AutoencoderKLCausal3D.load_config(vae_path)
    vae = AutoencoderKLCausal3D.from_config(config)

    vae_ckpt = Path(vae_path) / "pytorch_model.pt"
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

    ckpt = torch.load(vae_ckpt, map_location=vae.device, weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {
                k.replace("vae.", ""): v
                for k, v in ckpt.items()
                if k.startswith("vae.")
        }
    vae.load_state_dict(ckpt)
    vae = vae.to(dtype=vae_precision)
    vae.requires_grad_(False)
    vae = vae.to(device)
    vae.eval()
    autocast_type = torch.float16
    fps = 24
    return vae, autocast_type, fps


def load_text_encoder(pretrained_model_name_or_path, device, logger, args):
    # prompt_template_video
    if args.text_encoder == "llava-llama-3-8b":
        text_encoder_path = os.path.join(pretrained_model_name_or_path, "text_encoder")
    else:
        text_encoder_path = None
        
    # Use the original TextEncoder for non-distributed mode
    text_encoder = TextEncoder(
        text_encoder_type=args.text_encoder,
        text_encoder_path=text_encoder_path,
        max_length=args.text_len if not args.use_ll_cap_dataset else args.text_len_ll_cap,
        text_encoder_precision="fp16",
        tokenizer_type=args.tokenizer,
        prompt_template=PROMPT_TEMPLATE[args.prompt_template] if args.prompt_template else None,
        prompt_template_video=PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video else None,
        hidden_state_skip_layer=2,
        apply_final_norm=False,
        reproduce=False,
        logger=logger,
        device=device,
        uncond_token=args.uncond_token,
    )
        
    text_encoder_2 = None
    if args.text_encoder_2:
        if args.text_encoder_2 == "clipL":
            text_encoder_path_2 = os.path.join(pretrained_model_name_or_path, "text_encoder_2")
        else:
            text_encoder_path_2 = None
        
        text_encoder_2 = TextEncoder(
            text_encoder_type=args.text_encoder_2,
            text_encoder_path=text_encoder_path_2,
            max_length=args.text_len_2,
            text_encoder_precision="fp16",
            tokenizer_type=args.tokenizer_2,
            reproduce=False,
            logger=logger,
            device=device,
        )

    return text_encoder, text_encoder_2


def load_t2a_text_encoder(args, device, logger):
    # In fact, this args is not args of train script, it is the inject config.
    if args.text_feat_type == "umt5-xxl":
        text_encoder = UMT5EncoderModel(
            text_len=args.max_text_length,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(args.t5_checkpoint_dir, args.t5_checkpoint),
            tokenizer_path=os.path.join(args.t5_checkpoint_dir, args.t5_tokenizer),
            shard_fn=None
        )
    elif args.text_feat_type in TEXT_ENCODER_PATH:
        text_encoder = TextEncoder(
            text_encoder_type=args.text_feat_type,
            text_encoder_path=None,
            max_length=args.max_text_length,
            text_encoder_precision=args.text_encoder_precision, # default fp16
            tokenizer_type=args.tokenizer_type,
            prompt_template=None,
            prompt_template_video=None,
            hidden_state_skip_layer=args.hidden_state_skip_layer, # default 2
            apply_final_norm=args.apply_final_norm, # default False
            reproduce=False,
            logger=logger,
            device=device,
        )
    else:
        raise ValueError("Please specify the text feature type")

    return text_encoder


from hymm.dataset.format_prompt import MultilingualPromptFormat


# def load_t2va_text_encoder(args, device, logger):
#     text_encoder, text_encoder_2 = load_text_encoder(args.pretrained_model_name_or_path, device, logger, args)
#     byt5_kwarg = None
#     if args.t2v_glyph_byT5_v2:
#         byt5_kwarg = load_glyph_byT5_v2(args, device)
#         video_byt5_prompt_format = MultilingualPromptFormat(
#         font_path=args.multilingual_prompt_format_font_path,
#         color_path=args.multilingual_prompt_format_color_path
#         )
#     else:
#         byt5_kwarg = None
#         video_byt5_prompt_format = None
    
#     t2a_text_encoder = load_t2a_text_encoder(args, device, logger)
    
#     return text_encoder, text_encoder_2, byt5_kwarg, video_byt5_prompt_format, t2a_text_encoder

import json

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    # 确保 device 是正确的格式
    if isinstance(device, int):
        device = f'cuda:{device}'
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_dac_bigvae_ckpt(ckpt_file, device):
    # 加载DAC BigVAE模型
    config_file = os.path.join(os.path.dirname(ckpt_file), "config.json")  
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    # torch.manual_seed(h.seed)
    # 导入BigVAE生成器
    from hymm.models.audio_encoders.dac_bigvae.dac.model.generator import DAC as DAC_Generator
    
    generator = DAC_Generator(
        encoder_rates=h.encoder_rates,
        decoder_rates=h.decoder_rates,
        encoder_dim=h.encoder_dim,
        latent_dim=h.latent_dim,
        decoder_dim=h.decoder_dim,
        n_codebooks=h.n_codebooks,
        sample_rate=h.sample_rate,
        continuous=h.continuous,
        bigvgan_decoder=h.bigvgan_decoder,
    ).to(device)

    state_dict_g = load_checkpoint(ckpt_file, device)
    missing_keys, unexpected_keys = generator.load_state_dict(state_dict_g["generator"], strict=False)

    # 打印加载信息
    if missing_keys:
        print(f"Warning: 缺少以下参数: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: 多余的参数（未使用）: {unexpected_keys}")
    print(f"成功加载参数，匹配参数数量: {len(state_dict_g['generator']) - len(unexpected_keys)}")    

    return generator

def load_t2a_vae(args, logger, device):
    vae_type = getattr(args, 'vae_type', 'bigvae')
    if vae_type == "bigvae":
        # BigVAE 初始化
        bigvae_model_path = getattr(args, 'dac_bigvae_model_path', None)
        if bigvae_model_path is None:
            raise ValueError("`dac_bigvae_model_path` must be provided for BigVAE extraction.")
        
        vae_model = load_dac_bigvae_ckpt(bigvae_model_path, device)
        vae_model.eval().requires_grad_(False)
                        
        if hasattr(torch, 'compile'):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info("Compiling VAE model")
            vae_model = torch.compile(vae_model)
        else:
            vae_model = vae_model
    elif vae_type == 'waveflowvae':
        from hymm.models.waveflowvae.bigvgan_flow_vae_export import init_vae_stat
        vae_path = getattr(args, 'vae_model_path', None)
        vae_stas = getattr(args, 'vae_model_stas', None)
        assert vae_path is not None and vae_stas is not None
        vae_model = init_vae_stat(vae_path, vae_stas)
        if hasattr(torch, 'compile'):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info("Compiling VAE model")
            vae_model = torch.compile(vae_model)
        else:
            vae_model = vae_model

    else:
        # 默认 DAC VAE 初始化
        from hymm.models.audio_encoders.dac.model.dac import DAC
        dac_ckp = getattr(args, 'dac_ckp', None)
        if dac_ckp is None:
            raise ValueError("`dac_ckp` must be provided for online VAE extraction.")
        
        vae_model = DAC.load(dac_ckp).to(device).eval().requires_grad_(False)
                
        if hasattr(torch, 'compile'):
            vae_model = torch.compile(vae_model)
        else:
            vae_model = vae_model

    vae_model.eval().requires_grad_(False)
    vae_model.to(device=device)

    return vae_model

def get_no_split_modules(transformer):
    from hymm.models.modules.Refrain import RefrainTransformer
    from hymm.models.modules.Aries import HYVideoAriesDiffusionTransformer
    if isinstance(transformer, RefrainTransformer):
        from hymm.models.modules.Refrain import MMDoubleStreamBlock, MMSingleStreamBlock
        return (MMDoubleStreamBlock, MMSingleStreamBlock)
    if isinstance(transformer, HYVideoAriesDiffusionTransformer):
        from hymm.models.modules.Aries import MMDoubleStreamBlock, MMSingleStreamBlock
        return (MMDoubleStreamBlock, MMSingleStreamBlock)
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")

def create_byt5(args):
    align_special_token = True
    direction_special_token = True
    byt5_max_length = args.byt5_max_length

    byt5_config = dict(
        byt5_name=args.byT5_google_path,
        special_token=True,
        color_special_token=True,
        font_special_token=True,
        color_ann_path=args.multilingual_prompt_format_color_path,
        font_ann_path=args.multilingual_prompt_format_font_path,
        multilingual=True,
    )

    huggingface_cache_dir = None
    byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(
        **byt5_config,
        huggingface_cache_dir=huggingface_cache_dir,
    )

    if args.byT5_ckpt_path is not None:
        byt5_state_dict = torch.load(args.byT5_ckpt_path, map_location="cpu")
        if 'state_dict' in byt5_state_dict:
            sd = byt5_state_dict["state_dict"]
            newsd = {}
            for k, v in sd.items():
                if k.startswith('module.text_tower.encoder.'):
                    newsd[k[len('module.text_tower.encoder.'):]] = v
            byt5_state_dict = newsd

        byt5_model_para = byt5_state_dict
        byt5_model.load_state_dict(byt5_model_para)

    return byt5_tokenizer, byt5_model, byt5_max_length

def load_glyph_byT5_v2(args, device):

    byt5_tokenizer, byt5_model, byt5_max_length = create_byt5(args)
    byt5_model = byt5_model.to(device=device)
    byt5_model.requires_grad_(False)

    byt5_kwarg = {"byt5_tokenizer": byt5_tokenizer,
                  "byt5_model": byt5_model,
                  "byt5_max_length": byt5_max_length}

    return byt5_kwarg

