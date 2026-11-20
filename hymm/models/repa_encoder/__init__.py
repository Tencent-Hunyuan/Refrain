from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.utils import ModelOutput
import numpy as np
from transformers import AutoImageProcessor, Dinov2Model, AutoProcessor, Qwen2_5_VLForConditionalGeneration, DINOv3ViTModel, Qwen3VLForConditionalGeneration
from PIL import Image

from ...constants import REPA_ENCODER_PATH
from ...utils.torch_utils import PRECISION_TO_TYPE

QWEN_REPA_ENCODERS = ("qwen-2.5vl-7b", "qwen-3vl-8b")

def load_repa_encoder(
    repa_encoder_type,
    repa_encoder_precision=None,
    repa_encoder_path=None,
    logger=None,
    device=None,
):

    if repa_encoder_type is not None:
        repa_encoder_path = REPA_ENCODER_PATH[repa_encoder_type]

    if repa_encoder_type == "siglip":
        repa_encoder = SiglipVisionModel.from_pretrained(
            repa_encoder_path, 
            subfolder='image_encoder'
        )
    elif "dinov2" in repa_encoder_type:
        repa_encoder = Dinov2Model.from_pretrained(repa_encoder_path)
    elif "dinov3" in repa_encoder_type:
        repa_encoder = DINOv3ViTModel.from_pretrained(repa_encoder_path, local_files_only=True,)
    elif repa_encoder_type in QWEN_REPA_ENCODERS:
        load_kwargs = {"trust_remote_code": True}
        preferred_dtype = PRECISION_TO_TYPE.get(repa_encoder_precision)
        if preferred_dtype is None:
            preferred_dtype = torch.bfloat16
        load_kwargs["torch_dtype"] = preferred_dtype
        if "qwen-3vl" in repa_encoder_type:
            print("Loading Qwen-3-VL vision encoder...")
            qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                repa_encoder_path,
                **load_kwargs,
            )
        elif "qwen-2.5vl" in repa_encoder_type:
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repa_encoder_path,
                **load_kwargs,
            )
        qwen_model.eval()
        if hasattr(qwen_model, "visual"):
            repa_encoder = qwen_model.visual
        elif hasattr(qwen_model, "vision_model"):
            repa_encoder = qwen_model.vision_model
        else:
            raise ValueError("Unsupported Qwen-VL vision structure: no visual encoder found")
    else:
        raise ValueError(f"Unsupported repa encoder type: {repa_encoder_type}")

    # Keep the REPA encoder in its original precision (e.g. fp32) for
    # higher-quality feature extraction; dtype alignment happens at loss time.
    repa_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Repa encoder to dtype: {repa_encoder.dtype}")

    if device is not None:
        repa_encoder = repa_encoder.to(device)

    return repa_encoder, repa_encoder_path


def load_image_processor(
    processor_type, 
    processor_path=None, 
    logger=None
):
    if processor_path is None:
        processor_path = REPA_ENCODER_PATH[processor_type]
    if logger is not None:
        logger.info(f"Loading image processor ({processor_type}) from: {processor_path}")

    if processor_type in QWEN_REPA_ENCODERS:
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")

    return processor, processor_path

@dataclass
class RepaEncoderModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class RepaEncoder(nn.Module):
    def __init__(
        self,
        repa_encoder_type: str,
        repa_encoder_precision: Optional[str] = None,
        repa_encoder_path: Optional[str] = None,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.repa_encoder_type = repa_encoder_type
        self.precision = repa_encoder_precision
        self.model_path = repa_encoder_path
        self.logger = logger

        self.model, _ = load_repa_encoder(
            repa_encoder_type=self.repa_encoder_type,
            repa_encoder_precision=self.precision,
            repa_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.processor = None
        self.processor_path = None
        if "dino" not in self.repa_encoder_type:
            self.processor, self.processor_path = load_image_processor(
                processor_type=repa_encoder_type,
                processor_path=repa_encoder_path,
                logger=self.logger,
            )
        if "dino" in self.repa_encoder_type:
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _as_pil_images(self, imgs):
        if isinstance(imgs, list) and all(isinstance(item, dict) and "image" in item for item in imgs):
            extracted = [item["image"] for item in imgs]
            normalized = []
            for img in extracted:
                if isinstance(img, Image.Image):
                    normalized.append(img)
                elif isinstance(img, np.ndarray):
                    normalized.extend(self._as_pil_images(img))
                else:
                    raise ValueError(f"Unsupported nested image entry: {type(img)}")
            return normalized
        if isinstance(imgs, list):
            pil_images = []
            for item in imgs:
                if isinstance(item, Image.Image):
                    pil_images.append(item)
                elif isinstance(item, dict) and isinstance(item.get("image"), Image.Image):
                    pil_images.append(item["image"])
                elif isinstance(item, np.ndarray):
                    pil_images.extend(self._as_pil_images(item))
                else:
                    raise ValueError(f"Unsupported image entry for Qwen encoder: {type(item)}")
            return pil_images
        if isinstance(imgs, np.ndarray):
            arr = imgs
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)
            if arr.ndim != 4:
                raise ValueError(f"Expected image tensor with 3 or 4 dims, got {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            return [Image.fromarray(arr_i) for arr_i in arr]
        raise ValueError(f"Unsupported image format for Qwen encoder: {type(imgs)}")

    def _process_features(self, last_hidden_state, b, t):
        is_video = (t is not None and t > 1) 
        if last_hidden_state.dim() == 2:
             last_hidden_state = last_hidden_state.unsqueeze(1)
        b_total, n_tokens, d_dim = last_hidden_state.shape

        if is_video:
            last_hidden_state = last_hidden_state.view(b, t, n_tokens, d_dim)
            feat_first = last_hidden_state[:, 0:1, :, :]
            padding = torch.zeros(
                b, 1, n_tokens, 3 * d_dim, 
                device=last_hidden_state.device, 
                dtype=last_hidden_state.dtype
            )
            feat_first_out = torch.cat([feat_first, padding], dim=-1)

            # Time unshuffle
            if t > 1:
                feat_rest = last_hidden_state[:, 1:, :, :]  # [B, T-1, N, D]
                t_rest = t - 1
                if t_rest % 4 == 0:
                    # Reshape: [B, T_rest/4, 4, N, D]
                    feat_rest = feat_rest.view(b, t_rest // 4, 4, n_tokens, d_dim)
                    # Permute: [B, T_rest/4, N, 4, D] (Time folded into D)
                    feat_rest = feat_rest.permute(0, 1, 3, 2, 4)
                    # Flatten: [B, T_rest/4, N, 4D]
                    feat_rest_out = feat_rest.reshape(b, t_rest // 4, n_tokens, 4 * d_dim)
                    
                    # Concat Time
                    last_hidden_state = torch.cat([feat_first_out, feat_rest_out], dim=1)
                else:
                    if self.logger:
                        self.logger.warning(f"Video frames {t} is not 4n+1, skipping time unshuffle logic.")
                    last_hidden_state = feat_first_out
            else:
                last_hidden_state = feat_first_out
            
            # Merge Time and Space: [B, T_new * N, 4D]
            last_hidden_state = last_hidden_state.flatten(1, 2)

        else:
            # Image Logic, used when image input during video training
            # Pad to match 4*D dimension
            padding = torch.zeros(
                b_total, n_tokens, 3 * d_dim, 
                device=last_hidden_state.device, 
                dtype=last_hidden_state.dtype
            )
            last_hidden_state = torch.cat([last_hidden_state, padding], dim=-1)
        
        return last_hidden_state

    def encode_images(self, images, texts=None, data_type="image"):
        # if use video data, encode video frames, otherwise encode images
        last_hidden_state = None
        pooler_output = None
        hidden_states = None

        is_video = False
        b, c, t, h_in, w_in = None, None, None, None, None

        if isinstance(images, torch.Tensor) and images.ndim == 5:
            is_video = True
            b, c, t, h_in, w_in = images.shape
            images = images.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h_in, w_in)
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            b = images.shape[0] # Batch size for images

        if self.repa_encoder_type in QWEN_REPA_ENCODERS:
            # Qwen Preprocessing
            _, _, h, w = images.shape
            # Qwen2.5-VL likes 28x multiples, Qwen3-VL likes 32x multiples
            if "qwen-2.5vl" in self.repa_encoder_type:
                patch_size = 28
            elif "qwen-3vl" in self.repa_encoder_type:
                patch_size = 32
            else:
                raise ValueError(f"Unsupported repa encoder type: {self.repa_encoder_type}")
            new_h = round(h / patch_size) * patch_size
            new_w = round(w / patch_size) * patch_size
            if new_h != h or new_w != w:
                images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
            
            # Convert to PIL for Processor
            tensor_pil = images.permute(0, 2, 3, 1).contiguous()
            imgs = tensor_pil.cpu().numpy()
            pil_images = self._as_pil_images(imgs)
            
            # Text Prompt Handling
            text_input_qwen = [""] * len(pil_images)
            if texts is not None and len(pil_images) > 0:
                text_list = list(texts) if isinstance(texts, (list, tuple)) else [texts]
                
                if is_video and len(text_list) == b:
                    # Expand texts for video frames
                    expanded_texts = []
                    for txt in text_list:
                        expanded_texts.extend([str(txt) if txt is not None else ""] * t)
                    text_input_qwen = expanded_texts
                elif len(text_list) == len(pil_images):
                    text_input_qwen = [str(t) if t is not None else "" for t in text_list]
                elif len(text_list) == 1:
                    text_input_qwen = [str(text_list[0]) if text_list[0] is not None else ""] * len(pil_images)
            
            processor_inputs = self.processor(
                text=text_input_qwen,
                images=pil_images,
                videos=None,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            processor_inputs = processor_inputs.to(device=self.model.device)
            pixel_values = processor_inputs.pixel_values.to(dtype=self.model.dtype)
            image_grid_thw = getattr(processor_inputs, "image_grid_thw", None)

            model_outputs = self.model(
                hidden_states=pixel_values,
                grid_thw=image_grid_thw,
            )

            # Extract Hidden States
            if isinstance(model_outputs, torch.Tensor):
                visual_hidden = model_outputs
            elif isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 0:
                visual_hidden = model_outputs[0]
            else:
                visual_hidden = getattr(model_outputs, "last_hidden_state", None)
                if visual_hidden is None and hasattr(model_outputs, "image_embeds"):
                     visual_hidden = getattr(model_outputs, "image_embeds")
            
            if visual_hidden is None:
                raise ValueError("Qwen-VL vision encoder did not return hidden states.")

            # Fixed length in the batch
            total_tokens, d_dim = visual_hidden.shape
            batch_size_total = len(pil_images)
            
            if batch_size_total == 0:
                 raise ValueError("Batch size is 0")
            if total_tokens % batch_size_total != 0:
                raise ValueError(f"Total tokens {total_tokens} not divisible by batch size {batch_size_total}.")

            n_tokens = total_tokens // batch_size_total
            last_hidden_state = visual_hidden.view(batch_size_total, n_tokens, d_dim)

        elif "dino" in self.repa_encoder_type:
            # DINOv2 Preprocessing
            h, w = images.shape[-2], images.shape[-1]
            # DINOv2 likes 14x multiples
            if "dinov2" in self.repa_encoder_type:
                new_h = round(h / 14) * 14
                new_w = round(w / 14) * 14
                
                if new_h != h or new_w != w:
                    images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
            
            # use registered buffer
            images = images.to(device=self.model.device, dtype=self.model.dtype)
            if self.imagenet_mean.device != images.device or self.imagenet_mean.dtype != images.dtype:
                self.imagenet_mean = self.imagenet_mean.to(device=images.device, dtype=images.dtype)
                self.imagenet_std = self.imagenet_std.to(device=images.device, dtype=images.dtype)
            images = (images - self.imagenet_mean) / self.imagenet_std

            
            outputs = self.model(pixel_values=images)
            
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_state = outputs.last_hidden_state
            elif hasattr(outputs, "vision_model_output"):
                last_hidden_state = outputs.vision_model_output.last_hidden_state
            else:
                last_hidden_state = outputs[0]
            
            if last_hidden_state is not None:
                # Remove CLS token
                if "dinov2" in self.repa_encoder_type:
                    last_hidden_state = last_hidden_state[:, 1:, :]
                elif "dinov3" in self.repa_encoder_type:
                    last_hidden_state = last_hidden_state[:, 5:, :]

        else:
            raise ValueError(f"Unsupported repa encoder type: {self.repa_encoder_type}")

        if last_hidden_state is not None and data_type=='image_video':
            current_t = t if is_video else None
            current_b = b if is_video else last_hidden_state.shape[0]
            
            last_hidden_state = self._process_features(
                last_hidden_state, 
                b=current_b, 
                t=current_t,
            )

        return RepaEncoderModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states
        )
    def encode_latents(self, latents, vae, reorg_token=False):
        """
        Encode latents by first converting to images, then encoding.
        This is the main function that replaces sigclip_vision_encode.
        
        Args:
            latents: Input latent tensors
            vae: VAE model for decoding latents to images
            
        Returns:
            Encoded image features
        """
        # Convert latents to images
        images = self.encode_latents_to_images(latents, vae, reorg_token)
        
        # Encode images
        outputs = self.encode_images(images)
        
        return outputs.last_hidden_state

    def forward(self, images):
        """
        Forward pass for direct image encoding.
        
        Args:
            images: Input images
            
        Returns:
            RepaEncoderModelOutput with encoded features
        """
        return self.encode_images(images) 
        
    def encode_latents_to_images(self, latents, vae, reorg_token=False):
        """
        Convert latents to images using VAE decoder.
        
        Args:
            latents: Input latents tensor
            vae: VAE model for decoding
            reorg_token: Whether to reorg the token
        Returns:
            images: Decoded images as numpy array
        """
        # Handle both 4D and 5D latents (for video, take first frame)
        first_image_latents = latents[:, :, 0, ...] if len(latents.shape) == 5 else latents
        first_image_latents = 1 / vae.config.scaling_factor * first_image_latents
        first_image = vae.decode(first_image_latents.unsqueeze(2).to(vae.dtype), return_dict=False)[0].cpu()
        first_image = first_image[:, :, 0, :, :]
        first_image = (first_image / 2 + 0.5).clamp(0, 1)
        first_image = (first_image * 255.0).clamp(0, 255.0)
        first_image = first_image.to(torch.uint8).numpy()
        first_image = first_image.transpose(0, 2, 3, 1)

        assert isinstance(first_image, np.ndarray)
        assert first_image.ndim == 4 and first_image.shape[3] == 3
        assert first_image.dtype == np.uint8

        return first_image