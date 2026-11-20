from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch

from hymm.utils.prompt_cleaning import strip_identity_tags_nested
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, LlavaForConditionalGeneration, \
    LlamaTokenizerFast, CLIPImageProcessor, Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput

from ...constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
from ...utils.torch_utils import PRECISION_TO_TYPE


def use_default(value, default):
    return value if value is not None else default


def load_text_encoder(
        text_encoder_type,
        text_encoder_precision=None,
        text_encoder_path=None,
        logger=None,
        device=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llava-llama-3-8b-all":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
    elif text_encoder_type == "qwen-3vl-8b":
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(text_encoder_path, 
                                                 attn_implementation="flash_attention_2",
                                                 dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    elif text_encoder_type in TEXT_ENCODER_PATH:  # TODO(contributor, Distinguish different text encoder types)
        text_encoder = AutoModel.from_pretrained(text_encoder_path,
                                                 low_cpu_mem_usage=True)

        if hasattr(text_encoder, 'language_model'):
            text_encoder = text_encoder.language_model
        text_encoder.final_layer_norm = text_encoder.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None:
        text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
        tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    processor = None
    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == 'llava-llama-3-8b-all':
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path, padding_side=padding_side)
        processor = CLIPImageProcessor.from_pretrained(tokenizer_path)
    elif tokenizer_type in TOKENIZER_PATH:  # TODO(contributor, Distinguish different tokenizer types)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None
    image_features: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        uncond_token: Optional[str] = None,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                    use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger
        self.uncond_token = uncond_token
        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                    isinstance(self.prompt_template, dict)
                    and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                        isinstance(self.prompt_template_video, dict)
                        and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or \
                "glm" in text_encoder_type or \
                "llava" in text_encoder_type or \
                "qwen" in text_encoder_type or \
                "gemma" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

        # pre-calculate crop_start for image and video
        if self.use_template and self.prompt_template is not None:
            self.text2tokens("a photo of a cat", data_type="image")
            if self.logger is not None:
                self.logger.info(f"crop_start for image: {self.prompt_template['crop_start']}")
        if self.use_video_template and self.prompt_template_video is not None:
            self.text2tokens("a photo of a cat", data_type="video")
            if self.logger is not None:
                self.logger.info(f"crop_start for video: {self.prompt_template_video['crop_start']}")

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        elif isinstance(template, list):
            # For JSON list template format (chat conversation)
            # Create a deep copy to avoid modifying the original template
            template_copy = deepcopy(template)
            for item in template_copy:
                if isinstance(item, dict) and "content" in item:
                    # Replace placeholder with text in the content field
                    item["content"] = item["content"].format(text if text else (" " if prevent_empty_text else ""))
            return template_copy
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def calculate_crop_start(self, tokenized_input):
        """
        Automatically calculate the crop_start position based on identifying user tokens.

        Args:
            tokenized_input: The output from the tokenizer containing input_ids

        Returns:
            int: The position where the actual prompt content begins (after user markers)
        """
        input_ids = tokenized_input["input_ids"][0].tolist()  # Get the first example's tokens

        # Common user markers in different LLM tokenizers
        user_markers = {
            "llava": "<|start_header_id|>user<|end_header_id|>\n\n",  # LLaVA models
            "qwen": "<|im_start|>user\n",  # Qwen models
        }

        # Get relevant marker based on model type
        marker = None
        for model_type, token_marker in user_markers.items():
            if model_type in self.text_encoder_type.lower():
                marker = token_marker
                break

        if marker is None:
            # Default fallback: return 0 (no cropping) if we can't identify the model type
            return 0

        # Tokenize just the marker to get its token IDs
        marker_tokens = self.tokenizer(marker, add_special_tokens=False)["input_ids"]

        # Find the end position of the marker in the input sequence
        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i:i + len(marker_tokens)] == marker_tokens:
                # Return the position after the marker
                # print(f"crop_start: {i + len(marker_tokens)}, {self.tokenizer.decode(tokenized_input["input_ids"][0][i:i+len(marker_tokens)+10])}") # check crop_start
                return i + len(marker_tokens)

        # If marker not found, try to find based on special tokens
        if hasattr(self.tokenizer, 'special_tokens_map'):
            # Check for user token or any other special token that might indicate user input start
            for token_name, token_value in self.tokenizer.special_tokens_map.items():
                if 'user' in token_name.lower():
                    user_token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                    if user_token_id in input_ids:
                        return input_ids.index(user_token_id) + 1

        # Default fallback: return 0 (no cropping)
        return 0

    def safe_tokenize(self, text, return_length=False, return_overflowing_tokens=False, return_attention_mask=True, **kwargs):
        try:
            return self.tokenizer(text, 
                                return_length=return_length, 
                                return_overflowing_tokens=return_overflowing_tokens, 
                                return_attention_mask=return_attention_mask, 
                                **kwargs)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            if isinstance(text, (list,tuple)):
                text = [t.encode('utf-8', errors='ignore').decode() for t in text]
            elif isinstance(text, str):
                text = text.encode('utf-8', errors='ignore').decode()
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
            print(f"fixed text: {text}")
            return self.tokenizer(text, 
                                return_length=return_length, 
                                return_overflowing_tokens=return_overflowing_tokens, 
                                return_attention_mask=return_attention_mask, 
                                **kwargs)

    def text2tokens(self, text, data_type="image", max_length=300):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        # Final safety boundary shared by training and inference. Metadata tags
        # must never be embedded even if an upstream dataset/prompt path skips
        # caption augmentation.
        text = strip_identity_tags_nested(text)

        crop_start = 0
        if self.use_template or self.use_video_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
                crop_start = self.prompt_template_video.get("crop_start", -1)
            elif data_type == "audio":
                prompt_template = "{}"
                crop_start = 0
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Second pass: tokenize with the proper max_length using the found crop_start
            kwargs = dict(
                truncation=True,
                max_length=max_length + (crop_start if crop_start > 0 else 0),
                padding="max_length",
                return_tensors="pt",
            )
            if isinstance(text, (list, tuple)):
                if self.uncond_token == '<|im_end|>':
                    text = [self.uncond_token if t == '' else t for t in text]
                    text = [
                        self.apply_text_to_template(one_text, prompt_template)
                        for one_text in text
                    ]
                    tokenized_output = self.safe_tokenize(text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **kwargs,
                    )
                else:
                    text = [
                            self.apply_text_to_template(one_text, prompt_template)
                            for one_text in text
                        ]
                    tokenized_output = self.safe_tokenize(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **kwargs,
                    )
            elif isinstance(text, str): # for training
                if self.uncond_token == '<|im_end|>':
                    text = self.uncond_token if text == '' else text
                    text = self.apply_text_to_template(text, prompt_template)
                    # try:
                    tokenized_output = self.safe_tokenize(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **kwargs,
                    )
                else:
                    text = self.apply_text_to_template(text, prompt_template)
                    tokenized_output = self.safe_tokenize(
                        text,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_attention_mask=True,
                        **kwargs,
                    )
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        return tokenized_output

    def encode(
            self,
            batch_encoding,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=None,
            hidden_state_skip_layer=None,
            return_texts=False,
            data_type="image",
            device=None,
            semantic_images=None,
            is_uncond=False
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)

        if self.text_encoder_type != "llava-llama-3-8b-all":
            attention_mask = (
                batch_encoding["attention_mask"].to(device) if use_attention_mask else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                                     or hidden_state_skip_layer is not None,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", 0)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", 0)
                elif data_type == "audio":
                    crop_start = 0
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = (
                        attention_mask[:, crop_start:] if use_attention_mask else None
                    )

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state, attention_mask, outputs.hidden_states
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)
        else:
            if semantic_images is not None:
                image_outputs = self.processor(semantic_images, return_tensors='pt')["pixel_values"].to(
                    self.model.device)
            else:
                image_outputs = None

            attention_mask = (
                batch_encoding["attention_mask"].to(device) if use_attention_mask else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                                     or hidden_state_skip_layer is not None,
                pixel_values=image_outputs
            )
            image_features = outputs.image_hidden_states
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == 'image':
                    crop_start = self.prompt_template.get("crop_start", -1)
                    crop_end = self.prompt_template_video.get('assistant_emb_start', -1)
                elif data_type == 'video':
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = crop_start - 1 + self.prompt_template_video.get("image_emb_len", 576)
                    image_crop_start = self.prompt_template_video.get("image_emb_start", 5)
                    image_crop_end = self.prompt_template_video.get('image_emb_end', 581)
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"] == self.prompt_template_video.get('double_return_token_id', 271))
                    last_double_return_token_indices = last_double_return_token_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1)[:, -1]
                    batch_indices = batch_indices.reshape(batch_encoding["input_ids"].shape[0], -1)[:, -1]
                    assistant_crop_start = last_double_return_token_indices - 1 + self.prompt_template_video.get(
                        "image_emb_len", 576) - 4
                    assistant_crop_end = last_double_return_token_indices - 1 + self.prompt_template_video.get(
                        "image_emb_len", 576)

                    attention_mask_assistant_crop_start = last_double_return_token_indices - 4
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                text_last_hidden_state = []
                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(torch.cat(
                        [last_hidden_state[i, text_crop_start:assistant_crop_start[i].item()],
                         last_hidden_state[i, assistant_crop_end[i].item():]]))
                    text_attention_mask.append(torch.cat(
                        [attention_mask[i, crop_start:attention_mask_assistant_crop_start[i].item()], attention_mask[i,
                                                                                                      attention_mask_assistant_crop_end[
                                                                                                          i].item():]]) if use_attention_mask else None)
                    image_last_hidden_state.append(last_hidden_state[i, image_crop_start:image_crop_end])
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0]).to(last_hidden_state.device).to(
                            attention_mask.dtype) if use_attention_mask else None)

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                # if is_uncond:
                #     image_last_hidden_state = torch.zeros_like(image_last_hidden_state).to(image_last_hidden_state.device)

                assert text_last_hidden_state.shape[0] == text_attention_mask.shape[0] and \
                       image_last_hidden_state.shape[0] == image_attention_mask.shape[0]

                last_hidden_state = torch.cat([image_last_hidden_state, text_last_hidden_state], dim=1)
                attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)

            if output_hidden_states:
                return TextEncoderModelOutput(last_hidden_state, attention_mask, image_features=image_features,
                                              hidden_states_list=outputs.hidden_states)
            return TextEncoderModelOutput(last_hidden_state, attention_mask, image_features=image_features)

    def forward(
            self,
            text,
            use_attention_mask=None,
            output_hidden_states=False,
            do_sample=False,
            hidden_state_skip_layer=None,
            return_texts=False,
    ):
        batch_encoding = self.text2tokens(text, max_length=self.max_length)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
