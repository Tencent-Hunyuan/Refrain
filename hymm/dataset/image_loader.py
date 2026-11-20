import random
import time
import json
from collections import defaultdict
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchvision.transforms as T
from PIL import Image
from hymm.dataset.arrow_index import (
    ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2, arrow_mapper,
)
from torch.utils.data import Dataset

from hymm.utils.helpers import to_2tuple
from hymm.dataset.format_prompt import MultilingualPromptFormat
from hymm.dataset.caps import load_caption_processor
import re
from Crypto.Cipher import AES
from hashlib import md5
import io
from PIL import Image
import os


def use_default(value, default):
    return value if value is not None else default


def evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    """与 openssl enc 的 EVP_BytesToKey 算法兼容（MD5 版）"""
    dtot = b''
    prev = b''
    while len(dtot) < (key_len + iv_len):
        prev = md5(prev + password + salt).digest()
        dtot += prev
    key = dtot[:key_len]
    iv = dtot[key_len:key_len + iv_len]
    return key, iv


def openssl_decrypt(enc_file: str, password: str = None):
    if password is None:
        password = os.environ["REFRAIN_DATA_PASSWORD"]
    with open(enc_file, 'rb') as f:
        enc_data = f.read()

    # OpenSSL 格式：前8字节是 "Salted__"，后8字节是 salt
    assert enc_data[:8] == b'Salted__'
    salt = enc_data[8:16]
    ciphertext = enc_data[16:]

    key_len, iv_len = 16, 16  # AES-128-CBC
    key, iv = evp_bytes_to_key(password.encode('utf-8'), salt, key_len, iv_len)

    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)

    # 去除 PKCS7 填充
    pad_len = decrypted[-1]
    decrypted_data = decrypted[:-pad_len]

    # with open(dec_file, 'wb') as f:
    #     f.write(decrypted_data)

    pil_image = Image.open(io.BytesIO(decrypted_data)).convert("RGB")
    return pil_image


def open_image_retry(image_path, retry=3):
    for i in range(retry):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                ret = img.copy()  # VERY
        except OSError:
            # incase cos rate exceed
            print(f"fail load {image_path}")
            time.sleep(random.random() + i)
            continue
        break
    return ret


class TextImageArrowStream(Dataset):
    def __init__(self,
                 args,
                 image_size=256,
                 crop_type='random',
                 uncond_p=0.0,
                 text_len=256,
                 text_encoder=None,
                 text_encoder_2=None,
                 index_file=None,
                 multireso=False,
                 index_kwargs=None,
                 byt5_tokenizer=None,
                 byt5_max_length=None,
                 byt5_uncond_p=None,
                 dataset_type=None,
                 debug=False,
                 logger=None,
                 ):
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.args = args
        self.image_size = to_2tuple(image_size)
        self.crop_type = crop_type
        self.uncond_p = uncond_p
        self.text_len = text_len
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.index_file = index_file
        self.multireso = multireso
        self.debug = debug
        self.byt5_tokenizer = byt5_tokenizer
        self.byt5_max_length = byt5_max_length
        self.byt5_uncond_p = byt5_uncond_p

        if index_kwargs is None:
            index_kwargs = {}

        self.image_caption_rate = index_kwargs.get('image_caption_rate', 0.0)

        self.image_text_col = index_kwargs.get('image_text_col', None)
        self.image_text_col_2 = index_kwargs.get("image_text_col_2", None)
        if self.image_caption_rate < 1.0 and self.image_text_col is None:
            raise ValueError("Missing `image_text_col` in `index_kwargs`")
        self.image_text_arrow_suffix = use_default(index_kwargs['image_text_arrow_suffix'], '_trans')

        self.image_caption_col = index_kwargs.get('image_caption_col', None)
        self.image_caption_col_2 = index_kwargs.get("image_caption_col_2", None)
        if self.image_caption_rate > 0.0 and self.image_caption_col is None:
            raise ValueError("Missing `image_caption_col` in `index_kwargs`")
        self.image_caption_arrow_suffix = use_default(index_kwargs['image_caption_arrow_suffix'], '')
        if self.image_caption_col_2 is not None and '@' in self.image_caption_col_2:
            self.image_caption_col_2, self.image_caption_arrow_suffix_2 = self.image_caption_col_2.split('@')

        self.caption_sample_ratio = index_kwargs.get('caption_sample_ratio', None)
        self.ocr_only_long_caption = index_kwargs.get('ocr_only_long_caption', False)
        self.text_remove_rate = index_kwargs.get('text_remove_rate', 0.0)
        self.num_replace_rate = index_kwargs.get('num_replace_rate', 0.0)
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            caption_processor = index_kwargs.get('caption_processor', None)
            # self.caption_sample_ratio = self.caption_sample_ratio.replace('_', ' ') # caption_process_v4 & caption_process_v3 不需要替换
            if caption_processor == "caption_process":
                kwargs = {
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'text_remove_rate': self.text_remove_rate,
                }
            elif caption_processor in ["caption_process_v4", "caption_process_v3"]:
                kwargs = {
                    'ocr_only_long_caption': self.ocr_only_long_caption,
                    'num_replace_rate': self.num_replace_rate,
                }
            else:
                kwargs = {}
            caption_sample_ratio = json.loads(self.caption_sample_ratio)
            self.caption_aug = load_caption_processor(name=caption_processor,
                                                      caption_sample_ratio=caption_sample_ratio,
                                                      logger=self.logger,
                                                      kwargs=kwargs)
        else:
            self.use_structural_caption = False

        # Prepare index manager
        index_load_kwargs = dict(batch_size=index_kwargs.get('batch_size', 1),
                                 world_size=index_kwargs.get('world_size', 1),
                                 sample_strategy=index_kwargs.get('index_strategy', 'uniform'),
                                 probability=index_kwargs.get('index_probability', None),
                                 ceph_base=json.loads(index_kwargs.get('ceph_base', '{}')),
                                 ceph_base_inv=json.loads(index_kwargs.get('ceph_base_inv', '{}')))

        shadow_file_fn = {}  # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        if self.image_caption_rate < 1.0:
            shadow_file_fn.update({'text': partial(arrow_mapper, suffix=self.image_text_arrow_suffix)})
        if self.image_caption_rate > 0.0:
            shadow_file_fn.update({'caption': partial(arrow_mapper, suffix=self.image_caption_arrow_suffix)})

        if self.args.image_caption_processor == "caption_process_v4":
            self.image_update_caption_version_zh = [ "caption_v4_ocr_zh", "caption_v4_zh"]
            self.image_update_caption_version_en = [ "caption_v4_ocr_en", "caption_v4_en"]
            shadow_file_fn.update({'ocr_similarity': partial(arrow_mapper, suffix="_ocr_similarity_shadow")})


        elif self.args.image_caption_processor == "caption_process_v3":
            self.image_update_caption_version_zh = ["caption_v3_2_ip_zh", "caption_v3_2_ocr_zh", "caption_v3_2_zh",
                                                "caption_v3_1_zh", "caption_v3_zh"]
            self.image_update_caption_version_en = ["caption_v3_2_ip_en", "caption_v3_2_ocr_en", "caption_v3_2_en",
                                                "caption_v3_1_en", "caption_v3_en"]
        else:
            raise ValueError(f"image_caption_processor: {self.args.image_caption_processor} is not supported")

        index_load_kwargs['shadow_file_fn'] = shadow_file_fn
        self.index_manager = self.load_index(**index_load_kwargs)
        if not self.args.multireso:
            self.index_manager.set_resolution_buckets(base_size=self.args.image_size[0], step=self.args.multireso_step, align=16)

        self.logger.info(f"Using {self.index_manager}")
        self.logger.info(f"    Text col: {self.image_text_col} (suffix={self.image_text_arrow_suffix})")
        self.logger.info(f"    Caption col: {self.image_caption_col} (suffix={self.image_caption_arrow_suffix})")
        self.logger.info(f"    Caption rate: {self.image_caption_rate}")
        self.logger.info(f"    Caption sample ratio: {self.caption_sample_ratio}")

        self.pil_image_to_tensor = T.Compose(
            [
                T.ToTensor(),  # convert to tensor and normalize to [0, 1]
                T.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.tensor_to_pil_image = T.Compose(
            [
                T.Normalize([-1], [2]),
                T.ToPILImage(),
            ]
        )

        self.logger.info(f"Image transform: {self.pil_image_to_tensor}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 100

        self.prompt_format = MultilingualPromptFormat(font_path=self.args.multilingual_prompt_format_font_path,
                                                      color_path=self.args.multilingual_prompt_format_color_path)

        image_caption_col_probs = json.loads(index_kwargs.get('image_caption_col_probs', '{}'))
        self.multi_caption_cols = []
        self.multi_caption_probs = []
        for col, prob in image_caption_col_probs.items():
            self.multi_caption_cols.append(getattr(self, col))  # map to image_caption_col
            self.multi_caption_probs.append(prob)
        self.image_cos_root = args.image_cos_root
        self.dataset_type = dataset_type

    def load_index(self,
                   batch_size,
                   world_size,
                   shadow_file_fn=None,
                   sample_strategy='uniform',
                   probability=None,
                   ceph_base=None,
                   ceph_base_inv=None,
                   ):
        multireso = self.multireso
        index_file = self.index_file
        if not isinstance(index_file, (list, tuple)):
            index_file = [index_file]

        self.logger.info(f"Loading dataset index: {index_file}")
        if multireso:
            if len(index_file) == 1:
                index_manager = MultiResolutionBucketIndexV2(
                    index_file[0], batch_size, world_size, shadow_file_fn=shadow_file_fn, ceph_base=ceph_base,
                    ceph_base_inv=ceph_base_inv,
                )
            else:
                index_manager = MultiMultiResolutionBucketIndexV2(
                    index_file, batch_size, world_size, shadow_file_fn=shadow_file_fn, ceph_base=ceph_base,
                    ceph_base_inv=ceph_base_inv,
                    sample_strategy=sample_strategy, probability=probability,
                )
        else:
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(
                    index_file[0], shadow_file_fn=shadow_file_fn,
                )
            else:
                index_manager = MultiIndexV2(
                    index_file, shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy, probability=probability)

        return index_manager

    def shuffle(self, seed, fast=False, **kwargs):
        # self.index_manager.shuffle(seed, fast=fast, **kwargs)
        pass # disable shuffle because data is already shuffled

    def replace_image_path_root(self, image_path):
        if self.image_cos_root is not None:
            image_path = self.image_cos_root + "/" + "/".join(image_path.split("/")[3:])
        return image_path

    def get_crypto_path_bucket(self, image_path):
        import urllib
        parsed_url = urllib.parse.urlparse(image_path)
        bucket = parsed_url.netloc.split('.')[0]
        return bucket

    def replace_crypto_image_path(self, image_path, url_cos):
        bucket = self.get_crypto_path_bucket(url_cos)
        image_path = self.image_cos_root + bucket + "/" + "/".join(image_path.split("/")[3:])
        image_path = image_path + '.enc'
        return image_path

    def get_raw_image(self, index, image_key="image"):
        image_path_original = ""
        image_id = ""
        try:
            if image_key == "image":
                try:
                    ret = self.index_manager.get_image(index, image_key)
                except KeyError:
                    if self.args.image_decrypt:
                        image_path = self.index_manager.get_attribute(index, "cache_image")
                        url_cos = self.index_manager.get_attribute(index, "url_cos")
                        image_path = self.replace_crypto_image_path(image_path, url_cos)
                        ret = openssl_decrypt(image_path)
                    else:
                        image_path = self.index_manager.get_attribute(index, "cache_image")
                        image_path = self.replace_image_path_root(image_path)
                        ret = open_image_retry(image_path, retry=1)
            else:
                # merge get image_path and image_id
                data = self.index_manager.get_data(index, ['md5', 'cache_image'], return_meta=False)
                image_path = data['cache_image']
                image_id = data['md5']
                image_path_original = image_path
                if self.args.image_decrypt:
                    url_cos = self.index_manager.get_attribute(index, "url_cos")
                    image_path = self.replace_crypto_image_path(image_path, url_cos)
                    ret = openssl_decrypt(image_path)
                else:
                    image_path = self.replace_image_path_root(image_path)
                    ret = open_image_retry(image_path, retry=1)
            image_flag = "normal"
        except Exception as e:
            # PIL.UnidentifiedImageError: cannot identify image file
            self.logger.error(f'{type(e)}: {e}')
            target_size = self.index_manager.get_target_size(index)  # (w, h)
            if target_size[0] > 0 and target_size[1] > 0:
                ret = Image.new("RGB", (target_size[1], target_size[0]), (128, 128, 128))
            else:
                ret = Image.new("RGB", (self.image_size[0], self.image_size[1]), (128, 128, 128))
            image_flag = "gray"
        return ret, image_flag, image_path_original, image_id

    def get_image_with_size(self, index):
        # Get raw image
        image, image_flag, image_path_original, image_id = self.get_raw_image(index, image_key='cache_image')
        origin_size = image.size  # (w_ori, h_ori)

        target_size = self.index_manager.get_target_size(index)  # (w_tgt, h_tgt)

        # Resize and crop
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type='random', resample=Image.Resampling.BICUBIC)

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            'origin_size': origin_size,
            'target_size': target_size,
            'crop_coords_xy': (crop_left, crop_top),
            'image_id': image_id,
            'image_path': image_path_original,
        }
        return image_tensor, kwargs, image_flag

    @staticmethod
    def get_text_tokens(text_encoder, description, text_len):
        text_inputs = text_encoder.text2tokens(description, data_type="image", max_length=text_len)
        text_ids = text_inputs["input_ids"].squeeze(0)
        text_mask = text_inputs["attention_mask"].squeeze(0)
        return text_ids, text_mask

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    def handle_exception_message(self, func, e, info=""):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f'{func.__name__} | {e.__class__.__name__}: {message} {info}')

    def try_ip_ocr_caption_v32(self, ind, lang):
        for idx, key in enumerate(self.image_update_caption_version_zh):
            try:
                text_zh = self.index_manager.get_attribute(ind, key)
                if text_zh is None:
                    continue

                text_en = self.index_manager.get_attribute(ind, self.image_update_caption_version_en[idx])
                if text_en is None:
                    continue

                if "caption_v4_ocr" in key:
                    precision = self.index_manager.get_attribute(ind, "ocr_similarity", shadow="ocr_similarity")
                    if precision < 0.1:
                        continue

                text = self.caption_aug.caption_aug(text_en, text_zh, lang)
                return text
            except Exception as e:
                import traceback
                self.logger.error(f'try_ip_ocr_caption_v32 error: {e}')
                self.logger.error(traceback.format_exc())
                continue
        raise ValueError(f'No valid caption found for index {ind}')

    def get_text(self, ind):
        text = ''
        try:
            if self.image_caption_rate > 0.0 and random.random() < self.image_caption_rate:
                if self.image_caption_col == "result":  # caption 分开存放
                    text = self.index_manager.get_attribute(ind, self.image_caption_col, shadow='caption')
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text)
                elif len(self.multi_caption_cols) > 0:  # 中英文caption
                    sel_col = random.choices(
                        self.multi_caption_cols, weights=self.multi_caption_probs, k=1
                    )[0]
                    if 'zh' in sel_col:
                        lang = 'zh'
                    elif 'en' in sel_col:
                        lang = 'en'
                    else:
                        raise ValueError(f"Invalid language: {sel_col}")
                    text = self.try_ip_ocr_caption_v32(ind, lang)
                # caption v2 video&image
                elif self.args.image_caption_processor == "caption_process" and self.image_caption_col in self.index_manager.get_columns(
                        ind):
                    text = self.index_manager.get_attribute(ind, self.image_caption_col)
                    if self.use_structural_caption:
                        text = self.caption_aug.caption_aug(text)
                # ocr caption only
                elif self.image_caption_col_2 is not None and self.image_caption_col_2 in self.index_manager.get_columns(
                        ind):
                    text = self.index_manager.get_attribute(ind, self.image_caption_col_2)
                else:
                    raise NotImplementedError("image caption type not supported")
                # print(f'#{torch.distributed.get_rank()}: text {text}, sel_col={sel_col}, lang={lang}, self.multi_caption_cols={self.multi_caption_cols}')
            else:
                if self.image_text_col in self.index_manager.get_columns(ind):
                    text = self.index_manager.get_attribute(ind, self.image_text_col)
                elif self.image_text_col_2 is not None and self.image_text_col_2 in self.index_manager.get_columns(ind):
                    text = self.index_manager.get_attribute(ind, self.image_text_col_2)
                else:
                    raise ValueError(f"No text column found for index {ind}")
        except Exception as e:
            import traceback
            error_info = traceback.format_exc()
            if self.args.raise_text_error:
                print(f'#{torch.distributed.get_rank()}: indices {ind}')
                raise e
            message = f'load indices {ind} text {text} error: {error_info}'
            self.handle_exception_message(self.get_text, e, message)
        text = str(text).strip()

        if text == '':
            text = 'Generate an image'

        # Remove meaningless characters
        text = text.replace('\\N', '').strip('，,')

        return text

    def __getitem__(self, ind):
        """
        Get image and text from a given index

        Args:
            ind (int): Index of the dataset

        Returns:
            image_tensor (torch.FloatTensor): Image tensor with shape (3, H, W)
            text_ids (torch.LongTensor): Tokenized text tensor with shape (text_len,)
            text_mask (torch.LongTensor): Attention mask tensor with shape (text_len,)
            kwargs (dict): Additional information
                origin_size (torch.LongTensor): Original size of the image (W, H)
                target_size (torch.LongTensor): Target size of the image (W, H)
                crop_coords_xy (torch.LongTensor): Crop coordinates (x, y)
                text (str): Original text
                index (torch.LongTensor): Index of the dataset
        """

        # Get text

        if random.random() < self.uncond_p:
            description = ""

            is_uncond = True
            orig_prompt = self.get_text(ind)
        else:
            description = self.get_text(ind)

            is_uncond = False
            orig_prompt = description

        image_tensor, kwargs, image_flag = self.get_image_with_size(ind)

        if self.args.glyph_byT5_v2:
            byt5_text_ids = torch.zeros((self.byt5_max_length), dtype=torch.int64)
            byt5_text_mask = torch.zeros((self.byt5_max_length), dtype=torch.int64)
            byt5_text_valid = torch.tensor(0, dtype=torch.int64)

            pattern = r'\"(.*?)\"|“(.*?)”'
            matches = re.findall(pattern, description)
            glyph_byT5_text_list = [match[0] or match[1] for match in matches]
            glyph_byT5_text_list = list(dict.fromkeys(glyph_byT5_text_list)) if len(
                glyph_byT5_text_list) > 1 else glyph_byT5_text_list

            if len(glyph_byT5_text_list) > 0:
                if random.random() < self.byt5_uncond_p:
                    glyph_byT5_text_list = ["" for _ in range(len(glyph_byT5_text_list))]

                text_prompt_style_list = [{'color': None, 'font-family': None} for _ in
                                          range(len(glyph_byT5_text_list))]
                glyph_byT5_text_formatted = self.prompt_format.format_prompt(glyph_byT5_text_list,
                                                                             text_prompt_style_list)

                byt5_text_ids, byt5_text_mask = self.get_byt5_text_tokens(self.byt5_tokenizer, self.byt5_max_length,
                                                                          glyph_byT5_text_formatted)
                byt5_text_ids = byt5_text_ids.squeeze(0)
                byt5_text_mask = byt5_text_mask.squeeze(0)
                byt5_text_valid = torch.tensor(1, dtype=torch.int64)

        # Get image
        if image_flag == "gray":
            description = "A gray image"
        kwargs['text'] = description
        kwargs['index'] = ind
        kwargs['type'] = "image"
        kwargs["dataset_type"] = self.dataset_type

        # Get text tokens
        text_ids, text_mask = self.get_text_tokens(self.text_encoder, description, self.text_len)
        if self.args.glyph_byT5_v2:
            if self.text_encoder_2 is None:
                return (
                    image_tensor,
                    torch.tensor(0),  # latents
                    text_ids.clone(),
                    text_mask.clone(),
                    byt5_text_ids.clone(),
                    byt5_text_mask.clone(),
                    byt5_text_valid,
                    {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
                )
            else:
                text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, description, self.text_len)
                return (
                    image_tensor,
                    torch.tensor(0),  # latents
                    text_ids.clone(),
                    text_mask.clone(),
                    text_ids_2.clone(),
                    text_mask_2.clone(),
                    byt5_text_ids.clone(),
                    byt5_text_mask.clone(),
                    byt5_text_valid,
                    {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
                )
        else:
            if self.text_encoder_2 is None:
                return (
                    image_tensor,
                    torch.tensor(0),  # latents
                    text_ids.clone(),
                    text_mask.clone(),
                    {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
                )
            else:
                text_ids_2, text_mask_2 = self.get_text_tokens(self.text_encoder_2, description)
                return (
                    image_tensor,
                    torch.tensor(0),  # latents
                    text_ids.clone(),
                    text_mask.clone(),
                    text_ids_2.clone(),
                    text_mask_2.clone(),
                    {k: torch.as_tensor(v) if not isinstance(v, str) else v for k, v in kwargs.items()},
                )

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 40960)
        return len(self.index_manager)

    @property
    def total_length(self):
        if len(self.index_file) == 1:
            return len(self.index_manager)
        else:
            return sum([len(bucket) for bucket in self.index_manager.buckets])
