import random
import json
from collections import defaultdict
from functools import partial

import torch
import torchvision.transforms as T
from PIL import Image
from hymm.dataset.arrow_index import (
    ArrowIndexV2, MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2, arrow_mapper,
)
from torch.utils.data import Dataset

from hymm.utils.helpers import to_2tuple
from hymm.dataset.caps import load_caption_processor


class VAL30kDataset(Dataset):
    def __init__(
        self,
        index_file=None,
        index_kwargs=None,
        image_size=256,
        crop_type='center',
        multireso=False,
        return_image=False,
        debug=False,
        logger=None
    ):
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.index_file = index_file

        if index_kwargs is None:
            index_kwargs = {}
        self.image_caption_rate = index_kwargs.get('image_caption_rate', 0.0)

        self.image_text_col = index_kwargs.get('image_text_col', None)
        if self.image_caption_rate < 1.0 and self.image_text_col is None:
            raise ValueError("Missing `image_text_col` in `index_kwargs`")
        self.image_text_arrow_suffix = index_kwargs.get('image_text_arrow_suffix', '_trans')

        self.image_caption_col = index_kwargs.get('image_caption_col', None)
        if self.image_caption_rate > 0.0 and self.image_caption_col is None:
            raise ValueError("Missing `image_caption_col` in `index_kwargs`")
        self.image_caption_arrow_suffix = index_kwargs.get('image_caption_arrow_suffix', '_caption')

        # Now if this argument is not None, we believe users are using QWEN caption.
        self.caption_sample_ratio = index_kwargs.get('caption_sample_ratio', None)
        if self.caption_sample_ratio is not None:
            self.use_structural_caption = True
            self.caption_sample_ratio = json.loads(self.caption_sample_ratio)
            CaptionAug = load_caption_processor(
                name="caption_process",
                caption_sample_ratio=self.caption_sample_ratio,
                logger=self.logger,
            )
            self.caption_aug = CaptionAug(self.caption_sample_ratio, logger=self.logger)
        else:
            self.use_structural_caption = False

        self.image_size = to_2tuple(image_size)
        self.crop_type = crop_type
        self.multireso = multireso
        self.return_image = return_image
        
        self.debug = debug

        # Prepare index manager
        index_load_kwargs = dict(
            batch_size=index_kwargs.get('batch_size', 1),
            world_size=index_kwargs.get('world_size', 1)
        )
        shadow_file_fn={} # keys of shadow_file_fn should be pass to get_attribute(shadow=key)
        if self.image_caption_rate < 1.0:
            shadow_file_fn.update({'text': partial(arrow_mapper, suffix=self.image_text_arrow_suffix)})
        if self.image_caption_rate > 0.0:
            shadow_file_fn.update({'caption': partial(arrow_mapper, suffix=self.image_caption_arrow_suffix)})
        index_load_kwargs['shadow_file_fn'] = shadow_file_fn

        self.index_manager = self.load_index(**index_load_kwargs)
        self.logger.info(f"Using {self.index_manager}")

        self.pil_image_to_tensor = T.Compose(
            [
                T.ToTensor(),               # convert to tensor and normalize to [0, 1]
                # T.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        self.logger.info(f"Image transform: {self.pil_image_to_tensor}")

        # Handle exception message. Avoid printing the same message multiple times.
        self.warnings = defaultdict(int)
        self.warning_max_times = 1

    def load_index(self,
                   batch_size,
                   world_size,
                   shadow_file_fn=None,
                   sample_strategy='uniform',
                   probability=None,
                   ):
        multireso = self.multireso
        index_file = self.index_file
        if not isinstance(index_file, (list, tuple)):
            index_file = [index_file]

        if multireso:
            if len(index_file) == 1:
                index_manager = MultiResolutionBucketIndexV2(
                    index_file[0], batch_size, world_size, shadow_file_fn=shadow_file_fn,
                )
            else:
                index_manager = MultiMultiResolutionBucketIndexV2(
                    index_file, batch_size, world_size, shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy, probability=probability,
                )
        else:
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(
                    index_file[0], shadow_file_fn=shadow_file_fn)
            else:
                index_manager = MultiIndexV2(
                    index_file, shadow_file_fn=shadow_file_fn,
                    sample_strategy=sample_strategy, probability=probability)

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="image"):
        try:
            ret = self.index_manager.get_image(index, image_key)
        except Exception as e:
            self.logger.error(f'{type(e)}: {e}')
            ret = Image.new("RGB", (self.image_size[0], self.image_size[1]), (128, 128, 128))
        return ret

    def get_image_with_size(self, index):
        image = self.get_raw_image(index, image_key='image')

        origin_size = image.size    # (w_ori, h_ori)

        if self.multireso:
            target_size = self.index_manager.get_target_size(index)     # (w_tgt, h_tgt)
        else:
            target_size = self.image_size[0], self.image_size[1]

        # hyvae use BILINEAR and BICUBIC to resize image. So here we use BICUBIC
        # TODO: maybe we can try LANCZOS
        image, (crop_left, crop_top) = self.index_manager.resize_and_crop(
            image, target_size, crop_type='random', resample=Image.Resampling.BICUBIC)

        image_tensor = self.pil_image_to_tensor(image)

        kwargs = {
            'origin_size': origin_size,
            'target_size': target_size,
            'crop_coords_xy': (crop_left, crop_top),
        }
        return image_tensor, kwargs

    def handle_exception_message(self, func, e):
        message = str(e)
        if self.warnings[message] < self.warning_max_times:
            self.warnings[message] += 1
            self.logger.error(f'{func.__name__} | {e.__class__.__name__}: {message}')

    def get_text(self, ind):
        try:
            if self.image_caption_rate > 0.0 and random.random() < self.image_caption_rate:
                text = self.index_manager.get_attribute(ind, self.image_caption_col, shadow='caption')
                if self.use_structural_caption:
                    text = self.caption_aug.caption_aug(text)
            else:
                text = self.index_manager.get_attribute(ind, self.image_text_col, shadow='text')
        except Exception as e:
            self.handle_exception_message(self.get_text, e)
            text = ''
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

        """

        # Get text
        description = self.get_text(ind)
        ret = {
            "id": ind,
            "image_id": ind,
            "prompt": description,
            "seed": ind,
        }
        # Get image
        if self.return_image:
            image_tensor, kwargs = self.get_image_with_size(ind)
            ret.update({"image": image_tensor.unsqueeze(0)})
        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        image_ids = []
        prompts = []
        seeds = []

        for i in range(batch_size):
            ids.append(batch[i]["id"])
            image_ids.append(batch[i]["image_id"])
            prompts.append(batch[i]["prompt"])
            seeds.append(batch[i]["seed"])

        ret = {
            "ids": ids,
            "image_ids": image_ids,
            "prompts": prompts,
            "seeds": seeds,
        }

        if "image" in batch[0].keys():
            images = []
            for i in range(batch_size):
                images.append(batch[i]["image"])
            images = torch.concat(images, dim=0)
            ret.update({"images": images})

        return ret

    def __len__(self):
        if self.debug:
            return min(len(self.index_manager), 256)
        return len(self.index_manager)

    @property
    def total_length(self):
        if len(self.index_file) == 1:
            return len(self.index_manager)
        else:
            return sum([len(bucket) for bucket in self.index_manager.buckets])
