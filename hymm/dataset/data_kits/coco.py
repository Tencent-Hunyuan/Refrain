from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset

from ..caps.caption_process import CaptionAug


# COCO2014 validation set
class COCOValDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.path = Path(path)
        self.image_files = list(self.path.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        with open(self.image_files[index], 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return {"image": self.transform(image), "image_id": self.image_files[index].name[:-4]}
    
    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        images = []
        image_ids = []

        for i in range(batch_size):
            images.append(batch[i]["image"])
            image_ids.append(batch[i]["image_id"])

        return {
            "images": torch.stack(images, dim=0),
            "image_ids": image_ids,
        }

# 30k samples randomly sampled from COCO2014 validation set
class COCO30kDataset(Dataset):
    def __init__(self, path, use_qwen_caption=False, caption_sample_ratio=None, debug=False):
        super().__init__()
        self.path = Path(path)
        self.use_qwen_caption = use_qwen_caption
        self.captions = json.load(open(self.path, "r"))["annotations"]
        if self.use_qwen_caption:
            self.caption_aug = CaptionAug(json.loads(caption_sample_ratio))
        if debug:
            self.captions = self.captions[:1024]

    def __len__(self):
        # return 120 # for test
        return len(self.captions)
    
    def __getitem__(self, index):
        ret = {
            "id": self.captions[index]["id"],
            # different captions may correspond to the same image_id
            "image_id": self.captions[index]["image_id"],
            "seed": self.captions[index]["seed"],
        }
        if self.use_qwen_caption:
            qwen_caption = self.captions[index]["qwen_caption"]
            prompt = self.caption_aug.caption_aug(qwen_caption)
            ret.update({"prompt": prompt})
        else:
            ret.update({"prompt": self.captions[index]["caption"]})

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

        return {
            "ids": ids,
            "image_ids": image_ids,
            "prompts": prompts,
            "seeds": seeds,
        }
