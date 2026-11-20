import os
import csv
from pathlib import Path
import json
from torch.utils.data import Dataset

def fix_nulls(s):
    for line in s:
        yield line.replace('\0', ' ')

def conditions_to_condition_spec(conditions, gt_latent_cos_path=None):
    """Translate a reference-editing JSON case's `conditions` field into the
    `{image_anchors, video_anchors}` schema consumed by the range_condition
    generation path in hymm/eval/eval_score_video_general.py. Returns None when
    there are no conditions (case then generates as plain t2v via the fallback).

    JSON schema (benchmark-style):
      conditions.segments[] = {seg_id, dst_range:[a,b], latent_path, edit_category,
                               src_range?/src_latent_range?}
      conditions.anchors[]  = {frame_idx, source:"gt"}  (uses case gt_latent_cos_path)
    """
    if not conditions:
        return None
    image_anchors, video_anchors = [], []
    for seg in (conditions.get("segments") or []):
        dr = seg.get("dst_range")
        if not dr or len(dr) != 2:
            continue
        van = {
            "dst_start": int(dr[0]),
            "dst_end": int(dr[1]),
            "latent_path": seg.get("latent_path", "") or "",
            "path": seg.get("path", "") or "",
            "seg_id": int(seg.get("seg_id", -1)),
        }
        if seg.get('require_native_latent'):
            van['require_native_latent'] = True
        sr = seg.get("src_range") or seg.get("src_latent_range")
        if sr and len(sr) == 2:
            van["src_range"] = [int(sr[0]), int(sr[1])]
        video_anchors.append(van)
    for anc in (conditions.get("anchors") or []):
        fi = anc.get("frame_idx")
        if fi is None:
            continue
        ia = {"anchor_frame": int(fi), "frame_idx": int(fi)}
        if anc.get('latent_path'):
            # A portable [48,1,H,W] slice of the original full-video VAE latent.
            # Keep PNG alongside it for inspection; the latent owns inference.
            ia.update(latent_path=anc['latent_path'], path=anc.get('path', ''),
                      source_frame=int(anc.get('source_frame', fi)),
                      latent_scale=anc.get('latent_scale', 'unscaled_vae_mode'),
                      require_native_latent=True)
        elif anc.get("source") == "gt" and gt_latent_cos_path:
            ia["gt_latent_cos_path"] = gt_latent_cos_path
            # Same cache root train.py joins gt_latent_cos_path against (train.py:1319).
            ia["gt_latent_cache_folder"] = os.environ.get("REFRAIN_LATENT_ROOT", "train/latents")
        elif anc.get("path"):
            ia["path"] = anc["path"]
        image_anchors.append(ia)
    if not image_anchors and not video_anchors:
        return None
    return {"image_anchors": image_anchors, "video_anchors": video_anchors}

class GeneralGenerationBenchDataset(Dataset):
    def __init__(self, csvs, debug=False):
        super().__init__()

        dataset = []
        for csv_path in csvs:
            if str(csv_path).endswith(".json"):
                # Reference-editing JSON with per-case `conditions`. Normalize each
                # case into a row dict matching the CSV column contract below.
                raw = json.load(open(csv_path, "r", encoding="utf-8"))
                cases = raw.get("cases", raw) if isinstance(raw, dict) else raw
                for c in cases:
                    c = json.loads(json.dumps(c))
                    base = Path(csv_path).resolve().parent
                    for ref in (c.get("conditions") or {}).get("segments", []) + (c.get("conditions") or {}).get("anchors", []):
                        for field in ("path", "latent_path"):
                            if ref.get(field) and not Path(ref[field]).is_absolute():
                                ref[field] = str(base / ref[field])
                    if c.get("gt_latent_cos_path") and not Path(c["gt_latent_cos_path"]).is_absolute():
                        c["gt_latent_cos_path"] = str(base / c["gt_latent_cos_path"])
                    cs = conditions_to_condition_spec(
                        c.get("conditions"), c.get("gt_latent_cos_path")
                    )
                    dataset.append({
                        "prompt": c.get("prompt", ""),
                        "index": c.get("index"),
                        "seed": c.get("seed", 2025),
                        "height": c.get("target_height"),
                        "width": c.get("target_width"),
                        "latent_target_length": c.get("latent_target_length"),
                        "condition_spec": (json.dumps(cs) if cs is not None else None),
                    })
            else:
                with open(csv_path, 'r', newline="\n", encoding='utf-8-sig') as csvfile:
                    dataset_i = list(csv.DictReader(fix_nulls(csvfile)))
                dataset.extend(dataset_i)

        self.id_list, self.prompt_list, self.seed_list, self.ref_image_list, self.height_list, self.width_list = [], [], [], [], [], []
        self.first_list, self.last_list = [], []
        self.condition_spec_list, self.latent_target_length_list = [], []

        for i in range(len(dataset)):

            prompt = dataset[i]["prompt"]
            index = dataset[i]["index"] if "index" in dataset[i] else i
            
            seed = dataset[i]["seed"] if "seed" in dataset[i] else 2025
            ref_image_path = dataset[i]["ref_image_path"] if "ref_image_path" in dataset[i] else None
                
            first = dataset[i]["first"] if "first" in dataset[i] else None
            last = dataset[i]["last"] if "last" in dataset[i] else None
            condition_spec = dataset[i]["condition_spec"] if "condition_spec" in dataset[i] else None
            latent_target_length = dataset[i]["latent_target_length"] if "latent_target_length" in dataset[i] else None

            height = dataset[i]["height"] if "height" in dataset[i] else None
            width = dataset[i]["width"] if "width" in dataset[i] else None
            
            # index may be a string(md5) or int
            try:
                self.id_list.append(int(index))
            except Exception as e:
                self.id_list.append(index)
            self.prompt_list.append(prompt)
            self.seed_list.append(int(seed))
            self.ref_image_list.append(ref_image_path)
            self.height_list.append(height)
            self.width_list.append(width)
            self.first_list.append(first)
            self.last_list.append(last)
            self.condition_spec_list.append(condition_spec)
            self.latent_target_length_list.append(latent_target_length)
            
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        ret = {
            "id": self.id_list[index],
            "prompt": self.prompt_list[index],
            "seed": self.seed_list[index],
            "ref_image_path": self.ref_image_list[index],
            "first": self.first_list[index],
            "last": self.last_list[index],
            "questions": "",
            "dependencies": "",
            "height": self.height_list[index],
            "width": self.width_list[index],
            "condition_spec": self.condition_spec_list[index],
            "latent_target_length": self.latent_target_length_list[index],
        }
        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        ids = []
        prompts = []
        seeds = []
        questions = []
        dependencies = []
        ref_image_paths = []
        heights = []
        widths = []
        firsts = []
        lasts = []
        condition_specs = []
        latent_target_lengths = []
        for i in range(batch_size):
            ids.append(batch[i]["id"])
            prompts.append(batch[i]["prompt"])
            seeds.append(batch[i]["seed"])
            questions.append(batch[i]["questions"])
            dependencies.append(batch[i]["dependencies"])
            ref_image_paths.append(batch[i]["ref_image_path"])
            heights.append(batch[i]["height"])
            widths.append(batch[i]["width"])
            firsts.append(batch[i]["first"])
            lasts.append(batch[i]["last"])
            condition_specs.append(batch[i]["condition_spec"])
            latent_target_lengths.append(batch[i]["latent_target_length"])
        ret = {
            "ids": ids,
            "prompts": prompts,
            "seeds": seeds,
            "questions": questions,
            "dependencies": dependencies,
            "ref_image_paths": ref_image_paths,
            "heights": heights,
            "widths": widths,
            "firsts": firsts,
            "lasts": lasts,
            "condition_spec": condition_specs,
            "latent_target_lengths": latent_target_lengths,
        }

        return ret



    # dataloader = DataLoader(
    #     dataset=dataset,
    #     pin_memory=False,
    #     collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    #     batch_size=batch_size,
    #     num_workers=0,
    # )
    # for batch_index, batch in enumerate(dataloader, start=1):
    #     if "prompt_embeds" in batch.keys():
    #         prompts = None
    #         prompt_embeds = batch["prompt_embeds"]
    #         attention_mask = batch["attention_mask"]
    #         negative_prompt_embeds = batch["negative_prompt_embeds"]
    #         negative_attention_mask = batch["negative_attention_mask"]
    #     else:
    #         prompts = batch["prompts"]
    #         prompt_embeds = None
    #         attention_mask = None
    #         negative_prompt_embeds = None
    #         negative_attention_mask = None
    #     print(f"prompts = {prompts}")
