import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, source, save_template, subset='', seed_type='auto', seed=1234, seed_plus=0,
                 pos_prompt="", skip_exist=False, logger=None, **kwargs):
        self.seed_type = seed_type
        self.seed = seed
        self.seed_plus = seed_plus

        # Check csv source type
        attr_dict = {}
        source_parts = source.split('@@')

        # Fill source file
        src_file = Path(source_parts[0])

        assert src_file.suffix in ['.csv', '.jsonl', '.xlsx'], f"Wrong dataset file suffix: {src_file.suffix}"

        if len(source_parts) > 1:
            for part in source_parts[1:]:
                key, value = part.split('=')
                attr_dict[key] = value

        # Load dataset file
        suffix = src_file.suffix.lstrip('.')
        if suffix == 'csv':
            self.table = pd.read_csv(src_file, keep_default_na=False, **kwargs)
        elif suffix == 'jsonl':
            self.table = pd.read_json(src_file, lines=True, **kwargs)
        elif suffix == 'xlsx':
            self.table = pd.read_excel(src_file, **kwargs)
        else:
            raise ValueError(f"Wrong dataset file suffix: {suffix}")

        # Check index_col exists (optional)
        index_col = attr_dict.get('index', "index")
        if index_col in self.table.columns:
            self.table.set_index(index_col, inplace=True, drop=False)

        # Check text_col exists (mandatory)
        text_col = attr_dict.get('text', "prompt")
        if text_col not in self.table.columns:
            raise ValueError(f'Missing text column `{text_col}` in table. {self.table.columns}')

        # Check seed_col exists (optional)
        seed_col = attr_dict.get('seed', "seed")
        logger.info(f"Using seed type: {self.seed_type}")
        if self.seed_type == 'auto':
            if seed_col in self.table.columns:
                self.seed_type = 'file'
            else:
                self.seed_type = 'fixed'
        if self.seed_type == 'fixed':
            logger.info(f'    Using fixed seed for all samples: {seed}')
        elif self.seed_type == 'random':
            logger.info(f'    Using random seed for all samples.')
        elif self.seed_type == 'file':
            if seed_col not in self.table.columns:
                raise ValueError(f'Missing seed column `{seed_col}` in table.')
            self.seed_col = seed_col
            logger.info(f'    Using seeds from column: {seed_col}')
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")

        # Select a subset of indices if needed.
        if len(subset) == 0:
            subset = list(self.table.index)
        else:
            subset = list(map(int, subset.split()))
            logger.info(f"Generate a subset with ids: {subset}")

        # Build input dict
        self.total_prompts = []
        self.total_prompt_dicts = []
        for id_, item in self.table.iterrows():
            if id_ not in subset:
                continue
            if not item[text_col]:
                logger.info(f"Empty prompt for id {id_}")
            else:
                prompt = str(item[text_col]) + pos_prompt
                self.total_prompts.append(prompt)

                p_tmp = {
                    "id": id_,
                    "text": prompt,
                    "seed": self.get_seed(id_),
                    'save_path': save_template.format(id_),
                }

                if skip_exist and Path(p_tmp['save_path']).exists():
                    continue
                self.total_prompt_dicts.append(p_tmp)

    def get_seed(self, idx, min_val=0, max_val=1_000_000):
        # Determine seeds
        if self.seed_type == 'file':
            seed = int(self.table.loc[idx, self.seed_col])
        elif self.seed_type == 'fixed':
            seed = int(self.seed)
        elif self.seed_type == 'random':
            seed = random.randint(min_val, max_val)
        else:
            raise ValueError(f"Wrong seed_type: {self.seed_type}")
        return seed + self.seed_plus

    def __len__(self):
        return len(self.total_prompt_dicts)

    def __getitem__(self, index):
        return self.total_prompt_dicts[index]
