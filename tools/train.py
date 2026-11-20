#!/usr/bin/env python3
"""Launch one Refrain training stage from configs/train.yaml."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
STAGES=('text_to_video','reference_training','variable_duration_training')


def load_stage(config, stage):
    import yaml
    raw=yaml.safe_load(Path(config).read_text())
    if stage not in raw['stages']:
        raise ValueError(f'Unknown stage {stage!r}; choose from {list(raw["stages"])}')
    values=dict(raw['shared'])
    values.update(raw['stages'][stage])
    return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=STAGES)
    p.add_argument('--config',type=Path,default=ROOT/'configs/train.yaml')
    p.add_argument('--resume')
    p.add_argument('--output')
    p.add_argument('--dry-run',action='store_true',help='Print arguments without importing Torch or launching training')
    a=p.parse_args()
    values=load_stage(a.config,a.stage)
    if a.resume: values['resume']=a.resume
    if a.output: values['output_dir']=a.output
    if a.dry_run:
        print(json.dumps(values,indent=2));return
    if not Path(values['dit_model_name_or_path']).is_file():
        raise FileNotFoundError('Provide the backbone checkpoint at '+values['dit_model_name_or_path'])
    if values.get('resume') and not Path(values['resume']).is_dir():
        raise FileNotFoundError('Provide the full resume checkpoint at '+values['resume'])
    from hymm.config import parse_args
    sys.argv=[sys.argv[0]]
    args=parse_args(namespace=argparse.Namespace(**values))
    for path in args.video_index_file:
        if not Path(path).is_file(): raise FileNotFoundError('Provide training index: '+path)
    from hymm.train import main as train
    train(args)
if __name__=='__main__':main()
