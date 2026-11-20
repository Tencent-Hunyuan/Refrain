#!/usr/bin/env python3
"""Evaluate arbitrary method outputs on the released benchmark, without stored results."""
from __future__ import annotations
import argparse, csv, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
STAGES=('transition','clip','semantic','regions','ta','reference_semantic','sdc')

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def mean(xs):
    return statistics.mean(xs) if xs else None

def timing_error(expected,observed,penalty=1.0):
    """Ordered matching in seconds; one-second insertion/deletion penalties."""
    if not observed:return None
    dp=[j*penalty for j in range(len(observed)+1)]
    for i,a in enumerate(expected,1):
        new=[i*penalty]
        for j,b in enumerate(observed,1):
            new.append(min(dp[j-1]+abs(a-b),dp[j]+penalty,new[-1]+penalty))
        dp=new
    return dp[-1]/max(len(expected),len(observed))

def rows(args):
    benchmark=json.loads(args.benchmark.read_text());base=args.benchmark.resolve().parent
    cases={int(c['index']):c for c in benchmark['cases']}
    obj=json.loads(args.videos.read_text());items=obj['rows'] if isinstance(obj,dict) else obj
    from evaluation.consistency import plan
    seen=set()
    for item in items:
        case=int(item['case']);method=item['method']
        if not method or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in method):
            raise ValueError('Use a simple alphanumeric method name')
        if (method,case) in seen:raise ValueError('Duplicate method/case')
        seen.add((method,case));c=cases[case]
        if case%args.shards!=args.rank:continue
        refs=json.loads(json.dumps(c['references']))
        for ref in refs:ref['path']=str(base/ref['path'])
        path=Path(item['video_path'])
        if not path.is_absolute():path=args.videos.resolve().parent/path
        r=dict(method=method,case=case,video_path=str(path),task_type=c['task_type'],
               expected_fps=benchmark['fps'],shot_spans=c['shot_spans'],references=refs,
               style=c['shared_style'],shot_texts=c['original_long_descriptions'],
               clip_texts=c['evaluation_local_shot_texts'],
               review=c['evaluation_review'],source_id=c.get('source_id'),
               sdc_fold=c.get('sdc_fold'),
               mae_applicable=item.get('mae_applicable',True),
               reference_metrics_applicable=item.get('reference_metrics_applicable',True),
               submitted_cut_seconds=item.get('submitted_cut_seconds'))
        r['units']=plan(r)
        if not r['reference_metrics_applicable']:
            r['units']=[u for u in r['units'] if not u['metric'].startswith('reference_')]
        yield r

def sample8(a,b):
    import numpy as np
    if not 0<=a<b:raise ValueError('Invalid interval')
    return np.rint(np.linspace(a,b-1,8)).astype(int).tolist()

def clip_features(model,frames,device):
    import cv2,numpy as np,torch
    x=np.stack([cv2.resize(f,(224,224)) for f in frames]).astype(np.float32)/255
    x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    with torch.inference_mode():
        return torch.nn.functional.normalize(model.encode_vision(torch.from_numpy(x).permute(0,3,1,2)[None].to(device),test=True).float(),dim=-1)

def reference_semantic(model,row,frames,args):
    from evaluation.media import read_frames
    from evaluation.coordinates import reference_interval
    result=[]
    if not row['reference_metrics_applicable']:return result
    for j,r in enumerate(row['references']):
        if r['kind']!='video':continue
        u=dict(metric='reference_semantic',reference=j,edit_category=r['edit_category'])
        try:
            a,b=reference_interval(r,row['shot_spans'][-1][1])
            if b>len(frames):raise ValueError('Incomplete destination interval')
            ri=sample8(0,r['media']['frames']);gi=sample8(a,b)
            ref,_=read_frames(r['path'],ri)
            x=clip_features(model,ref,args.device);y=clip_features(model,[frames[k] for k in gi],args.device)
            u.update(status='ok',value=float((x@y.T).item()),reference_frames=ri,destination_frames=gi)
        except Exception as e:u.update(status='failed',error=str(e))
        result.append(u)
    return result

def transition(model,row,frames,args):
    import numpy as np
    from PIL import Image
    from evaluation.style_features.transitions import infer
    small=np.stack([np.asarray(Image.fromarray(f).resize((48,27),Image.Resampling.BILINEAR)) for f in frames])
    cuts,probabilities=infer(model,small,.8)
    expected=row['submitted_cut_seconds']
    if expected is None:expected=[4*b/row['expected_fps'] for _,b in row['shot_spans'][:-1]]
    observed=[(k+1)/row['expected_fps'] for k in cuts]
    value=timing_error(expected,observed) if row['mae_applicable'] else None
    return dict(cut_frames=cuts,frame_count=len(frames),fps=row['expected_fps'],
                units=[dict(metric='MAE',value=value,status='ok' if value is not None else 'inapplicable')])

def load(stage,args):
    import torch
    from evaluation.encoders import load_viclip,load_videoalign
    if stage in ('semantic','reference_semantic'):return load_viclip(args)
    if stage=='ta':return load_videoalign(args)
    if stage=='regions':
        from evaluation.consistency import Regions
        return Regions(args)
    if stage=='clip':
        from evaluation.media import load_clip
        return load_clip(args)
    if stage=='transition':
        sys.path.insert(0,str(args.transnet_root))
        from transnetv2_pytorch import TransNetV2
        model=TransNetV2().eval()
        model.load_state_dict(torch.load(args.transnet,map_location='cpu',weights_only=True))
        return model.to(args.device)
    if stage=='sdc':
        from evaluation.style_features.csd import CsdDualHeadExtractor
        from evaluation.style_consistency.content import DinoGPU
        from evaluation.style_consistency.style import VggAdaINGPU
        from evaluation.style_consistency.bank.build_bank import load_bank_set,load_source_folds
        from evaluation.style_consistency.bank.compact_bank import load_compact_bank
        compact=(args.bank/'metadata.json').is_file()
        banks,meta=load_compact_bank(args.bank) if compact else load_bank_set(args.bank)
        if meta.get('csd_head')!='module.last_layer_style':
            raise ValueError('Bank must explicitly identify corrected CSD STYLE head')
        # A full bank carries its source partition; otherwise each request's sdc_fold is used.
        folds=load_source_folds(args.bank,as_lookup=True) if (args.bank/'partition').is_dir() else {}
        return dict(csd=CsdDualHeadExtractor(args.csd,8,torch.device(args.device)),
                    dino=DinoGPU(args.dino,8,torch.device(args.device)),
                    vgg=VggAdaINGPU(args.vgg,8,torch.device(args.device)),banks=banks,folds=folds)

def sdc(model,row,frames,args):
    import numpy as np
    from evaluation.style_features.csd import sample_rgb_indices,pair_distances
    from evaluation.style_consistency.scoring import query_examples
    from evaluation.style_consistency.bank.array_bank import score_pairs_many
    path=args.output/f'{row["method"]}-{row["case"]}-transition.json'
    info=json.loads(path.read_text())
    if info['status']!='ok':raise ValueError('Transition extraction failed')
    boundaries=[0,*[x+1 for x in info['cut_frames']],len(frames)]
    spans=list(zip(boundaries[:-1],boundaries[1:]))
    if len(spans)<2 or any(b-a<8 for a,b in spans):return dict(units=[dict(metric='SDC',status='inapplicable',value=None)])
    indices=sample_rgb_indices(spans,len(frames),4,3)
    images=[frames[i] for group in indices for i in group];positions=[];offset=0
    for group in indices:positions.append(list(range(offset,offset+len(group))));offset+=len(group)
    # CSD extractor returns STYLE FIRST, content head second. DINO supplies content.
    style,_=model['csd'](images);content=model['dino'](images);vgg=model['vgg'](images)
    def geometry(features):
        within,cross=pair_distances(features,content,positions,indices)
        return dict(sample_indices=indices,within_pairs=within,cross_pairs=cross)
    a,b=geometry(style),geometry(vgg)
    csd_pairs=query_examples(a,a);vgg_pairs=query_examples(a,b)
    pairs=[dict(content_energy=x['content_energy'],csd_style_energy=x['style_energy'],vgg_style_energy=y['style_energy']) for x,y in zip(csd_pairs,vgg_pairs)]
    source=row['source_id']
    if not source:raise ValueError('Supply source_id for source-fold exclusion')
    fold=model['folds'].get(source)
    if fold is None:fold=row.get('sdc_fold')
    if fold not in range(5):raise ValueError('Supply sdc_fold in [0,4] for the request')
    scored=score_pairs_many(pairs,model['banks'],['real_env0','real_env1','real_env2'],fold)
    return dict(units=[dict(metric='SDC',status='ok',value=mean([q['pair_score'] for q in scored]))],
                shot_sample_indices=indices,excluded_fold=fold,pairs=scored)

def worker(args):
    from evaluation.media import read_frames,clip_score
    from evaluation.consistency import semantic_score,ta_score,region_score
    tasks=list(rows(args))
    model=None
    for row in tasks:
        dst=args.output/f'{row["method"]}-{row["case"]}-{args.stage}.json'
        if dst.exists():
            old=json.loads(dst.read_text())
            if old.get('inputs')!=row:raise ValueError('Inputs changed; choose a new output directory')
            if old.get('status')=='ok' and all(u.get('status') in ('ok','inapplicable') for u in old.get('units',[])):continue
        if model is None:model=load(args.stage,args)
        record=dict(method=row['method'],case=row['case'],inputs=row)
        try:
            frames,fps=read_frames(row['video_path'])
            if abs(fps-row['expected_fps'])>1e-4:raise ValueError('FPS mismatch; retiming is prohibited')
            if args.stage=='transition':data=transition(model,row,frames,args)
            elif args.stage=='sdc':data=sdc(model,row,frames,args)
            elif args.stage=='reference_semantic':data=dict(units=reference_semantic(model,row,frames,args))
            elif args.stage=='clip':
                data=clip_score(model,row,frames,args)
                data['units']=[dict(metric='CLIP-T',value=q['clip_t'],status='ok') for q in data['shots'] if q['status']=='ok']
            else:
                units,audit={'semantic':semantic_score,'ta':ta_score,'regions':region_score}[args.stage](model,row,frames,args)
                data=dict(units=units)
            record.update(status='ok',**data)
        except Exception as e:record.update(status='failed',error=repr(e))
        save(dst,record);print(args.stage,row['method'],row['case'],record['status'],flush=True)

def summarize(args):
    masks=json.loads((ROOT/'configs/metric_eligibility.json').read_text())['eligible_units']
    masks={m:{(int(c),u) for c,u in values} for m,values in masks.items()}
    totals=defaultdict(list);coverage=defaultdict(lambda:dict(videos=0,units=0));failures=[]
    for stage in STAGES:
        for p in args.output.glob(f'*-{stage}.json'):
            r=json.loads(p.read_text())
            if r.get('status')!='ok':failures.append(r);continue
            by=defaultdict(list)
            for u in r.get('units',[]):
                if u['metric'] in masks and (int(r['case']),u.get('unit')) not in masks[u['metric']]:continue
                if u.get('status')=='ok' and u.get('value') is not None:by[u['metric']].append(u['value'])
                elif u.get('status') not in ('ok','inapplicable'):failures.append(dict(method=r['method'],case=r['case'],stage=stage,unit=u))
            for metric,values in by.items():
                key=(r['method'],metric);totals[key].append(mean(values));coverage[key]['videos']+=1;coverage[key]['units']+=len(values)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['method','metric','mean','videos','units']);w.writeheader()
        for (method,metric),v in sorted(totals.items()):w.writerow(dict(method=method,metric=metric,mean=mean(v),**coverage[method,metric]))
    save(args.output/'failures.json',failures)
    print(args.output/'summary.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['worker','summarize'])
    p.add_argument('--stage',choices=STAGES,default='transition')
    p.add_argument('--benchmark',type=Path,default=ROOT/'benchmark/requests.json')
    p.add_argument('--videos',type=Path,default=Path('videos/videos.json'))
    p.add_argument('--output',type=Path,default=Path('metrics'))
    p.add_argument('--weights',type=Path,default=ROOT/'weights/evaluation')
    p.add_argument('--bank',type=Path,default=ROOT/'evaluation/assets/sdc_bank')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--shards',type=int,default=1);p.add_argument('--rank',type=int,default=0)
    args=p.parse_args()
    if not 0<=args.rank<args.shards:p.error('Invalid shard assignment')
    args.models=args.weights/'quality_eval';args.clip=args.weights/'clip/ViT-L-14.pt'
    args.detector=args.weights/'grounding-dino';args.sam=args.weights/'sam';args.dino=args.weights/'dinov2-small'
    args.csd=args.weights/'csd/csd-vit-l.pth';args.vgg=args.weights/'vgg16-397923af.pth'
    args.transnet_root=args.weights/'transnetv2';args.transnet=args.transnet_root/'model.pt'
    (worker if args.action=='worker' else summarize)(args)
if __name__=='__main__':main()
