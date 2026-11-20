#!/usr/bin/env python3
"""Extract real-video features and fit the SDC within-shot calibration bank.

A new source corpus defines a new calibration. Exact manuscript numbers require
its fixed released bank; a newly fitted bank is not interchangeable numerically.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def extract(args):
    import torch
    from evaluation.run import load,transition
    from evaluation.media import read_frames
    from evaluation.style_features.csd import CsdDualHeadExtractor
    from evaluation.style_consistency.content import DinoGPU
    from evaluation.style_consistency.style import VggAdaINGPU
    args.transnet_root=args.weights/'transnetv2';args.transnet=args.transnet_root/'model.pt'
    detector=load('transition',args)
    csd=CsdDualHeadExtractor(args.weights/'csd/csd-vit-l.pth',8,torch.device(args.device))
    dino=DinoGPU(args.weights/'dinov2-small',8,torch.device(args.device))
    vgg=VggAdaINGPU(args.weights/'vgg16-397923af.pth',8,torch.device(args.device))
    items=json.loads(args.manifest.read_text());args.output.mkdir(parents=True,exist_ok=True)
    failed=[]
    for index,r in enumerate(items):
        if index%args.shards!=args.rank:continue
        source=r['source_id']
        if len(source)!=32 or any(c not in '0123456789abcdef' for c in source):raise ValueError('source_id must be a 32-character hexadecimal identity')
        dst=args.output/(source+'.npz')
        if dst.exists():continue
        try:
            video=Path(r['video_path'])
            if not video.is_absolute():video=args.manifest.resolve().parent/video
            frames,fps=read_frames(video)
            query=dict(expected_fps=fps,submitted_cut_seconds=[],shot_spans=[],mae_applicable=False)
            cuts=transition(detector,query,frames,args)['cut_frames']
            ends=[0,*[k+1 for k in cuts],len(frames)];groups=[]
            for a,b in zip(ends[:-1],ends[1:]):
                n=min(4,b-a-6)
                if n>=2:groups.append(np.rint(np.linspace(a+3,b-4,n)).astype(int).tolist())
            if not groups:raise ValueError('No usable shot interior')
            images=[frames[i] for group in groups for i in group]
            style,content_head=csd(images)
            with dst.with_suffix('.tmp').open('wb') as f:
                np.savez(f,source_id=source,counts=np.array([len(g) for g in groups]),
                         frame_indices=np.array([i for g in groups for i in g]),
                         csd=style.astype(np.float32),csd_content=content_head.astype(np.float32),
                         dino=dino(images).astype(np.float32),vgg=vgg(images).astype(np.float32))
            dst.with_suffix('.tmp').replace(dst)
        except Exception as e:failed.append(dict(source_id=source,error=str(e)))
    (args.output/f'failures_{args.rank}.json').write_text(json.dumps(failed,indent=2))

def energies(x,counts):
    from evaluation.style_consistency.scoring import balanced_splits,energy
    result=[];offset=0
    for n in counts:
        z=x[offset:offset+n];offset+=n
        # Identical float32 cosine distances with explicit zero diagonal.
        distance=np.zeros((n,n),dtype=np.float64)
        for a in range(n):
            for b in range(a+1,n):
                distance[a,b]=distance[b,a]=max(1.0-float(z[a]@z[b]),0.0)
        for a,b in balanced_splits(int(n)):
            result.append(energy(distance[np.ix_(a,b)],distance[np.ix_(a,a)],distance[np.ix_(b,b)]))
    return result

def fit(args):
    from evaluation.style_consistency.scoring import balanced_splits
    from evaluation.style_consistency.bank.array_bank import ArrayConditionalResidualStyleBank
    paths=sorted(args.features.glob('*.npz'))
    if not paths:raise ValueError('No calibration features')
    sources=[];rows=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as f:
            sources.append(str(f['source_id']))
            counts=f['counts'];weights=[]
            for n in counts:weights.extend([1/(len(counts)*len(balanced_splits(int(n))))]*len(balanced_splits(int(n))))
            rows.append((energies(f['dino'],counts),energies(f['csd'],counts),energies(f['vgg'],counts),weights))
    if len(set(sources))!=len(sources):raise ValueError('Repeated source identities')
    n=len(rows);permutation=np.arange(n);np.random.default_rng(20260908).shuffle(permutation)
    groups=np.empty(n,dtype=np.uint8);folds=np.empty(n,dtype=np.uint8)
    groups[permutation]=np.arange(n)%3;folds[permutation]=np.arange(n)%5
    root=args.output;root.mkdir(parents=True,exist_ok=True);partition=root/'partition';partition.mkdir(exist_ok=True)
    md5=np.asarray(sources,dtype='S32')
    for name,array in dict(md5=md5,md5_sorter=np.argsort(md5,kind='mergesort'),fold=folds,group=groups,permutation=permutation).items():np.save(partition/(name+'.npy'),array,allow_pickle=False)
    cells=[]
    for fold in range(5):
        for group in range(3):
            selected=[r for i,r in enumerate(rows) if folds[i]!=fold and groups[i]==group]
            if not selected:raise ValueError('Insufficient independent source videos')
            content=np.concatenate([r[0] for r in selected]);weights=np.concatenate([np.asarray(r[3])/len(selected) for r in selected])
            for head,j in [('csd',1),('vgg',2)]:
                relative=f'cells/{head}_fold{fold}_group{group}';target=root/relative
                provenance=dict(head=head,fold=fold,group=group)
                if target.exists():
                    bank=ArrayConditionalResidualStyleBank.load(target)
                    if bank.provenance!=provenance:raise ValueError('Existing cell uses different feature data')
                else:
                    bank=ArrayConditionalResidualStyleBank.fit(content,np.concatenate([r[j] for r in selected]),weights)
                    bank.save(target,provenance=provenance)
                cells.append(dict(head=head,fold=fold,group=group,path=relative))
    metadata=dict(complete=True,bank_id='real_style_first',csd_head='module.last_layer_style',
                  n_videos=n,cells=cells)
    (root/'bank_inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['extract','fit'])
    p.add_argument('--manifest',type=Path,default=Path('train/real_videos.json'))
    p.add_argument('--features',type=Path,default=Path('outputs/calibration_features'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,default=ROOT/'weights/evaluation')
    p.add_argument('--device',default='cuda:0');p.add_argument('--shards',type=int,default=1);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.rank<a.shards:p.error('Invalid shard assignment')
    (extract if a.action=='extract' else fit)(a)
if __name__=='__main__':main()
