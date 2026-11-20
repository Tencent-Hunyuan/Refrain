from pathlib import Path
import itertools, statistics, re
from evaluation.media import read_frames, intervals, media_info
from evaluation.coordinates import reference_interval
ROOT = Path(__file__).resolve().parents[1]
METRICS = ["intershot_semantic", "intershot_subject", "intershot_scene", "reference_subject", "reference_background", "TA"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def keyframes(n):
    assert n > 0
    return sorted(set([0, (n - 1) // 2, n - 1]))


def plan(row):
    r = row['review']; units = []
    for a, b in r['semantic_pairs']:
        units.append(dict(metric='intershot_semantic', unit=f's{a+1}-s{b+1}', a=a, b=b))
    for a, b, e in r['subject_pairs']:
        units.append(dict(metric='intershot_subject', unit=f's{a+1}-s{b+1}:e{e}', a=a, b=b, entity=e))
    for a, b in r['scene_pairs']:
        units.append(dict(metric='intershot_scene', unit=f's{a+1}-s{b+1}', a=a, b=b))
    if row['method'] != 'vace_sparse_shotwise':
        for j, ref in enumerate(row['references']):
            shot = ref['shot']
            if ref['preserve_subject']:
                for e in r['entities'][shot]:
                    units.append(dict(metric='reference_subject', unit=f'r{j}:e{e}', reference=j, a=shot, entity=e))
            if ref['preserve_background']:
                units.append(dict(metric='reference_background', unit=f'r{j}', reference=j, a=shot))
    for a in range(len(row['shot_spans'])):
        units.append(dict(metric='TA', unit=f's{a+1}', a=a))
    return units


def stage_units(row, stage):
    names = {'semantic': {'intershot_semantic'}, 'ta': {'TA'}, 'regions': set(METRICS[1:5])}[stage]
    return [u for u in row['units'] if u['metric'] in names]


def scored(unit, value=None, status='ok', **kw):
    return dict(**unit, value=value, status=status, **kw)


def semantic_score(model, row, frames, args):
    import cv2
    import numpy as np
    import torch
    spans = intervals(row, len(frames)); features = {}; result = []
    for u in stage_units(row, 'semantic'):
        try:
            for s in [u['a'], u['b']]:
                if s in features: continue
                a, b = spans[s]
                if b > len(frames): raise ValueError('incomplete_requested_shot')
                ids = np.rint(np.linspace(a, b-1, 8)).astype(int).tolist()
                x = np.stack([cv2.resize(frames[k], (224,224)) for k in ids]).astype(np.float32)/255
                x = (x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
                with torch.inference_mode():
                    tensor = torch.from_numpy(x).permute(0,3,1,2)[None].to(args.device)
                    f = torch.nn.functional.normalize(model.encode_vision(tensor, test=True).float(), dim=-1)
                features[s] = (f, ids)
            a,b=u['a'],u['b']
            result.append(scored(u, float((features[a][0] @ features[b][0].T).item()),
                sample_indices=[features[a][1], features[b][1]]))
        except Exception as e: result.append(scored(u,status='measurement_failed',error=repr(e)))
    return result, {}


def ta_score(model, row, frames, args):
    import numpy as np
    import torch
    from hymm.models.reward_models.videoalign import vision_process as vp
    current = {}; sample = {}
    def reader(ele):
        images = current['frames']
        n = vp.smart_nframes(ele, total_frames=len(images), video_fps=row['expected_fps'])
        ids = torch.linspace(0,len(images)-1,n).round().long().tolist()
        sample['ids'] = ids
        return torch.from_numpy(np.stack([images[i] for i in ids])).permute(0,3,1,2)
    vp.VIDEO_READER_BACKENDS['long300_memory'] = reader
    vp.FORCE_QWENVL_VIDEO_READER = 'long300_memory'
    vp.get_video_reader_backend.cache_clear()
    spans = intervals(row,len(frames)); result = []
    for u in stage_units(row, 'ta'):
        s=u['a']; a,b=spans[s]
        try:
            if b > len(frames): raise ValueError('incomplete_requested_shot')
            current['frames'] = frames[a:b]
            text = row['style'] + '\n' + row['shot_texts'][s]
            with torch.inference_mode():
                # Published checkpoint jointly produces three heads. Retain TA only;
                # do not run separate VQ/MQ jobs or report their values.
                value = model.reward(['long300_memory_clip'],[text],use_norm=True)[0]['TA']
            result.append(scored(u,float(value),sample_indices=[a+i for i in sample['ids']],
                                 prompt=text,normalization='published VideoAlign TA mean/std'))
        except Exception as e: result.append(scored(u,status='measurement_failed',error=repr(e)))
        finally: current.clear()
    return result, {}


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    return inter/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def select_box(detections, margin=.05):
    """Independent text grounding; never choose by reference/DINO similarity."""
    detections=sorted(detections,key=lambda d:d['score'],reverse=True)
    unique=[]
    for d in detections:
        if all(iou(d['box'],p['box']) < .7 for p in unique): unique.append(d)
    if not unique: return None,'missing_detection'
    if len(unique)>1 and unique[0]['score']-unique[1]['score'] < margin:
        return None,'ambiguous_detection'
    return unique[0], 'ok'


def label_matches_subject(query,label):
    """Do not localize a person's glasses/coat instead of the requested person."""
    if '.' in query or re.search(r'\b(legs|hands|eyes|feet|ear|beard)\s*$',query): return True
    nouns=re.findall(r'\b(man|woman|girl|boy|person|child|chef|doctor|soldier|officer|guard|pilot|'
        r'firefighter|student|rider|jockey|scholar|maid|captain|warrior|gymnast|conductor|diver|'
        r'paramedic|clerk|surgeon|worker|presenter|prince|emperor|violinist|ninja)\b',query.lower())
    return not nouns or any(re.search(r'\b'+re.escape(n)+r'\b',label.lower()) for n in nouns)


class Regions:
    def __init__(self,args):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
        from transformers import AutoImageProcessor, AutoModel
        self.args=args; self.torch=torch
        self.dp=AutoProcessor.from_pretrained(args.detector,local_files_only=True)
        self.det=AutoModelForZeroShotObjectDetection.from_pretrained(args.detector,local_files_only=True).to(args.device).eval()
        self.sp=SamProcessor.from_pretrained(args.sam,local_files_only=True)
        self.sam=SamModel.from_pretrained(args.sam,local_files_only=True).to(args.device).eval()
        self.ip=AutoImageProcessor.from_pretrained(args.dino,local_files_only=True,use_fast=False)
        self.dino=AutoModel.from_pretrained(args.dino,local_files_only=True).to(args.device).eval()

    def detect(self,image,query):
        torch=self.torch
        inputs=self.dp(images=image,text=query.lower().rstrip('.')+'.',return_tensors='pt').to(self.args.device)
        with torch.inference_mode(): out=self.det(**inputs)
        data=self.dp.post_process_grounded_object_detection(out,inputs.input_ids,
            box_threshold=.30,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
        return [dict(box=b.tolist(),score=float(s),label=label)
                for b,s,label in zip(data['boxes'].cpu(),data['scores'].cpu(),data['labels'])
                if b[2]-b[0]>=8 and b[3]-b[1]>=8 and label_matches_subject(query,label)]

    def masks(self,image,boxes):
        torch=self.torch
        inputs=self.sp(images=image,input_boxes=[boxes],return_tensors='pt').to(self.args.device)
        with torch.inference_mode(): out=self.sam(**inputs,multimask_output=False)
        masks=self.sp.image_processor.post_process_masks(out.pred_masks.cpu(),
            inputs['original_sizes'].cpu(),inputs['reshaped_input_sizes'].cpu())[0][:,0]
        return masks.numpy().astype(bool)

    def encode(self,image):
        torch=self.torch
        inputs=self.ip(images=image,return_tensors='pt').to(self.args.device)
        with torch.inference_mode(): f=self.dino(**inputs).last_hidden_state[:,0].float()
        return torch.nn.functional.normalize(f,dim=-1)[0].cpu().numpy()

    def extract(self,frame,entities,need_background):
        import numpy as np
        from PIL import Image
        image=Image.fromarray(frame); selected={}; audit={}; features={}
        all_boxes=[]
        for e,q in entities.items():
            ds=self.detect(image,q); chosen,status=select_box(ds)
            audit[e]=dict(query=q,status=status,candidates=ds)
            all_boxes += [d['box'] for d in ds]
            if chosen is not None: selected[e]=chosen['box']
        # A detector assigning the same person to two labels cannot resolve identity.
        collisions=set()
        for a,b in itertools.combinations(selected,2):
            if iou(selected[a],selected[b])>.7: collisions.update([a,b])
        for e in collisions: selected.pop(e);audit[e]['status']='identity_collision'
        if need_background:
            for d in self.detect(image,'person. animal. car. boat. aircraft. robot.'):
                all_boxes.append(d['box'])
        boxes=list(selected.values())
        fg=[]
        for b in all_boxes:
            if all(iou(b,p)<.8 for p in fg): fg.append(b)
        masks=self.masks(image,boxes+fg) if boxes or fg else np.empty((0,*frame.shape[:2]),dtype=bool)
        for i,(e,b) in enumerate(selected.items()):
            mask=masks[i]; x0,y0,x1,y1=np.rint(b).astype(int)
            x0=max(0,x0);y0=max(0,y0);x1=min(image.width,x1);y1=min(image.height,y1)
            if mask[y0:y1,x0:x1].sum()<64:
                audit[e]['status']='empty_or_tiny_mask';continue
            crop=frame[y0:y1,x0:x1].copy(); crop[~mask[y0:y1,x0:x1]]=127
            features[e]=self.encode(Image.fromarray(crop))
            audit[e].update(box=b,mask_pixels=int(mask.sum()),status='ok')
        background=None
        if need_background:
            union=masks[len(boxes):].any(0) if fg else np.zeros(frame.shape[:2],bool)
            retained=float((~union).mean())
            # If none of the expected foreground is localized, background isolation
            # failed. Do not quietly score an unmasked full frame.
            if entities and not selected: audit['background']={'status':'foreground_unresolved'}
            elif retained<.2: audit['background']={'status':'insufficient_visible_background','fraction':retained}
            else:
                masked=frame.copy();masked[union]=127
                background=self.encode(Image.fromarray(masked))
                audit['background']={'status':'ok','fraction':retained,'foreground_boxes':fg}
        return dict(subject=features,background=background,audit=audit)


def feature_mean_cosine(left,right):
    import numpy as np
    if not left or not right: return None
    # Every available keyframe pair; no best-reference or nearest-neighbor selection.
    return float((np.stack(left) @ np.stack(right).T).mean())


def reference_keyframes(ref,frames,*,timeline_end=None):
    if ref.get('asset_error'): raise ValueError(ref['asset_error'])
    if ref['kind']=='image':
        t=ref['pixel_frame']
        if t>=len(frames): raise ValueError('anchor_destination_missing')
        from PIL import Image
        import numpy as np
        r=[np.asarray(Image.open(ref['path']).convert('RGB'))]
        return r,[frames[t]],dict(reference_indices=[t],destination_indices=[t],
            reference_frame=t,reference_frames=1,target_frames=1,target_latents=1)
    a,b=reference_interval(ref,timeline_end)
    if b>len(frames): raise ValueError('full_requested_destination_missing')
    info=media_info(ref['path']); assert info==ref['media'],'Reference changed'
    ri=keyframes(info['frames']);gi=[a+i for i in keyframes(b-a)]
    r,fps=read_frames(ref['path'],ri)
    assert abs(fps-24)<1e-6
    return r,[frames[i] for i in gi],dict(reference_indices=ri,destination_indices=gi,
        reference_frame=0,reference_frames=info['frames'],target_frames=b-a,
        target_latents=ref['target_latents'],length_relation=ref['length_relation'])


def region_score(model,row,frames,args):
    needed=stage_units(row,'regions'); spans=intervals(row,len(frames))
    cache={}; audits={}; refs={}; ref_errors={}
    def extract(images,entities,bg,label):
        result=[model.extract(im,entities,bg) for im in images]
        audits[label]=[r['audit'] for r in result]
        return result
    def shot(s):
        if s not in cache:
            a,b=spans[s]
            if b>len(frames):raise ValueError('incomplete_requested_shot')
            ids=[a+i for i in keyframes(b-a)]
            cache[s]=extract([frames[i] for i in ids],row['review']['entities'][s],
                             row['review']['scenes'][s]!='-',f'shot{s+1}')
            audits[f'shot{s+1}_indices']=ids
        return cache[s]
    def ref_pair(j):
        if j in ref_errors:raise ValueError(ref_errors[j])
        if j not in refs:
            try:
                ref=row['references'][j]; s=ref['shot'];r,g,meta=reference_keyframes(
                    ref,frames,timeline_end=row['shot_spans'][-1][1])
                # Anchor comparisons are only at their exact timestamp; video
                # references use independent native first/middle/last positions.
                bg=ref['preserve_background'];ent=row['review']['entities'][s]
                refs[j]=(extract(r,ent,bg,f'ref{j}'),extract(g,ent,bg,f'ref{j}_output'))
                audits[f'ref{j}_coordinates']=meta
            except Exception as e:ref_errors[j]=repr(e);raise
        return refs[j]
    results=[]
    for u in needed:
        try:
            if u['metric'].startswith('reference_'):left,right=ref_pair(u['reference'])
            else:left,right=shot(u['a']),shot(u['b'])
            if 'subject' in u['metric']:
                e=u['entity'];lf=[f['subject'][e] for f in left if e in f['subject']]
                rf=[f['subject'][e] for f in right if e in f['subject']]
            else:
                lf=[f['background'] for f in left if f['background'] is not None]
                rf=[f['background'] for f in right if f['background'] is not None]
            value=feature_mean_cosine(lf,rf)
            results.append(scored(u,value,status='ok' if value is not None else 'measurement_failed',
                valid_keyframes=[len(lf),len(rf)],expected_keyframes=[len(left),len(right)],
                failure_reason=None if value is not None else 'expected_region_missing_or_ambiguous'))
        except Exception as e:results.append(scored(u,status='measurement_failed',error=repr(e)))
    return results,audits

