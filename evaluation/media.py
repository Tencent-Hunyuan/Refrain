from functools import lru_cache
from pathlib import Path
import re, math, statistics
from evaluation.coordinates import reference_interval, shot_intervals
ROOT = Path(__file__).resolve().parents[1]
SPAN = re.compile(r"\[(\d+)-(\d+)\]")


def parse(text):
    matches = list(SPAN.finditer(text))
    assert len(matches) >= 2, 'Expected timed multi-shot caption'
    spans = [[int(m[1]), int(m[2])] for m in matches]
    texts = [text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
             for i, m in enumerate(matches)]
    assert spans[0][0] == 0 and all(a < b for a, b in spans)
    assert all(spans[i][1] == spans[i+1][0] for i in range(len(spans)-1))
    assert all(texts)
    return spans, texts


@lru_cache(maxsize=2000)
def media_info(path):
    import av
    with av.open(str(path)) as f:
        s = f.streams.video[0]
        assert s.frames > 0 and s.average_rate
        return dict(frames=s.frames, fps=float(s.average_rate), width=s.width, height=s.height)


def read_frames(path, indices=None, gt_half=False):
    import av
    import numpy as np
    requested = None if indices is None else set(indices)
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        graph = None
        if gt_half:
            graph = av.filter.Graph()
            src = graph.add_buffer(template=stream)
            crop = graph.add('crop', 'iw/2:ih:iw/2:0')
            sink = graph.add('buffersink')
            src.link_to(crop); crop.link_to(sink); graph.configure()
        fps = float(stream.average_rate)
        n = 0
        for i, frame in enumerate(container.decode(video=0)):
            n = i+1
            if requested is None or i in requested:
                if gt_half:
                    src.push(frame); frame = sink.pull()
                frames[i] = frame.to_ndarray(format='rgb24')
            if requested is not None and len(frames) == len(requested):
                break
    if indices is not None:
        assert set(indices) <= frames.keys(), (path, 'selected reference frame unavailable', indices)
        return [frames[i] for i in indices], fps
    return list(frames.values()), fps


def sample_indices(n, k=8):
    import numpy as np
    assert n > 0
    return np.rint(np.linspace(0,n-1,k)).astype(int).tolist()


def intervals(row, n):
    # n never changes the request: genuinely short videos still fail coverage.
    return shot_intervals(row)


def clip_score(bundle, row, frames, args):
    import numpy as np
    import torch
    from PIL import Image
    model, pre, clip = bundle
    scores, matrix, detail = [], [], []
    # No style prefix in local CLIP-T, matching the old local-content definition.
    with torch.inference_mode():
        encoded=[chunked_text_feature(t,clip._tokenizer,model.encode_text,77,args.device) for t in row.get('clip_texts', row['shot_texts'])]
        texts=torch.cat([x[0] for x in encoded])
        for i,(a,b) in enumerate(intervals(row,len(frames))):
            if b > len(frames):
                detail.append(dict(shot=i,status='incomplete_output'));continue
            ids = [a+j for j in sample_indices(b-a)]
            x = torch.stack([pre(Image.fromarray(frames[j])) for j in ids]).to(args.device)
            f = torch.nn.functional.normalize(model.encode_image(x).float(),dim=-1)
            sims = (f@texts.T).mean(0).cpu().numpy()
            scores.append(float(sims[i])); matrix.append((i,sims))
            detail.append(dict(shot=i,status='ok',sample_indices=ids,clip_t=float(sims[i]),text_chunks=encoded[i][1]))
    return dict(clip_t=average(scores),clip_t_shot_hit=average([float(int(x.argmax())==i) for i,x in matrix]),
                completed_shots=len(scores),requested_shots=len(row['shot_spans']),shots=detail)





def average(values):
    values = [x for x in values if x is not None]
    return statistics.fmean(values) if values else None


def load_clip(args):
    from hymm.metrics_video.clip_score.clip import clip
    model, preprocess = clip.load(str(args.clip),
                                  device=args.device, jit=False)
    return model.eval(), preprocess, clip


def chunked_text_feature(text, tokenizer, encoder, context, device):
    """Keep all caption tokens; record when native context needs >1 chunk."""
    import torch
    payload=tokenizer.encode(text)
    chunks=[payload[i:i+context-2] for i in range(0,len(payload),context-2)] or [[]]
    ids=torch.zeros((len(chunks),context),dtype=torch.long,device=device)
    for i,chunk in enumerate(chunks):
        values=[tokenizer.encoder['<|startoftext|>'],*chunk,tokenizer.encoder['<|endoftext|>']]
        ids[i,:len(values)]=torch.tensor(values,device=device)
    features=torch.nn.functional.normalize(encoder(ids).float(),dim=-1)
    return torch.nn.functional.normalize(features.mean(0,keepdim=True),dim=-1),len(chunks)
