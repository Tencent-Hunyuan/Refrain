from pathlib import Path
import importlib.util, importlib, os, sys, types, json
ROOT = Path(__file__).resolve().parents[1]


def load_viclip(args):
    import shutil
    import torch
    package = Path(importlib.util.find_spec('vbench').origin).parent
    # Import the unchanged ViCLIP implementation without VBench's full suite.
    for name, directory in [('vbench', package), ('vbench.third_party', package/'third_party')]:
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    cache = args.models / 'vbench_cache'
    (cache / 'ViCLIP').mkdir(parents=True, exist_ok=True)
    # Many workers share this one cache path. A plain copyfile is not atomic, so a
    # concurrent reader can open a half-written gzip and die on EOFError. Copy to a
    # per-process temp file in the same directory, then os.replace (atomic within a
    # filesystem) so every reader sees either the old complete file or the new one.
    # os.replace is atomic, but replacing the SAME path from every worker still
    # swaps the inode under a concurrent reader: on the contributor ceph-fuse mount that
    # surfaced as `OSError 116 Stale file handle` and killed 46 of 96 workers.
    # Write it only if it is missing, and give each worker its own cache dir so
    # no two processes ever contend for one inode.
    # Node-local so 96 workers never touch the shared mount for this at all.
    cache = Path(os.environ.get('VBENCH_CACHE_ROOT', cache)) / f'w{os.getpid()}'
    (cache / 'ViCLIP').mkdir(parents=True, exist_ok=True)
    bpe = cache / 'ViCLIP/bpe_simple_vocab_16e6.txt.gz'
    source = ROOT/'evaluation/assets/ViCLIP/bpe_simple_vocab_16e6.txt.gz'
    if not bpe.exists() or bpe.stat().st_size != source.stat().st_size:
        staging = bpe.with_suffix(f'.gz.{os.getpid()}.tmp')
        shutil.copyfile(source, staging)
        os.replace(staging, bpe)
    utils = types.ModuleType('vbench.utils')
    utils.CACHE_DIR = str(cache)
    sys.modules['vbench.utils'] = utils
    cls = importlib.import_module('vbench.third_party.ViCLIP.viclip').ViCLIP
    return cls(pretrain=str(args.models/'viclip.pth')).eval().to(args.device)


def load_videoalign(args):
    import gc
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from hymm.models.reward_models.videoalign.inference import VideoVLMRewardInference
    from hymm.models.reward_models.videoalign.trainer import Qwen2VLRewardModelBT
    from hymm.models.reward_models.videoalign.train_reward import find_target_linear_names
    from hymm.models.reward_models.videoalign.data import DataConfig
    from hymm.models.reward_models.videoalign import vision_process
    from hymm.models.reward_models.videoalign.utils import _fix_checkpoint_key_mapping, _fix_checkpoint_key_mapping_v2
    cfg = json.loads((args.models/'videoalign_config.json').read_text())
    processor = AutoProcessor.from_pretrained(args.models/'qwen_config', padding_side='right')
    special = ['<|VQ_reward|>', '<|MQ_reward|>', '<|TA_reward|>']
    processor.tokenizer.add_special_tokens({'additional_special_tokens': special})
    config = AutoConfig.from_pretrained(args.models/'qwen_config')
    config.vocab_size = len(processor.tokenizer)
    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        model = Qwen2VLRewardModelBT(config, output_dim=1, reward_token='special',
                                    special_token_ids=processor.tokenizer.convert_tokens_to_ids(special))
    pc = cfg['peft_lora_config']
    targets = find_target_linear_names(model, num_lora_modules=pc['num_lora_modules'],
                                      lora_namespan_exclude=pc['lora_namespan_exclude'])
    with init_empty_weights():
        model = get_peft_model(model, LoraConfig(target_modules=targets, r=pc['lora_r'],
                    lora_alpha=pc['lora_alpha'], lora_dropout=pc['lora_dropout'],
                    task_type=pc['lora_task_type'], use_rslora=pc['use_rslora'], bias='none',
                    modules_to_save=pc['lora_modules_to_save']))
    state = torch.load(str(args.models/'videoalign.pth'), map_location='cpu', mmap=True)
    keys = set(model.state_dict())
    candidates = [('unchanged', state), ('v1', _fix_checkpoint_key_mapping(state)),
                  ('v2', _fix_checkpoint_key_mapping_v2(state))]
    match = next(((name, s) for name, s in candidates if set(s) == keys), None)
    if match is None:
        raise ValueError('VideoAlign checkpoint key mismatch; refusing partial or random weights')
    mapping_name = match[0]
    model.load_state_dict(match[1], strict=True, assign=True)
    del state, candidates, match
    gc.collect()
    model.config.tokenizer_padding_side = 'right'
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.use_cache = False
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    model = model.eval().to(device=args.device, dtype=dtype)
    scorer = VideoVLMRewardInference.__new__(VideoVLMRewardInference)
    scorer.device, scorer.model, scorer.processor = args.device, model, processor
    data = {k:v for k,v in cfg['data_config'].items() if k not in ['meta_data','data_dir']}
    scorer.data_config = DataConfig(**data)
    scorer.inference_config = cfg['inference_config']
    # Keep the published frame indices and image transform while decoding one
    # frame at a time. torchvision.io otherwise materializes the entire movie.
    def read_sampled_frames(ele):
        import av
        import numpy as np
        assert ele['sample_type'] == 'uniform'
        assert 'video_start' not in ele and 'video_end' not in ele
        path = ele['video'].removeprefix('file://')
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            total = stream.frames
            assert total > 0, 'A known frame count is required for exact uniform sampling'
            count = vision_process.smart_nframes(ele, total_frames=total,
                                                 video_fps=float(stream.average_rate))
            indices = torch.linspace(0, total-1, count).round().long().tolist()
            wanted = set(indices)
            selected = {}
            decoded = 0
            for i, frame in enumerate(container.decode(video=0)):
                decoded += 1
                if i in wanted:
                    selected[i] = frame.to_ndarray(format='rgb24')
            assert decoded == total, ('Video frame count mismatch', decoded, total)
        return torch.from_numpy(np.stack([selected[i] for i in indices])).permute(0, 3, 1, 2)
    vision_process.VIDEO_READER_BACKENDS['streaming_pyav'] = read_sampled_frames
    vision_process.FORCE_QWENVL_VIDEO_READER = 'streaming_pyav'
    vision_process.get_video_reader_backend.cache_clear()
    print('VideoAlign strict full checkpoint loaded, key mapping:', mapping_name, flush=True)
    return scorer

