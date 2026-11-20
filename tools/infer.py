#!/usr/bin/env python3
"""Run benchmark requests sequentially on one eight-GPU sequence-parallel group."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASK_TYPES = ('t2v', 'ti2v', 'v2v', 'tiv2v')


def prepare_benchmark(path, samples_per_type=None, case_ids=None,
                      prompt_policy='control-only', prompt_subset='all', reference_input='rgb'):
    from tools.prompt_policy import select_prompt_subset, restore_reference_descriptions
    if prompt_policy not in ('control-only', 'described', 'prepared'):
        raise ValueError(f'Unknown prompt policy: {prompt_policy}')
    raw = json.loads(path.read_text())
    cases = raw['cases'] if isinstance(raw, dict) else raw
    if not cases or len({c['index'] for c in cases}) != len(cases):
        raise ValueError('Benchmark must contain unique, nonempty case IDs')
    cases = select_prompt_subset(cases, prompt_subset)
    if samples_per_type is not None:
        if samples_per_type < 1:
            raise ValueError('--samples-per-type must be positive')
        selected = []
        for kind in TASK_TYPES:
            group = [c for c in cases if c['task_type'] == kind]
            if len(group) < samples_per_type:
                raise ValueError(f'Not enough {kind} cases: {len(group)}')
            selected.extend(group[:samples_per_type])
        cases = selected
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c['index'] in wanted]
        if {c['index'] for c in cases} != wanted:
            raise ValueError('Requested case IDs are absent from the selected benchmark')
    # Only resolve paths; preserve timing, seeds, prompts, and native media.
    for c in cases:
        conditions = c.get('conditions') or {}
        refs = conditions.get('segments', []) + conditions.get('anchors', [])
        for ref in refs:
            if reference_input == 'native-latent':
                if not ref.get('latent_path'):
                    raise ValueError(f'Case {c["index"]} has no native latent mapping; use requests.json')
                ref['require_native_latent'] = True
            elif reference_input == 'rgb':
                ref.pop('latent_path', None)
                ref.pop('require_native_latent', None)
            else:
                raise ValueError(f'Unknown reference input: {reference_input}')
            for field in ('path', 'latent_path'):
                if ref.get(field) and not Path(ref[field]).is_absolute():
                    ref[field] = str(path.parent / ref[field])
        if c.get('gt_latent_cos_path') and not Path(c['gt_latent_cos_path']).is_absolute():
            c['gt_latent_cos_path'] = str(path.parent / c['gt_latent_cos_path'])
        from hymm.utils.physrope_prompt_rewrite import rewrite_bracket_prompt_for_reference_conditions
        from hymm.utils.prompt_cleaning import strip_identity_tags
        conditions = c.get('conditions') or {}
        c['original_prompt'] = c['prompt']
        if prompt_policy == 'prepared':
            if not c.get('prepared_prompt') or not c.get('prompt_role'):
                raise ValueError(f'Case {c["index"]} needs an explicit prepared prompt and experiment role')
            c['prompt'] = c['prepared_prompt']
            c['prompt_policy'] = prompt_policy
            c['reference_input'] = reference_input
            continue
        prompt = strip_identity_tags(c['prompt'])
        if conditions.get('segments') or conditions.get('anchors'):
            prompt, _ = rewrite_bracket_prompt_for_reference_conditions(
                prompt,
                condition_prompt_seg_ids=[s['seg_id'] for s in conditions.get('segments', []) if 'seg_id' in s],
                condition_prompt_ranges=[s['dst_range'] for s in conditions.get('segments', [])],
                conditions=conditions,
                global_prompt_prefix=strip_identity_tags(c.get('global_prompt_prefix', '')))
        c['prompt'] = strip_identity_tags(prompt)
        if prompt_policy == 'described':
            c['prompt'], restored = restore_reference_descriptions(c, c['prompt'])
            c['prompt_policy'] = prompt_policy
            c['restored_description_seg_ids'] = restored
        if reference_input == 'native-latent':
            c['reference_input'] = reference_input
    return {'cases': cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/infer.yaml')
    p.add_argument('--benchmark', type=Path, default=ROOT / 'benchmark/requests.json')
    p.add_argument('--output', type=Path, default=ROOT / 'videos')
    p.add_argument('--sp-size', type=int, choices=[8], default=8)
    p.add_argument('--samples-per-type', type=int)
    p.add_argument('--case-ids', type=int, nargs='+')
    from tools.prompt_policy import add_prompt_arguments, add_reference_arguments
    add_prompt_arguments(p)
    add_reference_arguments(p)
    p.set_defaults(prompt_policy="prepared", reference_input="native-latent")
    p.add_argument('--save-tensors', action='store_true', help='Save pre-video-compression RGB for numerical comparison')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    import yaml
    benchmark = prepare_benchmark(a.benchmark.resolve(), a.samples_per_type, a.case_ids,
                                  a.prompt_policy, a.prompt_subset, a.reference_input)
    values = yaml.safe_load(a.config.read_text())
    output = a.output.resolve()
    selected_path = output / 'requests.json'
    values.update(dit_weight=str(a.checkpoint.resolve()), resume=None,
                  output_dir=str(output), save_path=str(output),
                  val_save_path=str(output / 'generation.json'),
                  video_csv=[str(selected_path)], scores=[str(selected_path)],
                  score_save_image=True, flat_video_output=True, val_yaml_path='', val_exp_name='Refrain',
                  inference_save_tensors=a.save_tensors, strict_reference_inputs=True)
    if values.get('sp_size') != 8:
        raise ValueError('The released inference path uses SP=8')
    if a.dry_run:
        print(json.dumps({'arguments': values, 'prompt_policy': a.prompt_policy,
                          'reference_input': a.reference_input,
                          'prompt_subset': a.prompt_subset,
                          'cases': [c['index'] for c in benchmark['cases']]}, indent=2))
        return
    if int(os.environ.get('WORLD_SIZE', '0')) != 8 or int(os.environ.get('LOCAL_WORLD_SIZE', '0')) != 8:
        raise ValueError('Use tools/infer.sh: one node, eight ranks, SP=8, DP=1')
    if not a.checkpoint.exists():
        raise FileNotFoundError(a.checkpoint)
    import torch
    import torch.distributed as dist
    from hymm.config import parse_args
    from hymm.eval.eval_score_video_general import initialize_distributed, ScoreEvaluator
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = parse_args(mode='eval', namespace=argparse.Namespace(**values))
    finally:
        sys.argv = original_argv
    dims, mesh, device = initialize_distributed(args)
    device = torch.device('cuda', device)
    try:
        if dist.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=True)
            if selected_path.exists() and json.loads(selected_path.read_text()) != benchmark:
                raise ValueError('Output contains a different benchmark/prompt policy; use a fresh output')
            selected_path.write_text(json.dumps(benchmark, indent=2))
            (output / 'inference_args.json').write_text(json.dumps(vars(args), indent=2))
            print(f"[Refrain] sequential inference SP=8 DP=1 cases={[c['index'] for c in benchmark['cases']]} "
                  f"types={dict(Counter(c['task_type'] for c in benchmark['cases']))}", flush=True)
        dist.barrier()
        evaluator = ScoreEvaluator.from_pretrained(
            args.dit_weight, args.pretrained_model_name_or_path, args=args,
            device=device, parallel_dims=dims, world_mesh=mesh)
        evaluator.initialize_scores(args.scores)
        evaluator.eval_video(video_size=args.video_size, batch_size=1,
                             save_path=args.val_save_path, save_base=output, ignore_max_size=True)
        if dist.get_rank() == 0:
            manifest = [dict(method='Refrain', case=c['index'],
                             video_path=f"{c['index']}.mp4")
                        for c in benchmark['cases']]
            missing = [r['case'] for r in manifest if not (output / r['video_path']).is_file()]
            if missing:
                raise RuntimeError(f'Generation incomplete; missing cases: {missing}')
            (output / 'videos.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
