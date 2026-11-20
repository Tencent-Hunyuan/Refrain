#!/usr/bin/env python3
"""Convert a model-only PyTorch distributed checkpoint to safetensors on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict


class ParallelFileReader(FileSystemReader):
    """Read independent shard files concurrently using the standard DCP reader."""
    def __init__(self, path, workers):
        super().__init__(path)
        self.workers = workers

    def read_data(self, plan, planner):
        groups = {}
        for item in plan.items:
            filename = self.storage_data[item.storage_index].relative_path
            groups.setdefault(filename, []).append(item)
        def read(items):
            return super(ParallelFileReader, self).read_data(
                replace(plan, items=items), planner).wait()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for index, _ in enumerate(pool.map(read, groups.values()), 1):
                if index % 8 == 0:
                    print(f'Loaded {index}/{len(groups)} shard files', flush=True)
        result = torch.futures.Future()
        result.set_result(None)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='Checkpoint directory or its weights directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('--read-workers', type=int, default=8)
    args = parser.parse_args()
    if args.read_workers < 1:
        parser.error("--read-workers must be positive")
    source = args.checkpoint
    if (source / 'weights' / '.metadata').is_file():
        source = source / 'weights'
    if not (source / '.metadata').is_file():
        parser.error('The input must contain a distributed checkpoint .metadata file')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    # safetensors >= 0.8 serializes CPU tensor pointers without a full-model bytes copy.
    import safetensors
    from packaging.version import Version
    if Version(safetensors.__version__) < Version('0.8.0'):
        parser.error('Install safetensors>=0.8.0 for memory-efficient serialization')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    started = time.monotonic()
    state = {}
    print(f'Loading model weights from {source}', flush=True)
    _load_state_dict(state, storage_reader=ParallelFileReader(source, args.read_workers),
                     planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    model = state.get('model', state)
    if not model or any(not isinstance(t, torch.Tensor) for t in model.values()):
        raise ValueError('Expected a flat model tensor dictionary, optionally nested under model')
    size = sum(t.numel() * t.element_size() for t in model.values())
    print(f'Loaded {len(model)} tensors, {size / 2**30:.2f} GiB; preserving dtypes', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    save_file(model, temporary, metadata={'format': 'pt'})
    print('Checking all tensor names, shapes, dtypes and exact values', flush=True)
    with safe_open(temporary, framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(model):
            raise RuntimeError('Tensor names changed during conversion')
        for index, (name, expected) in enumerate(model.items(), 1):
            actual = saved.get_tensor(name)
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f'Tensor metadata mismatch: {name}')
            if not torch.equal(actual, expected):
                raise RuntimeError(f'Tensor values differ or contain NaN: {name}')
            del actual
            if index % 128 == 0:
                print(f'Verified {index}/{len(model)} tensors', flush=True)
    os.replace(temporary, args.output)
    print(f'COMPLETE: {args.output} ({time.monotonic() - started:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
