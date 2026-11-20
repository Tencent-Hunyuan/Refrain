#!/usr/bin/env python3
"""Install and validate the released Refrain Benchmark requests against extracted inputs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'benchmark')
    args = parser.parse_args()
    benchmark = json.loads((ROOT / 'configs/benchmark.json').read_text())
    policy = json.loads((ROOT / 'configs/prompt_groups.json').read_text())
    cases = benchmark['cases']
    assert len(cases) == len({c['index'] for c in cases}) == 300
    missing = []
    for case in cases:
        conditions = case.get('conditions') or {}
        for reference in conditions.get('segments', []) + conditions.get('anchors', []):
            for field in ('path', 'latent_path'):
                path = Path(reference[field])
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError(f'Nonportable reference: {path}')
                if not (args.input_root / path).is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'Extract benchmark.zip into the repository root: {len(missing)} missing inputs; first: {missing[:5]}')
    for kind in ('v2v', 'tiv2v'):
        ids = {c['index'] for c in cases if c['task_type'] == kind}
        assert len(ids & set(policy['described_case_ids'])) == 80
        assert len(ids & set(policy['control_only_case_ids'])) == 40
    output = args.input_root / 'requests.json'
    payload = json.dumps(benchmark, ensure_ascii=False, indent=2) + '\n'
    tmp = output.with_suffix('.tmp')
    tmp.write_text(payload)
    tmp.replace(output)
    print(f'Validated 300 requests: V2V 80/40, TIV2V 80/40; wrote {output}')


if __name__ == '__main__':
    main()
