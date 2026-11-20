"""Fixed Refrain Benchmark prompt availability, independent of media and timing."""
import json
from functools import lru_cache
from pathlib import Path
import re

POLICY_PATH = Path(__file__).resolve().parents[1] / 'configs/prompt_groups.json'
SPAN = re.compile(r'\[(\d+)-(\d+)\]\s*')


def add_prompt_arguments(parser):
    parser.add_argument('--prompt-policy', choices=['control-only', 'described', 'prepared'],
                        default='control-only', help='described restores prose on the fixed 80+80 requests')
    parser.add_argument('--prompt-subset', choices=['all', 'described160', 'control80'],
                        default='all', help='Select the published fixed prompt-availability groups')


def add_reference_arguments(parser):
    parser.add_argument('--reference-input', choices=['rgb', 'native-latent'], default='rgb',
                        help='Use retained MP4/PNG or the exact original cached video/anchor NPY tensors')


@lru_cache(maxsize=1)
def load_policy():
    data = json.loads(POLICY_PATH.read_text())
    described, omitted = set(data['described_case_ids']), set(data['control_only_case_ids'])
    if len(described) != 160 or len(omitted) != 80 or described & omitted:
        raise ValueError('Malformed fixed Refrain Benchmark prompt policy')
    rows = {r['case']: r for r in data['cases']}
    if set(rows) != described or len(data['cases']) != 160:
        raise ValueError('Missing/duplicate original prompt descriptions')
    for kind in ('v2v', 'tiv2v'):
        if sum(r['task_type'] == kind for r in rows.values()) != 80:
            raise ValueError(f'Expected 80 described requests for {kind}')
    return data


def select_prompt_subset(cases, subset):
    if subset == 'all':
        return cases
    data = load_policy()
    field = {'described160': 'described_case_ids', 'control80': 'control_only_case_ids'}[subset]
    wanted = set(data[field])
    if not wanted <= {c['index'] for c in cases}:
        raise ValueError(f'Benchmark is missing cases in fixed {subset} subset')
    return [c for c in cases if c['index'] in wanted]


def restore_reference_descriptions(case, prompt):
    """Insert prose before existing hints; never rebuild non-reference shots."""
    from hymm.utils.physrope_prompt_rewrite import REFERENCE_VIDEO_SEGMENT_HINT
    from hymm.utils.prompt_cleaning import strip_identity_tags
    data = load_policy()
    if case['index'] not in data['benchmark_case_ids']:
        raise ValueError('described requires a case from the fixed Refrain Benchmark benchmark')
    rows = {r['case']: r for r in data['cases']}
    if case['index'] not in rows:
        return prompt, []
    row = rows[case['index']]
    matches = list(SPAN.finditer(prompt))
    if row['task_type'] != case['task_type'] or row['shot_spans'] != [
            [int(m[1]), int(m[2])] for m in matches]:
        raise ValueError(f'Original prompt identity/timing differs: {case["index"]}')
    segments = (case.get('conditions') or {}).get('segments', [])
    if {str(s['seg_id']) for s in segments} != set(row['descriptions_by_segment']):
        raise ValueError(f'Reference shot identity differs: {case["index"]}')
    # Apply backwards so offsets into the effective control-only prompt stay valid.
    for key, description in sorted(row['descriptions_by_segment'].items(),
                                   key=lambda item: int(item[0]), reverse=True):
        index = int(key) - 1
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        body = prompt[start:end]
        if body.count(REFERENCE_VIDEO_SEGMENT_HINT) != 1:
            raise ValueError(f'Missing/duplicate reference hint: {case["index"]} shot {key}')
        body = body.replace(REFERENCE_VIDEO_SEGMENT_HINT,
                            strip_identity_tags(description).strip() + ' ' + REFERENCE_VIDEO_SEGMENT_HINT, 1)
        prompt = prompt[:start] + body + prompt[end:]
    return prompt, sorted(map(int, row['descriptions_by_segment']))
