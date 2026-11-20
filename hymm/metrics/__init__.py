"""Load optional metric dependencies only when their metric is requested."""
from importlib import import_module

_MODULES = {
    'FIDMetric': '.fid.metric',
    'InceptionScoreMetric': '.inception_score.metric',
    'CLIPScoreMetric': '.clip_score.metric',
    'HPSv2Metric': '.hpsv2.metric',
    'T2ICompBenchMetric': '.t2i_compbench.metric',
    'GeneralGenerationBenchMetric': '.general_generation_bench.metric',
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
