"""Public dense Refrain training engine."""
from .parallel_engine import BaseParallelEngine


class RefrainEngine(BaseParallelEngine):
    """DP/FSDP + sequence parallelism for the three published training stages."""
