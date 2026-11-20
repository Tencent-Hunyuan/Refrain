"""Distributed samplers with resumable per-rank offsets and resolution batches."""
import math
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class DistributedSamplerWithStartIndex(DistributedSampler):
    """Resume after a per-rank count; shuffle complete epochs before slicing."""
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0,
                 drop_last=False, start_index=0, use_numpy_indices=False, batch_size=1):
        if num_replicas is None and not dist.is_initialized(): num_replicas = 1
        if rank is None and not dist.is_initialized(): rank = 0
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = int(batch_size)
        if self.batch_size < 1: raise ValueError("batch_size must be positive")
        self.start_index = start_index

    @property
    def start_index(self): return self._start_index
    @start_index.setter
    def start_index(self, value):
        value = int(value)
        if value < 0: raise ValueError("start_index must be nonnegative")
        self._start_index = value

    def _indices(self):
        if self.batch_size == 1:
            return list(super().__iter__())
        # Bucket indexes store contiguous batches of the same resolution.
        n = len(self.dataset)
        batches = [list(range(i, min(i + self.batch_size, n))) for i in range(0, n, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size: batches.pop()
        if not batches: return []
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        if self.drop_last:
            batches = batches[:len(batches) // self.num_replicas * self.num_replicas]
        else:
            total = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
            batches = (batches * math.ceil(total / len(batches)))[:total]
        return [i for b in batches[self.rank::self.num_replicas] for i in b]

    def __iter__(self): return iter(self._indices()[self.start_index:])
    def __len__(self):
        if self.batch_size == 1:
            return max(0, super().__len__() - self.start_index)
        return max(0, len(self._indices()) - self.start_index)


class BlockDistributedSampler(DistributedSamplerWithStartIndex):
    """Assign contiguous blocks from an index already shuffled by its dataset."""
    def _indices(self):
        n = len(self.dataset)
        block = n // (self.num_replicas * self.batch_size) * self.batch_size
        return list(range(self.rank * block, (self.rank + 1) * block))
    def __len__(self): return max(0, len(self._indices()) - self.start_index)


class IndexBatchSampler(Sampler):
    """Batch an Arrow index without mixing incompatible target resolutions."""
    def __init__(self, index_manager, sampler, batch_size, drop_last=False, multireso=False):
        self.index_manager, self.sampler = index_manager, sampler
        self.batch_size, self.drop_last = batch_size, drop_last
        self.multireso = multireso

    def __iter__(self):
        pending = OrderedDict()
        for index in self.sampler:
            size = tuple(self.index_manager.get_target_size(index)) if hasattr(self.index_manager, "get_target_size") else None
            if callable(self.batch_size) and size is None:
                raise ValueError("Dynamic batches require target resolution metadata.")
            limit = int(self.batch_size(size) if callable(self.batch_size) else self.batch_size)
            if limit < 1: raise ValueError("batch_size must be positive")
            batch = pending.setdefault(size, [])
            batch.append(index)
            if len(batch) == limit:
                yield list(batch)
                batch.clear()
        if not self.drop_last:
            yield from (batch for batch in pending.values() if batch)

    def __len__(self): return sum(1 for _ in self.__iter__())
