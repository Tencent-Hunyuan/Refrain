from typing import List, Union, Tuple, Dict

import torch
import torch.distributed as dist


class BaseMetric(object):
    dataset_name: str
    results: list

    def load_model(self):
        raise NotImplementedError

    def release_model(self):
        raise NotImplementedError

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def all_gather_results(self):
        if not dist.is_available() or not dist.is_initialized():
            return self.results
        else:
            gather_results_list = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(gather_results_list, self.results)
            gather_results = []
            for results in gather_results_list:
                gather_results.extend(results)
            return gather_results

    def compute_metrics(self, results) -> Union[Tuple[float, int], Dict[str, Tuple[float, int]]]:
        """
        Calculate average score of all results.

        Args:
            results (list): list of results

        Returns:
            score (tuple or dict): average score
        """
        raise NotImplementedError

    def reset(self):
        self.results = []
