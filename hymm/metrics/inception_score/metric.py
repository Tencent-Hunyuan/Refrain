import numpy as np
import torch

from ..module.base_metric import BaseMetric
from hymm.metrics.module.inception import InceptionV3


def calculate_inception_score(feature, splits=10, shuffle=True, rng_seed=2024):
    assert torch.is_tensor(feature) and feature.dim() == 2
    B = feature.shape[0]
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(B), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * B // splits) : ((i + 1) * B // splits), :]
        log_p_chunk = log_p[(i * B // splits) : ((i + 1) * B // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return float(np.mean(scores)), float(np.std(scores))


class InceptionScoreMetric(BaseMetric):
    def __init__(self, dims, inception_path, dataset_name="COCO", max_size=256):
        super().__init__()
        self.inception_path = inception_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self._inception = None
        self.dataset_name = dataset_name
        self.max_size = max_size
        self.results = []

    def load_model(self):
        self._inception = InceptionV3(
            [self.block_idx], inception_score=True, inception_path=self.inception_path).to(self.device)
        self._inception.eval()

    def release_model(self):
        self._inception = None

    @property
    def inception(self):
        if self._inception is None:
            self.load_model()
        return self._inception

    # b x c x h x w
    @torch.no_grad()
    def process(self, images, **kwargs):
        with torch.no_grad():
            # b x 1008
            prediction = self.inception(images)[0]
        prediction = prediction.cpu().numpy()
        self.results.append(prediction)

    def compute_metrics(self, results):
        # n x 1008
        predictions = np.concatenate(results, axis=0)
        count = predictions.shape[0]
        print("InceptionScoreMetric: predictions.shape is", predictions.shape)
        inception_score_mean, inception_score_std = calculate_inception_score(torch.from_numpy(predictions))
        return [float(inception_score_mean), float(inception_score_std)], int(count)
