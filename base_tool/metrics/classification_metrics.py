import torch

from base_tool.metrics.base_metric import BaseMetric
from base_tool.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class AccuracyMetric(BaseMetric):
    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def __call__(self, prediction, target):
        predicted_class = torch.argmax(prediction, dim=1)
        return (predicted_class == target.long()).float().mean().item()
