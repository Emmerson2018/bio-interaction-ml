import torch
from base_tool.metrics.base_metric import BaseMetric
from base_tool.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class MSEMetric(BaseMetric):
    def __init__(self):
        super(MSEMetric, self).__init__()

    def __call__(self, prediction, target):
        return torch.nn.functional.mse_loss(prediction, target).item()

@METRIC_REGISTRY.register()
class MAEMetric(BaseMetric):
    def __init__(self):
        super(MAEMetric, self).__init__()

    def __call__(self, prediction, target):
        return torch.nn.functional.l1_loss(prediction, target).item()
