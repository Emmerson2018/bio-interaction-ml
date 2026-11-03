import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from base_tool.metrics.base_metric import BaseMetric
from base_tool.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
class AccuracyMetric(BaseMetric):
    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def __call__(self, prediction, target):
        predicted_class = torch.argmax(prediction, dim=1)
        return (predicted_class == target.long()).float().mean().item()


@METRIC_REGISTRY.register()
class MacroF1Metric(BaseMetric):
    def __init__(self):
        super(MacroF1Metric, self).__init__()

    def __call__(self, prediction, target):
        pred = torch.argmax(prediction, dim=1).detach().cpu().numpy()
        tgt = target.long().detach().cpu().numpy()
        return float(f1_score(tgt, pred, average='macro', zero_division=0))


@METRIC_REGISTRY.register()
class BalancedAccuracyMetric(BaseMetric):
    def __init__(self):
        super(BalancedAccuracyMetric, self).__init__()

    def __call__(self, prediction, target):
        pred = torch.argmax(prediction, dim=1).detach().cpu().numpy()
        tgt = target.long().detach().cpu().numpy()
        return float(balanced_accuracy_score(tgt, pred))
