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


def compute_classification_metrics(y_true, y_pred, class_names, confidences=None):
    y_true = [int(value) for value in y_true]
    y_pred = [int(value) for value in y_pred]
    class_names = list(class_names)
    num_classes = len(class_names)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for target, prediction in zip(y_true, y_pred):
        confusion[target][prediction] += 1

    per_class = {}
    total_correct = 0
    total_support = 0
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0

    for idx, class_name in enumerate(class_names):
        tp = confusion[idx][idx]
        fp = sum(confusion[row][idx] for row in range(num_classes) if row != idx)
        fn = sum(confusion[idx][col] for col in range(num_classes) if col != idx)
        support = sum(confusion[idx])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
        }
        total_correct += tp
        total_support += support
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    metrics = {
        'accuracy': total_correct / total_support if total_support else 0.0,
        'per_class': per_class,
        'macro_f1': macro_f1_sum / num_classes if num_classes else 0.0,
        'weighted_f1': weighted_f1_sum / total_support if total_support else 0.0,
        'confusion_matrix': confusion,
        'num_samples': total_support,
        'classes': class_names,
    }
    if confidences is not None:
        confidences = [float(value) for value in confidences]
        metrics['confidence_mean'] = sum(confidences) / len(confidences) if confidences else 0.0

    return metrics
