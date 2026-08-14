import torch.nn as nn
from torchvision import models

from base_tool.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TorchvisionClassifier(nn.Module):
    def __init__(self, architecture='resnet18', num_classes=2, weights=None, freeze_backbone=False, dropout=0.0):
        super(TorchvisionClassifier, self).__init__()
        self.model = models.get_model(architecture, weights=weights)
        self._replace_classifier(num_classes, dropout=float(dropout))

        if freeze_backbone:
            self._freeze_backbone()

    def _replace_classifier(self, num_classes, dropout=0.0):
        head = lambda in_f: nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes)) if dropout > 0 else nn.Linear(in_f, num_classes)

        if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            self.model.fc = head(in_features)
            return

        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                self.model.classifier = head(in_features)
                return
            if isinstance(classifier, nn.Sequential):
                for idx in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[idx], nn.Linear):
                        in_features = classifier[idx].in_features
                        classifier[idx] = head(in_features)
                        return

        raise ValueError('Unsupported torchvision architecture classifier head.')

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        classifier = getattr(self.model, 'fc', None) or getattr(self.model, 'classifier', None)
        if classifier is None:
            raise ValueError('Cannot freeze backbone because classifier head was not found.')
        for param in classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
