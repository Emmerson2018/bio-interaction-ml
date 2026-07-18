import warnings

import torch.nn as nn
from torchvision import models

from base_tool.utils.registry import ARCH_REGISTRY


SUPPORTED_BACKBONES = {
    'resnet18',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'efficientnet_b0',
}

LEGACY_IMAGENET_PREPROCESSING = {
    'input_size': 224,
    'resize_size': 224,
    'crop_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'interpolation': 'bilinear',
    'color_order': 'RGB',
    'source': 'legacy_imagenet_default',
}


def normalize_weights(weights):
    if weights == '':
        return None
    return weights


def resolve_backbone(backbone=None, architecture=None):
    if backbone and architecture and backbone != architecture:
        raise ValueError(
            f'Conflicting backbone configuration: backbone={backbone!r} '
            f'and deprecated architecture={architecture!r}. Use only backbone.'
        )
    if backbone is None and architecture is not None:
        warnings.warn(
            'network_g.architecture is deprecated; use network_g.backbone instead.',
            DeprecationWarning,
            stacklevel=3,
        )
        backbone = architecture
    return backbone or 'resnet18'


def resolve_weights_enum(backbone, weights):
    weights = normalize_weights(weights)
    if weights is None:
        return None
    weights_enum = models.get_model_weights(backbone)
    if isinstance(weights, str):
        if weights == 'DEFAULT':
            return weights_enum.DEFAULT
        try:
            return weights_enum[weights]
        except KeyError as exc:
            raise ValueError(f'Unsupported weights value {weights!r} for {backbone}.') from exc
    return weights


def _as_size(value, default=224):
    if isinstance(value, (list, tuple)):
        return int(value[0])
    if value is None:
        return int(default)
    return int(value)


def resolve_torchvision_preprocessing(backbone, weights=None, preprocessing=None):
    preprocessing = dict(preprocessing or {})
    resolved_weights = resolve_weights_enum(backbone, weights)
    if resolved_weights is not None:
        transforms = resolved_weights.transforms()
        crop_size = _as_size(getattr(transforms, 'crop_size', 224))
        resize_size = _as_size(getattr(transforms, 'resize_size', crop_size), default=crop_size)
        defaults = {
            'input_size': crop_size,
            'resize_size': resize_size,
            'crop_size': crop_size,
            'mean': list(getattr(transforms, 'mean', LEGACY_IMAGENET_PREPROCESSING['mean'])),
            'std': list(getattr(transforms, 'std', LEGACY_IMAGENET_PREPROCESSING['std'])),
            'interpolation': str(getattr(transforms, 'interpolation', 'bilinear')).split('.')[-1].lower(),
            'color_order': 'RGB',
            'source': f'torchvision_weights:{backbone}:{resolved_weights.name}',
        }
    else:
        defaults = dict(LEGACY_IMAGENET_PREPROCESSING)
        defaults['source'] = 'yaml_or_legacy_weights_null'

    defaults.update(preprocessing)
    defaults['input_size'] = _as_size(defaults.get('input_size', defaults.get('crop_size', 224)))
    defaults['resize_size'] = _as_size(defaults.get('resize_size', defaults['input_size']))
    defaults['crop_size'] = _as_size(defaults.get('crop_size', defaults['input_size']))
    defaults['mean'] = [float(value) for value in defaults['mean']]
    defaults['std'] = [float(value) for value in defaults['std']]
    return defaults


@ARCH_REGISTRY.register()
class TorchvisionClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        architecture=None,
        num_classes=2,
        weights=None,
        classifier_dropout=None,
        freeze_backbone=False,
        preprocessing=None,
    ):
        super(TorchvisionClassifier, self).__init__()
        self.backbone = resolve_backbone(backbone, architecture)
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f'Unsupported backbone: {self.backbone}. '
                f'Supported backbones: {sorted(SUPPORTED_BACKBONES)}'
            )

        self.num_classes = int(num_classes)
        self.weights = normalize_weights(weights)
        self.classifier_dropout = 0.0 if classifier_dropout is None else float(classifier_dropout)
        self.preprocessing = resolve_torchvision_preprocessing(self.backbone, self.weights, preprocessing)
        self.model = models.get_model(self.backbone, weights=resolve_weights_enum(self.backbone, self.weights))
        self._replace_classifier(self.num_classes)
        self.backbone_frozen = False
        self.freeze_batchnorm = bool(freeze_backbone)
        if freeze_backbone:
            self.freeze_backbone()

    def _replace_classifier(self, num_classes):
        if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
            in_features = self.model.fc.in_features
            self.model.fc = self._build_classifier(in_features, num_classes)
            return self.model.fc

        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                in_features = classifier.in_features
                self.model.classifier = self._build_classifier(in_features, num_classes)
                return self.model.classifier
            if isinstance(classifier, nn.Sequential):
                for idx in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[idx], nn.Linear):
                        in_features = classifier[idx].in_features
                        classifier[idx] = nn.Linear(in_features, num_classes)
                        self._configure_or_insert_dropout(classifier, idx)
                        return self.model.classifier

        raise ValueError('Unsupported torchvision architecture classifier head.')

    def _build_classifier(self, in_features, num_classes):
        linear = nn.Linear(in_features, num_classes)
        if self.classifier_dropout > 0:
            return nn.Sequential(nn.Dropout(p=self.classifier_dropout), linear)
        return linear

    def _configure_or_insert_dropout(self, classifier, linear_idx):
        dropout_indices = [idx for idx, module in enumerate(classifier) if isinstance(module, nn.Dropout)]
        if dropout_indices:
            for idx in dropout_indices:
                classifier[idx].p = self.classifier_dropout
            return
        if self.classifier_dropout <= 0:
            return
        modules = list(classifier.children())
        modules.insert(linear_idx, nn.Dropout(p=self.classifier_dropout))
        self.model.classifier = nn.Sequential(*modules)

    def _classifier_module(self):
        classifier = getattr(self.model, 'fc', None) or getattr(self.model, 'classifier', None)
        if classifier is None:
            raise ValueError('Classifier head was not found.')
        return classifier

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self._classifier_module().parameters():
            param.requires_grad = True
        self.backbone_frozen = True
        self.freeze_batchnorm = True
        self._set_backbone_batchnorm_eval()

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        self.freeze_batchnorm = False

    def _set_backbone_batchnorm_eval(self):
        classifier_prefixes = ('fc', 'classifier')
        for name, module in self.model.named_modules():
            if name.startswith(classifier_prefixes):
                continue
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode=True):
        super(TorchvisionClassifier, self).train(mode)
        if mode and self.freeze_batchnorm:
            self._set_backbone_batchnorm_eval()
        return self

    def get_metadata(self):
        return {
            'type': self.__class__.__name__,
            'backbone': self.backbone,
            'num_classes': self.num_classes,
            'weights': self.weights,
            'classifier_dropout': self.classifier_dropout,
            'preprocessing': dict(self.preprocessing),
            'backbone_frozen': self.backbone_frozen,
            'freeze_batchnorm': self.freeze_batchnorm,
        }

    def forward(self, x):
        return self.model(x)
