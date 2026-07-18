import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from base_tool.archs.image_classifier import SUPPORTED_BACKBONES, TorchvisionClassifier


def _last_linear(module):
    linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    return linear_layers[-1]


@pytest.mark.parametrize('backbone', sorted(SUPPORTED_BACKBONES))
def test_create_supported_backbones_forward_shape(backbone):
    model = TorchvisionClassifier(
        backbone=backbone,
        num_classes=3,
        weights=None,
        preprocessing={'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    )
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 3)


@pytest.mark.parametrize('backbone', sorted(SUPPORTED_BACKBONES))
def test_final_classifier_layer_uses_num_classes(backbone):
    model = TorchvisionClassifier(
        backbone=backbone,
        num_classes=5,
        weights=None,
        classifier_dropout=0.3,
        preprocessing={'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    )

    classifier = getattr(model.model, 'fc', None) or getattr(model.model, 'classifier')
    assert _last_linear(classifier).out_features == 5
    dropout_layers = [layer for layer in classifier.modules() if isinstance(layer, nn.Dropout)]
    assert dropout_layers
    assert all(layer.p == pytest.approx(0.3) for layer in dropout_layers)


def test_weights_default_resolves_preprocessing_without_download():
    fake_transform = SimpleNamespace(
        crop_size=192,
        resize_size=232,
        mean=[0.1, 0.2, 0.3],
        std=[0.4, 0.5, 0.6],
        interpolation='bilinear',
    )
    fake_weight = SimpleNamespace(name='DEFAULT', transforms=lambda: fake_transform)
    fake_weights_enum = SimpleNamespace(DEFAULT=fake_weight)

    fake_model = nn.Sequential()
    fake_model.classifier = nn.Linear(4, 1000)

    with patch('base_tool.archs.image_classifier.models.get_model_weights', return_value=fake_weights_enum), patch(
        'base_tool.archs.image_classifier.models.get_model', return_value=fake_model
    ) as get_model:
        model = TorchvisionClassifier(backbone='resnet18', num_classes=7, weights='DEFAULT')

    get_model.assert_called_once_with('resnet18', weights=fake_weight)
    assert model.preprocessing['input_size'] == 192
    assert model.preprocessing['resize_size'] == 232
    assert model.preprocessing['mean'] == [0.1, 0.2, 0.3]
    assert _last_linear(model.model.classifier).out_features == 7


def test_weights_null_uses_explicit_yaml_preprocessing():
    model = TorchvisionClassifier(
        backbone='resnet18',
        num_classes=2,
        weights=None,
        preprocessing={'input_size': 128, 'mean': [0.2, 0.3, 0.4], 'std': [0.5, 0.6, 0.7]},
    )

    assert model.preprocessing['input_size'] == 128
    assert model.preprocessing['mean'] == [0.2, 0.3, 0.4]
    assert model.preprocessing['std'] == [0.5, 0.6, 0.7]


def test_freeze_and_unfreeze_backbone():
    model = TorchvisionClassifier(
        backbone='resnet18',
        num_classes=2,
        weights=None,
        freeze_backbone=True,
        preprocessing={'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
    )

    classifier = getattr(model.model, 'fc')
    assert any(param.requires_grad for param in classifier.parameters())
    assert any(not param.requires_grad for name, param in model.model.named_parameters() if not name.startswith('fc'))
    batch_norm_layers = [
        module for module in model.model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    model.train()
    assert batch_norm_layers
    assert all(not module.training for module in batch_norm_layers)

    model.unfreeze_backbone()

    assert all(param.requires_grad for param in model.parameters())


def test_legacy_architecture_warns_and_conflict_fails():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = TorchvisionClassifier(
            architecture='resnet18',
            num_classes=2,
            weights=None,
            preprocessing={'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        )

    assert model.backbone == 'resnet18'
    assert any(item.category is DeprecationWarning for item in caught)

    with pytest.raises(ValueError, match='Conflicting backbone configuration'):
        TorchvisionClassifier(backbone='resnet18', architecture='efficientnet_b0', num_classes=2)
