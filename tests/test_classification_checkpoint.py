import torch

from base_tool.models.classification_model import ClassificationModel


def _opt(tmp_path, freeze_backbone=False):
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return {
        'name': 'test',
        'is_train': True,
        'path': {'experiments_root': str(tmp_path)},
        'model_type': 'ClassificationModel',
        'network_g': {
            'type': 'TorchvisionClassifier',
            'backbone': 'resnet18',
            'num_classes': 2,
            'weights': None,
            'freeze_backbone': freeze_backbone,
            'preprocessing': {'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        },
        'train': {
            'seed': 42,
            'deterministic': False,
            'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.0},
            'scheduler': {'type': 'StepLR', 'step_size': 1, 'gamma': 0.5},
        },
    }


def test_checkpoint_saves_classes_and_preprocessing(tmp_path):
    model = ClassificationModel(_opt(tmp_path))
    model.set_dataset_metadata(
        {
            'class_to_idx': {'capivara': 0, 'sapo': 1},
            'idx_to_class': {'0': 'capivara', '1': 'sapo'},
            'preprocessing': {'input_size': 64, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        }
    )

    model.save(epoch=0, current_iter=0, filename='test.pth', metric_results={'accuracy': 1.0})

    checkpoint = torch.load(tmp_path / 'models' / 'test.pth', map_location='cpu')
    assert checkpoint['class_to_idx'] == {'capivara': 0, 'sapo': 1}
    assert checkpoint['idx_to_class'] == {'0': 'capivara', '1': 'sapo'}
    assert checkpoint['preprocessing']['input_size'] == 64
    assert checkpoint['metrics'] == {'accuracy': 1.0}
    assert checkpoint['training_metadata']['seed'] == 42


def test_optimizer_keeps_all_parameters_when_backbone_is_frozen(tmp_path):
    model = ClassificationModel(_opt(tmp_path, freeze_backbone=True))
    total_params = sum(1 for _ in model.net.parameters())
    optimizer_params = sum(len(group['params']) for group in model.optimizer_g.param_groups)

    assert optimizer_params == total_params
    assert any(not param.requires_grad for param in model.net.parameters())

    model.net.unfreeze_backbone()

    assert all(param.requires_grad for param in model.net.parameters())
