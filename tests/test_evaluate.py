import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

import base_tool.evaluate as evaluate_module
from base_tool.evaluate import _compute_metrics, evaluate


class ThresholdModel(nn.Module):
    def load_state_dict(self, state_dict, strict=True):
        return None

    def forward(self, x):
        means = x.mean(dim=(1, 2, 3))
        logits = torch.stack([1.0 - means, means], dim=1) * 10.0
        return logits


def _write_image(path, value):
    Image.new('RGB', (16, 16), color=(value, value, value)).save(path)


def _write_opt(path, dataset_root, num_classes=2):
    opt = {
        'name': 'Eval_Test',
        'model_type': 'ClassificationModel',
        'datasets': {
            'train': {'type': 'ImageFolderClassificationDataset', 'root': str(dataset_root / 'train')},
            'val': {'type': 'ImageFolderClassificationDataset', 'root': str(dataset_root / 'val')},
            'test': {'type': 'ImageFolderClassificationDataset', 'root': str(dataset_root / 'test')},
        },
        'network_g': {
            'type': 'TorchvisionClassifier',
            'backbone': 'resnet18',
            'num_classes': num_classes,
            'weights': None,
        },
    }
    path.write_text(yaml.safe_dump(opt), encoding='utf-8')


def _make_split(root, split='test'):
    capivara = root / split / 'capivara'
    sapo = root / split / 'sapo'
    capivara.mkdir(parents=True)
    sapo.mkdir(parents=True)
    _write_image(capivara / 'capivara.png', 0)
    _write_image(sapo / 'sapo.png', 255)


def _new_checkpoint(path, class_to_idx=None, preprocessing=None):
    class_to_idx = class_to_idx or {'capivara': 0, 'sapo': 1}
    idx_to_class = {str(idx): name for name, idx in class_to_idx.items()}
    torch.save(
        {
            'network': {},
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'preprocessing': preprocessing or {'input_size': 16, 'mean': [0, 0, 0], 'std': [1, 1, 1]},
        },
        path,
    )


def test_metrics_perfect():
    metrics = _compute_metrics([0, 1], [0, 1], [0.9, 0.8], ['capivara', 'sapo'])

    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['macro_f1'] == pytest.approx(1.0)
    assert metrics['weighted_f1'] == pytest.approx(1.0)
    assert metrics['per_class']['capivara']['precision'] == pytest.approx(1.0)
    assert metrics['per_class']['sapo']['recall'] == pytest.approx(1.0)


def test_metrics_with_errors():
    metrics = _compute_metrics([0, 0, 1, 1], [0, 1, 1, 0], [0.9, 0.7, 0.8, 0.6], ['capivara', 'sapo'])

    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['per_class']['capivara']['precision'] == pytest.approx(0.5)
    assert metrics['per_class']['capivara']['recall'] == pytest.approx(0.5)
    assert metrics['macro_f1'] == pytest.approx(0.5)


def test_evaluate_generates_artifacts_and_uses_checkpoint_class_order(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'data'
    _make_split(dataset_root)
    opt_path = tmp_path / 'opt.yml'
    _write_opt(opt_path, dataset_root)
    checkpoint_path = tmp_path / 'checkpoint.pth'
    _new_checkpoint(checkpoint_path, class_to_idx={'capivara': 0, 'sapo': 1})
    monkeypatch.setattr(evaluate_module, 'build_network', lambda opt: ThresholdModel())

    result = evaluate(str(opt_path), str(checkpoint_path), 'test')

    assert result['metrics']['accuracy'] == pytest.approx(1.0)
    assert result['metrics']['classes'] == ['capivara', 'sapo']
    for artifact in result['artifacts'].values():
        assert artifact
        assert Path(artifact).exists()
    with open(result['artifacts']['predictions'], newline='', encoding='utf-8') as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row['target'] for row in rows] == ['capivara', 'sapo']


def test_evaluate_legacy_checkpoint_falls_back_to_split_classes(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'data'
    _make_split(dataset_root)
    opt_path = tmp_path / 'opt.yml'
    _write_opt(opt_path, dataset_root)
    checkpoint_path = tmp_path / 'legacy.pth'
    torch.save({'network': {}}, checkpoint_path)
    monkeypatch.setattr(evaluate_module, 'build_network', lambda opt: ThresholdModel())

    result = evaluate(str(opt_path), str(checkpoint_path), 'test')

    assert result['metrics']['classes'] == ['capivara', 'sapo']
    assert result['metrics']['accuracy'] == pytest.approx(1.0)


def test_evaluate_missing_split_fails(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'data'
    opt_path = tmp_path / 'opt.yml'
    _write_opt(opt_path, dataset_root)
    checkpoint_path = tmp_path / 'checkpoint.pth'
    _new_checkpoint(checkpoint_path)
    monkeypatch.setattr(evaluate_module, 'build_network', lambda opt: ThresholdModel())

    with pytest.raises(FileNotFoundError, match='Split directory not found'):
        evaluate(str(opt_path), str(checkpoint_path), 'test')


def test_evaluate_class_count_mismatch_fails(tmp_path, monkeypatch):
    dataset_root = tmp_path / 'data'
    _make_split(dataset_root)
    opt_path = tmp_path / 'opt.yml'
    _write_opt(opt_path, dataset_root, num_classes=2)
    checkpoint_path = tmp_path / 'checkpoint.pth'
    _new_checkpoint(checkpoint_path, class_to_idx={'capivara': 0, 'sapo': 1, 'unknown': 2})
    monkeypatch.setattr(evaluate_module, 'build_network', lambda opt: ThresholdModel())

    with pytest.raises(ValueError, match='Number of classes mismatch'):
        evaluate(str(opt_path), str(checkpoint_path), 'test')
