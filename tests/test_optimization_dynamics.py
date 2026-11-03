import torch
from base_tool.metrics.classification_metrics import AccuracyMetric, BalancedAccuracyMetric, MacroF1Metric
from base_tool.models.classification_model import ClassificationModel


def test_metrics_evaluation():
    # Simula predições para 15 classes
    pred = torch.randn(30, 15)
    target = torch.randint(0, 15, (30,))

    acc = AccuracyMetric()(pred, target)
    f1 = MacroF1Metric()(pred, target)
    bal_acc = BalancedAccuracyMetric()(pred, target)

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= bal_acc <= 1.0
    print("Metrics test passed!")


def test_two_stage_and_differential_lr():
    opt = {
        'is_train': True,
        'network_g': {
            'type': 'TorchvisionClassifier',
            'architecture': 'resnet18',
            'num_classes': 15,
            'weights': None,
            'freeze_backbone': False,
            'dropout': 0.1,
        },
        'train': {
            'total_epochs': 10,
            'two_stage': {
                'enabled': True,
                'freeze_epochs': 2,
                'head_lr': 1e-3,
            },
            'differential_lr': {
                'enabled': True,
                'backbone_shallow_lr': 1e-5,
                'backbone_deep_lr': 5e-5,
                'head_lr': 5e-4,
            },
            'weight_decay': 1e-2,
            'scheduler': {
                'type': 'CosineAnnealingWarmup',
                'warmup_epochs': 2,
                'min_lr': 1e-6,
            },
            'regularization': {
                'label_smoothing': 0.1,
                'mixup_cutmix': {
                    'enabled': True,
                    'mixup_alpha': 0.8,
                    'cutmix_alpha': 1.0,
                },
            },
        },
    }

    model = ClassificationModel(opt)

    # 1. Verificar Estágio 1 (Backbone Congelado)
    for name, param in model.net.named_parameters():
        if 'fc' in name:
            assert param.requires_grad is True, f"Head parameter {name} should be trainable in stage 1"
        else:
            assert param.requires_grad is False, f"Backbone parameter {name} should be frozen in stage 1"

    # Verificar que o otimizador tem apenas 1 grupo no estágio 1 e inicia com o warmup factor de 0.1
    assert len(model.optimizer_g.param_groups) == 1
    assert abs(model.optimizer_g.param_groups[0]['lr'] - 1e-4) < 1e-6  # 1e-3 * 0.1 warmup

    # Simular passo de treino
    dummy_x = torch.randn(4, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 3])
    model.feed_data({'x': dummy_x, 'y': dummy_y})
    model.optimize_parameters(current_iter=1)

    # 2. Simular avanço de épocas para acionar a transição de estágio
    model.update_learning_rate(current_iter=1)  # epoch 1
    model.update_learning_rate(current_iter=2)  # epoch 2 (atinge freeze_epochs=2 -> unfreeze!)

    # 3. Verificar Estágio 2 (Backbone Descongelado e 3 Grupos de LR)
    for name, param in model.net.named_parameters():
        assert param.requires_grad is True, f"Parameter {name} should be trainable in stage 2"

    assert len(model.optimizer_g.param_groups) == 3
    shallow_lr = model.optimizer_g.param_groups[0]['lr']
    deep_lr = model.optimizer_g.param_groups[1]['lr']
    head_lr = model.optimizer_g.param_groups[2]['lr']

    assert shallow_lr == 1e-5
    assert deep_lr == 5e-5
    assert head_lr == 5e-4
    assert shallow_lr < deep_lr < head_lr

    print("Two-stage and differential learning rate test passed!")


if __name__ == '__main__':
    test_metrics_evaluation()
    test_two_stage_and_differential_lr()
