import pytest

from base_tool.metrics.classification_metrics import compute_classification_metrics
from base_tool.train import _select_monitored_value, _update_best_state


def test_validation_metrics_perfect_epoch():
    metrics = compute_classification_metrics(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        ['capivara', 'sapo'],
        [0.99, 0.98, 0.97, 0.96],
    )

    assert metrics['accuracy'] == pytest.approx(1.0)
    assert metrics['macro_f1'] == pytest.approx(1.0)
    assert metrics['weighted_f1'] == pytest.approx(1.0)
    assert metrics['per_class']['capivara']['precision'] == pytest.approx(1.0)
    assert metrics['per_class']['sapo']['recall'] == pytest.approx(1.0)
    assert metrics['confidence_mean'] == pytest.approx(0.975)


def test_validation_metrics_with_errors():
    metrics = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        ['capivara', 'sapo'],
        [0.90, 0.60, 0.80, 0.55],
    )

    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['macro_f1'] == pytest.approx(0.5)
    assert metrics['weighted_f1'] == pytest.approx(0.5)
    assert metrics['per_class']['capivara']['precision'] == pytest.approx(0.5)
    assert metrics['per_class']['capivara']['recall'] == pytest.approx(0.5)
    assert metrics['per_class']['capivara']['f1'] == pytest.approx(0.5)
    assert metrics['per_class']['sapo']['f1'] == pytest.approx(0.5)


def test_monitor_selection_supports_generic_and_legacy_keys():
    metrics = {'val_macro_f1': 0.81, 'val_accuracy': 0.75}
    assert _select_monitored_value(metrics, 'val_macro_f1') == pytest.approx(0.81)
    assert _select_monitored_value(metrics, 'macro_f1') == pytest.approx(0.81)
    assert _select_monitored_value(metrics, 'accuracy') == pytest.approx(0.75)


def test_best_checkpoint_selection_uses_monitored_metric():
    best_value, should_save, monitored_value = _update_best_state(
        None,
        {'val_macro_f1': 0.6},
        'val_macro_f1',
        'max',
    )
    assert should_save is True
    assert best_value == pytest.approx(0.6)
    assert monitored_value == pytest.approx(0.6)

    best_value, should_save, monitored_value = _update_best_state(
        best_value,
        {'val_macro_f1': 0.55},
        'val_macro_f1',
        'max',
    )
    assert should_save is False
    assert best_value == pytest.approx(0.6)
    assert monitored_value == pytest.approx(0.55)
