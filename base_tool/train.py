import argparse
import os
import time

import torch

from base_tool.data import build_dataset, build_dataloader
from base_tool.metrics import build_metric
from base_tool.metrics.classification_metrics import compute_classification_metrics
from base_tool.models import build_model
from base_tool.utils.logger import get_env_info, get_root_logger
from base_tool.utils.options import parse_options
from base_tool.utils.seed import set_random_seed
from base_tool.visualization import build_visualizer


def _is_better(value, best_value, mode):
    if best_value is None:
        return True
    if mode == 'max':
        return value > best_value
    if mode == 'min':
        return value < best_value
    raise ValueError(f'Unsupported checkpoint mode: {mode}')


def _select_monitored_value(metric_results, monitor):
    candidates = [monitor]
    if monitor.startswith('val_'):
        candidates.append(monitor[4:])
    else:
        candidates.append(f'val_{monitor}')
    for key in candidates:
        if key in metric_results:
            return metric_results[key]
    return None


def _update_best_state(best_value, metric_results, monitor, mode):
    monitored_value = _select_monitored_value(metric_results, monitor)
    if monitored_value is None:
        return best_value, False, None
    if _is_better(monitored_value, best_value, mode):
        return monitored_value, True, monitored_value
    return best_value, False, monitored_value


def _save_model(model, epoch, current_iter, filename=None, metric_results=None):
    try:
        model.save(epoch, current_iter, filename=filename, metric_results=metric_results)
    except TypeError:
        model.save(epoch, current_iter)


def _class_names_from_metadata(metadata):
    if not metadata:
        return []
    classes = metadata.get('classes')
    if classes:
        return list(classes)
    idx_to_class = metadata.get('idx_to_class') or {}
    if idx_to_class:
        return [idx_to_class[str(idx)] for idx in sorted(int(key) for key in idx_to_class.keys())]
    return []


def _validate_num_classes(opt, train_metadata):
    classes = _class_names_from_metadata(train_metadata)
    if not classes:
        return
    configured = opt.get('network_g', {}).get('num_classes')
    if configured is None:
        opt.setdefault('network_g', {})['num_classes'] = len(classes)
        return
    if int(configured) != len(classes):
        raise ValueError(
            f"network_g.num_classes={configured} diverges from dataset classes={len(classes)}: {classes}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Caminho para o arquivo YAML de opcoes.')
    args = parser.parse_args()

    opt = parse_options(args.opt, is_train=True)
    if 'num_epochs' in opt.get('train', {}) and 'total_epochs' not in opt.get('train', {}):
        opt['train']['total_epochs'] = opt['train']['num_epochs']
    set_random_seed(
        opt.get('train', {}).get('seed'),
        deterministic=opt.get('train', {}).get('deterministic', False),
    )

    log_file = os.path.join(opt['path']['experiments_root'], f"train_{opt['name']}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())

    train_loader = None
    val_loader = None
    train_metadata = {}
    val_metadata = {}
    for phase, dataset_opt in opt['datasets'].items():
        dataset_opt.setdefault('network_g', opt.get('network_g', {}))
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            if hasattr(train_set, 'get_metadata'):
                train_metadata = train_set.get_metadata()
            train_loader = build_dataloader(train_set, opt, phase='train')
            logger.info(f"Dataset de treino [{train_set.__class__.__name__}] criado. Total: {len(train_set)}")
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            if hasattr(val_set, 'get_metadata'):
                val_metadata = val_set.get_metadata()
            val_loader = build_dataloader(val_set, opt, phase='val')
            logger.info(f"Dataset de validacao [{val_set.__class__.__name__}] criado.")

    _validate_num_classes(opt, train_metadata)
    model = build_model(opt)
    if opt.get('runtime', {}).get('require_cuda') and model.device.type != 'cuda':
        raise RuntimeError(f'CUDA was required, but effective model device is {model.device}.')
    if model.device.type == 'cuda':
        logger.info(f"Dispositivo efetivo: {model.device} | GPU: {torch.cuda.get_device_name(model.device)}")
        torch.cuda.reset_peak_memory_stats(model.device)
    if hasattr(model, 'set_dataset_metadata'):
        model.set_dataset_metadata(train_metadata)
    logger.info(f"Modelo [{model.__class__.__name__}] construido.")

    visualizers = []
    if 'visualization' in opt:
        for viz_name, viz_opt in opt['visualization'].items():
            viz_opt['path'] = opt['path']
            viz_opt['name'] = opt['name']
            visualizers.append(build_visualizer(viz_opt))
            logger.info(f"Visualizador [{viz_name}] adicionado.")

    metrics = {}
    if 'train' in opt and 'metrics' in opt['train']:
        for metric_name, metric_opt in opt['train']['metrics'].items():
            metrics[metric_name] = build_metric(metric_opt)
            logger.info(f"Metrica [{metric_name}] adicionada.")

    total_epochs = opt['train'].get('total_epochs', 100)
    current_iter = 0
    checkpoint_opt = opt['train'].get('checkpoint', {})
    monitor = checkpoint_opt.get('monitor', 'val_accuracy')
    mode = checkpoint_opt.get('mode', 'max')
    save_best = checkpoint_opt.get('save_best', False)
    save_last = checkpoint_opt.get('save_last', False)
    early_opt = opt['train'].get('early_stopping', {})
    early_enabled = early_opt.get('enabled', False)
    early_patience = int(early_opt.get('patience', 10))
    best_value = None
    epochs_without_improvement = 0

    logger.info(f"Iniciando treinamento por {total_epochs} epocas...")
    train_started_at = time.perf_counter()

    for epoch in range(total_epochs):
        if hasattr(model, 'on_epoch_start'):
            model.on_epoch_start(epoch)
        metric_results = {}

        for data in train_loader:
            current_iter += 1
            model.feed_data(data)
            model.optimize_parameters(current_iter)

            if current_iter % opt['logger'].get('print_freq', 100) == 0:
                losses = model.get_current_losses()
                lrs = model.get_current_learning_rate()
                viz_payload = {**losses}
                for idx, lr in enumerate(lrs):
                    viz_payload[f'lr_{idx}'] = lr
                logger.info(f"[Epoca {epoch}][Iter {current_iter}] Train Metrics: {viz_payload}")
                for viz in visualizers:
                    viz.visualize(current_iter, viz_payload)

        if val_loader is not None and (epoch + 1) % opt['train'].get('val_freq', 1) == 0:
            class_names = _class_names_from_metadata(train_metadata) or _class_names_from_metadata(val_metadata)
            if not class_names:
                raise ValueError('Unable to resolve validation class names from dataset metadata.')
            val_targets = []
            val_predictions = []
            val_confidences = []
            for val_data in val_loader:
                model.feed_data(val_data)
                model.test()
                visuals = model.get_current_visuals()
                logits = visuals['prediction']
                targets = visuals['target']
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                val_targets.extend(targets.tolist())
                val_predictions.extend(predictions.tolist())
                val_confidences.extend(confidences.tolist())

            val_metrics = compute_classification_metrics(val_targets, val_predictions, class_names, val_confidences)
            metric_results = {
                'val_accuracy': val_metrics['accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'val_confidence_mean': val_metrics['confidence_mean'],
                'val_confusion_matrix': val_metrics['confusion_matrix'],
                'val_per_class': val_metrics['per_class'],
                'val_num_samples': val_metrics['num_samples'],
                'val_classes': val_metrics['classes'],
            }
            for class_name, values in val_metrics['per_class'].items():
                metric_results[f'val_precision_{class_name}'] = values['precision']
                metric_results[f'val_recall_{class_name}'] = values['recall']
                metric_results[f'val_f1_{class_name}'] = values['f1']
                metric_results[f'val_support_{class_name}'] = values['support']

            logger.info(f"--- [Validacao Epoca {epoch}] Metricas: {metric_results} ---")
            for viz in visualizers:
                viz_payload = {key: value for key, value in metric_results.items() if isinstance(value, (int, float))}
                viz.visualize(current_iter, viz_payload)

        best_value, should_save_best, monitored_value = _update_best_state(best_value, metric_results, monitor, mode)
        if should_save_best:
            epochs_without_improvement = 0
            if save_best:
                _save_model(model, epoch, current_iter, filename='best.pth', metric_results=metric_results)
        elif monitored_value is not None:
            epochs_without_improvement += 1

        model.update_learning_rate(current_iter)

        if save_last:
            _save_model(model, epoch, current_iter, filename='last.pth', metric_results=metric_results)

        if (epoch + 1) % opt['train'].get('save_checkpoint_freq', 10) == 0:
            _save_model(model, epoch, current_iter, metric_results=metric_results)

        if early_enabled and epochs_without_improvement >= early_patience:
            logger.info(f"Early stopping acionado na epoca {epoch}.")
            break

    elapsed_seconds = time.perf_counter() - train_started_at
    if getattr(model, 'device', torch.device('cpu')).type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(model.device) / 1024 / 1024
        logger.info(f'Peak VRAM MB: {peak_vram_mb:.3f}')
    logger.info(f'Train time seconds: {elapsed_seconds:.3f}')
    logger.info('Treinamento concluido!')


if __name__ == '__main__':
    main()
