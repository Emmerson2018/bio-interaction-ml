import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import torch
from base_tool.utils.options import parse_options
from base_tool.utils.logger import get_root_logger, get_env_info
from base_tool.data import build_dataset, build_dataloader
from base_tool.models import build_model
from base_tool.visualization import build_visualizer
from base_tool.metrics import build_metric


def run_validation(model, val_loader, metrics, use_ema=False):
    metric_results = {}
    all_val_predictions = []
    all_val_probabilities = []
    all_val_targets = []

    for val_data in val_loader:
        model.feed_data(val_data)
        model.test(use_ema=use_ema)

        visuals = model.get_current_visuals()
        all_val_predictions.append(visuals['prediction'])
        all_val_probabilities.append(visuals['probabilities'])
        all_val_targets.append(visuals['target'])

    if len(all_val_predictions) > 0:
        cat_preds = torch.cat(all_val_predictions, dim=0)
        cat_targets = torch.cat(all_val_targets, dim=0)
        for name, metric_fn in metrics.items():
            metric_results[name] = metric_fn(cat_preds, cat_targets)

    return metric_results, all_val_probabilities, all_val_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Caminho para o arquivo YAML de opcoes.')
    args = parser.parse_args()

    # 1. Parse Options
    opt = parse_options(args.opt, is_train=True)

    # 2. Logger
    log_file = os.path.join(opt['path']['experiments_root'], f"train_{opt['name']}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())

    # 3. Build Datasets & Dataloaders
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(train_set, opt, phase='train')
            logger.info(f"Dataset de treino [{train_set.__class__.__name__}] criado. Total: {len(train_set)}")
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, opt, phase='val')
            logger.info(f"Dataset de validacao [{val_set.__class__.__name__}] criado.")

    # 4. Build Model
    model = build_model(opt)
    logger.info(f"Modelo [{model.__class__.__name__}] construido.")

    # 5. Build Visualizers
    visualizers = []
    if 'visualization' in opt:
        for viz_name, viz_opt in opt['visualization'].items():
            viz_opt['path'] = opt['path']
            viz_opt['name'] = opt['name']
            visualizers.append(build_visualizer(viz_opt))
            logger.info(f"Visualizador [{viz_name}] adicionado.")

    # 6. Build Metrics
    metrics = {}
    if 'train' in opt and 'metrics' in opt['train']:
        for metric_name, metric_opt in opt['train']['metrics'].items():
            metrics[metric_name] = build_metric(metric_opt)
            logger.info(f"Metrica [{metric_name}] adicionada.")

    # 7. Training Loop
    total_epochs = opt['train'].get('total_epochs', 100)
    current_iter = 0
    best_val_macro_f1 = -1.0

    logger.info(f"Iniciando treinamento por {total_epochs} epocas...")

    for epoch in range(total_epochs):
        for i, data in enumerate(train_loader):
            current_iter += 1

            model.feed_data(data)
            model.optimize_parameters(current_iter)

            # Log e Visualizacao (Treino)
            if current_iter % opt['logger'].get('print_freq', 100) == 0:
                losses = model.get_current_losses()
                lrs = model.get_current_learning_rate()

                viz_payload = {**losses}
                for idx, lr in enumerate(lrs):
                    viz_payload[f'lr_{idx}'] = lr

                logger.info(f"[Epoca {epoch}][Iter {current_iter}] Train Metrics: {viz_payload}")

                for viz in visualizers:
                    viz.visualize(current_iter, viz_payload)

        # Validacao no final da epoca
        if val_loader is not None and (epoch + 1) % opt['train'].get('val_freq', 1) == 0:
            metric_results, all_val_probabilities, all_val_targets = run_validation(
                model, val_loader, metrics, use_ema=False
            )
            logger.info(f"--- [Validacao Epoca {epoch} (Raw)] Metricas: {metric_results} ---")

            # Validar tambem pesos EMA se ativo
            primary_macro_f1 = metric_results.get('macro_f1', metric_results.get('accuracy', 0.0))
            if hasattr(model, 'ema') and model.ema is not None:
                ema_results, _, _ = run_validation(model, val_loader, metrics, use_ema=True)
                logger.info(f"--- [Validacao Epoca {epoch} (EMA)] Metricas: {ema_results} ---")
                ema_f1 = ema_results.get('macro_f1', ema_results.get('accuracy', 0.0))
                if ema_f1 > primary_macro_f1:
                    primary_macro_f1 = ema_f1
                    metric_results = ema_results

            # Checkpoint do melhor modelo
            if primary_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = primary_macro_f1
                logger.info(f"🏆 [Novo Recorde] Epoca {epoch} alcancou Macro-F1 = {best_val_macro_f1:.4f}! Salvando best_macro_f1.pth")
                model.save(epoch, current_iter, name='best_macro_f1.pth')

            # Enviar metricas de validacao para visualizadores
            val_viz_payload = {f'val_{k}': v for k, v in metric_results.items()}
            if len(all_val_probabilities) > 0:
                val_viz_payload['val_probabilities'] = torch.cat(all_val_probabilities, dim=0)
                val_viz_payload['val_targets'] = torch.cat(all_val_targets, dim=0)
            val_viz_payload['epoch'] = epoch
            val_viz_payload['total_epochs'] = total_epochs
            if hasattr(val_loader.dataset, 'classes'):
                val_viz_payload['classes'] = val_loader.dataset.classes

            for viz in visualizers:
                viz.visualize(current_iter, val_viz_payload)

        model.update_learning_rate(current_iter)

        if (epoch + 1) % opt['train'].get('save_checkpoint_freq', 10) == 0:
            model.save(epoch, current_iter)
            model.save(epoch, current_iter, name='latest.pth')

    logger.info(f"Treinamento concluido! Melhor Macro-F1 alcancado: {best_val_macro_f1:.4f}")


if __name__ == '__main__':
    main()
