import argparse
import logging
import os
import torch
from base_tool.utils.options import parse_options
from base_tool.utils.logger import get_root_logger, get_env_info
from base_tool.data import build_dataset, build_dataloader
from base_tool.models import build_model
from base_tool.visualization import build_visualizer
from base_tool.metrics import build_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Caminho para o arquivo YAML de opções.')
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
            logger.info(f"Dataset de validação [{val_set.__class__.__name__}] criado.")

    # 4. Build Model
    model = build_model(opt)
    logger.info(f"Modelo [{model.__class__.__name__}] construído.")

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
            logger.info(f"Métrica [{metric_name}] adicionada.")

    # 7. Training Loop
    total_epochs = opt['train'].get('total_epochs', 100)
    current_iter = 0
    
    logger.info(f"Iniciando treinamento por {total_epochs} épocas...")
    
    for epoch in range(total_epochs):
        model.optimizers[0].param_groups[0]['lr'] # Apenas check
        
        for i, data in enumerate(train_loader):
            current_iter += 1
            
            model.feed_data(data)
            model.optimize_parameters(current_iter)
            
            # Log e Visualização (Treino)
            if current_iter % opt['logger'].get('print_freq', 100) == 0:
                losses = model.get_current_losses()
                lrs = model.get_current_learning_rate()
                
                viz_payload = {**losses}
                for idx, lr in enumerate(lrs):
                    viz_payload[f'lr_{idx}'] = lr
                
                logger.info(f"[Época {epoch}][Iter {current_iter}] Train Metrics: {viz_payload}")
                
                for viz in visualizers:
                    viz.visualize(current_iter, viz_payload)

        # Validação no final da época
        if val_loader is not None and (epoch + 1) % opt['train'].get('val_freq', 1) == 0:
            metric_results = {name: 0 for name in metrics.keys()}
            num_val_batches = 0
            
            for val_data in val_loader:
                model.feed_data(val_data)
                model.test()
                
                visuals = model.get_current_visuals()
                for name, metric_fn in metrics.items():
                    metric_results[name] += metric_fn(visuals['prediction'], visuals['target'])
                num_val_batches += 1
            
            # Média das métricas
            for name in metric_results:
                metric_results[name] /= num_val_batches
                
            logger.info(f"--- [Validação Época {epoch}] Métricas: {metric_results} ---")
            
            # Também enviar métricas de validação para os visualizadores
            for viz in visualizers:
                viz.visualize(current_iter, {f'val_{k}': v for k, v in metric_results.items()})

        model.update_learning_rate(current_iter)
        
        if (epoch + 1) % opt['train'].get('save_checkpoint_freq', 10) == 0:
            model.save(epoch, current_iter)

    logger.info("Treinamento concluído!")

if __name__ == '__main__':
    main()
