import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import json
import logging
import random
import shutil
import numpy as np
import torch

from base_tool.utils.options import parse_options
from base_tool.utils.logger import get_root_logger, get_env_info
from base_tool.data import build_dataset, build_dataloader
from base_tool.models import build_model
from base_tool.visualization import build_visualizer
from base_tool.metrics import build_metric


def set_random_seed(seed: int):
    """Garante reprodutibilidade estatística por semente nas repetições."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


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


def train_single_run(opt, run_idx: int, total_runs: int, seed: int, logger):
    """Executa uma única repetição independente do experimento com semente controlada."""
    set_random_seed(seed)
    run_prefix = f"[Run {run_idx}/{total_runs} (Seed {seed})]" if total_runs > 1 else ""

    logger.info(f"{run_prefix} Configurando datasets e dataloaders...")
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = build_dataset(dataset_opt)
            train_loader = build_dataloader(train_set, opt, phase='train')
            logger.info(f"{run_prefix} Dataset de treino [{train_set.__class__.__name__}] criado. Total: {len(train_set)}")
        elif phase == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(val_set, opt, phase='val')
            logger.info(f"{run_prefix} Dataset de validacao [{val_set.__class__.__name__}] criado. Total: {len(val_set)}")

    # Constrói o modelo (pesos inicializados estocasticamente sob a semente da run)
    model = build_model(opt)
    logger.info(f"{run_prefix} Modelo [{model.__class__.__name__}] construido com sucesso.")

    # Constrói os visualizadores
    visualizers = []
    if 'visualization' in opt:
        viz_base_dir = opt['path'].get('visualization', os.path.join(opt['path']['experiments_root'], 'visualization'))
        for viz_name, viz_opt in opt['visualization'].items():
            viz_opt_copy = copy.deepcopy(viz_opt)
            viz_opt_copy['path'] = copy.deepcopy(opt['path'])
            if total_runs > 1:
                run_viz_dir = os.path.join(viz_base_dir, f"run_{run_idx}")
                os.makedirs(run_viz_dir, exist_ok=True)
                viz_opt_copy['path']['visualization'] = run_viz_dir
            else:
                viz_opt_copy['path']['visualization'] = viz_base_dir
            viz_opt_copy['name'] = opt['name']
            visualizers.append(build_visualizer(viz_opt_copy))

    # Constrói métricas
    metrics = {}
    if 'train' in opt and 'metrics' in opt['train']:
        for metric_name, metric_opt in opt['train']['metrics'].items():
            metrics[metric_name] = build_metric(metric_opt)

    total_epochs = opt['train'].get('total_epochs', 100)
    current_iter = 0
    best_val_macro_f1 = -1.0
    best_val_acc = -1.0
    best_epoch = -1

    logger.info(f"{run_prefix} Iniciando treinamento por {total_epochs} epocas...")

    for epoch in range(total_epochs):
        for i, data in enumerate(train_loader):
            current_iter += 1
            model.feed_data(data)
            model.optimize_parameters(current_iter)

            # Log e Visualização de treino
            if current_iter % opt['logger'].get('print_freq', 100) == 0:
                losses = model.get_current_losses()
                lrs = model.get_current_learning_rate()
                viz_payload = {**losses}
                for idx, lr in enumerate(lrs):
                    viz_payload[f'lr_{idx}'] = lr

                logger.info(f"{run_prefix}[Epoca {epoch}][Iter {current_iter}] Train Metrics: {viz_payload}")
                for viz in visualizers:
                    viz.visualize(current_iter, viz_payload)

        # Validação ao final da época
        if val_loader is not None and (epoch + 1) % opt['train'].get('val_freq', 1) == 0:
            metric_results, all_val_probabilities, all_val_targets = run_validation(
                model, val_loader, metrics, use_ema=False
            )
            logger.info(f"{run_prefix}--- [Validacao Epoca {epoch} (Raw)] Metricas: {metric_results} ---")

            primary_macro_f1 = metric_results.get('macro_f1', metric_results.get('accuracy', 0.0))
            current_acc = metric_results.get('acc', metric_results.get('accuracy', 0.0))

            if hasattr(model, 'ema') and model.ema is not None:
                ema_results, _, _ = run_validation(model, val_loader, metrics, use_ema=True)
                logger.info(f"{run_prefix}--- [Validacao Epoca {epoch} (EMA)] Metricas: {ema_results} ---")
                ema_f1 = ema_results.get('macro_f1', ema_results.get('accuracy', 0.0))
                if ema_f1 > primary_macro_f1:
                    primary_macro_f1 = ema_f1
                    current_acc = ema_results.get('acc', ema_results.get('accuracy', 0.0))
                    metric_results = ema_results

            if primary_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = primary_macro_f1
                best_val_acc = current_acc
                best_epoch = epoch
                logger.info(f"{run_prefix} 🏆 [Novo Recorde da Run] Epoca {epoch}: Macro-F1 = {best_val_macro_f1:.4f}, Acc = {best_val_acc:.4f}!")
                
                if total_runs > 1:
                    model.save(epoch, current_iter, name=f'best_macro_f1_run{run_idx}.pth')
                else:
                    model.save(epoch, current_iter, name='best_macro_f1.pth')

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
            if total_runs > 1:
                model.save(epoch, current_iter, name=f'latest_run{run_idx}.pth')
            else:
                model.save(epoch, current_iter, name='latest.pth')

    logger.info(f"{run_prefix} Concluído! Melhor Macro-F1: {best_val_macro_f1:.4f} (Acc: {best_val_acc:.4f}) na Epoca {best_epoch}")

    return {
        'run': run_idx,
        'seed': seed,
        'best_macro_f1': float(best_val_macro_f1),
        'best_accuracy': float(best_val_acc),
        'best_epoch': int(best_epoch)
    }


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento com Suporte a Repetições Estatísticas Multi-Run.")
    parser.add_argument('-opt', type=str, required=True, help='Caminho para o arquivo YAML de opcoes.')
    parser.add_argument('--runs', '--repeats', dest='runs', type=int, default=1, help='Numero de repeticoes/runs independentes do experimento (default: 1).')
    parser.add_argument('--base-seed', type=int, default=42, help='Semente aleatoria base para as repeticoes (default: 42).')
    args = parser.parse_args()

    # 1. Parse Options
    opt = parse_options(args.opt, is_train=True)

    # 2. Logger Central
    log_file = os.path.join(opt['path']['experiments_root'], f"train_{opt['name']}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(get_env_info())
    logger.info(f"Configuração carregada: {args.opt} | Repetições Solicitadas: {args.runs} | Seed Base: {args.base_seed}")

    models_dir = os.path.join(opt['path']['experiments_root'], 'models')
    os.makedirs(models_dir, exist_ok=True)

    runs_results = []
    best_overall_macro_f1 = -1.0
    best_run_idx = 1

    # 3. Loop de Repetições Estatísticas
    for r in range(1, args.runs + 1):
        run_seed = args.base_seed + (r - 1) * 100
        if args.runs > 1:
            logger.info("\n" + "=" * 65)
            logger.info(f"🔄 INICIANDO REPETIÇÃO {r}/{args.runs} — Experimento: {opt['name']} (Seed: {run_seed})")
            logger.info("=" * 65)

        run_summary = train_single_run(opt, run_idx=r, total_runs=args.runs, seed=run_seed, logger=logger)
        runs_results.append(run_summary)

        # Se esta run bateu o recorde global do experimento, atualiza best_macro_f1.pth principal
        if run_summary['best_macro_f1'] > best_overall_macro_f1:
            best_overall_macro_f1 = run_summary['best_macro_f1']
            best_run_idx = r
            if args.runs > 1:
                run_pth = os.path.join(models_dir, f'best_macro_f1_run{r}.pth')
                main_best_pth = os.path.join(models_dir, 'best_macro_f1.pth')
                if os.path.exists(run_pth):
                    shutil.copy2(run_pth, main_best_pth)
                    logger.info(f"🏆 Checkpoint campeão global atualizado com pesos da Run {r} (Macro-F1 = {best_overall_macro_f1:.4f})")

    # 4. Cálculo de Estatísticas (Média e Desvio Padrão / Variância)
    f1_list = [res['best_macro_f1'] for res in runs_results]
    acc_list = [res['best_accuracy'] for res in runs_results]

    f1_mean = float(np.mean(f1_list))
    f1_std = float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
    acc_mean = float(np.mean(acc_list))
    acc_std = float(np.std(acc_list, ddof=1)) if len(acc_list) > 1 else 0.0

    summary_payload = {
        'experiment_name': opt['name'],
        'num_runs': args.runs,
        'base_seed': args.base_seed,
        'best_run_idx': best_run_idx,
        'runs': runs_results,
        'metrics_summary': {
            'macro_f1_mean': round(f1_mean, 4),
            'macro_f1_std': round(f1_std, 4),
            'macro_f1_min': round(float(np.min(f1_list)), 4),
            'macro_f1_max': round(float(np.max(f1_list)), 4),
            'accuracy_mean': round(acc_mean, 4),
            'accuracy_std': round(acc_std, 4),
            'accuracy_min': round(float(np.min(acc_list)), 4),
            'accuracy_max': round(float(np.max(acc_list)), 4),
        }
    }

    # Salva resumo JSON na raiz do experimento
    summary_path = os.path.join(opt['path']['experiments_root'], 'runs_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # 5. Emissão do Relatório Estatístico no Log
    logger.info("\n" + "=" * 65)
    logger.info(f"📊 RESUMO ESTATÍSTICO DE REPETIÇÕES ({args.runs} RUNS) — {opt['name']}")
    logger.info("=" * 65)
    for res in runs_results:
        logger.info(f"  • Run {res['run']} (Seed {res['seed']}): Macro-F1 = {res['best_macro_f1']*100:.2f}%, Acc = {res['best_accuracy']*100:.2f}% (Epoca {res['best_epoch']})")
    logger.info("-" * 65)
    logger.info(f"  📈 Macro-F1 Médio: {f1_mean * 100:.2f}% ± {f1_std * 100:.2f}% (Min: {np.min(f1_list)*100:.2f}%, Max: {np.max(f1_list)*100:.2f}%)")
    logger.info(f"  📈 Acurácia Média: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}% (Min: {np.min(acc_list)*100:.2f}%, Max: {np.max(acc_list)*100:.2f}%)")
    logger.info(f"  💾 Resumo consolidado salvo em: {summary_path}")
    logger.info("=" * 65 + "\n")


if __name__ == '__main__':
    main()
