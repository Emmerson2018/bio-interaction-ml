import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import optuna
import torch
import yaml

from base_tool.data import build_dataloader, build_dataset
from base_tool.metrics import build_metric
from base_tool.models import build_model
from base_tool.utils.options import parse_options


def objective(trial, base_opt, num_epochs=6):
    opt = copy.deepcopy(base_opt)

    # Sugestão de Hiperparâmetros via Optuna TPE
    head_lr = trial.suggest_float('head_lr', 1e-4, 2e-3, log=True)
    backbone_ratio = trial.suggest_float('backbone_ratio', 0.03, 0.20)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.05, 0.35)

    deep_lr = head_lr * backbone_ratio
    shallow_lr = deep_lr * 0.33

    opt['network_g']['dropout'] = dropout
    opt['train']['weight_decay'] = weight_decay
    opt['train']['two_stage']['head_lr'] = head_lr
    opt['train']['two_stage']['freeze_epochs'] = 2
    opt['train']['differential_lr']['head_lr'] = head_lr
    opt['train']['differential_lr']['backbone_deep_lr'] = deep_lr
    opt['train']['differential_lr']['backbone_shallow_lr'] = shallow_lr
    opt['train']['total_epochs'] = num_epochs
    opt['train']['scheduler']['warmup_epochs'] = 2

    # Construir datasets e dataloaders
    train_set = build_dataset(opt['datasets']['train'])
    val_set = build_dataset(opt['datasets']['val'])
    train_loader = build_dataloader(train_set, opt, phase='train')
    val_loader = build_dataloader(val_set, opt, phase='val')

    model = build_model(opt)
    macro_f1_metric = build_metric({'type': 'MacroF1Metric'})

    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        model.net.train()
        for i, data in enumerate(train_loader):
            model.feed_data(data)
            model.optimize_parameters(i)

        # Avaliação no dataset de validação
        all_preds = []
        all_targets = []
        for val_data in val_loader:
            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()
            all_preds.append(visuals['prediction'])
            all_targets.append(visuals['target'])

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        val_f1 = macro_f1_metric(cat_preds, cat_targets)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        model.update_learning_rate(epoch)

        # Informar Optuna para Successive Halving / ASHA Pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description='HPO Multi-fidelidade para ResNet18 Baseline.')
    parser.add_argument('-opt', type=str, default='options/exp1_resnet18_baseline.yml')
    parser.add_argument('--n-trials', type=int, default=15, help='Número total de trials no estudo.')
    parser.add_argument('--epochs-per-trial', type=int, default=5, help='Número de épocas por trial na busca rápida.')
    args = parser.parse_args()

    opt = parse_options(args.opt, is_train=True)
    out_dir = Path(opt['path']['experiments_root'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"Iniciando HPO Multi-Fidelidade (Optuna ASHA) com {args.n_trials} trials...")
    print(f"==========================================")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3)

    study = optuna.create_study(
        study_name='resnet18_baseline_hpo',
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, opt, num_epochs=args.epochs_per_trial),
        n_trials=args.n_trials,
        timeout=1800,  # 30 minutos max
    )

    print(f"\n[HPO Concluído]")
    print(f"Melhor Trial: {study.best_trial.number}")
    print(f"Melhor Macro-F1: {study.best_value:.4f}")
    print(f"Melhores Parâmetros: {study.best_params}")

    summary_path = out_dir / 'hpo_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump({
            'best_trial': study.best_trial.number,
            'best_macro_f1': study.best_value,
            'best_params': study.best_params,
            'trials_count': len(study.trials),
        }, f, indent=2)

    print(f"Resumo HPO gravado em: {summary_path}")


if __name__ == '__main__':
    main()
