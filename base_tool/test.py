import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from base_tool.data import build_dataloader, build_dataset
from base_tool.metrics import build_metric
from base_tool.models import build_model
from base_tool.utils.logger import get_env_info, get_root_logger
from base_tool.utils.options import parse_options
from base_tool.visualization import build_visualizer


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on the test dataset.')
    parser.add_argument('-opt', type=str, required=True, help='Path to the options YAML config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained .pth checkpoint file.')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom directory to save test results.')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[1]
    opt = parse_options(args.opt, is_train=False)

    exp_root = Path(opt['path']['experiments_root'])
    if args.output_dir:
        test_res_dir = _resolve_path(root_path, args.output_dir)
    else:
        test_res_dir = exp_root / 'test_results'

    test_res_dir.mkdir(parents=True, exist_ok=True)

    log_file = test_res_dir / f"test_{opt['name']}.log"
    logger = get_root_logger(log_file=str(log_file))
    logger.info(get_env_info())
    logger.info(f"Starting test evaluation for experiment: {opt['name']}")

    # 1. Build Test Dataset & DataLoader
    test_opt = opt['datasets'].get('test', opt['datasets'].get('val'))
    if not test_opt:
        raise ValueError("No 'test' or 'val' dataset configuration found in YAML options.")

    test_set = build_dataset(test_opt)
    batch_size = test_opt.get('batch_size_per_gpu', 32)
    num_workers = test_opt.get('num_worker_per_gpu', 4)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    logger.info(f"Test dataset [{test_set.__class__.__name__}] created. Total samples: {len(test_set)}")

    # 2. Build Model & Load Checkpoint
    model = build_model(opt)
    checkpoint_path = _resolve_path(root_path, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading model weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    state_dict = checkpoint.get('network', checkpoint)
    model.net.load_state_dict(state_dict)
    model.net.eval()

    # 3. Build Metrics
    metrics = {}
    if 'train' in opt and 'metrics' in opt['train']:
        for metric_name, metric_opt in opt['train']['metrics'].items():
            metrics[metric_name] = build_metric(metric_opt)

    # 4. Run Evaluation Loop
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss()

    logger.info("Running inference on test dataset...")

    with torch.no_grad():
        for data in test_loader:
            model.feed_data(data)
            model.test()

            visuals = model.get_current_visuals()
            logits = visuals['prediction']
            probs = visuals['probabilities']
            preds = torch.argmax(probs, dim=1)
            targets = visuals['target']

            # Ensure targets do not exceed model output class count
            valid_mask = (targets >= 0) & (targets < logits.shape[1])
            if not valid_mask.all():
                if not valid_mask.any():
                    continue
                logits = logits[valid_mask]
                targets = targets[valid_mask]
                preds = preds[valid_mask]
                probs = probs[valid_mask]

            loss = criterion(logits.to(model.device), targets.to(model.device))
            total_loss += loss.item()
            num_batches += 1

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    avg_loss = total_loss / max(num_batches, 1)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    # 5. Calculate Classification Report & Confusion Matrix
    unique_indices = sorted(list(set(all_targets).union(set(all_preds))))
    if hasattr(test_set, 'classes') and len(test_set.classes) >= max(unique_indices) + 1:
        class_names = [test_set.classes[i] for i in unique_indices]
    else:
        class_names = [f"class_{i}" for i in unique_indices]

    report_dict = classification_report(all_targets, all_preds, labels=unique_indices, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_targets, all_preds, labels=unique_indices, target_names=class_names, zero_division=0)
    conf_mat = confusion_matrix(all_targets, all_preds, labels=unique_indices)

    acc = float(report_dict.get('accuracy', 0.0))

    print("\n" + "=" * 60)
    print(f" TEST EVALUATION RESULTS - {opt['name']}")
    print("=" * 60)
    print(f" Test Loss:     {avg_loss:.4f}")
    print(f" Test Accuracy: {acc * 100:.2f}%\n")
    print(report_text)
    print("=" * 60 + "\n")

    logger.info(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"\n{report_text}")

    # 6. Save Results to Disk
    report_file = test_res_dir / 'classification_report.txt'
    with report_file.open('w', encoding='utf-8') as f:
        f.write(f"Experiment: {opt['name']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report_text)

    summary_data = {
        'experiment_name': opt['name'],
        'checkpoint': str(checkpoint_path),
        'test_loss': avg_loss,
        'test_accuracy': acc,
        'metrics_per_class': report_dict,
    }
    summary_file = test_res_dir / 'test_summary.json'
    with summary_file.open('w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

    # 7. Generate Visualization Graphics
    # A. Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Test Confusion Matrix - {opt['name']}\nAccuracy: {acc * 100:.2f}%")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = test_res_dir / 'test_confusion_matrix.png'
    plt.savefig(str(cm_path), dpi=300)
    plt.close()

    # B. Per-Class F1 / Precision / Recall Bar Plot
    plt.figure(figsize=(10, 5))
    x_indices = np.arange(len(class_names))
    f1_scores = [report_dict[c]['f1-score'] for c in class_names if c in report_dict]
    precisions = [report_dict[c]['precision'] for c in class_names if c in report_dict]
    recalls = [report_dict[c]['recall'] for c in class_names if c in report_dict]

    width = 0.25
    plt.bar(x_indices - width, f1_scores, width, label='F1-Score', color='#1f77b4')
    plt.bar(x_indices, precisions, width, label='Precision', color='#ff7f0e')
    plt.bar(x_indices + width, recalls, width, label='Recall', color='#2ca02c')

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title(f"Test Per-Class Performance - {opt['name']}")
    plt.xticks(x_indices, class_names, rotation=25)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    f1_path = test_res_dir / 'test_per_class_metrics.png'
    plt.savefig(str(f1_path), dpi=300)
    plt.close()

    # C. Prediction Confidence Boxplot Plot
    plt.figure(figsize=(9, 5))
    max_probs = np.max(all_probs, axis=1)
    correct_mask = (all_preds == all_targets)

    box_data = [max_probs[correct_mask], max_probs[~correct_mask]] if len(max_probs[~correct_mask]) > 0 else [max_probs[correct_mask]]
    box_labels = ['Correct Predictions', 'Incorrect Predictions'] if len(max_probs[~correct_mask]) > 0 else ['Correct Predictions']

    sns.boxplot(data=box_data)
    plt.xticks(range(len(box_labels)), box_labels)
    plt.ylabel('Model Confidence (Max Probability)')
    plt.title(f"Test Prediction Confidence Distribution - {opt['name']}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    conf_box_path = test_res_dir / 'test_confidence_boxplot.png'
    plt.savefig(str(conf_box_path), dpi=300)
    plt.close()

    logger.info(f"Test results saved to: {test_res_dir}")
    print(f"Test results successfully saved to: {test_res_dir}")


if __name__ == '__main__':
    main()
