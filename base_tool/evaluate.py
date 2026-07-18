import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

from base_tool.archs import build_network
from base_tool.data.image_folder_dataset import IMG_EXTENSIONS
from base_tool.data.preprocessing import build_image_transform, resolve_preprocessing
from base_tool.metrics.classification_metrics import compute_classification_metrics


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        return {'network': checkpoint}
    return checkpoint


def _normalize_idx_to_class(idx_to_class):
    return {int(idx): name for idx, name in idx_to_class.items()}


def _classes_from_checkpoint(checkpoint):
    idx_to_class = checkpoint.get('idx_to_class') or checkpoint.get('dataset_metadata', {}).get('idx_to_class')
    class_to_idx = checkpoint.get('class_to_idx') or checkpoint.get('dataset_metadata', {}).get('class_to_idx')
    if idx_to_class and class_to_idx:
        idx_to_class = _normalize_idx_to_class(idx_to_class)
        return dict(class_to_idx), idx_to_class
    return None, None


def _split_root(opt, split, root_path):
    datasets = opt.get('datasets', {})
    if split in datasets:
        return _resolve_path(root_path, datasets[split]['root'])
    if 'train' not in datasets:
        raise ValueError(f'Split {split!r} not found and train split is not configured.')
    train_root = _resolve_path(root_path, datasets['train']['root'])
    if train_root.name in {'train', 'val', 'test'}:
        return train_root.parent / split
    return train_root.parent / split


def _samples_from_split(split_root, class_to_idx):
    if not split_root.exists():
        raise FileNotFoundError(f'Split directory not found: {split_root}')
    samples = []
    for class_name, class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((path, class_index))
    if not samples:
        raise FileNotFoundError(f'No images found for configured classes in {split_root}')
    return samples


def _classes_from_split(split_root):
    classes = sorted([path.name for path in split_root.iterdir() if path.is_dir()])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def _compute_metrics(y_true, y_pred, confidences, class_names):
    return compute_classification_metrics(y_true, y_pred, class_names, confidences)


def _write_predictions(path, rows):
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'path',
                'target',
                'prediction',
                'confidence',
                'top2_prediction',
                'top2_confidence',
                'margin',
                'correct',
                'top2_correct',
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path, metrics):
    with path.open('w', encoding='utf-8') as file:
        file.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        file.write(f"macro_f1: {metrics['macro_f1']:.6f}\n")
        file.write(f"weighted_f1: {metrics['weighted_f1']:.6f}\n")
        file.write(f"confidence_mean: {metrics['confidence_mean']:.6f}\n\n")
        file.write('class,precision,recall,f1,support\n')
        for class_name, values in metrics['per_class'].items():
            file.write(
                f"{class_name},{values['precision']:.6f},{values['recall']:.6f},"
                f"{values['f1']:.6f},{values['support']}\n"
            )


def _write_confusion_matrix(path, confusion, class_names):
    fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))
    ax.imshow(confusion, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')
    for row_idx, row in enumerate(confusion):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(opt_path, checkpoint_path, split):
    if split not in {'train', 'val', 'test'}:
        raise ValueError(f"Unsupported split {split!r}. Use train, val or test.")

    root_path = Path(__file__).resolve().parents[1]
    with open(opt_path, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)
    opt.setdefault(
        'path',
        {
            'root': str(root_path),
            'experiments_root': str(root_path / 'experiments' / opt['name']),
        },
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = _load_checkpoint(_resolve_path(root_path, checkpoint_path), device)
    split_root = _split_root(opt, split, root_path)
    class_to_idx, idx_to_class = _classes_from_checkpoint(checkpoint)
    if class_to_idx is None:
        class_to_idx, idx_to_class = _classes_from_split(split_root)
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]

    output_classes = opt['network_g'].get('num_classes', len(class_names))
    if int(output_classes) != len(class_names):
        raise ValueError(
            f'Number of classes mismatch: model config has {output_classes}, '
            f'but checkpoint/dataset has {len(class_names)} classes.'
        )

    net = build_network(opt['network_g']).to(device)
    state_dict = checkpoint.get('network', checkpoint)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    preprocessing = checkpoint.get('preprocessing') or resolve_preprocessing(opt)
    transform = build_image_transform(preprocessing, augment=False)
    samples = _samples_from_split(split_root, class_to_idx)

    y_true = []
    y_pred = []
    confidences = []
    top2_correct = 0
    prediction_rows = []
    with torch.no_grad():
        for image_path, target in samples:
            with Image.open(image_path) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
            logits = net(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            top_values, top_indices = torch.topk(probabilities, k=min(2, probabilities.numel()))
            confidence = top_values[0]
            prediction = top_indices[0]
            prediction_index = int(prediction.item())
            confidence_value = float(confidence.item())
            second_index = int(top_indices[1].item()) if top_indices.numel() > 1 else prediction_index
            second_confidence = float(top_values[1].item()) if top_values.numel() > 1 else 0.0
            is_top2_correct = int(target) in [int(index.item()) for index in top_indices]
            top2_correct += int(is_top2_correct)
            y_true.append(int(target))
            y_pred.append(prediction_index)
            confidences.append(confidence_value)
            prediction_rows.append(
                {
                    'path': str(image_path),
                    'target': idx_to_class[int(target)],
                    'prediction': idx_to_class[prediction_index],
                    'confidence': f'{confidence_value:.8f}',
                    'top2_prediction': idx_to_class[second_index],
                    'top2_confidence': f'{second_confidence:.8f}',
                    'margin': f'{confidence_value - second_confidence:.8f}',
                    'correct': str(prediction_index == int(target)).lower(),
                    'top2_correct': str(is_top2_correct).lower(),
                }
            )

    metrics = _compute_metrics(y_true, y_pred, confidences, class_names)
    metrics['classes'] = class_names
    metrics['split'] = split
    metrics['top2_accuracy'] = top2_correct / len(samples) if samples else 0.0

    evaluation_dir = _resolve_path(root_path, opt['path']['experiments_root']) / 'evaluation'
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_dir / f'{split}_metrics.json'
    predictions_path = evaluation_dir / f'{split}_predictions.csv'
    report_path = evaluation_dir / 'classification_report.txt'
    confusion_path = evaluation_dir / 'confusion_matrix.png'

    with metrics_path.open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    _write_predictions(predictions_path, prediction_rows)
    _write_report(report_path, metrics)
    _write_confusion_matrix(confusion_path, metrics['confusion_matrix'], class_names)

    return {
        'metrics': metrics,
        'artifacts': {
            'metrics': str(metrics_path),
            'predictions': str(predictions_path),
            'report': str(report_path),
            'confusion_matrix': str(confusion_path),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate an image classifier checkpoint.')
    parser.add_argument('--opt', required=True, help='Path to the training YAML.')
    parser.add_argument('--checkpoint', required=True, help='Path to a checkpoint.')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    args = parser.parse_args()

    result = evaluate(args.opt, args.checkpoint, args.split)
    print(json.dumps(result['metrics'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
