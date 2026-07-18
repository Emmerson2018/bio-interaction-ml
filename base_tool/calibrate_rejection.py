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
from base_tool.evaluate import _classes_from_checkpoint, _load_checkpoint, _resolve_path


def _parse_grid(value, default):
    if value is None:
        return list(default)
    if ':' in value:
        start, stop, step = [float(item) for item in value.split(':')]
        values = []
        current = start
        while current <= stop + 1e-9:
            values.append(round(current, 10))
            current += step
        return values
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def _image_paths(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Dataset directory not found: {root}')
    return sorted(
        path for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def _known_samples(root, class_to_idx):
    root = Path(root)
    samples = []
    for class_name, class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in _image_paths(class_dir):
            samples.append((path, int(class_index), class_name))
    if not samples:
        raise FileNotFoundError(f'No known-class images found under {root}')
    return samples


def _unknown_samples(root):
    samples = [(path, -1, 'unknown') for path in _image_paths(root)]
    if not samples:
        raise FileNotFoundError(f'No unknown images found under {root}')
    return samples


def _predict_samples(net, transform, samples, idx_to_class, device):
    rows = []
    with torch.no_grad():
        for image_path, target_index, target_label in samples:
            with Image.open(image_path) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
            probabilities = torch.softmax(net(tensor), dim=1)[0]
            top_values, top_indices = torch.topk(probabilities, k=min(2, probabilities.numel()))
            top1_index = int(top_indices[0].item())
            top2_index = int(top_indices[1].item()) if top_indices.numel() > 1 else top1_index
            top1_confidence = float(top_values[0].item())
            top2_confidence = float(top_values[1].item()) if top_values.numel() > 1 else 0.0
            rows.append(
                {
                    'path': str(image_path),
                    'target_index': int(target_index),
                    'target_label': target_label,
                    'top1_index': top1_index,
                    'top1_class': idx_to_class[top1_index],
                    'top1_confidence': top1_confidence,
                    'top2_index': top2_index,
                    'top2_class': idx_to_class[top2_index],
                    'top2_confidence': top2_confidence,
                    'margin': top1_confidence - top2_confidence,
                }
            )
    return rows


def _apply_policy(row, confidence_threshold, margin_threshold):
    if row['top1_confidence'] < confidence_threshold:
        return False, 'low_confidence'
    if row['margin'] < margin_threshold:
        return False, 'low_margin'
    return True, ''


def _metrics_for_thresholds(rows, confidence_threshold, margin_threshold):
    known_rows = [row for row in rows if row['target_label'] != 'unknown']
    unknown_rows = [row for row in rows if row['target_label'] == 'unknown']
    accepted_known = []
    false_rejects = 0
    false_accepts = 0
    correct_accepted_known = 0

    for row in known_rows:
        accepted, _ = _apply_policy(row, confidence_threshold, margin_threshold)
        if accepted:
            accepted_known.append(row)
            correct_accepted_known += int(row['top1_index'] == row['target_index'])
        else:
            false_rejects += 1

    for row in unknown_rows:
        accepted, _ = _apply_policy(row, confidence_threshold, margin_threshold)
        false_accepts += int(accepted)

    total = len(rows)
    accepted_total = len(accepted_known) + false_accepts
    return {
        'confidence_threshold': confidence_threshold,
        'margin_threshold': margin_threshold,
        'false_accept_rate': false_accepts / len(unknown_rows) if unknown_rows else 0.0,
        'false_reject_rate': false_rejects / len(known_rows) if known_rows else 0.0,
        'unknown_recall': (len(unknown_rows) - false_accepts) / len(unknown_rows) if unknown_rows else 0.0,
        'known_recall': len(accepted_known) / len(known_rows) if known_rows else 0.0,
        'coverage': accepted_total / total if total else 0.0,
        'selective_accuracy': correct_accepted_known / accepted_total if accepted_total else 0.0,
        'accepted_known': len(accepted_known),
        'false_accepts': false_accepts,
        'false_rejects': false_rejects,
        'total_known': len(known_rows),
        'total_unknown': len(unknown_rows),
    }


def _select_threshold(metrics_rows, max_far):
    feasible = [row for row in metrics_rows if row['false_accept_rate'] <= max_far]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            row['coverage'],
            row['selective_accuracy'],
            row['unknown_recall'],
            -row['false_reject_rate'],
        ),
    )


def _write_confusion_matrix(path, rows, class_names, selected):
    labels = list(class_names) + ['unknown']
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        accepted, _ = _apply_policy(row, selected['confidence_threshold'], selected['margin_threshold'])
        predicted = row['top1_class'] if accepted else 'unknown'
        target = row['target_label']
        matrix[index[target]][index[predicted]] += 1

    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(5, len(labels))))
    ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')
    for row_idx, values in enumerate(matrix):
        for col_idx, value in enumerate(values):
            ax.text(col_idx, row_idx, str(value), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return matrix


def _write_score_distributions(confidence_path, margin_path, rows):
    known_confidence = [row['top1_confidence'] for row in rows if row['target_label'] != 'unknown']
    unknown_confidence = [row['top1_confidence'] for row in rows if row['target_label'] == 'unknown']
    known_margin = [row['margin'] for row in rows if row['target_label'] != 'unknown']
    unknown_margin = [row['margin'] for row in rows if row['target_label'] == 'unknown']

    def draw(path, known_values, unknown_values, title, xlabel):
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = 20
        if known_values:
            ax.hist(known_values, bins=bins, alpha=0.65, label='known')
        if unknown_values:
            ax.hist(unknown_values, bins=bins, alpha=0.65, label='unknown')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('count')
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    draw(confidence_path, known_confidence, unknown_confidence, 'Top-1 confidence distribution', 'top1 confidence')
    draw(margin_path, known_margin, unknown_margin, 'Top-1 minus Top-2 margin distribution', 'margin')


def calibrate(args):
    root_path = Path(__file__).resolve().parents[1]
    with open(args.opt, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device requested, but torch.cuda.is_available() is false.')

    checkpoint = _load_checkpoint(_resolve_path(root_path, args.checkpoint), device)
    class_to_idx, idx_to_class = _classes_from_checkpoint(checkpoint)
    if class_to_idx is None:
        raise ValueError('Checkpoint must include class_to_idx and idx_to_class for rejection calibration.')

    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    net = build_network(opt['network_g']).to(device)
    net.load_state_dict(checkpoint.get('network', checkpoint), strict=False)
    net.eval()

    preprocessing = checkpoint.get('preprocessing') or resolve_preprocessing(opt)
    transform = build_image_transform(preprocessing, augment=False)
    samples = _known_samples(args.known_root, class_to_idx) + _unknown_samples(args.unknown_root)
    rows = _predict_samples(net, transform, samples, idx_to_class, device)

    confidence_grid = _parse_grid(args.confidence_grid, [round(x / 100, 2) for x in range(50, 100, 5)])
    margin_grid = _parse_grid(args.margin_grid, [round(x / 100, 2) for x in range(5, 55, 5)])
    metrics_rows = [
        _metrics_for_thresholds(rows, confidence, margin)
        for confidence in confidence_grid
        for margin in margin_grid
    ]
    selected = _select_threshold(metrics_rows, args.max_false_accept_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / 'calibration_grid.csv'
    with grid_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    metrics_path = output_dir / 'rejection_metrics.json'
    thresholds_path = output_dir / 'thresholds.json'
    confusion_path = output_dir / 'rejection_confusion_matrix.png'
    confidence_distribution_path = output_dir / 'confidence_distribution.png'
    margin_distribution_path = output_dir / 'margin_distribution.png'
    _write_score_distributions(confidence_distribution_path, margin_distribution_path, rows)
    selected_payload = {
        'policy': 'top1_confidence_and_margin',
        'confidence_threshold': None,
        'margin_threshold': None,
        'fallback_label': 'unknown',
        'selection_criterion': {
            'max_false_accept_rate': args.max_false_accept_rate,
            'objective': 'maximize_coverage',
        },
        'metrics': {},
    }
    confusion_matrix = None
    if selected is not None:
        selected_payload.update(
            {
                'confidence_threshold': selected['confidence_threshold'],
                'margin_threshold': selected['margin_threshold'],
                'metrics': {
                    'false_accept_rate': selected['false_accept_rate'],
                    'false_reject_rate': selected['false_reject_rate'],
                    'coverage': selected['coverage'],
                    'selective_accuracy': selected['selective_accuracy'],
                    'unknown_recall': selected['unknown_recall'],
                    'known_recall': selected['known_recall'],
                },
            }
        )
        confusion_matrix = _write_confusion_matrix(confusion_path, rows, class_names, selected)

    metrics_payload = {
        'selected': selected,
        'num_known': sum(1 for row in rows if row['target_label'] != 'unknown'),
        'num_unknown': sum(1 for row in rows if row['target_label'] == 'unknown'),
        'classes': class_names,
        'confusion_matrix_with_unknown': confusion_matrix,
    }
    with metrics_path.open('w', encoding='utf-8') as file:
        json.dump(metrics_payload, file, indent=2, ensure_ascii=False)
    with thresholds_path.open('w', encoding='utf-8') as file:
        json.dump(selected_payload, file, indent=2, ensure_ascii=False)

    return {
        'thresholds': str(thresholds_path),
        'grid': str(grid_path),
        'metrics': str(metrics_path),
        'confusion_matrix': str(confusion_path) if selected is not None else None,
        'confidence_distribution': str(confidence_distribution_path),
        'margin_distribution': str(margin_distribution_path),
        'selected': selected_payload,
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate unknown rejection thresholds.')
    parser.add_argument('--opt', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--known-root', required=True)
    parser.add_argument('--unknown-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--confidence-grid', default='0.50:0.95:0.05')
    parser.add_argument('--margin-grid', default='0.05:0.50:0.05')
    parser.add_argument('--max-false-accept-rate', type=float, default=0.05)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    print(json.dumps(calibrate(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
