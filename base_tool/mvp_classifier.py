import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageStat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
SPLITS = ('train', 'validation', 'test')
SPLIT_DIR_ALIASES = {
    'train': 'train',
    'validation': 'validation',
    'val': 'validation',
    'test': 'test',
}
KNOWN_CLASS_ALIASES = {
    'unknow': 'unknown',
    'unknown': 'unknown',
}


@dataclass
class SampleRecord:
    filepath: str
    class_name: str
    class_index: int
    media_type: str
    source_file: str
    source_group: str
    timestamp_seconds: str
    split: str = ''
    original_path: str = ''


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value):
    return ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in value)


def normalize_class_name(folder_name):
    normalized = folder_name.strip().lower()
    normalized = (
        normalized.replace(' ', '_')
        .replace('-', '_')
        .replace('ç', 'c')
        .replace('ã', 'a')
        .replace('á', 'a')
        .replace('à', 'a')
        .replace('â', 'a')
        .replace('é', 'e')
        .replace('ê', 'e')
        .replace('í', 'i')
        .replace('ó', 'o')
        .replace('õ', 'o')
        .replace('ú', 'u')
    )
    return KNOWN_CLASS_ALIASES.get(normalized, normalized)


def _has_media(path):
    for item in path.rglob('*'):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            return True
    return False


def discover_input_root(explicit_root=None):
    if explicit_root:
        return Path(explicit_root)
    candidates = [Path('datasets/imagens_reais'), Path('datasets/real_mvp'), Path('datasets')]
    for candidate in candidates:
        if not candidate.exists():
            continue
        class_dirs = [path for path in candidate.iterdir() if path.is_dir() and _has_media(path)]
        if len(class_dirs) >= 2:
            return candidate
    raise FileNotFoundError('No folder-class dataset root was found. Pass --input-root explicitly.')


def _validate_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = image.convert('RGB')
            if image.width <= 0 or image.height <= 0:
                return False, 'empty_dimensions'
            if path.stat().st_size == 0:
                return False, 'empty_file'
            if max(ImageStat.Stat(image).stddev) < 0.5:
                return False, 'near_uniform_empty'
        return True, ''
    except Exception as exc:
        return False, f'illegible:{exc}'


def _video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return {'fps': fps, 'frame_count': frame_count, 'duration_seconds': frame_count / fps, 'width': width, 'height': height}


def _read_frame_at_second(video_path, second):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _class_dirs(input_root):
    pairs = []
    for path in sorted(Path(input_root).iterdir()):
        if path.is_dir() and _has_media(path):
            pairs.append((normalize_class_name(path.name), path))
    unique = {}
    for class_name, path in pairs:
        unique.setdefault(class_name, []).append(path)
    return [(class_name, paths) for class_name, paths in sorted(unique.items())]


def _is_split_dataset(input_root):
    root = Path(input_root)
    return (root / 'train').is_dir() and ((root / 'val').is_dir() or (root / 'validation').is_dir()) and (root / 'test').is_dir()


def _split_dir(input_root, split):
    root = Path(input_root)
    for dirname, canonical in SPLIT_DIR_ALIASES.items():
        if canonical == split and (root / dirname).is_dir():
            return root / dirname
    return None


def _read_class_index(input_root):
    class_index_path = Path(input_root) / 'class_index.csv'
    if not class_index_path.exists():
        return None
    rows = _read_manifest(class_index_path)
    class_to_idx = {}
    for row in rows:
        class_to_idx[row['class_name']] = int(row['class_index'])
    return dict(sorted(class_to_idx.items(), key=lambda item: item[1]))


def _discover_split_classes(input_root):
    class_to_idx = _read_class_index(input_root)
    if class_to_idx:
        return list(class_to_idx), class_to_idx
    classes = set()
    for split in SPLITS:
        split_path = _split_dir(input_root, split)
        if split_path is None:
            continue
        for path in split_path.iterdir():
            if path.is_dir() and _has_media(path):
                classes.add(normalize_class_name(path.name))
    ordered = sorted(classes)
    return ordered, {class_name: idx for idx, class_name in enumerate(ordered)}


def prepare_existing_split_dataset(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    classes, class_to_idx = _discover_split_classes(input_root)
    if not classes:
        raise FileNotFoundError(f'No class folders were found in split dataset {input_root}')

    manifest_rows = []
    reports = {'corrupt_or_removed': [], 'duplicates': [], 'class_folder_mapping': {}}
    seen_hashes = {}
    media_counts = {class_name: {'images': 0, 'videos': 0, 'frames': 0} for class_name in classes}
    split_summary = defaultdict(lambda: defaultdict(int))

    for split in SPLITS:
        split_path = _split_dir(input_root, split)
        if split_path is None:
            continue
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                continue
            reports['class_folder_mapping'].setdefault(class_name, []).append(str(class_path))
            for path in sorted(class_path.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                valid, reason = _validate_image(path)
                if not valid:
                    reports['corrupt_or_removed'].append({'filepath': str(path), 'reason': reason})
                    continue
                digest = _sha256(path)
                if digest in seen_hashes:
                    reports['duplicates'].append({'filepath': str(path), 'duplicate_of': seen_hashes[digest], 'sha256': digest})
                    continue
                seen_hashes[digest] = str(path)
                media_counts[class_name]['images'] += 1
                split_summary[split][class_name] += 1
                manifest_rows.append({
                    'filepath': str(path),
                    'class_name': class_name,
                    'class_index': class_to_idx[class_name],
                    'media_type': 'image',
                    'source_file': str(path),
                    'source_group': f'image:{digest}',
                    'timestamp_seconds': '',
                    'split': split,
                    'original_path': str(path),
                })

    missing_train = [class_name for class_name in classes if split_summary['train'].get(class_name, 0) == 0]
    if missing_train:
        raise RuntimeError('Classes without training samples: ' + ', '.join(missing_train))

    fieldnames = ['filepath', 'class_name', 'class_index', 'media_type', 'source_file', 'source_group', 'timestamp_seconds', 'split', 'original_path']
    _write_csv(output_root / 'dataset_manifest.csv', manifest_rows, fieldnames)
    split_summary = {split: dict(split_summary[split]) for split in SPLITS}
    dataset_summary = {
        'input_root': str(input_root),
        'prepared_root': str(input_root),
        'classes': classes,
        'class_to_idx': class_to_idx,
        'media_counts': media_counts,
        'total_samples': len(manifest_rows),
        'frames_extracted': 0,
        'video_exclusions_count': 0,
        'corrupt_or_removed_count': len(reports['corrupt_or_removed']),
        'duplicates_count': len(reports['duplicates']),
        'limitations': [],
        'reports': reports,
        'layout': 'existing_train_val_test',
    }
    (output_root / 'dataset_summary.json').write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_root / 'split_summary.json').write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return dataset_summary, split_summary


def _split_groups(group_records, seed):
    groups = list(group_records)
    rng = random.Random(seed)
    rng.shuffle(groups)
    total = len(groups)
    if total == 0:
        return {}
    if total == 1:
        counts = {'train': 1, 'validation': 0, 'test': 0}
    elif total == 2:
        counts = {'train': 1, 'validation': 1, 'test': 0}
    else:
        train_count = max(1, round(total * 0.70))
        val_count = max(1, round(total * 0.15))
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            train_count = max(1, train_count - 1)
        while train_count + val_count + test_count > total:
            train_count = max(1, train_count - 1)
        counts = {'train': train_count, 'validation': val_count, 'test': test_count}
    split_map = {}
    cursor = 0
    for split, count in counts.items():
        for group_id in groups[cursor:cursor + count]:
            split_map[group_id] = split
        cursor += count
    return split_map


def prepare_folder_dataset(input_root, output_root, seed=42, extract_video_fps=1.0):
    input_root = Path(input_root)
    output_root = Path(output_root)
    prepared_root = output_root / 'prepared_dataset'
    frame_root = output_root / 'extracted_frames'
    reports = {
        'corrupt_or_removed': [],
        'duplicates': [],
        'video_exclusions': [],
        'class_folder_mapping': {},
        'special_unknown_video': None,
    }
    class_sources = _class_dirs(input_root)
    if not class_sources:
        raise FileNotFoundError(f'No class folders with media were found in {input_root}')
    classes = [class_name for class_name, _ in class_sources]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    reports['class_folder_mapping'] = {
        class_name: [str(path) for path in paths] for class_name, paths in class_sources
    }

    unknown_videos = []
    for class_name, paths in class_sources:
        if class_name == 'unknown':
            for folder in paths:
                unknown_videos.extend(sorted(path for path in folder.rglob('*') if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS))
    special_unknown_video = unknown_videos[0] if unknown_videos else None
    reports['special_unknown_video'] = str(special_unknown_video) if special_unknown_video else None

    records = []
    seen_hashes = {}
    media_counts = {class_name: {'images': 0, 'videos': 0, 'frames': 0} for class_name in classes}
    for class_name, folders in class_sources:
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    valid, reason = _validate_image(path)
                    if not valid:
                        reports['corrupt_or_removed'].append({'filepath': str(path), 'reason': reason})
                        continue
                    digest = _sha256(path)
                    if digest in seen_hashes:
                        reports['duplicates'].append({'filepath': str(path), 'duplicate_of': seen_hashes[digest], 'sha256': digest})
                        continue
                    seen_hashes[digest] = str(path)
                    source_group = f'image:{digest}'
                    media_counts[class_name]['images'] += 1
                    records.append(SampleRecord(
                        filepath='',
                        class_name=class_name,
                        class_index=class_to_idx[class_name],
                        media_type='image',
                        source_file=str(path),
                        source_group=source_group,
                        timestamp_seconds='',
                        original_path=str(path),
                    ))
                elif suffix in VIDEO_EXTENSIONS:
                    info = _video_info(path)
                    if info is None:
                        reports['corrupt_or_removed'].append({'filepath': str(path), 'reason': 'video_unreadable'})
                        continue
                    media_counts[class_name]['videos'] += 1
                    video_digest = _sha256(path)
                    source_group = f'video:{video_digest}'
                    max_second = max(0, int(math.floor(info['duration_seconds'])))
                    if path == special_unknown_video:
                        seconds = list(range(0, min(6, max_second) + 1))
                        for second in range(7, max_second + 1):
                            reports['video_exclusions'].append({
                                'source_file': str(path),
                                'timestamp_seconds': second,
                                'reason': 'ambiguous_scene_after_00_07_fish_and_sapo',
                            })
                    else:
                        seconds = list(range(0, max_second + 1, max(1, int(round(1.0 / extract_video_fps)))))
                    for second in seconds:
                        image = _read_frame_at_second(path, second)
                        if image is None:
                            reports['corrupt_or_removed'].append({'filepath': str(path), 'reason': f'frame_unreadable_at_{second}s'})
                            continue
                        frame_dir = frame_root / class_name
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        frame_name = f'{_safe_name(path.stem)}_{second:06d}s.jpg'
                        frame_path = frame_dir / frame_name
                        image.save(frame_path, quality=95)
                        digest = _sha256(frame_path)
                        if digest in seen_hashes:
                            reports['duplicates'].append({'filepath': str(frame_path), 'duplicate_of': seen_hashes[digest], 'sha256': digest})
                            continue
                        seen_hashes[digest] = str(frame_path)
                        media_counts[class_name]['frames'] += 1
                        records.append(SampleRecord(
                            filepath='',
                            class_name=class_name,
                            class_index=class_to_idx[class_name],
                            media_type='video_frame',
                            source_file=str(path),
                            source_group=source_group,
                            timestamp_seconds=str(second),
                            original_path=str(frame_path),
                        ))

    grouped_by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped_by_class[record.class_name][record.source_group].append(record)

    limitations = []
    for class_name in classes:
        group_count = len(grouped_by_class[class_name])
        if group_count < 3:
            limitations.append({
                'class_name': class_name,
                'source_groups': group_count,
                'message': 'Class has too few source groups to appear in train, validation and test.',
            })

    for class_name in classes:
        split_map = _split_groups(list(grouped_by_class[class_name]), seed + class_to_idx[class_name])
        if not any(split == 'train' for split in split_map.values()):
            raise RuntimeError(f'Class {class_name} has no training sample after split.')
        for source_group, class_records in grouped_by_class[class_name].items():
            split = split_map[source_group]
            for record in class_records:
                record.split = split

    manifest_rows = []
    for record in records:
        split_dir = prepared_root / record.split / record.class_name
        split_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(record.original_path)
        target = split_dir / source_path.name
        shutil.copy2(source_path, target)
        record.filepath = str(target)
        manifest_rows.append(record.__dict__)

    fieldnames = ['filepath', 'class_name', 'class_index', 'media_type', 'source_file', 'source_group', 'timestamp_seconds', 'split', 'original_path']
    _write_csv(output_root / 'dataset_manifest.csv', manifest_rows, fieldnames)

    split_summary = defaultdict(lambda: defaultdict(int))
    for row in manifest_rows:
        split_summary[row['split']][row['class_name']] += 1
    split_summary = {split: dict(split_summary[split]) for split in SPLITS}
    dataset_summary = {
        'input_root': str(input_root),
        'prepared_root': str(prepared_root),
        'classes': classes,
        'class_to_idx': class_to_idx,
        'media_counts': media_counts,
        'total_samples': len(manifest_rows),
        'frames_extracted': sum(counts['frames'] for counts in media_counts.values()),
        'video_exclusions_count': len(reports['video_exclusions']),
        'corrupt_or_removed_count': len(reports['corrupt_or_removed']),
        'duplicates_count': len(reports['duplicates']),
        'limitations': limitations,
        'reports': reports,
    }
    (output_root / 'dataset_summary.json').write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_root / 'split_summary.json').write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return dataset_summary, split_summary


class ManifestDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = list(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['filepath']) as image:
            image = image.convert('RGB')
            tensor = self.transform(image)
        return tensor, int(row['class_index']), row['filepath']


def _read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def _build_model(architecture, weights_name, num_classes):
    if architecture == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if weights_name == 'DEFAULT' else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        preprocessing = weights.transforms() if weights else None
        return model, preprocessing
    if architecture == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if weights_name == 'DEFAULT' else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        preprocessing = weights.transforms() if weights else None
        return model, preprocessing
    raise ValueError(f'Unsupported MVP architecture: {architecture}')


def _transforms(input_size, mean, std, train=False):
    ops = [transforms.Resize((input_size, input_size))]
    if train:
        ops.extend([
            transforms.RandomResizedCrop(input_size, scale=(0.90, 1.0)),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def _epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['train_loss'] for row in history], label='train')
    axes[0].plot(epochs, [row['validation_loss'] for row in history], label='validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(epochs, [row['train_accuracy'] for row in history], label='train')
    axes[1].plot(epochs, [row['validation_accuracy'] for row in history], label='validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def train_model(output_root, experiment_name='mvp_classifier', seed=42, input_size=224, batch_size=16, epochs=30, patience=5):
    output_root = Path(output_root)
    manifest = _read_manifest(output_root / 'dataset_manifest.csv')
    classes = sorted({row['class_name'] for row in manifest}, key=lambda c: int(next(row['class_index'] for row in manifest if row['class_name'] == c)))
    num_classes = len(classes)
    architecture = 'mobilenet_v3_small'
    weights_name = 'DEFAULT'
    model, preprocessing = _build_model(architecture, weights_name, num_classes)
    mean = list(getattr(preprocessing, 'mean', [0.485, 0.456, 0.406])) if preprocessing else [0.485, 0.456, 0.406]
    std = list(getattr(preprocessing, 'std', [0.229, 0.224, 0.225])) if preprocessing else [0.229, 0.224, 0.225]
    train_rows = [row for row in manifest if row['split'] == 'train']
    val_rows = [row for row in manifest if row['split'] == 'validation']
    test_rows = [row for row in manifest if row['split'] == 'test']

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_ds = ManifestDataset(train_rows, _transforms(input_size, mean, std, train=True))
    val_ds = ManifestDataset(val_rows, _transforms(input_size, mean, std, train=False))
    test_ds = ManifestDataset(test_rows, _transforms(input_size, mean, std, train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    counts = Counter(int(row['class_index']) for row in train_rows)
    if counts and max(counts.values()) / max(1, min(counts.values())) >= 2.0:
        weights = torch.tensor([sum(counts.values()) / max(1, counts.get(idx, 1)) for idx in range(num_classes)], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        class_weights_used = True
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights_used = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    experiment_root = Path('experiments') / experiment_name
    models_dir = experiment_root / 'models'
    logs_dir = experiment_root / 'logs'
    vis_dir = experiment_root / 'visualization'
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / 'best_model.pth'
    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _epoch(model, val_loader, criterion, device, None) if val_rows else (float('nan'), float('nan'))
        scheduler.step()
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                'network': model.state_dict(),
                'architecture': architecture,
                'weights': weights_name,
                'classes': classes,
                'class_to_idx': {class_name: idx for idx, class_name in enumerate(classes)},
                'idx_to_class': {idx: class_name for idx, class_name in enumerate(classes)},
                'input_size': input_size,
                'normalization_mean': mean,
                'normalization_std': std,
                'seed': seed,
                'epoch': epoch,
                'validation_loss': val_loss,
                'validation_accuracy': val_acc,
            }, best_path)
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.perf_counter() - started

    _write_csv(output_root / 'training_history.csv', history, ['epoch', 'train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy', 'learning_rate'])
    shutil.copy2(output_root / 'training_history.csv', logs_dir / 'training_history.csv')
    _plot_history(history, output_root / 'training_curves.png')
    shutil.copy2(output_root / 'training_curves.png', vis_dir / 'training_curves.png')
    metadata = {
        'architecture': architecture,
        'weights': weights_name,
        'input_size': input_size,
        'normalization_mean': mean,
        'normalization_std': std,
        'num_classes': num_classes,
        'classes': classes,
        'seed': seed,
        'epochs_ran': len(history),
        'best_epoch': best_epoch,
        'best_validation_loss': best_val_loss,
        'device': str(device),
        'class_weights_used': class_weights_used,
        'train_seconds': round(train_seconds, 3),
        'checkpoint': str(best_path),
    }
    (output_root / 'training_summary.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return metadata, test_ds


def _load_best_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model, _ = _build_model(checkpoint['architecture'], None, len(checkpoint['classes']))
    model.load_state_dict(checkpoint['network'])
    return model, checkpoint


def evaluate_model(output_root, checkpoint_path):
    output_root = Path(output_root)
    model, checkpoint = _load_best_model(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    rows = [row for row in _read_manifest(output_root / 'dataset_manifest.csv') if row['split'] == 'test']
    classes = checkpoint['classes']
    transform = _transforms(checkpoint['input_size'], checkpoint['normalization_mean'], checkpoint['normalization_std'], train=False)
    dataset = ManifestDataset(rows, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    y_true, y_pred, confidences, prediction_rows = [], [], [], []
    class_conf = defaultdict(list)
    top3_correct = 0
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            top_values, top_indices = torch.topk(probs, k=min(3, len(classes)), dim=1)
            for idx in range(len(paths)):
                target = int(y[idx].item())
                pred = int(top_indices[idx, 0].item())
                conf = float(top_values[idx, 0].item())
                y_true.append(target)
                y_pred.append(pred)
                confidences.append(conf)
                class_conf[classes[target]].append(conf)
                top3_correct += int(target in [int(v.item()) for v in top_indices[idx]])
                prediction_rows.append({
                    'filepath': paths[idx],
                    'target': classes[target],
                    'prediction': classes[pred],
                    'confidence': f'{conf:.8f}',
                    'top3': ';'.join(classes[int(v.item())] for v in top_indices[idx]),
                    'correct': str(target == pred).lower(),
                })

    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) if y_true else 0.0,
        'macro_f1': f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0) if y_true else 0.0,
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0) if y_true else 0.0,
        'top3_accuracy': top3_correct / max(1, len(y_true)),
        'num_test_samples': len(y_true),
        'per_class': {},
        'confusion_matrix': cm.tolist(),
    }
    per_class_rows = []
    for idx, class_name in enumerate(classes):
        correct = int(cm[idx, idx])
        total = int(support[idx])
        errors = total - correct
        row = {
            'class_name': class_name,
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': total,
            'correct': correct,
            'errors': errors,
            'confidence_mean': float(np.mean(class_conf[class_name])) if class_conf[class_name] else None,
        }
        metrics['per_class'][class_name] = row
        per_class_rows.append(row)

    _write_csv(output_root / 'per_class_metrics.csv', per_class_rows, ['class_name', 'precision', 'recall', 'f1', 'support', 'correct', 'errors', 'confidence_mean'])
    _write_csv(output_root / 'misclassified_examples.csv', [row for row in prediction_rows if row['correct'] == 'false'], ['filepath', 'target', 'prediction', 'confidence', 'top3', 'correct'])
    _write_csv(output_root / 'predictions.csv', prediction_rows, ['filepath', 'target', 'prediction', 'confidence', 'top3', 'correct'])
    with (output_root / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['target\\prediction'] + classes)
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([classes[idx]] + row)
    _plot_confusion(cm, classes, output_root / 'confusion_matrix.png')
    (output_root / 'test_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    return metrics


def _plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(max(5, len(classes)), max(4, len(classes))))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return ''


def _file_hash(path):
    return _sha256(path) if Path(path).exists() else ''


def export_mobile(output_root, checkpoint_path):
    output_root = Path(output_root)
    export_root = output_root / 'mobile_export'
    export_root.mkdir(parents=True, exist_ok=True)
    model, checkpoint = _load_best_model(checkpoint_path)
    model.eval()
    labels_path = export_root / 'labels.txt'
    labels_path.write_text('\n'.join(checkpoint['classes']) + '\n', encoding='utf-8')
    onnx_path = export_root / 'model.onnx'
    tflite_path = export_root / 'model.tflite'
    tflite_float16_path = export_root / 'model_float16.tflite'
    export_errors = []
    dummy = torch.randn(1, 3, checkpoint['input_size'], checkpoint['input_size'])
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['input'],
            output_names=['logits'],
            dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
            opset_version=18,
            dynamo=False,
        )
    except Exception as exc:
        export_errors.append({'format': 'onnx', 'error': str(exc)})
    tflite_conversion = _convert_onnx_to_tflite(onnx_path, export_root, tflite_path, tflite_float16_path) if onnx_path.exists() else {'status': 'skipped', 'reason': 'ONNX model not available.'}
    if tflite_conversion['status'] != 'ok':
        export_errors.append({'format': 'tflite', 'error': tflite_conversion.get('reason', 'TFLite conversion failed.')})
    parity = _parity_check(output_root, model, checkpoint, onnx_path if onnx_path.exists() else None)
    tflite_parity = _tflite_parity_check(output_root, model, checkpoint, tflite_path if tflite_path.exists() else None)
    manifest = {
        'model_name': 'mvp_animals_classifier',
        'model_version': '0.1.0-mvp',
        'architecture': checkpoint['architecture'],
        'input_width': checkpoint['input_size'],
        'input_height': checkpoint['input_size'],
        'input_channels': 3,
        'input_dtype': 'float32',
        'color_order': 'RGB',
        'resize_strategy': 'resize_to_224x224',
        'normalization_mean': checkpoint['normalization_mean'],
        'normalization_std': checkpoint['normalization_std'],
        'class_count': len(checkpoint['classes']),
        'labels_file': str(labels_path),
        'output_type': 'logits',
        'top_k': min(3, len(checkpoint['classes'])),
        'fallback_policy': {
            'type': 'top1_probability_threshold',
            'confidence_threshold': None,
            'fallback_label': 'unknown',
            'fallback_message': 'Nao consegui identificar a figura.',
            'threshold_status': 'pending_real_validation',
        },
        'quantization_type': 'float32' if tflite_path.exists() else 'none',
        'source_checkpoint': str(checkpoint_path),
        'dataset_manifest_hash': _file_hash(output_root / 'dataset_manifest.csv'),
        'git_commit_hash': _git_commit(),
        'exported_files': {
            'tflite': str(tflite_path) if tflite_path.exists() else '',
            'tflite_float16': str(tflite_float16_path) if tflite_float16_path.exists() else '',
            'onnx': str(onnx_path) if onnx_path.exists() else '',
            'labels': str(labels_path),
        },
        'conversion': {
            'onnx_to_tflite': tflite_conversion,
        },
        'export_errors': export_errors,
        'parity': parity,
        'tflite_parity': tflite_parity,
    }
    (export_root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def _parity_check(output_root, torch_model, checkpoint, onnx_path):
    rows = [row for row in _read_manifest(Path(output_root) / 'dataset_manifest.csv') if row['split'] == 'test'][:8]
    if not onnx_path:
        return {'status': 'skipped', 'reason': 'ONNX model not available.'}
    try:
        import onnxruntime as ort
    except Exception as exc:
        return {'status': 'skipped', 'reason': f'onnxruntime not available: {exc}'}
    transform = _transforms(checkpoint['input_size'], checkpoint['normalization_mean'], checkpoint['normalization_std'], train=False)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    same = 0
    max_score_diff = 0.0
    comparisons = []
    torch_model.eval()
    with torch.no_grad():
        for row in rows:
            with Image.open(row['filepath']) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            torch_scores = torch.softmax(torch_model(tensor), dim=1).numpy()[0]
            onnx_scores = torch.softmax(torch.tensor(session.run(None, {'input': tensor.numpy()})[0]), dim=1).numpy()[0]
            torch_pred = int(np.argmax(torch_scores))
            onnx_pred = int(np.argmax(onnx_scores))
            diff = float(np.max(np.abs(torch_scores - onnx_scores)))
            same += int(torch_pred == onnx_pred)
            max_score_diff = max(max_score_diff, diff)
            comparisons.append({
                'filepath': row['filepath'],
                'torch_prediction': checkpoint['classes'][torch_pred],
                'onnx_prediction': checkpoint['classes'][onnx_pred],
                'max_score_diff': diff,
            })
    return {
        'status': 'ok',
        'samples': len(rows),
        'same_predictions': same,
        'same_prediction_rate': same / max(1, len(rows)),
        'max_score_diff': max_score_diff,
        'comparisons': comparisons,
    }


def _convert_onnx_to_tflite(onnx_path, export_root, tflite_path, tflite_float16_path):
    if not onnx_path.exists():
        return {'status': 'skipped', 'reason': 'ONNX model not available.'}
    converted_root = export_root / 'tf_converted'
    command = [
        sys.executable,
        '-m',
        'onnx2tf',
        '-i',
        str(onnx_path),
        '-o',
        str(converted_root),
        '-n',
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return {'status': 'failed', 'reason': str(exc), 'command': command}
    if result.returncode != 0:
        return {
            'status': 'failed',
            'reason': (result.stderr or result.stdout)[-2000:],
            'command': command,
        }
    float32_source = converted_root / 'model_float32.tflite'
    float16_source = converted_root / 'model_float16.tflite'
    if not float32_source.exists():
        return {
            'status': 'failed',
            'reason': f'Expected TFLite file was not created: {float32_source}',
            'command': command,
        }
    shutil.copy2(float32_source, tflite_path)
    if float16_source.exists():
        shutil.copy2(float16_source, tflite_float16_path)
    return {
        'status': 'ok',
        'saved_model_dir': str(converted_root),
        'float32_tflite': str(tflite_path),
        'float16_tflite': str(tflite_float16_path) if tflite_float16_path.exists() else '',
        'command': command,
    }


def _parity_rows(output_root, classes):
    rows = [row for row in _read_manifest(Path(output_root) / 'dataset_manifest.csv') if row['split'] == 'test']
    selected = []
    seen = set()
    for row in rows:
        class_name = row['class_name']
        if class_name in seen:
            continue
        selected.append(row)
        seen.add(class_name)
        if len(selected) == len(classes):
            break
    return selected or rows[:8]


def _tflite_parity_check(output_root, torch_model, checkpoint, tflite_path):
    if not tflite_path:
        return {'status': 'skipped', 'reason': 'TFLite model not available.'}
    try:
        import tensorflow as tf
    except Exception as exc:
        return {'status': 'skipped', 'reason': f'TensorFlow not available: {exc}'}
    rows = _parity_rows(output_root, checkpoint['classes'])
    transform = _transforms(checkpoint['input_size'], checkpoint['normalization_mean'], checkpoint['normalization_std'], train=False)
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    mean = np.array(checkpoint['normalization_mean'], dtype=np.float32)
    std = np.array(checkpoint['normalization_std'], dtype=np.float32)
    same = 0
    max_logit_diff = 0.0
    comparisons = []
    torch_model.eval()
    with torch.no_grad():
        for row in rows:
            with Image.open(row['filepath']) as image:
                image = image.convert('RGB')
                torch_logits = torch_model(transform(image).unsqueeze(0)).detach().cpu().numpy()[0]
                array = np.asarray(
                    image.resize((checkpoint['input_size'], checkpoint['input_size'])),
                    dtype=np.float32,
                ) / 255.0
            array = ((array - mean) / std)[None, ...].astype(np.float32)
            interpreter.set_tensor(input_details['index'], array)
            interpreter.invoke()
            tflite_logits = interpreter.get_tensor(output_details['index'])[0]
            torch_pred = int(np.argmax(torch_logits))
            tflite_pred = int(np.argmax(tflite_logits))
            diff = float(np.max(np.abs(torch_logits - tflite_logits)))
            same += int(torch_pred == tflite_pred)
            max_logit_diff = max(max_logit_diff, diff)
            comparisons.append({
                'filepath': row['filepath'],
                'true_class': row['class_name'],
                'torch_prediction': checkpoint['classes'][torch_pred],
                'tflite_prediction': checkpoint['classes'][tflite_pred],
                'max_logit_diff': diff,
            })
    report = {
        'status': 'ok',
        'samples': len(rows),
        'same_predictions': same,
        'same_prediction_rate': same / max(1, len(rows)),
        'max_logit_diff': max_logit_diff,
        'input_details': {
            'name': input_details['name'],
            'shape': input_details['shape'].tolist(),
            'dtype': str(input_details['dtype']),
        },
        'output_details': {
            'name': output_details['name'],
            'shape': output_details['shape'].tolist(),
            'dtype': str(output_details['dtype']),
        },
        'comparisons': comparisons,
    }
    export_root = Path(output_root) / 'mobile_export'
    (export_root / 'tflite_parity_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def _interpret(history, metrics):
    if not history:
        return []
    last = history[-1]
    messages = []
    train_acc = float(last['train_accuracy'])
    val_acc = float(last['validation_accuracy'])
    train_loss_values = [float(row['train_loss']) for row in history]
    val_loss_values = [float(row['validation_loss']) for row in history]
    if train_acc - val_acc > 0.15:
        messages.append(f'Possivel overfitting: train accuracy {train_acc:.3f} bem acima de validation accuracy {val_acc:.3f}.')
    if len(history) >= 4 and val_loss_values[-1] > min(val_loss_values[:-1]) * 1.20 and train_loss_values[-1] < train_loss_values[0]:
        messages.append('Possivel overfitting: validation loss subiu enquanto train loss caiu.')
    if train_acc < 0.60 and val_acc < 0.60:
        messages.append('Possivel underfitting: train e validation accuracy permanecem baixas.')
    if not messages:
        messages.append('Nao ha evidencia forte de overfitting ou underfitting apenas pelas curvas atuais.')
    low_recall = [name for name, row in metrics['per_class'].items() if row['support'] > 0 and row['recall'] < 0.70]
    low_precision = [name for name, row in metrics['per_class'].items() if row['support'] > 0 and row['precision'] < 0.70]
    if low_recall:
        messages.append('Classes com recall baixo: ' + ', '.join(low_recall))
    if low_precision:
        messages.append('Classes com precision baixa: ' + ', '.join(low_precision))
    return messages


def _final_report(output_root, dataset_summary, split_summary, training_summary, metrics, export_manifest):
    output_root = Path(output_root)
    history = []
    with (output_root / 'training_history.csv').open(newline='', encoding='utf-8') as file:
        history = list(csv.DictReader(file))
    cm = metrics['confusion_matrix']
    classes = dataset_summary['classes']
    confusions = []
    for i, target in enumerate(classes):
        for j, pred in enumerate(classes):
            if i != j and cm[i][j] > 0:
                confusions.append((cm[i][j], target, pred))
    confusions.sort(reverse=True)
    lines = ['# MVP classifier report', '']
    lines.append(f"Dataset detectado: `{dataset_summary['input_root']}`")
    lines.append(f"Classes: {', '.join(classes)}")
    lines.append(f"Arquitetura: {training_summary['architecture']} | pesos: {training_summary['weights']}")
    lines.append(f"Melhor epoca: {training_summary['best_epoch']}")
    lines.append('')
    lines.append('## Distribuicao')
    lines.append(json.dumps(split_summary, indent=2, ensure_ascii=False))
    lines.append('')
    lines.append('## Metricas globais')
    lines.append(f"- accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"- macro F1: {metrics['macro_f1']:.4f}")
    lines.append(f"- weighted F1: {metrics['weighted_f1']:.4f}")
    lines.append(f"- top-3 accuracy: {metrics['top3_accuracy']:.4f}")
    lines.append('')
    lines.append('## Metricas por classe')
    for class_name, row in metrics['per_class'].items():
        lines.append(f"- {class_name}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}, support={row['support']}, acertos={row['correct']}, erros={row['errors']}")
    lines.append('')
    lines.append('## Principais confusoes')
    if confusions:
        for count, target, pred in confusions[:10]:
            lines.append(f"- {target} -> {pred}: {count}")
    else:
        lines.append('- Nenhuma confusao no conjunto de teste.')
    lines.append('')
    lines.append('## Interpretacao')
    for message in _interpret(history, metrics):
        lines.append(f'- {message}')
    lines.append('')
    lines.append('## Unknown')
    if 'unknown' in classes:
        lines.append('Neste MVP, `unknown` e apenas uma classe supervisionada com os objetos negativos disponiveis. Isso nao garante reconhecimento de todo objeto desconhecido possivel.')
    else:
        lines.append('Este modelo foi treinado apenas com classes conhecidas. Para o aplicativo, `unknown` deve ser tratado como fallback de inferencia: se a maior probabilidade ficar abaixo do corte definido na validacao real, retornar `unknown` e exibir mensagem de nao identificacao.')
    lines.append('')
    lines.append('## Exportacao')
    lines.append(json.dumps(export_manifest.get('exported_files', {}), indent=2, ensure_ascii=False))
    if export_manifest.get('export_errors'):
        lines.append('Erros/limitacoes de exportacao:')
        lines.append(json.dumps(export_manifest['export_errors'], indent=2, ensure_ascii=False))
    (output_root / 'final_report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_pipeline(args):
    input_root = discover_input_root(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if _is_split_dataset(input_root):
        dataset_summary, split_summary = prepare_existing_split_dataset(input_root, output_root)
    else:
        dataset_summary, split_summary = prepare_folder_dataset(input_root, output_root, args.seed, args.extract_video_fps)
    training_summary, _ = train_model(
        output_root,
        experiment_name=args.experiment_name,
        seed=args.seed,
        input_size=args.input_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    metrics = evaluate_model(output_root, training_summary['checkpoint'])
    export_manifest = export_mobile(output_root, training_summary['checkpoint'])
    _final_report(output_root, dataset_summary, split_summary, training_summary, metrics, export_manifest)
    return {
        'input_root': str(input_root),
        'classes': dataset_summary['classes'],
        'dataset_summary': str(output_root / 'dataset_summary.json'),
        'split_summary': str(output_root / 'split_summary.json'),
        'training_summary': str(output_root / 'training_summary.json'),
        'test_metrics': str(output_root / 'test_metrics.json'),
        'final_report': str(output_root / 'final_report.md'),
        'mobile_export': str(output_root / 'mobile_export'),
    }


def main():
    parser = argparse.ArgumentParser(description='Run the simplified MVP image-classification pipeline.')
    parser.add_argument('--input-root', default=None)
    parser.add_argument('--output-root', default='reports/mvp_classifier')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extract-video-fps', type=float, default=1.0)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--experiment-name', default='mvp_classifier')
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
