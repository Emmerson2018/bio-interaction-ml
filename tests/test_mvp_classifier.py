import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from base_tool.mvp_classifier import (
    discover_input_root,
    normalize_class_name,
    prepare_existing_split_dataset,
    prepare_folder_dataset,
)


def _write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :, 0] = value
    array[:, :, 1] = np.arange(32, dtype=np.uint8)[None, :]
    array[:, :, 2] = np.arange(32, dtype=np.uint8)[:, None]
    Image.fromarray(array).save(path)


def _write_video(path, seconds=9, fps=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (32, 32))
    for frame_idx in range(seconds * fps):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = (frame_idx * 17) % 255
        frame[:, :, 1] = np.arange(32, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.arange(32, dtype=np.uint8)[:, None]
        writer.write(frame)
    writer.release()


def _read_manifest(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def test_normalize_class_name_aliases_unknown_typo():
    assert normalize_class_name('UNKNOW') == 'unknown'
    assert normalize_class_name('SAPO') == 'sapo'
    assert normalize_class_name('Bicho Preguiça') == 'bicho_preguica'


def test_discover_input_root_finds_folder_dataset(tmp_path):
    root = tmp_path / 'dataset'
    _write_image(root / 'SAPO' / 'a.jpg', 40)
    _write_image(root / 'UNKNOW' / 'b.jpg', 90)
    assert discover_input_root(root) == root


def test_prepare_folder_dataset_extracts_video_frames_and_manifest(tmp_path):
    root = tmp_path / 'dataset'
    for idx in range(3):
        _write_image(root / 'SAPO' / f'sapo_{idx}.jpg', 30 + idx)
        _write_image(root / 'UNKNOW' / f'unknown_{idx}.jpg', 90 + idx)
    _write_video(root / 'SAPO' / 'sapo_video.mp4', seconds=3, fps=2)
    output = tmp_path / 'reports'

    summary, split_summary = prepare_folder_dataset(root, output, seed=42, extract_video_fps=1)
    manifest = _read_manifest(output / 'dataset_manifest.csv')

    assert summary['classes'] == ['sapo', 'unknown']
    assert any(row['media_type'] == 'video_frame' for row in manifest)
    assert {'filepath', 'class_name', 'class_index', 'media_type', 'source_file', 'source_group', 'timestamp_seconds', 'split'} <= set(manifest[0])
    assert set(split_summary) == {'train', 'validation', 'test'}


def test_unknown_special_video_keeps_only_seconds_zero_to_six(tmp_path):
    root = tmp_path / 'dataset'
    for idx in range(3):
        _write_image(root / 'SAPO' / f'sapo_{idx}.jpg', 30 + idx)
        _write_image(root / 'UNKNOW' / f'unknown_{idx}.jpg', 90 + idx)
    _write_video(root / 'UNKNOW' / 'unknown_video.mp4', seconds=10, fps=2)
    output = tmp_path / 'reports'

    summary, _ = prepare_folder_dataset(root, output, seed=42, extract_video_fps=1)
    manifest = _read_manifest(output / 'dataset_manifest.csv')
    unknown_frame_seconds = sorted(
        int(row['timestamp_seconds'])
        for row in manifest
        if row['class_name'] == 'unknown' and row['media_type'] == 'video_frame'
    )

    assert unknown_frame_seconds == [0, 1, 2, 3, 4, 5, 6]
    exclusions = summary['reports']['video_exclusions']
    assert exclusions
    assert all(item['timestamp_seconds'] >= 7 for item in exclusions)
    assert summary['reports']['special_unknown_video'].endswith('unknown_video.mp4')


def test_video_source_group_does_not_cross_splits(tmp_path):
    root = tmp_path / 'dataset'
    for idx in range(4):
        _write_image(root / 'SAPO' / f'sapo_{idx}.jpg', 40 + idx)
        _write_image(root / 'UNKNOW' / f'unknown_{idx}.jpg', 100 + idx)
    _write_video(root / 'SAPO' / 'sapo_video.mp4', seconds=5, fps=2)
    output = tmp_path / 'reports'

    prepare_folder_dataset(root, output, seed=42, extract_video_fps=1)
    manifest = _read_manifest(output / 'dataset_manifest.csv')
    splits_by_group = {}
    for row in manifest:
        splits_by_group.setdefault(row['source_group'], set()).add(row['split'])

    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_class_index_mapping_is_deterministic(tmp_path):
    root = tmp_path / 'dataset'
    _write_image(root / 'SAPO' / 'a.jpg', 40)
    _write_image(root / 'UNKNOW' / 'b.jpg', 90)
    output = tmp_path / 'reports'

    summary, _ = prepare_folder_dataset(root, output, seed=42, extract_video_fps=1)
    manifest = _read_manifest(output / 'dataset_manifest.csv')

    assert summary['classes'] == ['sapo', 'unknown']
    assert {row['class_name']: int(row['class_index']) for row in manifest} == {'sapo': 0, 'unknown': 1}


def test_existing_split_dataset_uses_class_index_and_val_alias(tmp_path):
    root = tmp_path / 'split_dataset'
    root.mkdir()
    (root / 'class_index.csv').write_text('class_name,class_index\ncapivara,0\nsapo,1\n', encoding='utf-8')
    for split_idx, split in enumerate(['train', 'val', 'test']):
        _write_image(root / split / 'capivara' / f'{split}_capivara.jpg', 40 + split_idx)
        _write_image(root / split / 'sapo' / f'{split}_sapo.jpg', 90 + split_idx)
    output = tmp_path / 'reports'

    summary, split_summary = prepare_existing_split_dataset(root, output)
    manifest = _read_manifest(output / 'dataset_manifest.csv')

    assert summary['layout'] == 'existing_train_val_test'
    assert summary['classes'] == ['capivara', 'sapo']
    assert {row['class_name']: int(row['class_index']) for row in manifest} == {'capivara': 0, 'sapo': 1}
    assert split_summary['validation'] == {'capivara': 1, 'sapo': 1}
