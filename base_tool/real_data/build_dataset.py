import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from base_tool.real_data.validate_metadata import (
    ALLOWED_SPLIT_ROLES,
    KNOWN_CLASSES,
    UNKNOWN_CLASS,
    validate_metadata,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def _safe_name(path):
    source = Path(path)
    stem = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in source.stem)
    return f'{stem}{source.suffix.lower()}'


def _destination(base_root, row):
    split_role = row['split_role'].strip()
    true_class = row['true_class'].strip()
    if split_role == 'calibration_known':
        return base_root / 'real_multiclass_v1' / 'calibration_known' / true_class
    if split_role == 'test_known':
        return base_root / 'real_multiclass_v1' / 'test_known' / true_class
    if split_role == 'calibration_unknown':
        return base_root / 'unknown_multiclass_v1' / 'calibration'
    if split_role == 'test_unknown':
        return base_root / 'unknown_multiclass_v1' / 'test'
    raise ValueError(f'Unsupported materialization split_role: {split_role}')


def build_dataset(metadata_path, output_root, copy_mode='copy', require_ready=False, resume=False):
    if copy_mode != 'copy':
        raise ValueError('Only --copy-mode copy is supported in this version.')

    metadata_path = Path(metadata_path)
    output_root = Path(output_root)
    validation = validate_metadata(metadata_path, require_ready=require_ready)
    if require_ready and not validation['valid']:
        raise ValueError('Metadata validation failed; refusing to materialize datasets.')

    rows = _read_rows(metadata_path)
    files_manifest = []
    excluded = []
    copied = []
    coverage = {class_name: {'calibration_known': 0, 'test_known': 0} for class_name in sorted(KNOWN_CLASSES)}
    unknown_coverage = {'calibration_unknown': 0, 'test_unknown': 0}

    for row in rows:
        split_role = row.get('split_role', '').strip()
        if split_role == 'exclude' or not split_role:
            excluded.append({'file_path': row.get('file_path', ''), 'reason': 'exclude_or_empty_split_role'})
            continue
        if split_role not in ALLOWED_SPLIT_ROLES:
            excluded.append({'file_path': row.get('file_path', ''), 'reason': f'invalid_split_role:{split_role}'})
            continue

        source_path = Path(row['file_path'])
        if not source_path.exists():
            excluded.append({'file_path': row.get('file_path', ''), 'reason': 'source_missing'})
            continue
        actual_hash = _sha256(source_path)
        if row.get('sha256') and actual_hash != row['sha256']:
            raise ValueError(f"Hash mismatch for {source_path}: metadata={row['sha256']} actual={actual_hash}")

        destination_dir = _destination(output_root, row)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / _safe_name(source_path)
        if destination_path.exists() and not resume:
            raise FileExistsError(f'Destination already exists: {destination_path}. Use --resume to keep existing files.')
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)

        copied.append(str(destination_path))
        if split_role in {'calibration_known', 'test_known'}:
            coverage[row['true_class'].strip()][split_role] += 1
        if split_role in {'calibration_unknown', 'test_unknown'}:
            unknown_coverage[split_role] += 1
        files_manifest.append({
            'source_path': str(source_path),
            'destination_path': str(destination_path),
            'sha256': actual_hash,
            'media_type': row.get('media_type', ''),
            'true_class': row.get('true_class', ''),
            'is_unknown': row.get('is_unknown', ''),
            'split_role': split_role,
            'session_id': row.get('session_id', ''),
            'source_group': row.get('source_group', ''),
            'source_video': row.get('source_video', ''),
            'frame_index': row.get('frame_index', ''),
            'timestamp_seconds': row.get('timestamp_seconds', ''),
            'device': row.get('device', ''),
            'view': row.get('view', ''),
            'background': row.get('background', ''),
            'lighting': row.get('lighting', ''),
            'notes': row.get('notes', ''),
        })

    manifest = {
        'dataset_name': 'real_multiclass_v1_with_unknown',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'metadata_path': str(metadata_path),
        'output_root': str(output_root),
        'copy_mode': copy_mode,
        'num_files_copied': len(copied),
        'num_files_excluded': len(excluded),
        'known_classes': sorted(KNOWN_CLASSES),
        'unknown_label': UNKNOWN_CLASS,
        'validation': {
            'valid': validation['valid'],
            'errors': validation['errors'],
            'warnings': validation['warnings'],
            'classes_without_coverage': validation['classes_without_coverage'],
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_root / 'coverage_report.json').write_text(
        json.dumps({'known': coverage, 'unknown': unknown_coverage, 'excluded': excluded}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    with (output_root / 'files_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'source_path',
            'destination_path',
            'sha256',
            'media_type',
            'true_class',
            'is_unknown',
            'split_role',
            'session_id',
            'source_group',
            'source_video',
            'frame_index',
            'timestamp_seconds',
            'device',
            'view',
            'background',
            'lighting',
            'notes',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_manifest)
    with (output_root / 'excluded_files.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['file_path', 'reason'])
        writer.writeheader()
        writer.writerows(excluded)

    return {
        'dataset_manifest': str(output_root / 'dataset_manifest.json'),
        'files_manifest': str(output_root / 'files_manifest.csv'),
        'coverage_report': str(output_root / 'coverage_report.json'),
        'excluded_files': str(output_root / 'excluded_files.csv'),
        'num_files_copied': len(copied),
        'num_files_excluded': len(excluded),
    }


def main():
    parser = argparse.ArgumentParser(description='Materialize real known/unknown datasets from validated metadata.')
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--copy-mode', default='copy', choices=['copy'])
    parser.add_argument('--require-ready', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = build_dataset(args.metadata, args.output_root, args.copy_mode, args.require_ready, args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
