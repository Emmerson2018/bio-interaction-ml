import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


KNOWN_CLASSES = {
    'boto_cor_de_rosa',
    'capivara',
    'onca_pintada',
    'sapo',  # ID historico dos datasets usados no experimento R1.
    'sapo_flecha',
}
UNKNOWN_CLASS = 'unknown'
ALLOWED_CLASSES = KNOWN_CLASSES | {UNKNOWN_CLASS}
ALLOWED_SPLIT_ROLES = {
    'calibration_known',
    'test_known',
    'calibration_unknown',
    'test_unknown',
    'exclude',
}
REQUIRED_FIELDS = {
    'true_class',
    'is_unknown',
    'split_role',
    'source_group',
    'session_id',
    'device',
    'view',
    'background',
    'lighting',
}
CALIBRATION_SPLITS = {'calibration_known', 'calibration_unknown'}
TEST_SPLITS = {'test_known', 'test_unknown'}
KNOWN_SPLITS = {'calibration_known', 'test_known'}
UNKNOWN_SPLITS = {'calibration_unknown', 'test_unknown'}


def _read_rows(metadata_path):
    with Path(metadata_path).open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def _truthy(value):
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'sim'}


def _split_from_path(file_path):
    normalized = str(file_path).replace('\\', '/')
    if 'datasets/real_multiclass_v1/calibration_known/' in normalized:
        return 'calibration_known'
    if 'datasets/real_multiclass_v1/test_known/' in normalized:
        return 'test_known'
    if 'datasets/unknown_multiclass_v1/calibration/' in normalized:
        return 'calibration_unknown'
    if 'datasets/unknown_multiclass_v1/test/' in normalized:
        return 'test_unknown'
    return 'unassigned'


def _row_split(row):
    split = str(row.get('split_role') or row.get('split') or row.get('dataset_split') or '').strip()
    return split or _split_from_path(row.get('file_path', ''))


def _split_group(split):
    if split in CALIBRATION_SPLITS:
        return 'calibration'
    if split in TEST_SPLITS:
        return 'test'
    return 'unassigned'


def _video_group(row):
    source_video = str(row.get('source_video') or row.get('video_id') or '').strip()
    if source_video:
        return source_video
    if str(row.get('media_type', '')).strip().lower() == 'video':
        return str(row.get('file_path', '')).strip()
    return ''


def _is_known_class(true_class):
    return true_class in KNOWN_CLASSES


def _is_unknown_class(true_class):
    return true_class == UNKNOWN_CLASS


def validate_metadata(metadata_path, require_ready=False):
    rows = _read_rows(metadata_path)
    errors = []
    warnings = []
    row_reports = []
    files_by_path = defaultdict(set)
    files_by_hash = defaultdict(set)
    sessions = defaultdict(set)
    source_groups = defaultdict(set)
    videos = defaultdict(set)
    roles_by_path = defaultdict(set)
    roles_by_hash = defaultdict(set)
    roles_by_video = defaultdict(set)
    coverage = {class_name: 0 for class_name in sorted(KNOWN_CLASSES)}
    unknown_count = 0

    for index, row in enumerate(rows, start=2):
        file_path = str(row.get('file_path', '')).strip()
        split = _row_split(row)
        group = _split_group(split)
        true_class = str(row.get('true_class', '')).strip()
        is_unknown_raw = str(row.get('is_unknown', '')).strip()
        missing_required = [field for field in sorted(REQUIRED_FIELDS) if not str(row.get(field, '')).strip()]
        source_group = str(row.get('source_group', '')).strip()

        if missing_required:
            message = {
                'row': index,
                'file_path': file_path,
                'code': 'missing_required_fields',
                'fields': missing_required,
            }
            if require_ready:
                errors.append(message)
            else:
                warnings.append(message)

        if split and split != 'unassigned' and split not in ALLOWED_SPLIT_ROLES:
            errors.append({
                'row': index,
                'file_path': file_path,
                'code': 'invalid_split_role',
                'value': split,
                'allowed': sorted(ALLOWED_SPLIT_ROLES),
            })

        if true_class and true_class not in ALLOWED_CLASSES:
            errors.append({
                'row': index,
                'file_path': file_path,
                'code': 'invalid_true_class',
                'value': true_class,
                'allowed': sorted(ALLOWED_CLASSES),
            })

        if is_unknown_raw:
            is_unknown = _truthy(is_unknown_raw)
            if _is_unknown_class(true_class) and not is_unknown:
                errors.append({
                    'row': index,
                    'file_path': file_path,
                    'code': 'is_unknown_false_for_unknown_class',
                })
            if _is_known_class(true_class) and is_unknown:
                errors.append({
                    'row': index,
                    'file_path': file_path,
                    'code': 'is_unknown_true_for_known_class',
                })

        if _is_known_class(true_class) and split in UNKNOWN_SPLITS:
            errors.append({
                'row': index,
                'file_path': file_path,
                'code': 'known_class_in_unknown_split',
                'true_class': true_class,
                'split_role': split,
            })
        if _is_unknown_class(true_class) and split in KNOWN_SPLITS:
            errors.append({
                'row': index,
                'file_path': file_path,
                'code': 'unknown_class_in_known_split',
                'true_class': true_class,
                'split_role': split,
            })
        if _is_known_class(true_class) and split != 'exclude':
            coverage[true_class] += 1
        if _is_unknown_class(true_class) and split != 'exclude':
            unknown_count += 1

        if file_path:
            files_by_path[file_path].add(group)
            if split not in {'', 'unassigned', 'exclude'}:
                roles_by_path[file_path].add(split)
        digest = str(row.get('sha256', '')).strip()
        if digest:
            files_by_hash[digest].add(group)
            if split not in {'', 'unassigned', 'exclude'}:
                roles_by_hash[digest].add(split)
        session_id = str(row.get('session_id', '')).strip()
        if session_id:
            sessions[session_id].add(group)
        if source_group:
            source_groups[source_group].add(group)
        video_group = _video_group(row)
        if video_group:
            videos[video_group].add(group)
            if split not in {'', 'unassigned', 'exclude'}:
                roles_by_video[video_group].add(split)

        row_reports.append({
            'row': index,
            'file_path': file_path,
            'split_role': split,
            'split_group': group,
            'true_class': true_class,
            'is_unknown': is_unknown_raw,
            'source_group': source_group,
            'missing_required_fields': missing_required,
        })

    for file_path, groups in files_by_path.items():
        if 'calibration' in groups and 'test' in groups:
            errors.append({'code': 'same_file_in_calibration_and_test', 'file_path': file_path, 'groups': sorted(groups)})
    for digest, groups in files_by_hash.items():
        if 'calibration' in groups and 'test' in groups:
            errors.append({'code': 'same_hash_in_calibration_and_test', 'sha256': digest, 'groups': sorted(groups)})
    for file_path, roles in roles_by_path.items():
        if len(roles) > 1:
            errors.append({'code': 'same_file_in_more_than_one_split', 'file_path': file_path, 'split_roles': sorted(roles)})
    for digest, roles in roles_by_hash.items():
        if len(roles) > 1:
            errors.append({'code': 'same_hash_in_more_than_one_split', 'sha256': digest, 'split_roles': sorted(roles)})
    for session_id, groups in sessions.items():
        if 'calibration' in groups and 'test' in groups:
            errors.append({'code': 'same_session_in_calibration_and_test', 'session_id': session_id, 'groups': sorted(groups)})
    for source_group, groups in source_groups.items():
        if 'calibration' in groups and 'test' in groups:
            errors.append({'code': 'same_source_group_in_calibration_and_test', 'source_group': source_group, 'groups': sorted(groups)})
    for video_group, groups in videos.items():
        if 'calibration' in groups and 'test' in groups:
            errors.append({'code': 'same_video_in_calibration_and_test', 'video_group': video_group, 'groups': sorted(groups)})
    for video_group, roles in roles_by_video.items():
        if len(roles) > 1:
            errors.append({'code': 'same_video_in_more_than_one_split', 'video_group': video_group, 'split_roles': sorted(roles)})

    classes_without_coverage = [class_name for class_name, count in coverage.items() if count == 0]

    return {
        'metadata_path': str(metadata_path),
        'valid': not errors,
        'require_ready': require_ready,
        'num_rows': len(rows),
        'allowed_true_class': sorted(ALLOWED_CLASSES),
        'allowed_split_role': sorted(ALLOWED_SPLIT_ROLES),
        'required_fields_before_evaluation': sorted(REQUIRED_FIELDS),
        'coverage_by_known_class': coverage,
        'unknown_count': unknown_count,
        'classes_without_coverage': classes_without_coverage,
        'errors': errors,
        'warnings': warnings,
        'rows': row_reports,
    }


def _write_csv(path, rows):
    fieldnames = ['row', 'file_path', 'split_role', 'split_group', 'true_class', 'is_unknown', 'source_group', 'missing_required_fields']
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output['missing_required_fields'] = ';'.join(output['missing_required_fields'])
            writer.writerow(output)


def main():
    parser = argparse.ArgumentParser(description='Validate real-media metadata before real/unknown evaluation.')
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-json', default='reports/multiclass_v1/real_metadata_validation.json')
    parser.add_argument('--output-csv', default='reports/multiclass_v1/real_metadata_validation_rows.csv')
    parser.add_argument('--require-ready', action='store_true')
    args = parser.parse_args()

    result = validate_metadata(args.metadata, require_ready=args.require_ready)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_csv(output_csv, result['rows'])
    print(json.dumps({key: result[key] for key in ['valid', 'num_rows', 'errors', 'warnings']}, indent=2, ensure_ascii=False))
    if not result['valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
