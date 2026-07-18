import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageStat


SUPPORTED_MODEL_SUFFIXES = {'.blend', '.blend1', '.fbx', '.obj', '.glb', '.gltf'}
DEFAULT_ANGLES = list(range(0, 360, 30))
PILOT_RADII = ['near', 'far']
RENDER_OPTION_KEYS = {
    'front_angle_degrees',
    'camera_radius',
    'pitch_degrees',
    'roll_degrees',
    'camera_lens',
    'light_energy',
    'scale',
    'vertical_offset',
    'view_angle_degrees',
}
DEFAULT_CONTACT_VIEWS = [
    {'name': 'front', 'view_angle_degrees': 0, 'camera_radius': 4.0, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'three_quarter_left', 'view_angle_degrees': -35, 'camera_radius': 4.0, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'three_quarter_right', 'view_angle_degrees': 35, 'camera_radius': 4.0, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'left', 'view_angle_degrees': -90, 'camera_radius': 4.1, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'right', 'view_angle_degrees': 90, 'camera_radius': 4.1, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'back', 'view_angle_degrees': 180, 'camera_radius': 4.2, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'close', 'view_angle_degrees': 0, 'camera_radius': 3.0, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'far', 'view_angle_degrees': 0, 'camera_radius': 5.4, 'pitch_degrees': 72, 'roll_degrees': 0},
    {'name': 'pitch_low', 'view_angle_degrees': 25, 'camera_radius': 4.2, 'pitch_degrees': 62, 'roll_degrees': 0},
    {'name': 'pitch_high', 'view_angle_degrees': -25, 'camera_radius': 4.2, 'pitch_degrees': 82, 'roll_degrees': 0},
    {'name': 'roll_left', 'view_angle_degrees': 20, 'camera_radius': 4.0, 'pitch_degrees': 72, 'roll_degrees': -6},
    {'name': 'roll_right', 'view_angle_degrees': -20, 'camera_radius': 4.0, 'pitch_degrees': 72, 'roll_degrees': 6},
]


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path):
    with path.open('r', encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader) or {}


def _load_class_mapping(root_path, config):
    mapping_file = _resolve_path(root_path, config['class_mapping_file'])
    data = _load_yaml(mapping_file)
    mapping = data.get('class_mapping', {})
    if not mapping:
        raise ValueError(f'No class_mapping found in {mapping_file}')
    canonical_values = list(mapping.values())
    duplicates = sorted({value for value in canonical_values if canonical_values.count(value) > 1})
    if duplicates:
        raise ValueError(f'Duplicate canonical class ids found: {duplicates}')
    return mapping_file, mapping


def _discover_models(source_root, class_mapping):
    rows = []
    mapped_folders = set(class_mapping)
    for source_folder, canonical_class in sorted(class_mapping.items(), key=lambda item: item[1]):
        folder = source_root / source_folder
        files = []
        if folder.exists():
            files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES)
        if not files:
            rows.append({
                'source_folder': source_folder,
                'source_file': '',
                'extension': '',
                'size_bytes': '',
                'sha256': '',
                'canonical_class': canonical_class,
                'status': 'missing_model_file',
                'notes': f'No supported model file found in {folder}',
                'path': None,
            })
            continue
        for index, path in enumerate(files):
            rows.append({
                'source_folder': source_folder,
                'source_file': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': path.stat().st_size,
                'sha256': _sha256(path),
                'canonical_class': canonical_class,
                'status': 'ok' if index == 0 else 'extra_model_file',
                'notes': '',
                'path': path,
            })
    for folder in sorted(path for path in source_root.iterdir() if path.is_dir() and path.name not in mapped_folders):
        files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES)
        for path in files:
            rows.append({
                'source_folder': folder.name,
                'source_file': path.name,
                'extension': path.suffix.lower(),
                'size_bytes': path.stat().st_size,
                'sha256': _sha256(path),
                'canonical_class': '',
                'status': 'unmapped_source_folder',
                'notes': 'Supported model file found, but the source folder is not present in class_mapping.',
                'path': path,
            })
    return rows


def _write_inventory(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'source_folder',
        'source_file',
        'extension',
        'size_bytes',
        'sha256',
        'canonical_class',
        'status',
        'notes',
        'object_count',
        'mesh_count',
        'has_armature',
        'bone_count',
        'has_constraints',
        'has_shape_keys',
        'action_count',
        'movable_parts',
        'embedded_props',
        'canonical_pose_required',
        'pose_variants_proposed',
    ]
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _write_class_index(path, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['class_name', 'class_index'])
        for index, class_name in enumerate(classes):
            writer.writerow([class_name, index])


def _find_blender_executable(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    path_blender = shutil.which('blender')
    if path_blender:
        candidates.append(Path(path_blender))
    candidates.extend(Path('C:/Program Files/Blender Foundation').glob('Blender */blender.exe'))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _renderer_command(config, render_script, render_args):
    backend = config.get('renderer_backend', 'python_bpy')
    if backend == 'python_bpy':
        python_executable = config.get('python_executable') or sys.executable
        return [python_executable, str(render_script), *render_args]
    if backend == 'blender_executable':
        blender = _find_blender_executable(config.get('blender_executable_path') or config.get('blender_executable'))
        if not blender:
            raise FileNotFoundError(
                'Blender executable was not found. Set blender_executable_path in the YAML or install Blender.'
            )
        return [str(blender), '--background', '--python', str(render_script), '--', *render_args]
    raise ValueError(f'Unsupported renderer_backend: {backend}')


def _parse_last_json_line(stdout):
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            return json.loads(line)
    raise json.JSONDecodeError('No JSON object found in renderer stdout.', stdout, 0)


def _run_renderer(config, render_script, model_path, output_path, seed, view_options):
    args = [
        '--model-path', str(model_path),
        '--output-path', str(output_path),
        '--image-size', str(config.get('image_size', 224)),
        '--seed', str(seed),
        '--render-engine', str(config.get('render_engine', 'BLENDER_EEVEE')),
        '--device', str(config.get('device', 'CPU')),
    ]
    for key, value in view_options.items():
        if key not in RENDER_OPTION_KEYS:
            continue
        args.extend([f'--{key.replace("_", "-")}', str(value)])
    command = _renderer_command(config, render_script, args)
    started_at = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    elapsed = time.perf_counter() - started_at
    payload = {}
    if completed.stdout.strip():
        try:
            payload = _parse_last_json_line(completed.stdout)
        except json.JSONDecodeError:
            payload = {'stdout': completed.stdout}
    payload['elapsed_seconds_total'] = round(elapsed, 3)
    payload['stderr'] = completed.stderr.strip()
    return payload


def _run_batch_renderer(config, render_script, plan_path, model_path, render_options=None):
    args = [
        '--model-path', str(model_path),
        '--batch-plan', str(plan_path),
        '--image-size', str(config.get('image_size', 224)),
        '--seed', str(config.get('seed', 42)),
        '--render-engine', str(config.get('render_engine', 'BLENDER_EEVEE')),
        '--device', str(config.get('device', 'CPU')),
    ]
    for key, value in (render_options or {}).items():
        if key not in RENDER_OPTION_KEYS:
            continue
        args.extend([f'--{key.replace("_", "-")}', str(value)])
    command = _renderer_command(config, render_script, args)
    started_at = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = _parse_last_json_line(completed.stdout)
    payload['elapsed_seconds_total'] = round(time.perf_counter() - started_at, 3)
    payload['stderr'] = completed.stderr.strip()
    return payload


def _inspect_model_structure(config, render_script, model_path):
    args = ['--model-path', str(model_path), '--output-path', os.devnull, '--inspect-only']
    command = _renderer_command(config, render_script, args)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if not completed.stdout.strip():
        raise RuntimeError(f'No inspection payload produced for {model_path}')
    return _parse_last_json_line(completed.stdout)


def _inspect_rows(config, rows, reports_root):
    render_script = Path(__file__).with_name('render_blender_model.py')
    structure_report = {}
    for row in rows:
        if not row.get('path'):
            continue
        try:
            payload = _inspect_model_structure(config, render_script, row['path'])
            row.update({
                'object_count': payload.get('object_count', ''),
                'mesh_count': payload.get('mesh_count', ''),
                'has_armature': payload.get('has_armature', ''),
                'bone_count': payload.get('bone_count', ''),
                'has_constraints': payload.get('has_constraints', ''),
                'has_shape_keys': payload.get('has_shape_keys', ''),
                'action_count': payload.get('action_count', ''),
                'movable_parts': ';'.join(payload.get('movable_parts') or []),
                'embedded_props': ';'.join(payload.get('embedded_props') or []),
                'canonical_pose_required': payload.get('canonical_pose_required', ''),
                'pose_variants_proposed': ';'.join(payload.get('pose_variants_proposed') or []),
            })
            structure_report[row['canonical_class'] or row['source_folder']] = payload
        except Exception as exc:
            row['notes'] = (row.get('notes') or '').strip()
            row['notes'] = f"{row['notes']} inspection_error={exc}".strip()
            structure_report[row['canonical_class'] or row['source_folder']] = {'status': 'error', 'error': str(exc)}

    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / 'model_structure_report.json'
    report_path.write_text(json.dumps(structure_report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report_path


def _image_quality(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        stat = ImageStat.Stat(image)
        extrema = image.getextrema()
        gray = image.convert('L')
        bbox = ImageChops.difference(gray, Image.new('L', gray.size, int(stat.mean[0]))).getbbox()
        occupied_ratio = 0.0
        if bbox:
            occupied_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (gray.size[0] * gray.size[1])
        return {
            'width': image.size[0],
            'height': image.size[1],
            'mean': [round(value, 3) for value in stat.mean],
            'stddev': [round(value, 3) for value in stat.stddev],
            'extrema': extrema,
            'occupied_ratio_approx': round(occupied_ratio, 4),
            'near_uniform': max(stat.stddev) < 2.0,
            'sha256': _sha256(path),
        }


def _make_contact_sheet(image_paths, output_path, labels=None, thumb_size=(224, 224), columns=4):
    labels = labels or [path.stem for path in image_paths]
    rows = (len(image_paths) + columns - 1) // columns
    label_height = 28
    sheet = Image.new('RGB', (columns * thumb_size[0], rows * (thumb_size[1] + label_height)), 'white')
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image.thumbnail(thumb_size)
            x0 = (index % columns) * thumb_size[0]
            y0 = (index // columns) * (thumb_size[1] + label_height)
            x = x0 + (thumb_size[0] - image.width) // 2
            y = y0 + (thumb_size[1] - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((x0 + 6, y0 + thumb_size[1] + 6), str(labels[index])[:32], fill='black')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _class_options(config, class_name):
    common = dict(config.get('render_options') or {})
    specific = dict((config.get('class_render_options') or {}).get(class_name) or {})
    merged = {**common, **specific}
    merged.setdefault('front_angle_degrees', 0.0)
    merged.setdefault('camera_radius', 4.0)
    merged.setdefault('pitch_degrees', 72.0)
    merged.setdefault('roll_degrees', 0.0)
    merged.setdefault('camera_lens', 55.0)
    merged.setdefault('light_energy', 700.0)
    merged.setdefault('scale', 1.0)
    merged.setdefault('vertical_offset', 0.0)
    return {key: value for key, value in merged.items() if value is not None}


def _split_plan(dataset_kind):
    if dataset_kind == 'pilot':
        return {'train': 16, 'val': 4, 'test': 4}
    if dataset_kind == 'full':
        return {'train': 480, 'val': 60, 'test': 60}
    raise ValueError(f'Unsupported dataset kind: {dataset_kind}')


def _dataset_output_root(root_path, config, dataset_kind):
    if dataset_kind == 'pilot':
        return _resolve_path(root_path, config.get('pilot_output_root', 'datasets/generated/animals_multiclass_v1_pilot'))
    return _resolve_path(root_path, config.get('output_root', 'datasets/generated/animals_multiclass_v1'))


def _camera_radii(options, dataset_kind, count):
    min_radius = float(options.get('camera_radius_min', options.get('camera_radius', 4.0)))
    max_radius = float(options.get('camera_radius_max', min_radius))
    if count <= 1:
        return [min_radius]
    if dataset_kind == 'pilot':
        return [min_radius, max_radius]
    values = []
    for idx in range(count):
        ratio = idx / max(1, count - 1)
        values.append(min_radius + (max_radius - min_radius) * ratio)
    return values


def _frame_specs_for_class(config, class_name, class_index, dataset_kind, split_counts, class_seed):
    options = _class_options(config, class_name)
    angles = list(config.get('angle_sweep_degrees', DEFAULT_ANGLES))
    frames = []
    global_index = 0
    pilot_cursor = 0
    pilot_combinations = []
    if dataset_kind == 'pilot':
        radii = _camera_radii(options, dataset_kind, len(PILOT_RADII))
        for radius in radii:
            for angle in angles:
                pilot_combinations.append((angle, radius))
    for split, split_count in split_counts.items():
        if dataset_kind == 'pilot':
            if len(pilot_combinations) < pilot_cursor + split_count:
                raise ValueError(f'Not enough pilot angle/radius combinations for {class_name}.')
            selected = pilot_combinations[pilot_cursor:pilot_cursor + split_count]
            pilot_cursor += split_count
        else:
            split_offsets = {'train': 0.17, 'val': 0.43, 'test': 0.71}
            min_radius = float(options.get('camera_radius_min', options.get('camera_radius', 4.0)))
            max_radius = float(options.get('camera_radius_max', min_radius))
            split_offset = split_offsets.get(split, 0.0)
            radius_span = max_radius - min_radius
            selected = []
            for idx in range(split_count):
                angle = angles[(idx * 5 + class_index * 2 + int(split_offset * 10)) % len(angles)]
                ratio = (idx + 0.5 + split_offset) / (split_count + 1.0)
                selected.append((angle, min_radius + radius_span * ratio))
        for local_index, (angle, camera_radius) in enumerate(selected):
            seed = class_seed + global_index
            frames.append({
                'class_name': class_name,
                'class_index': class_index,
                'split': split,
                'seed': seed,
                'view_angle_degrees': float(angle),
                'front_angle_degrees': float(options.get('front_angle_degrees', 0.0)),
                'pitch_degrees': float(options.get('pitch_degrees', 72.0)),
                'roll_degrees': float(options.get('roll_degrees', 0.0)),
                'camera_radius': float(camera_radius),
                'object_rotation': float(options.get('front_angle_degrees', 0.0)),
                'object_scale': float(options.get('scale', 1.0)),
                'vertical_offset': float(options.get('vertical_offset', 0.0)),
                'pose_id': (options.get('pose_ids') or ['neutral'])[0],
                'articulation_state': 'neutral',
                'interactive_objects': ';'.join(options.get('interactive_objects') or []),
                'frame_index': global_index,
            })
            global_index += 1
    return frames


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _dataset_manifest(config, dataset_kind, classes, rows, output_root, source_rows):
    split_counts = {}
    for row in rows:
        split_counts.setdefault(row['split'], {})
        split_counts[row['split']].setdefault(row['class_name'], 0)
        split_counts[row['split']][row['class_name']] += 1
    return {
        'dataset_name': output_root.name,
        'dataset_version': '1.0.0-pilot' if dataset_kind == 'pilot' else '1.0.0',
        'task_type': 'image_classification',
        'source': 'blender_synthetic',
        'classes': classes,
        'num_classes': len(classes),
        'seed': int(config.get('seed', 42)),
        'image_size': int(config.get('image_size', 224)),
        'splits': split_counts,
        'renderer_backend': config.get('renderer_backend', 'python_bpy'),
        'render_engine': config.get('render_engine', 'BLENDER_EEVEE'),
        'source_models': [
            {
                'class_name': row['canonical_class'],
                'source_folder': row['source_folder'],
                'source_file': row['source_file'],
                'sha256': row['sha256'],
            }
            for row in source_rows
            if row.get('canonical_class')
        ],
        'generation_git_commit': _git_commit_hash(),
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def _git_commit_hash():
    try:
        completed = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True)
        return completed.stdout.strip()
    except Exception:
        return ''


def _summarize_dataset(rows, classes):
    summary_rows = []
    for split in sorted({row['split'] for row in rows}):
        for class_name in classes:
            class_rows = [row for row in rows if row['split'] == split and row['class_name'] == class_name]
            summary_rows.append({
                'split': split,
                'class_name': class_name,
                'count': len(class_rows),
                'near_uniform_count': sum(str(row.get('near_uniform')).lower() == 'true' for row in class_rows),
                'mean_frame_occupancy': round(
                    sum(float(row.get('frame_occupancy') or 0.0) for row in class_rows) / len(class_rows),
                    4,
                ) if class_rows else 0.0,
            })
    return summary_rows


def _duplicate_rows(rows):
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row.get('image_sha256'), []).append(row)
    duplicates = []
    for digest, digest_rows in by_hash.items():
        if digest and len(digest_rows) > 1:
            for row in digest_rows:
                duplicates.append({
                    'image_sha256': digest,
                    'image_path': row['image_path'],
                    'class_name': row['class_name'],
                    'split': row['split'],
                })
    return duplicates


def _corruption_rows(rows, output_root):
    corruptions = []
    for row in rows:
        image_path = output_root / row['image_path']
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            corruptions.append({
                'image_path': row['image_path'],
                'class_name': row['class_name'],
                'split': row['split'],
                'error': str(exc),
            })
    return corruptions


def _make_class_contact_sheets(rows, output_root, reports_root, classes):
    sheet_root = reports_root / 'class_contact_sheets'
    for class_name in classes:
        image_paths = []
        labels = []
        for row in rows:
            if row['class_name'] == class_name and len(image_paths) < 12:
                image_paths.append(output_root / row['image_path'])
                labels.append(f"{row['split']} {row['view_angle']}")
        if image_paths:
            _make_contact_sheet(image_paths, sheet_root / f'{class_name}.png', labels, columns=4)


def _generate_dataset(root_path, config, rows, classes, dataset_kind):
    render_script = Path(__file__).with_name('render_blender_model.py')
    output_root = _dataset_output_root(root_path, config, dataset_kind)
    reports_root = _resolve_path(root_path, config.get('reports_root', 'reports/multiclass_v1'))
    split_counts = _split_plan(dataset_kind)
    seed = int(config.get('seed', 42))
    primary_rows = [row for row in rows if row.get('path') and row['status'] == 'ok' and row.get('canonical_class')]
    row_by_class = {row['canonical_class']: row for row in primary_rows}
    missing = sorted(set(classes) - set(row_by_class))
    if missing:
        raise ValueError(f'Missing model rows for classes: {missing}')

    output_root.mkdir(parents=True, exist_ok=True)
    _write_class_index(output_root / 'class_index.csv', classes)
    all_manifest_rows = []
    render_payloads = {}

    for class_index, class_name in enumerate(classes):
        source_row = row_by_class[class_name]
        options = _class_options(config, class_name)
        class_seed = seed + class_index * 100000
        frame_specs = _frame_specs_for_class(config, class_name, class_index, dataset_kind, split_counts, class_seed)
        batch_frames = []
        for spec in frame_specs:
            file_name = f"{class_name}_{spec['split']}_{spec['frame_index']:04d}.png"
            rel_path = Path(spec['split']) / class_name / file_name
            output_path = output_root / rel_path
            batch_frames.append({
                'output_path': str(output_path),
                'seed': spec['seed'],
                'view_angle_degrees': spec['view_angle_degrees'],
                'front_angle_degrees': spec['front_angle_degrees'],
                'pitch_degrees': spec['pitch_degrees'],
                'roll_degrees': spec['roll_degrees'],
                'camera_radius': spec['camera_radius'],
            })
            spec['image_path'] = str(rel_path).replace('\\', '/')

        plan_path = output_root / '_render_plans' / f'{class_name}.json'
        _write_json(plan_path, {'model_path': str(source_row['path']), 'frames': batch_frames})
        render_payloads[class_name] = _run_batch_renderer(config, render_script, plan_path, source_row['path'], options)

        for spec in frame_specs:
            image_path = output_root / spec['image_path']
            quality = _image_quality(image_path)
            all_manifest_rows.append({
                'image_path': spec['image_path'],
                'class_name': class_name,
                'class_index': class_index,
                'split': spec['split'],
                'seed': spec['seed'],
                'source_model': str(source_row['path']),
                'source_model_sha256': source_row['sha256'],
                'view_angle': spec['view_angle_degrees'],
                'pitch': spec['pitch_degrees'],
                'roll': spec['roll_degrees'],
                'camera_radius': spec['camera_radius'],
                'frame_occupancy': quality['occupied_ratio_approx'],
                'near_uniform': quality['near_uniform'],
                'object_rotation': spec['object_rotation'],
                'object_scale': spec['object_scale'],
                'vertical_offset': spec['vertical_offset'],
                'pose_id': spec['pose_id'],
                'articulation_state': spec['articulation_state'],
                'interactive_objects': spec['interactive_objects'],
                'light_config': f"energy={_class_options(config, class_name).get('light_energy', 700.0)}",
                'background_config': f"seed={spec['seed']}",
                'render_engine': render_payloads[class_name].get('render_engine', config.get('render_engine')),
                'bpy_version': render_payloads[class_name].get('bpy_version', ''),
                'image_sha256': quality['sha256'],
            })

    manifest_fields = [
        'image_path',
        'class_name',
        'class_index',
        'split',
        'seed',
        'source_model',
        'source_model_sha256',
        'view_angle',
        'pitch',
        'roll',
        'camera_radius',
        'frame_occupancy',
        'near_uniform',
        'object_rotation',
        'object_scale',
        'vertical_offset',
        'pose_id',
        'articulation_state',
        'interactive_objects',
        'light_config',
        'background_config',
        'render_engine',
        'bpy_version',
        'image_sha256',
    ]
    _write_csv(output_root / 'images_manifest.csv', manifest_fields, all_manifest_rows)
    _write_json(output_root / 'dataset_manifest.json', _dataset_manifest(config, dataset_kind, classes, all_manifest_rows, output_root, primary_rows))
    summary_rows = _summarize_dataset(all_manifest_rows, classes)
    summary_csv_path = reports_root / f'{dataset_kind}_dataset_summary.csv'
    summary_json_path = reports_root / f'{dataset_kind}_dataset_summary.json'
    _write_csv(summary_csv_path, ['split', 'class_name', 'count', 'near_uniform_count', 'mean_frame_occupancy'], summary_rows)
    _write_json(summary_json_path, {'rows': summary_rows})
    duplicate_rows = _duplicate_rows(all_manifest_rows)
    duplicate_path = reports_root / f'{dataset_kind}_duplicate_report.csv'
    _write_csv(duplicate_path, ['image_sha256', 'image_path', 'class_name', 'split'], duplicate_rows)
    corruption_rows = _corruption_rows(all_manifest_rows, output_root)
    corruption_path = reports_root / f'{dataset_kind}_corruption_report.csv'
    _write_csv(corruption_path, ['image_path', 'class_name', 'split', 'error'], corruption_rows)
    if dataset_kind == 'full':
        _write_csv(reports_root / 'dataset_summary.csv', ['split', 'class_name', 'count', 'near_uniform_count', 'mean_frame_occupancy'], summary_rows)
        _write_json(reports_root / 'dataset_summary.json', {'rows': summary_rows})
        _write_csv(reports_root / 'duplicate_report.csv', ['image_sha256', 'image_path', 'class_name', 'split'], duplicate_rows)
        _write_csv(reports_root / 'corruption_report.csv', ['image_path', 'class_name', 'split', 'error'], corruption_rows)
    _write_json(output_root / 'render_payloads.json', render_payloads)
    _make_class_contact_sheets(all_manifest_rows, output_root, reports_root / dataset_kind, classes)
    return {
        'output_root': str(output_root),
        'dataset_manifest': str(output_root / 'dataset_manifest.json'),
        'images_manifest': str(output_root / 'images_manifest.csv'),
        'class_index': str(output_root / 'class_index.csv'),
        'summary': str(reports_root / f'{dataset_kind}_dataset_summary.json'),
        'duplicates': str(duplicate_path),
        'corruptions': str(corruption_path),
        'num_images': len(all_manifest_rows),
        'duplicates_count': len(duplicate_rows),
        'corruption_count': len(corruption_rows),
    }


def _render_diagnostics(root_path, config, rows, selected_classes=None):
    render_script = Path(__file__).with_name('render_blender_model.py')
    smoke_root = _resolve_path(root_path, config.get('smoke_output_root', 'datasets/generated/smoke_multiclass_v1'))
    diagnostics_root = _resolve_path(root_path, config.get('diagnostics_output_root', 'datasets/generated/diagnostics_multiclass_v1'))
    reports_root = _resolve_path(root_path, config.get('reports_root', 'reports/multiclass_v1'))
    seed = int(config.get('seed', 42))
    report_path = reports_root / 'render_smoke_report.json'
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            report = {'created_at': datetime.now(timezone.utc).isoformat(), 'classes': {}}
    else:
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'classes': {}}
    report['updated_at'] = datetime.now(timezone.utc).isoformat()

    primary_rows = [row for row in rows if row.get('path') and row['status'] == 'ok']
    if selected_classes:
        primary_rows = [row for row in primary_rows if row['canonical_class'] in selected_classes]
    for class_offset, row in enumerate(primary_rows):
        class_name = row['canonical_class']
        model_path = row['path']
        options = _class_options(config, class_name)
        class_report = {'source_file': str(model_path), 'smoke': {}, 'angle_sweep': {}, 'contact_sheet': {}}

        try:
            smoke_path = smoke_root / class_name / f'{class_name}_smoke.png'
            payload = _run_renderer(config, render_script, model_path, smoke_path, seed + class_offset, options)
            class_report['smoke'] = {
                'status': 'ok',
                'path': str(smoke_path),
                'renderer': payload,
                'quality': _image_quality(smoke_path),
            }
        except Exception as exc:
            class_report['smoke'] = {'status': 'error', 'error': str(exc)}
            report['classes'][class_name] = class_report
            continue

        angle_paths = []
        angle_labels = []
        angle_dir = diagnostics_root / 'angle_frames' / class_name
        for angle in config.get('angle_sweep_degrees', DEFAULT_ANGLES):
            frame_path = angle_dir / f'{class_name}_angle_{int(angle):03d}.png'
            angle_options = dict(options)
            angle_options['view_angle_degrees'] = angle
            try:
                _run_renderer(config, render_script, model_path, frame_path, seed + class_offset + int(angle), angle_options)
                angle_paths.append(frame_path)
                angle_labels.append(f'{angle} deg')
            except Exception as exc:
                class_report.setdefault('angle_errors', []).append({'angle': angle, 'error': str(exc)})
        if angle_paths:
            sheet_path = diagnostics_root / f'{class_name}_angle_sweep.png'
            _make_contact_sheet(angle_paths, sheet_path, angle_labels, columns=4)
            class_report['angle_sweep'] = {'status': 'ok', 'path': str(sheet_path), 'frames': [str(path) for path in angle_paths]}

        contact_paths = []
        contact_labels = []
        contact_dir = diagnostics_root / 'contact_frames' / class_name
        for index, view in enumerate(config.get('contact_views', DEFAULT_CONTACT_VIEWS)):
            frame_path = contact_dir / f'{class_name}_{index:02d}_{view["name"]}.png'
            view_options = dict(options)
            view_options.update({key: value for key, value in view.items() if key != 'name'})
            try:
                _run_renderer(config, render_script, model_path, frame_path, seed + class_offset + 1000 + index, view_options)
                contact_paths.append(frame_path)
                contact_labels.append(view['name'])
            except Exception as exc:
                class_report.setdefault('contact_errors', []).append({'view': view['name'], 'error': str(exc)})
        if contact_paths:
            sheet_path = diagnostics_root / f'{class_name}_contact_sheet.png'
            _make_contact_sheet(contact_paths, sheet_path, contact_labels, columns=4)
            class_report['contact_sheet'] = {'status': 'ok', 'path': str(sheet_path), 'frames': [str(path) for path in contact_paths]}

        if row.get('canonical_pose_required') is True or str(row.get('canonical_pose_required')).lower() == 'true':
            pose_sheet_path = diagnostics_root / f'{class_name}_pose_contact_sheet.png'
            pose_source = contact_paths[:1] or angle_paths[:1]
            if pose_source:
                _make_contact_sheet(pose_source, pose_sheet_path, ['neutral_pose_review_required'], columns=1)
                class_report['pose_contact_sheet'] = {
                    'status': 'review_required',
                    'path': str(pose_sheet_path),
                    'note': 'The model contains rigging, actions, constraints, or shape keys. Only the neutral imported pose was rendered automatically.',
                }

        report['classes'][class_name] = class_report

    reports_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Generate Blender synthetic dataset diagnostics.')
    parser.add_argument('-opt', '--options', required=True, help='Path to the generation YAML config.')
    parser.add_argument('--mode', choices=['inventory', 'diagnostics', 'dataset'], default='diagnostics')
    parser.add_argument('--classes', help='Comma-separated canonical class ids to render in diagnostics mode.')
    parser.add_argument('--dataset-kind', choices=['pilot', 'full'], default='pilot')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    config = _load_yaml(_resolve_path(root_path, args.options))
    if 'synthetic' in config:
        config = config['synthetic']

    source_root = _resolve_path(root_path, config['source_root'])
    reports_root = _resolve_path(root_path, config.get('reports_root', 'reports/multiclass_v1'))
    mapping_file, class_mapping = _load_class_mapping(root_path, config)
    rows = _discover_models(source_root, class_mapping)
    structure_report_path = _inspect_rows(config, rows, reports_root)
    classes = sorted(set(class_mapping.values()))

    inventory_path = reports_root / 'blender_models_inventory.csv'
    _write_inventory(inventory_path, rows)
    _write_class_index(reports_root / 'class_index.csv', classes)

    result = {
        'class_mapping_file': str(mapping_file),
        'inventory': str(inventory_path),
        'class_index': str(reports_root / 'class_index.csv'),
        'model_structure_report': str(structure_report_path),
        'classes': classes,
    }
    if args.mode == 'diagnostics':
        selected_classes = None
        if args.classes:
            selected_classes = {value.strip() for value in args.classes.split(',') if value.strip()}
            unknown = sorted(selected_classes - set(classes))
            if unknown:
                raise ValueError(f'Unknown classes requested for diagnostics: {unknown}')
            result['selected_classes'] = sorted(selected_classes)
        result['render_smoke_report'] = str(_render_diagnostics(root_path, config, rows, selected_classes))
    elif args.mode == 'dataset':
        result['dataset'] = _generate_dataset(root_path, config, rows, classes, args.dataset_kind)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
