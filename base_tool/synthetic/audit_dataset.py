import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageStat


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def _edge_pixels(gray):
    width, height = gray.size
    pixels = gray.load()
    values = []
    for x in range(width):
        values.append(pixels[x, 0])
        values.append(pixels[x, height - 1])
    for y in range(height):
        values.append(pixels[0, y])
        values.append(pixels[width - 1, y])
    return values


def compute_frame_occupancy(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        gray = rgb.convert('L')
        stat = ImageStat.Stat(gray)
        edge_values = _edge_pixels(gray)
        edge_mean = sum(edge_values) / len(edge_values)
        edge_var = sum((value - edge_mean) ** 2 for value in edge_values) / len(edge_values)
        edge_std = edge_var ** 0.5

        # The synthetic renders use a smooth gray background and light clay models.
        # The old mean-difference method selected the full background gradient.
        threshold = edge_mean + max(8.0, edge_std * 2.5)
        pixels = gray.load()
        xs = []
        ys = []
        for y in range(gray.height):
            for x in range(gray.width):
                if pixels[x, y] > threshold:
                    xs.append(x)
                    ys.append(y)

        if not xs:
            return {
                'frame_occupancy': 0.0,
                'bbox_left': '',
                'bbox_top': '',
                'bbox_right': '',
                'bbox_bottom': '',
                'foreground_pixels': 0,
                'threshold': round(threshold, 3),
                'near_uniform': max(ImageStat.Stat(rgb).stddev) < 2.0,
            }

        left, right = min(xs), max(xs) + 1
        top, bottom = min(ys), max(ys) + 1
        bbox_area = (right - left) * (bottom - top)
        image_area = gray.width * gray.height
        return {
            'frame_occupancy': round(bbox_area / image_area, 4),
            'bbox_left': left,
            'bbox_top': top,
            'bbox_right': right,
            'bbox_bottom': bottom,
            'foreground_pixels': len(xs),
            'threshold': round(threshold, 3),
            'near_uniform': max(ImageStat.Stat(rgb).stddev) < 2.0,
        }


def _perceptual_hash(path, hash_size=8, highfreq_factor=4):
    import numpy as np
    from scipy.fftpack import dct

    size = hash_size * highfreq_factor
    with Image.open(path) as image:
        gray = image.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)
    dct_values = dct(dct(pixels, axis=0, norm='ortho'), axis=1, norm='ortho')
    low_freq = dct_values[:hash_size, :hash_size]
    flattened = low_freq.flatten()[1:]
    median = float(np.median(flattened))
    bits = 0
    for value in flattened:
        bits = (bits << 1) | int(value > median)
    return bits


def _hamming(a, b):
    return (a ^ b).bit_count()


def _parameter_key(row, include_background=True):
    fields = [
        'class_name',
        'source_model_sha256',
        'view_angle',
        'pitch',
        'roll',
        'camera_radius',
        'object_rotation',
        'object_scale',
        'vertical_offset',
        'pose_id',
        'articulation_state',
        'interactive_objects',
        'light_config',
    ]
    if include_background:
        fields.append('background_config')
    return tuple(row.get(field, '') for field in fields)


def audit_dataset(dataset_root, reports_root, update_manifest=False):
    dataset_root = Path(dataset_root)
    reports_root = Path(reports_root)
    manifest_path = dataset_root / 'images_manifest.csv'
    rows = _read_csv(manifest_path)

    exact_hashes = {}
    phashes = []
    parameter_rows = {}
    geometry_parameter_rows = {}
    for row in rows:
        image_path = dataset_root / row['image_path']
        quality = compute_frame_occupancy(image_path)
        row['frame_occupancy'] = quality['frame_occupancy']
        row['bbox_left'] = quality['bbox_left']
        row['bbox_top'] = quality['bbox_top']
        row['bbox_right'] = quality['bbox_right']
        row['bbox_bottom'] = quality['bbox_bottom']
        row['foreground_pixels'] = quality['foreground_pixels']
        row['occupancy_threshold'] = quality['threshold']
        row['near_uniform'] = quality['near_uniform']
        row['image_sha256'] = row.get('image_sha256') or _sha256(image_path)
        exact_hashes.setdefault(row['image_sha256'], []).append(row)
        phashes.append((row, _perceptual_hash(image_path)))
        parameter_rows.setdefault(_parameter_key(row, include_background=True), []).append(row)
        geometry_parameter_rows.setdefault(_parameter_key(row, include_background=False), []).append(row)

    if update_manifest:
        fieldnames = list(rows[0].keys())
        for field in [
            'bbox_left',
            'bbox_top',
            'bbox_right',
            'bbox_bottom',
            'foreground_pixels',
            'occupancy_threshold',
        ]:
            if field not in fieldnames:
                fieldnames.append(field)
        _write_csv(manifest_path, fieldnames, rows)

    exact_duplicates = []
    for digest, digest_rows in exact_hashes.items():
        if len(digest_rows) > 1:
            for row in digest_rows:
                exact_duplicates.append({
                    'image_sha256': digest,
                    'image_path': row['image_path'],
                    'class_name': row['class_name'],
                    'split': row['split'],
                })

    near_duplicates = []
    for idx, (row_a, hash_a) in enumerate(phashes):
        for row_b, hash_b in phashes[idx + 1:]:
            if row_a['split'] == row_b['split']:
                continue
            distance = _hamming(hash_a, hash_b)
            if distance <= 8:
                near_duplicates.append({
                    'image_path_a': row_a['image_path'],
                    'split_a': row_a['split'],
                    'class_a': row_a['class_name'],
                    'image_path_b': row_b['image_path'],
                    'split_b': row_b['split'],
                    'class_b': row_b['class_name'],
                    'hamming_distance': distance,
                })

    parameter_collisions = []
    def append_collisions(source, collision_type):
        for key, key_rows in source.items():
            splits = sorted({row['split'] for row in key_rows})
            if len(splits) <= 1:
                continue
            for row in key_rows:
                parameter_collisions.append({
                    'collision_type': collision_type,
                    'image_path': row['image_path'],
                    'class_name': row['class_name'],
                    'split': row['split'],
                    'collision_splits': ';'.join(splits),
                    'parameter_key': '|'.join(str(value) for value in key),
                })

    append_collisions(parameter_rows, 'full_parameters')
    append_collisions(geometry_parameter_rows, 'geometry_without_background')

    classes = sorted({row['class_name'] for row in rows})
    summary = []
    for split in sorted({row['split'] for row in rows}):
        for class_name in classes:
            subset = [row for row in rows if row['split'] == split and row['class_name'] == class_name]
            if not subset:
                continue
            summary.append({
                'split': split,
                'class_name': class_name,
                'count': len(subset),
                'near_uniform_count': sum(str(row.get('near_uniform')).lower() == 'true' for row in subset),
                'mean_frame_occupancy': round(sum(float(row['frame_occupancy']) for row in subset) / len(subset), 4),
                'min_frame_occupancy': round(min(float(row['frame_occupancy']) for row in subset), 4),
                'max_frame_occupancy': round(max(float(row['frame_occupancy']) for row in subset), 4),
            })

    _write_csv(reports_root / 'duplicate_report.csv', ['image_sha256', 'image_path', 'class_name', 'split'], exact_duplicates)
    _write_csv(
        reports_root / 'near_duplicate_report.csv',
        ['image_path_a', 'split_a', 'class_a', 'image_path_b', 'split_b', 'class_b', 'hamming_distance'],
        near_duplicates,
    )
    _write_csv(
        reports_root / 'split_parameter_collision_report.csv',
        ['collision_type', 'image_path', 'class_name', 'split', 'collision_splits', 'parameter_key'],
        parameter_collisions,
    )
    _write_csv(
        reports_root / 'dataset_summary.csv',
        ['split', 'class_name', 'count', 'near_uniform_count', 'mean_frame_occupancy', 'min_frame_occupancy', 'max_frame_occupancy'],
        summary,
    )
    (reports_root / 'dataset_summary.json').write_text(json.dumps({'rows': summary}, indent=2, ensure_ascii=False), encoding='utf-8')

    return {
        'num_images': len(rows),
        'exact_duplicates': len(exact_duplicates),
        'near_duplicates': len(near_duplicates),
        'parameter_collisions': len(parameter_collisions),
        'summary': summary,
    }


def main():
    parser = argparse.ArgumentParser(description='Audit a generated image classification dataset.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--reports-root', required=True)
    parser.add_argument('--update-manifest', action='store_true')
    args = parser.parse_args()

    result = audit_dataset(args.dataset_root, args.reports_root, update_manifest=args.update_manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
