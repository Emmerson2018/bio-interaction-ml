import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from base_tool.real_data.validate_metadata import validate_metadata

try:
    import cv2
except Exception:  # pragma: no cover - depends on optional OpenCV install
    cv2 = None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def _truthy(value):
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'sim'}


def _phash(image, hash_size=8, highfreq_factor=4):
    import numpy as np
    from scipy.fftpack import dct

    size = hash_size * highfreq_factor
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


def _split_dir(output_root, row):
    split_role = row['split_role'].strip()
    true_class = row['true_class'].strip()
    if split_role == 'calibration_known':
        return output_root / 'real_multiclass_v1' / 'calibration_known' / true_class
    if split_role == 'test_known':
        return output_root / 'real_multiclass_v1' / 'test_known' / true_class
    if split_role == 'calibration_unknown':
        return output_root / 'unknown_multiclass_v1' / 'calibration'
    if split_role == 'test_unknown':
        return output_root / 'unknown_multiclass_v1' / 'test'
    raise ValueError(f'Cannot extract frames for split_role={split_role!r}')


def extract_frames(metadata_path, output_root, fps=1.0, phash_threshold=4, require_ready=False, resume=False):
    if cv2 is None:
        raise RuntimeError('OpenCV is required for frame extraction. Install opencv-python in the active environment.')
    metadata_path = Path(metadata_path)
    output_root = Path(output_root)
    validation = validate_metadata(metadata_path, require_ready=require_ready)
    if require_ready and not validation['valid']:
        raise ValueError('Metadata validation failed; refusing to extract frames.')
    rows = _read_rows(metadata_path)
    manifest_rows = []
    skipped = []

    for row in rows:
        if row.get('media_type') != 'video':
            continue
        required = ['true_class', 'is_unknown', 'session_id', 'source_group', 'split_role']
        missing = [field for field in required if not str(row.get(field, '')).strip()]
        if missing:
            skipped.append({'file_path': row.get('file_path', ''), 'reason': f"missing_required:{';'.join(missing)}"})
            continue
        if row['split_role'] == 'exclude':
            skipped.append({'file_path': row.get('file_path', ''), 'reason': 'exclude'})
            continue

        video_path = Path(row['file_path'])
        if not video_path.exists():
            skipped.append({'file_path': row.get('file_path', ''), 'reason': 'source_missing'})
            continue
        if row.get('sha256') and _sha256(video_path) != row['sha256']:
            raise ValueError(f'Hash mismatch for {video_path}')

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            skipped.append({'file_path': str(video_path), 'reason': 'video_open_failed'})
            continue
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            capture.release()
            skipped.append({'file_path': str(video_path), 'reason': 'invalid_fps'})
            continue
        step_frames = max(1, int(round(source_fps / fps)))
        output_dir = _split_dir(output_root, row)
        output_dir.mkdir(parents=True, exist_ok=True)
        previous_hashes = []
        frame_index = 0
        extracted_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step_frames != 0:
                frame_index += 1
                continue
            timestamp_seconds = frame_index / source_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            frame_hash = _phash(image)
            if any(_hamming(frame_hash, previous) <= phash_threshold for previous in previous_hashes):
                frame_index += 1
                continue
            previous_hashes.append(frame_hash)
            stem = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in video_path.stem)
            output_path = output_dir / f'{stem}_frame_{frame_index:06d}.jpg'
            if output_path.exists() and not resume:
                raise FileExistsError(f'Frame already exists: {output_path}. Use --resume to keep existing files.')
            if not output_path.exists():
                image.save(output_path, quality=95)
            manifest_rows.append({
                'source_video': str(video_path),
                'frame_path': str(output_path),
                'frame_index': frame_index,
                'timestamp_seconds': round(timestamp_seconds, 3),
                'sha256': _sha256(output_path),
                'true_class': row['true_class'],
                'is_unknown': row['is_unknown'],
                'split_role': row['split_role'],
                'session_id': row['session_id'],
                'source_group': row['source_group'],
                'device': row.get('device', ''),
                'view': row.get('view', ''),
                'background': row.get('background', ''),
                'lighting': row.get('lighting', ''),
            })
            extracted_index += 1
            frame_index += 1
        capture.release()
        if extracted_index == 0:
            skipped.append({'file_path': str(video_path), 'reason': 'no_frames_after_phash_filter'})

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'extracted_frames_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'source_video',
            'frame_path',
            'frame_index',
            'timestamp_seconds',
            'sha256',
            'true_class',
            'is_unknown',
            'split_role',
            'session_id',
            'source_group',
            'device',
            'view',
            'background',
            'lighting',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    skipped_path = output_root / 'extracted_frames_skipped.csv'
    with skipped_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['file_path', 'reason'])
        writer.writeheader()
        writer.writerows(skipped)
    return {
        'manifest': str(manifest_path),
        'skipped': str(skipped_path),
        'num_frames': len(manifest_rows),
        'num_skipped_videos': len(skipped),
    }


def main():
    parser = argparse.ArgumentParser(description='Extract real video frames without mixing metadata groups.')
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--phash-threshold', type=int, default=4)
    parser.add_argument('--require-ready', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = extract_frames(args.metadata, args.output_root, args.fps, args.phash_threshold, args.require_ready, args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
