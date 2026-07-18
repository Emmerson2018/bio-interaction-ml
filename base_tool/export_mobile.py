import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from PIL import Image

from base_tool.archs import build_network
from base_tool.data.preprocessing import build_image_transform


UNKNOWN_ID = 'unknown'


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ordered_checkpoint_ids(checkpoint):
    idx_to_class = checkpoint.get('idx_to_class') or checkpoint.get('dataset_metadata', {}).get('idx_to_class')
    if not idx_to_class:
        raise ValueError('Checkpoint does not contain idx_to_class metadata.')
    normalized = {int(index): class_id for index, class_id in idx_to_class.items()}
    expected = list(range(len(normalized)))
    if sorted(normalized) != expected:
        raise ValueError(f'Checkpoint class indices must be contiguous: {sorted(normalized)}')
    return [normalized[index] for index in expected]


def load_content_ids(content_db):
    payload = json.loads(Path(content_db).read_text(encoding='utf-8-sig'))
    if not isinstance(payload, list):
        raise ValueError('The app content database must be a JSON list.')
    ids = [item.get('id') for item in payload if isinstance(item, dict)]
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError('Every app content item must have a non-empty string id.')
    if len(ids) != len(set(ids)):
        raise ValueError('The app content database contains duplicate ids.')
    return set(ids)


def resolve_deployment_ids(trained_ids, content_ids, aliases=None):
    aliases = dict(aliases or {})
    deployment_ids = []
    for trained_id in trained_ids:
        deployment_id = aliases.get(trained_id, trained_id)
        if deployment_id != UNKNOWN_ID and deployment_id not in content_ids:
            raise ValueError(
                f'Model id {trained_id!r} resolves to {deployment_id!r}, '
                'which does not exist in the app content database.'
            )
        deployment_ids.append(deployment_id)
    if len(deployment_ids) != len(set(deployment_ids)):
        raise ValueError(f'Deployment ids must be unique: {deployment_ids}')
    return deployment_ids


def _network_options(checkpoint):
    metadata = checkpoint.get('model_metadata') or {}
    if not metadata:
        raise ValueError('Checkpoint does not contain model_metadata.')
    return {
        'type': metadata.get('type', 'TorchvisionClassifier'),
        'backbone': metadata['backbone'],
        'num_classes': int(metadata['num_classes']),
        'weights': None,
        'classifier_dropout': float(metadata.get('classifier_dropout', 0.0)),
        'preprocessing': checkpoint.get('preprocessing') or metadata.get('preprocessing'),
    }


def _load_model(checkpoint):
    model = build_network(_network_options(checkpoint))
    model.load_state_dict(checkpoint['network'], strict=True)
    model.eval()
    return model


def _write_deployment_checkpoint(checkpoint, deployment_ids, output_path):
    payload = dict(checkpoint)
    class_to_idx = {class_id: index for index, class_id in enumerate(deployment_ids)}
    idx_to_class = {str(index): class_id for index, class_id in enumerate(deployment_ids)}
    payload['class_to_idx'] = class_to_idx
    payload['idx_to_class'] = idx_to_class
    dataset_metadata = dict(payload.get('dataset_metadata') or {})
    dataset_metadata['class_to_idx'] = class_to_idx
    dataset_metadata['idx_to_class'] = idx_to_class
    payload['dataset_metadata'] = dataset_metadata
    payload['deployment_metadata'] = {
        'class_ids_source': 'app_content_database',
        'class_ids': deployment_ids,
    }
    torch.save(payload, output_path)


def _sample_images(test_root, trained_ids, limit=100):
    samples = []
    for class_id in trained_ids:
        class_root = test_root / class_id
        if not class_root.exists():
            continue
        for image in sorted(class_root.iterdir()):
            if image.is_file():
                samples.append(image)
            if len(samples) >= limit:
                return samples
    return samples


def _onnx_parity(model, onnx_path, image_paths, preprocessing):
    transform = build_image_transform(preprocessing, augment=False)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    comparisons = []
    max_logit_diff = 0.0
    same_predictions = 0
    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            torch_logits = model(tensor).cpu().numpy()
            onnx_logits = session.run(None, {'input': tensor.numpy()})[0]
            difference = float(np.max(np.abs(torch_logits - onnx_logits)))
            same = int(np.argmax(torch_logits) == np.argmax(onnx_logits))
            max_logit_diff = max(max_logit_diff, difference)
            same_predictions += same
            comparisons.append({'image': str(image_path), 'same_prediction': bool(same), 'max_logit_diff': difference})
    return {
        'samples': len(comparisons),
        'same_predictions': same_predictions,
        'same_prediction_rate': same_predictions / max(1, len(comparisons)),
        'max_logit_diff': max_logit_diff,
        'comparisons': comparisons,
    }


def _convert_to_tflite(onnx_path, output_root):
    converted_root = output_root / 'tf_converted'
    if converted_root.exists():
        shutil.rmtree(converted_root)
    converter_sample = output_root / 'calibration_image_sample_data_20x128x128x3_float32.npy'
    if not converter_sample.exists():
        sample = np.random.default_rng(42).random((20, 128, 128, 3), dtype=np.float32)
        np.save(converter_sample, sample)
    command = [
        sys.executable,
        '-m',
        'onnx2tf',
        '-i',
        str(onnx_path),
        '-o',
        str(converted_root),
        '-dsm',
        '-n',
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, cwd=output_root)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout)[-4000:])
    source = converted_root / 'model_float32.tflite'
    if not source.exists():
        raise FileNotFoundError(f'onnx2tf did not create {source}')
    destination = output_root / 'model.tflite'
    shutil.copy2(source, destination)
    return destination


def _tflite_parity(model, tflite_path, image_paths, preprocessing):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    size = int(preprocessing['input_size'])
    mean = np.asarray(preprocessing['mean'], dtype=np.float32)
    std = np.asarray(preprocessing['std'], dtype=np.float32)
    comparisons = []
    max_logit_diff = 0.0
    same_predictions = 0
    transform = build_image_transform(preprocessing, augment=False)
    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as source_image:
                image = source_image.convert('RGB')
                torch_logits = model(transform(image).unsqueeze(0)).cpu().numpy()
                resized = image.resize((size, size), Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) / 255.0
            input_array = ((array - mean) / std)[None, ...].astype(np.float32)
            interpreter.set_tensor(input_details['index'], input_array)
            interpreter.invoke()
            tflite_logits = interpreter.get_tensor(output_details['index'])
            difference = float(np.max(np.abs(torch_logits - tflite_logits)))
            same = int(np.argmax(torch_logits) == np.argmax(tflite_logits))
            max_logit_diff = max(max_logit_diff, difference)
            same_predictions += same
            comparisons.append({'image': str(image_path), 'same_prediction': bool(same), 'max_logit_diff': difference})
    return {
        'samples': len(comparisons),
        'same_predictions': same_predictions,
        'same_prediction_rate': same_predictions / max(1, len(comparisons)),
        'max_logit_diff': max_logit_diff,
        'input_shape': input_details['shape'].tolist(),
        'output_shape': output_details['shape'].tolist(),
        'comparisons': comparisons,
    }


def export_mobile(config_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    checkpoint_path = _resolve(repo_root, config['checkpoint'])
    content_db = _resolve(repo_root, config['app_content_db'])
    output_root = _resolve(repo_root, config['output_root'])
    test_root = _resolve(repo_root, config['test_root'])
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    trained_ids = _ordered_checkpoint_ids(checkpoint)
    deployment_ids = resolve_deployment_ids(trained_ids, load_content_ids(content_db), config.get('id_aliases'))
    model = _load_model(checkpoint)
    preprocessing = checkpoint['preprocessing']

    deployment_checkpoint = output_root / 'model_app_ids.pth'
    _write_deployment_checkpoint(checkpoint, deployment_ids, deployment_checkpoint)

    input_size = int(preprocessing['input_size'])
    onnx_path = output_root / 'model.onnx'
    torch.onnx.export(
        model,
        torch.randn(1, 3, input_size, input_size),
        onnx_path,
        input_names=['input'],
        output_names=['logits'],
        opset_version=18,
        dynamo=False,
    )
    images = _sample_images(test_root, trained_ids)
    onnx_parity = _onnx_parity(model, onnx_path, images, preprocessing)
    if onnx_parity['same_prediction_rate'] != 1.0:
        raise RuntimeError(f'ONNX parity failed: {onnx_parity}')

    tflite_path = _convert_to_tflite(onnx_path, output_root)
    tflite_parity = _tflite_parity(model, tflite_path, images, preprocessing)
    if tflite_parity['same_prediction_rate'] != 1.0:
        raise RuntimeError(f'TFLite parity failed: {tflite_parity}')

    manifest = {
        'model_version': config['model_version'],
        'architecture': checkpoint['model_metadata']['backbone'],
        'class_ids': deployment_ids,
        'class_ids_source': 'assets/content/animais.json',
        'unknown_id': UNKNOWN_ID,
        'input': {
            'width': input_size,
            'height': input_size,
            'channels': 3,
            'dtype': 'float32',
            'color_order': preprocessing['color_order'],
            'normalization_mean': preprocessing['mean'],
            'normalization_std': preprocessing['std'],
        },
        'output': {'type': 'logits', 'count': len(deployment_ids)},
        'acceptance': {
            'confidence_threshold': float(config['acceptance']['confidence_threshold']),
            'entropy_threshold': float(config['acceptance']['entropy_threshold']),
            'required_consistent_frames': int(config['acceptance']['required_consistent_frames']),
        },
    }
    manifest_path = output_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    report = {
        'source_checkpoint': str(checkpoint_path),
        'deployment_checkpoint': str(deployment_checkpoint),
        'model': str(tflite_path),
        'onnx': str(onnx_path),
        'manifest': str(manifest_path),
        'parity': {'onnx': onnx_parity, 'tflite': tflite_parity},
    }
    report_path = output_root / 'export_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return {
        'model': str(tflite_path),
        'onnx': str(onnx_path),
        'manifest': str(manifest_path),
        'report': str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Export a trained classifier using app database ids.')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(export_mobile(args.config), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
