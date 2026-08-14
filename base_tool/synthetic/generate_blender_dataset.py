import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml


SUPPORTED_MODEL_SUFFIXES = {'.blend', '.blend1', '.fbx', '.obj', '.glb', '.gltf'}


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _find_model_files(source_dir):
    files = []
    for path in source_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
            files.append(path)
    return sorted(files)


def _class_name_for_model(source_dir, model_path):
    relative = model_path.relative_to(source_dir)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return model_path.stem


def _class_render_options(synthetic_opt, class_name):
    common_options = synthetic_opt.get('render_options', {})
    class_options = synthetic_opt.get('class_render_options', {}).get(class_name, {})
    return {**common_options, **class_options}


def _split_counts(total, splits):
    counts = {}
    remaining = total
    split_items = list(splits.items())
    for idx, (name, ratio) in enumerate(split_items):
        if idx == len(split_items) - 1:
            counts[name] = remaining
        else:
            value = int(total * float(ratio))
            counts[name] = value
            remaining -= value
    return counts


def _run_renderer(synthetic_opt, render_script, args):
    backend = synthetic_opt.get('renderer_backend', 'python_bpy')
    if backend == 'python_bpy':
        python_executable = synthetic_opt.get('python_executable', sys.executable)
        command = [
            python_executable,
            str(render_script),
            *args,
        ]
    elif backend == 'blender_executable':
        blender_executable = synthetic_opt.get('blender_executable', 'blender')
        command = [
            blender_executable,
            '--background',
            '--python',
            str(render_script),
            '--',
            *args,
        ]
    else:
        raise ValueError(f'Unsupported renderer backend: {backend}')

    subprocess.run(command, check=True)


def _write_class_index(output_dir, classes):
    path = output_dir / 'class_index.csv'
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['class_name', 'class_index'])
        for index, class_name in enumerate(classes):
            writer.writerow([class_name, index])


def main():
    parser = argparse.ArgumentParser(description='Generate an image dataset from Blender model files.')
    parser.add_argument('-opt', '--options', required=True, help='Path to the synthetic dataset YAML config.')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    with open(args.options, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    synthetic_opt = opt['synthetic']
    source_dir = _resolve_path(root_path, synthetic_opt['source_dir'])
    output_dir = _resolve_path(root_path, synthetic_opt['output_dir'])
    samples_per_model = int(synthetic_opt.get('samples_per_model', 100))
    image_size = int(synthetic_opt.get('image_size', 224))
    seed = int(synthetic_opt.get('seed', 123))
    splits = synthetic_opt.get('splits', {'train': 0.8, 'val': 0.1, 'test': 0.1})
    camera_radius_min = float(synthetic_opt.get('camera_radius_min', 3.0))
    camera_radius_max = float(synthetic_opt.get('camera_radius_max', 5.5))
    render_engine = synthetic_opt.get('render_engine', 'CYCLES')
    device = synthetic_opt.get('device', 'CUDA')
    
    psychedelic_bg_prob = float(synthetic_opt.get('psychedelic_bg_prob', 0.0))
    light_count_min = int(synthetic_opt.get('light_count_min', 2))
    light_count_max = int(synthetic_opt.get('light_count_max', 2))
    occlusion_prob = float(synthetic_opt.get('occlusion_prob', 0.0))
    occlusion_count_max = int(synthetic_opt.get('occlusion_count_max', 0))
    occlusion_coverage_max = float(synthetic_opt.get('occlusion_coverage_max', 0.0))
    noise_prob = float(synthetic_opt.get('noise_prob', 0.0))
    noise_types = synthetic_opt.get('noise_types', [])
    noise_intensity_min = float(synthetic_opt.get('noise_intensity_min', 0.0))
    noise_intensity_max = float(synthetic_opt.get('noise_intensity_max', 0.0))

    render_script = Path(__file__).with_name('render_blender_model.py')
    model_files = _find_model_files(source_dir)
    if not model_files:
        raise FileNotFoundError(f'No supported model files found in {source_dir}')

    classes = sorted({_class_name_for_model(source_dir, model_path) for model_path in model_files})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_class_index(output_dir, classes)

    for model_index, model_path in enumerate(model_files):
        class_name = _class_name_for_model(source_dir, model_path)
        render_options = _class_render_options(synthetic_opt, class_name)
        counts = _split_counts(samples_per_model, splits)
        for split_name, count in counts.items():
            if count <= 0:
                continue
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            start_index = len(list(split_class_dir.glob('*.png')))
            render_args = [
                '--model-path', str(model_path),
                '--output-dir', str(split_class_dir),
                '--class-name', class_name,
                '--split', split_name,
                '--samples', str(count),
                '--start-index', str(start_index),
                '--image-size', str(image_size),
                '--seed', str(seed + (model_index * 10000000) + {'train': 0, 'val': 1000000, 'test': 2000000}.get(split_name, 0) + start_index),
                '--camera-radius-min', str(camera_radius_min),
                '--camera-radius-max', str(camera_radius_max),
                '--render-engine', render_engine,
                '--device', device,
                '--psychedelic-bg-prob', str(psychedelic_bg_prob),
                '--light-count-min', str(light_count_min),
                '--light-count-max', str(light_count_max),
                '--occlusion-prob', str(occlusion_prob),
                '--occlusion-count-max', str(occlusion_count_max),
                '--occlusion-coverage-max', str(occlusion_coverage_max),
                '--noise-prob', str(noise_prob),
                '--noise-intensity-min', str(noise_intensity_min),
                '--noise-intensity-max', str(noise_intensity_max),
            ]
            if noise_types:
                render_args.extend(['--noise-types', ','.join(noise_types)])
            if render_options:
                render_args.extend([
                    '--view-strategy', render_options.get('view_strategy', 'random_360'),
                    '--front-angle-degrees', str(render_options.get('front_angle_degrees', 0.0)),
                    '--close-view-ratio', str(render_options.get('close_view_ratio', 0.25)),
                    '--far-view-ratio', str(render_options.get('far_view_ratio', 0.20)),
                ])
            print(f'Rendering {count} images: {class_name}/{split_name} from {model_path.name}')
            _run_renderer(synthetic_opt, render_script, render_args)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f'Renderer failed with exit code {exc.returncode}', file=sys.stderr)
        raise
