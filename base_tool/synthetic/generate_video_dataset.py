import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms.v2 as v2
import yaml
from tqdm import tqdm

from base_tool.data.dataset_splitter import VideoDatasetSplitter, compute_file_md5

SUPPORTED_VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


def _write_class_index(output_dir, classes):
    path = output_dir / 'class_index.csv'
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['class_name', 'class_index'])
        for index, class_name in enumerate(classes):
            writer.writerow([class_name, index])


def build_gpu_augmentor(aug_opt, image_size):
    rot_deg = float(aug_opt.get('rotation_degrees', 20.0))
    shear_deg = float(aug_opt.get('affine_shear', 10.0))
    persp_dist = float(aug_opt.get('perspective_distortion', 0.15))
    color_j = float(aug_opt.get('color_jitter', 0.25))
    blur_p = float(aug_opt.get('blur_prob', 0.3))

    transform_list = [
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomRotation(degrees=(-rot_deg, rot_deg)),
        v2.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            shear=(-shear_deg, shear_deg),
        ),
        v2.RandomPerspective(distortion_scale=persp_dist, p=0.5),
        v2.ColorJitter(
            brightness=color_j,
            contrast=color_j,
            saturation=color_j,
            hue=min(0.08, color_j * 0.25),
        ),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
            p=blur_p,
        ),
    ]

    return v2.Compose(transform_list)


def apply_gaussian_noise_gpu(img_tensor, aug_opt):
    noise_p = float(aug_opt.get('noise_prob', 0.3))
    if torch.rand(1, device=img_tensor.device).item() < noise_p:
        noise_min = float(aug_opt.get('noise_intensity_min', 0.01))
        noise_max = float(aug_opt.get('noise_intensity_max', 0.03))
        std = random.uniform(noise_min, noise_max)
        noise = torch.randn_like(img_tensor) * std
        img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
    return img_tensor


def extract_decimated_frames(
    video_path,
    target_fps=2.0,
    start_ratio=0.0,
    end_ratio=1.0,
    min_delta=0.02,
    max_frames=None,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}", file=sys.stderr)
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, round(orig_fps / target_fps))
    start_idx = int(total_frames * start_ratio)
    end_idx = int(total_frames * end_ratio)

    frames = []
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= end_idx:
            break

        if frame_idx >= start_idx and (frame_idx - start_idx) % step == 0:
            small_gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and min_delta > 0:
                diff = np.mean(np.abs(small_gray.astype(float) - prev_gray.astype(float))) / 255.0
                if diff < min_delta:
                    frame_idx += 1
                    continue

            prev_gray = small_gray
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_sec = frame_idx / orig_fps if orig_fps > 0 else 0.0
            frames.append((frame_idx, timestamp_sec, frame_rgb))

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def save_tensor_as_png(tensor_float, output_path):
    tensor_uint8 = (torch.clamp(tensor_float, 0.0, 1.0) * 255.0).to(torch.uint8).cpu()
    io.write_png(tensor_uint8, str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Generate a leak-free augmented image dataset from videos.')
    parser.add_argument('-opt', '--options', required=True, help='Path to the video dataset YAML config.')
    parser.add_argument('--max-frames', type=int, default=None, help='Limit max extracted frames per video (useful for testing).')
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    with open(args.options, mode='r', encoding='utf-8') as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    video_opt = opt.get('video', opt.get('synthetic', {}))
    source_dir = _resolve_path(root_path, video_opt['source_dir'])
    output_dir = _resolve_path(root_path, video_opt['output_dir'])
    image_size = int(video_opt.get('image_size', 224))
    seed = int(video_opt.get('seed', 20260814))
    device_str = video_opt.get('device', 'cuda')
    aug_per_frame = int(video_opt.get('augmentations_per_frame', 3))
    target_fps = float(video_opt.get('target_fps', 2.0))
    min_delta = float(video_opt.get('min_frame_delta', 0.02))
    gap_ratio = float(video_opt.get('gap_ratio', 0.10))
    val_ratio_for_shared = float(video_opt.get('val_ratio_for_shared', 0.45))
    aug_opt = video_opt.get('augmentations', {})

    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    splitter = VideoDatasetSplitter(
        source_dir=source_dir,
        gap_ratio=gap_ratio,
        val_ratio_for_shared=val_ratio_for_shared,
        seed=seed,
    )
    split_plans = splitter.build_split_plan()
    classes = sorted(list(set(p.class_name for p in split_plans)))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_class_index(output_dir, classes)

    gpu_augmentor = build_gpu_augmentor(aug_opt, image_size)
    resize_orig = v2.Resize((image_size, image_size), antialias=True)

    manifest_records = []
    split_counts = {'train': 0, 'val': 0, 'test': 0}

    print(f"\nIniciando extração sanitizada para {len(classes)} classes e {len(split_plans)} planos de particionamento...")

    for plan in tqdm(split_plans, desc="Processando planos de vídeo"):
        split_class_dir = output_dir / plan.split / plan.class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_decimated_frames(
            plan.video_path,
            target_fps=target_fps,
            start_ratio=plan.start_ratio,
            end_ratio=plan.end_ratio,
            min_delta=min_delta,
            max_frames=args.max_frames,
        )

        for frame_idx, timestamp_sec, frame_np in frames:
            frame_id = f"{plan.video_path.stem}_f{frame_idx:05d}"

            img_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(device)

            orig_resized = resize_orig(img_tensor)
            orig_filename = f"{frame_id}_orig.png"
            orig_path = split_class_dir / orig_filename
            save_tensor_as_png(orig_resized, orig_path)
            split_counts[plan.split] += 1

            manifest_records.append({
                'filename': orig_filename,
                'path': str(orig_path.relative_to(output_dir)),
                'class_name': plan.class_name,
                'split': plan.split,
                'video_name': plan.video_path.name,
                'video_hash': plan.content_hash,
                'frame_idx': frame_idx,
                'timestamp_sec': round(timestamp_sec, 3),
                'is_augmented': False,
            })

            if plan.split == 'train' and aug_per_frame > 0:
                for aug_idx in range(1, aug_per_frame + 1):
                    aug_tensor = gpu_augmentor(img_tensor)
                    aug_tensor = apply_gaussian_noise_gpu(aug_tensor, aug_opt)
                    aug_filename = f"{frame_id}_aug{aug_idx}.png"
                    aug_path = split_class_dir / aug_filename
                    save_tensor_as_png(aug_tensor, aug_path)
                    split_counts[plan.split] += 1

                    manifest_records.append({
                        'filename': aug_filename,
                        'path': str(aug_path.relative_to(output_dir)),
                        'class_name': plan.class_name,
                        'split': plan.split,
                        'video_name': plan.video_path.name,
                        'video_hash': plan.content_hash,
                        'frame_idx': frame_idx,
                        'timestamp_sec': round(timestamp_sec, 3),
                        'is_augmented': True,
                    })

    manifest_path = output_dir / 'dataset_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump({
            'total_images': len(manifest_records),
            'split_counts': split_counts,
            'classes': classes,
            'records': manifest_records,
        }, f, indent=2)

    print(f"\n==========================================")
    print(f"Dataset sanitizado gerado com sucesso em '{output_dir}'!")
    print(f"Total de imagens: {len(manifest_records)}")
    print(f"  - Treino: {split_counts['train']} imagens (originais + {aug_per_frame} aumentações)")
    print(f"  - Validação: {split_counts['val']} imagens (puras/sem aumento)")
    print(f"  - Teste: {split_counts['test']} imagens (puras/sem aumento)")
    print(f"Manifesto auditável gravado em: {manifest_path}")
    print(f"==========================================")


if __name__ == '__main__':
    main()
