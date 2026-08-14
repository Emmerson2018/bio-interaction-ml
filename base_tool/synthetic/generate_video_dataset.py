import argparse
import csv
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

SUPPORTED_VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def _resolve_path(root_path, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return root_path / path


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


def _write_class_index(output_dir, classes):
    path = output_dir / 'class_index.csv'
    with path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['class_name', 'class_index'])
        for index, class_name in enumerate(classes):
            writer.writerow([class_name, index])


def build_gpu_augmentor(aug_opt, image_size):
    rot_deg = float(aug_opt.get('rotation_degrees', 30.0))
    shear_deg = float(aug_opt.get('affine_shear', 15.0))
    persp_dist = float(aug_opt.get('perspective_distortion', 0.2))
    color_j = float(aug_opt.get('color_jitter', 0.3))
    blur_p = float(aug_opt.get('blur_prob', 0.5))

    transform_list = [
        v2.Resize((image_size, image_size), antialias=True),
        v2.RandomRotation(degrees=(-rot_deg, rot_deg)),
        v2.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            shear=(-shear_deg, shear_deg)
        ),
        v2.RandomPerspective(distortion_scale=persp_dist, p=0.7),
        v2.ColorJitter(
            brightness=color_j,
            contrast=color_j,
            saturation=color_j,
            hue=min(0.1, color_j * 0.3)
        ),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
            p=blur_p
        ),
    ]

    return v2.Compose(transform_list)


def apply_gaussian_noise_gpu(img_tensor, aug_opt):
    noise_p = float(aug_opt.get('noise_prob', 0.5))
    if torch.rand(1, device=img_tensor.device).item() < noise_p:
        noise_min = float(aug_opt.get('noise_intensity_min', 0.01))
        noise_max = float(aug_opt.get('noise_intensity_max', 0.05))
        std = random.uniform(noise_min, noise_max)
        noise = torch.randn_like(img_tensor) * std
        img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
    return img_tensor


def extract_frames_from_video(video_path, frame_step=1, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}", file=sys.stderr)
        return []

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            # OpenCV captures in BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def save_tensor_as_png(tensor_float, output_path):
    # tensor_float: (C, H, W) in range [0, 1] on device
    tensor_uint8 = (torch.clamp(tensor_float, 0.0, 1.0) * 255.0).to(torch.uint8).cpu()
    io.write_png(tensor_uint8, str(output_path))


def main():
    parser = argparse.ArgumentParser(description='Generate an augmented image dataset from videos.')
    parser.add_argument('-opt', '--options', required=True, help='Path to the video dataset YAML config.')
    parser.add_argument('--max-frames', type=int, default=None, help='Limit max extracted frames per video (useful for quick testing).')
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
    aug_per_frame = int(video_opt.get('augmentations_per_frame', 4))
    frame_step = int(video_opt.get('frame_step', 1))
    splits = video_opt.get('splits', {'train': 0.6, 'val': 0.2, 'test': 0.2})
    aug_opt = video_opt.get('augmentations', {})

    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device('cpu')
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Find class directories
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {source_dir}")

    classes = [d.name for d in class_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_class_index(output_dir, classes)

    gpu_augmentor = build_gpu_augmentor(aug_opt, image_size)
    resize_orig = v2.Resize((image_size, image_size), antialias=True)

    total_generated = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        video_files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES])
        if not video_files:
            print(f"Skipping empty class directory: {class_name}")
            continue

        print(f"\nProcessing class '{class_name}' ({len(video_files)} videos)...")
        all_frames = []
        for vid_file in video_files:
            frames = extract_frames_from_video(vid_file, frame_step=frame_step, max_frames=args.max_frames)
            print(f"  Loaded {len(frames)} frames from {vid_file.name}")
            all_frames.extend([(vid_file.stem, frame_idx, f) for frame_idx, f in enumerate(frames)])

        if not all_frames:
            print(f"No frames extracted for class '{class_name}'.")
            continue

        # Shuffle frames randomly before splitting
        random.shuffle(all_frames)

        # Split count calculation
        split_counts = _split_counts(len(all_frames), splits)

        start_idx = 0
        for split_name, count in split_counts.items():
            if count <= 0:
                continue

            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            split_frames = all_frames[start_idx : start_idx + count]
            start_idx += count

            print(f"  Generating {len(split_frames)} base frames (+{aug_per_frame} augs each) for split '{split_name}'...")

            for frame_info in tqdm(split_frames, desc=f"{class_name}/{split_name}"):
                vid_name, frame_idx, frame_np = frame_info
                frame_id = f"{vid_name}_f{frame_idx:05d}"

                # Convert to FloatTensor (C, H, W) on GPU [0.0, 1.0]
                img_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.to(device)

                # Base sample (Original resized)
                orig_resized = resize_orig(img_tensor)
                orig_path = split_class_dir / f"{frame_id}_orig.png"
                save_tensor_as_png(orig_resized, orig_path)
                total_generated += 1

                # Generate 4 augmented samples
                for aug_idx in range(1, aug_per_frame + 1):
                    aug_tensor = gpu_augmentor(img_tensor)
                    aug_tensor = apply_gaussian_noise_gpu(aug_tensor, aug_opt)
                    aug_path = split_class_dir / f"{frame_id}_aug{aug_idx}.png"
                    save_tensor_as_png(aug_tensor, aug_path)
                    total_generated += 1

    print(f"\nSuccessfully generated {total_generated} images across all splits in '{output_dir}'.")


if __name__ == '__main__':
    main()
