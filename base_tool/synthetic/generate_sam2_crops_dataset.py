import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
import yaml
from tqdm import tqdm
from ultralytics import SAM

from base_tool.data.dataset_splitter import VideoDatasetSplitter
from base_tool.synthetic.generate_video_dataset import (
    _resolve_path,
    _write_class_index,
    extract_decimated_frames,
    save_tensor_as_png,
)


def extract_square_crop(frame_bgr, box, margin: float = 0.15, target_size: int = 224):
    """
    Recorta uma ROI quadrada centralizada na Bounding Box com margem dinâmica (+15% padding),
    mantendo aspect ratio 1:1 sem distorção anamórfica, e redimensiona para target_size x target_size.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(bw, bh) * (1.0 + margin)
    half = side / 2.0

    # Ajusta o centro se o quadrado couber na dimensão para evitar distorção de aspect ratio
    if side <= w:
        if cx - half < 0:
            cx = half
        elif cx + half > w:
            cx = w - half
    if side <= h:
        if cy - half < 0:
            cy = half
        elif cy + half > h:
            cy = h - half

    nx1 = max(0, int(round(cx - half)))
    ny1 = max(0, int(round(cy - half)))
    nx2 = min(w, int(round(cx + half)))
    ny2 = min(h, int(round(cy + half)))

    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        crop = extract_center_crop(frame_bgr, target_size=target_size)
        return crop, [0, 0, w, h]

    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized, [nx1, ny1, nx2, ny2]


def extract_center_crop(frame_bgr, target_size: int = 224):
    """Recorte central quadrado para vídeos sem animal (fundo puro)."""
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    crop = frame_bgr[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description="Gerador de dataset recortado na fauna via SAM 2 Point-Prompt.")
    parser.add_argument("-opt", "--options", required=True, help="Caminho para o arquivo YAML de opções.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limite de quadros por vídeo para testes.")
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    with open(args.options, mode="r", encoding="utf-8") as file:
        opt = yaml.load(file, Loader=yaml.FullLoader)

    video_opt = opt.get("video", {})
    source_dir = _resolve_path(root_path, video_opt["source_dir"])
    output_dir = _resolve_path(root_path, video_opt["output_dir"])
    annotations_file = _resolve_path(root_path, video_opt.get("annotations_file", "assets/annotations/video_point_prompts.json"))
    sam_model_path = video_opt.get("sam_model", "sam2_t.pt")
    crop_margin = float(video_opt.get("crop_margin", 0.15))
    image_size = int(video_opt.get("image_size", 224))
    seed = int(video_opt.get("seed", 20260814))
    device_str = video_opt.get("device", "cuda")
    target_fps = float(video_opt.get("target_fps", 2.0))
    min_delta = float(video_opt.get("min_frame_delta", 0.02))
    gap_ratio = float(video_opt.get("gap_ratio", 0.10))
    val_ratio_for_shared = float(video_opt.get("val_ratio_for_shared", 0.45))

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Diretório fonte não encontrado: {source_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Arquivo de anotações não encontrado: {annotations_file}")

    with annotations_file.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"✓ Carregadas {len(annotations)} anotações de âncora de '{annotations_file}'.")

    print(f"Carregando modelo SAM 2: '{sam_model_path}'...")
    sam_model = SAM(sam_model_path)

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

    manifest_records = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    print(f"\nIniciando extração SAM 2 para {len(classes)} classes e {len(split_plans)} planos de particionamento...")

    for plan in tqdm(split_plans, desc="Processando planos com SAM 2"):
        split_class_dir = output_dir / plan.split / plan.class_name
        split_class_dir.mkdir(parents=True, exist_ok=True)

        rel_key = f"{plan.class_name}/{plan.video_path.name}"
        prompt_data = annotations.get(rel_key, {})
        has_animal = prompt_data.get("has_animal", True)
        point_norm = prompt_data.get("point_norm")
        point_px = prompt_data.get("point_px")

        frames = extract_decimated_frames(
            plan.video_path,
            target_fps=target_fps,
            start_ratio=plan.start_ratio,
            end_ratio=plan.end_ratio,
            min_delta=min_delta,
            max_frames=args.max_frames,
        )

        last_box = None

        for frame_idx, timestamp_sec, frame_rgb in frames:
            frame_id = f"{plan.video_path.stem}_f{frame_idx:05d}"
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            h_f, w_f = frame_bgr.shape[:2]

            if point_norm is not None:
                pt_x = int(round(point_norm[0] * w_f))
                pt_y = int(round(point_norm[1] * h_f))
                active_point = [pt_x, pt_y]
            elif point_px is not None:
                active_point = point_px
            else:
                active_point = None

            if not has_animal or active_point is None:
                # Caso classe sem animal (fundo puro)
                crop_bgr = extract_center_crop(frame_bgr, target_size=image_size)
                applied_box = [0, 0, w_f, h_f]
                detection_mode = "center_crop_fundo"
            else:
                # Segmentação SAM 2
                try:
                    is_first = (last_box is None)
                    if is_first:
                        # 1º quadro: usar o ponto de âncora anotado pelo usuário
                        results = sam_model(frame_bgr, points=[active_point], labels=[1], verbose=False)
                    else:
                        # Quadros subsequentes: propagar via Bounding Box prompt
                        results = sam_model(frame_bgr, bboxes=[last_box], verbose=False)

                    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                        detected_box = results[0].boxes.xyxy[0].cpu().tolist()
                        detection_mode = "sam2_point" if is_first else "sam2_track"
                        last_box = detected_box
                    elif last_box is not None:
                        detected_box = last_box
                        detection_mode = "sam2_box_fallback"
                    else:
                        detected_box = [w_f * 0.2, h_f * 0.2, w_f * 0.8, h_f * 0.8]
                        detection_mode = "center_fallback"

                    crop_bgr, applied_box = extract_square_crop(
                        frame_bgr, detected_box, margin=crop_margin, target_size=image_size
                    )
                except Exception as e:
                    crop_bgr = extract_center_crop(frame_bgr, target_size=image_size)
                    applied_box = [0, 0, w_f, h_f]
                    detection_mode = f"error_fallback_{e}"

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0

            crop_filename = f"{frame_id}_crop.png"
            crop_path = split_class_dir / crop_filename
            save_tensor_as_png(crop_tensor, crop_path)
            split_counts[plan.split] += 1

            manifest_records.append({
                "filename": crop_filename,
                "path": str(crop_path.relative_to(output_dir)),
                "class_name": plan.class_name,
                "split": plan.split,
                "video_name": plan.video_path.name,
                "video_hash": plan.content_hash,
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 3),
                "has_animal": has_animal,
                "applied_crop_box": applied_box,
                "detection_mode": detection_mode,
            })

    manifest_path = output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(manifest_records),
            "split_counts": split_counts,
            "classes": classes,
            "records": manifest_records,
        }, f, indent=2)

    print("\n==========================================")
    print(f"Dataset SAM 2 recortado gerado com sucesso em '{output_dir}'!")
    print(f"Total de imagens: {len(manifest_records)}")
    print(f"  - Treino: {split_counts['train']} imagens (ROI SAM 2)")
    print(f"  - Validação: {split_counts['val']} imagens (ROI SAM 2)")
    print(f"  - Teste: {split_counts['test']} imagens (ROI SAM 2)")
    print(f"Manifesto gravado em: {manifest_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
