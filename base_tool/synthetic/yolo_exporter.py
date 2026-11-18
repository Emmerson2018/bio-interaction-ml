import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from base_tool.data.dataset_splitter import VideoDatasetSplitter
from base_tool.models.sam2_video_tracker import SAM2VideoTracker


def mask_to_yolo_polygons(
    mask_u8: np.ndarray,
    frame_w: int,
    frame_h: int,
    eps_ratio: float = 0.004,
    min_area_px: float = 25.0,
) -> List[List[float]]:
    """
    Converte uma mascara binaria (H, W) em lista de poligonos normalizados YOLO-Seg [x1, y1, x2, y2, ...].
    """
    if mask_u8 is None or mask_u8.size == 0 or frame_w <= 0 or frame_h <= 0:
        return []
    m_bin = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    areas = [float(cv2.contourArea(c)) for c in contours]
    max_area = max(areas) if areas else 0.0
    polys: List[List[float]] = []

    for cnt, area in zip(contours, areas):
        if area < min_area_px and area < max_area:
            continue
        peri = float(cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, max(1.0, float(eps_ratio) * peri), True)
        pts = approx.reshape(-1, 2) if len(approx) >= 3 else cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        coords: List[float] = []
        for x, y in pts:
            coords.append(float(np.clip(float(x) / float(frame_w), 0.0, 1.0)))
            coords.append(float(np.clip(float(y) / float(frame_h), 0.0, 1.0)))
        polys.append(coords)
    return polys


def box_to_yolo_detect(
    box_xyxy: Sequence[float],
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Converte [x1, y1, x2, y2] em (cx, cy, w, h) normalizados para YOLO-Detect."""
    if box_xyxy is None or len(box_xyxy) != 4 or frame_w <= 0 or frame_h <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = float(np.clip(x1, 0.0, float(frame_w)))
    y1 = float(np.clip(y1, 0.0, float(frame_h)))
    x2 = float(np.clip(x2, 0.0, float(frame_w)))
    y2 = float(np.clip(y2, 0.0, float(frame_h)))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return None
    cx = (x1 + x2) * 0.5 / float(frame_w)
    cy = (y1 + y2) * 0.5 / float(frame_h)
    return (
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(bw / float(frame_w), 0.0, 1.0)),
        float(np.clip(bh / float(frame_h), 0.0, 1.0)),
    )


def write_yolo_data_yaml(yolo_dir: Path, animal_classes: Sequence[str]) -> Path:
    """Grava o arquivo data.yaml portatil (sem path absoluto) excluindo `fundo_sem_animal`."""
    yolo_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(animal_classes),
        "names": {int(i): str(c) for i, c in enumerate(animal_classes)},
    }
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def export_yolo_sample(
    yolo_dir: Path,
    split: str,
    sample_stem: str,
    frame_bgr: np.ndarray,
    class_id: Optional[int],
    mask_u8: Optional[np.ndarray],
    box_xyxy: Optional[Sequence[float]],
    yolo_task: str = "segment",
    has_animal: bool = True,
    jpeg_quality: int = 95,
) -> bool:
    """
    Salva imagem completa (`images/<split>/<stem>.jpg`) e rotulo `.txt` (`labels/<split>/<stem>.txt`).
    Para `fundo_sem_animal` (`has_animal=False`), escreve `.txt` vazio (hard negative).
    """
    h_f, w_f = frame_bgr.shape[:2]
    lines: List[str] = []
    if has_animal:
        if class_id is None:
            return False
        if yolo_task == "detect":
            bb = box_to_yolo_detect(box_xyxy, w_f, h_f)
            if bb is None:
                return False
            lines.append(f"{int(class_id)} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
        else:
            polys = mask_to_yolo_polygons(mask_u8, w_f, h_f)
            if not polys:
                return False
            for poly in polys:
                lines.append(f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in poly))

    img_dir = yolo_dir / "images" / split
    lbl_dir = yolo_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(img_dir / f"{sample_stem}.jpg"),
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    (lbl_dir / f"{sample_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def export_yolo_from_manifest(
    manifest_path: Path,
    source_dir: Path,
    annotations_file: Path,
    yolo_dir: Path,
    sam_model_path: str = "sam2_t.pt",
    device_str: Optional[str] = None,
    yolo_task: str = "segment",
    sim_thresh: float = 0.52,
    gap_ratio: float = 0.10,
    val_ratio_for_shared: float = 0.45,
    seed: int = 20260814,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gera dataset YOLO-Detect / YOLO-Seg diretamente a partir dos frames ja filtrados em `dataset_manifest.json`,
    evitando re-executar passadas densas 10fps por 4 horas (~15-20x mais rapido).
    """
    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = mdata.get("records") or []
    annotations = json.loads(annotations_file.read_text(encoding="utf-8"))

    all_classes = sorted({r["class_name"] for r in records})
    animal_classes = sorted([c for c in all_classes if c != "fundo_sem_animal"])
    cls_to_id = {c: i for i, c in enumerate(animal_classes)}

    splitter = VideoDatasetSplitter(
        source_dir=source_dir,
        gap_ratio=gap_ratio,
        val_ratio_for_shared=val_ratio_for_shared,
        seed=seed,
    )
    plan_lookup = {
        (p.split, p.class_name, p.video_path.name): p for p in splitter.build_split_plan()
    }

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(rec["split"], rec["class_name"], rec["video_name"])].append(rec)

    dev = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    tracker = SAM2VideoTracker(
        sam_model_path=sam_model_path,
        device=dev,
        sim_threshold=sim_thresh,
        use_puzzle_v4=False,
        enable_roi_auditor=True,
    )

    write_yolo_data_yaml(yolo_dir, animal_classes)
    split_counts = {"train": 0, "val": 0, "test": 0}
    exported = 0
    skipped = 0

    for (split, cls_name, vid_name), recs in tqdm(
        sorted(grouped.items()), desc=f"Exportando YOLO ({yolo_task}) via manifesto"
    ):
        plan = plan_lookup.get((split, cls_name, vid_name))
        vpath = plan.video_path if plan is not None else (source_dir / cls_name / vid_name)
        if not vpath.exists():
            skipped += len(recs)
            continue

        recs_sorted = sorted(recs, key=lambda r: int(r["frame_idx"]))
        if max_frames is not None:
            recs_sorted = recs_sorted[: int(max_frames)]

        rel_key = f"{cls_name}/{vid_name}"
        prompt_data = annotations.get(rel_key, {})
        has_animal = bool(prompt_data.get("has_animal", cls_name != "fundo_sem_animal"))

        tracker.reset()
        if has_animal and prompt_data:
            s_ratio = float(plan.start_ratio) if plan is not None else 0.0
            e_ratio = float(plan.end_ratio) if plan is not None else 1.0
            tracker.initialize_from_video_annotation(
                vpath,
                prompt_data,
                start_ratio=s_ratio,
                end_ratio=e_ratio,
                split_name=split,
                enforce_split_bounds=True,
                annotations_dir=annotations_file.parent,
            )

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            skipped += len(recs_sorted)
            continue

        for rec in recs_sorted:
            f_idx = int(rec["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                skipped += 1
                continue

            stem = f"{cls_name}_{Path(vid_name).stem}_f{f_idx:05d}"
            if not has_animal:
                if export_yolo_sample(
                    yolo_dir=yolo_dir,
                    split=split,
                    sample_stem=stem,
                    frame_bgr=frame_bgr,
                    class_id=None,
                    mask_u8=None,
                    box_xyxy=None,
                    yolo_task=yolo_task,
                    has_animal=False,
                ):
                    exported += 1
                    split_counts[split] = split_counts.get(split, 0) + 1
                continue

            if not tracker.is_initialized:
                skipped += 1
                continue

            step_res = tracker.step_frame(frame_bgr, frame_idx=f_idx, use_puzzle_v4=False)
            if step_res.get("is_valid") and step_res.get("mask") is not None:
                if export_yolo_sample(
                    yolo_dir=yolo_dir,
                    split=split,
                    sample_stem=stem,
                    frame_bgr=frame_bgr,
                    class_id=cls_to_id.get(cls_name),
                    mask_u8=step_res["mask"],
                    box_xyxy=step_res.get("box"),
                    yolo_task=yolo_task,
                    has_animal=True,
                ):
                    exported += 1
                    split_counts[split] = split_counts.get(split, 0) + 1
                else:
                    skipped += 1
            else:
                skipped += 1

        cap.release()

    summary = {
        "yolo_dir": str(yolo_dir),
        "yolo_task": yolo_task,
        "animal_classes": animal_classes,
        "exported": exported,
        "skipped": skipped,
        "split_counts": split_counts,
    }
    (yolo_dir / "yolo_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✓ YOLO ({yolo_task}) exportado em '{yolo_dir}': {exported} imagens ({split_counts}), {skipped} descartadas.")
    return summary
