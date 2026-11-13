import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import SAM

from base_tool.models.exemplar_verifier import ExemplarVerifier


SUPPORTED_VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def extract_square_crop(
    img_bgr: np.ndarray,
    box_xyxy: List[float],
    margin: float = 0.15,
    target_size: int = 224,
) -> Tuple[np.ndarray, List[int]]:
    """Recorta a ROI com padding simétrico quadrado (+15% de margem) e redimensiona."""
    h_orig, w_orig = img_bgr.shape[:2]
    bx1, by1, bx2, by2 = box_xyxy
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    side = max(bw, bh) * (1.0 + margin * 2.0)

    x1 = max(0, int(round(cx - side / 2.0)))
    y1 = max(0, int(round(cy - side / 2.0)))
    x2 = min(w_orig, int(round(cx + side / 2.0)))
    y2 = min(h_orig, int(round(cy + side / 2.0)))

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        side_fallback = int(min(h_orig, w_orig) * 0.8)
        cx_f, cy_f = w_orig // 2, h_orig // 2
        x1 = max(0, cx_f - side_fallback // 2)
        y1 = max(0, cy_f - side_fallback // 2)
        x2 = min(w_orig, cx_f + side_fallback // 2)
        y2 = min(h_orig, cy_f + side_fallback // 2)
        crop = img_bgr[y1:y2, x1:x2]

    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized, [x1, y1, x2, y2]


def draw_hud(
    display_img: np.ndarray,
    info_lines: List[str],
    warning_text: Optional[str] = None,
    is_paused: bool = False,
    is_occluded: bool = False,
):
    """Desenha painel HUD superior semi-transparente com informações e atalhos."""
    h, w = display_img.shape[:2]
    header_h = 80 if not warning_text else 105
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

    status_color = (0, 0, 255) if is_occluded else ((0, 165, 255) if is_paused else (0, 230, 100))
    cv2.line(display_img, (0, header_h), (w, header_h), status_color, 2)

    y_offset = 24
    for i, line in enumerate(info_lines):
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 255)
        elif "OCLUSAO" in line:
            color = (0, 80, 255)
        elif "RASTREAMENTO ATIVO" in line:
            color = (0, 230, 100)

        cv2.putText(
            display_img,
            line,
            (16, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            1,
            cv2.LINE_AA,
        )
        y_offset += 24

    if warning_text:
        cv2.putText(
            display_img,
            warning_text,
            (16, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )


def overlay_mask_and_box(
    frame_bgr: np.ndarray,
    mask_np: Optional[np.ndarray],
    box_xyxy: Optional[List[float]],
    point_px: Optional[Tuple[int, int]],
    show_mask: bool = True,
    show_box: bool = True,
    is_valid: bool = True,
) -> np.ndarray:
    """Aplica máscara colorida semi-transparente, contornos e bounding box com status de validade."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1. Overlay da máscara
    if show_mask and mask_np is not None:
        mask_binary = (mask_np > 0).astype(np.uint8)
        if mask_binary.shape[:2] != (h, w):
            mask_binary = cv2.resize(mask_binary, (w, h), interpolation=cv2.INTER_NEAREST)

        # Cor da máscara: Verde Esmeralda se válido, Laranja/Vermelho se ocluso
        mask_color = [0, 220, 100] if is_valid else [0, 50, 230]
        color_mask = np.zeros_like(out, dtype=np.uint8)
        color_mask[mask_binary == 1] = mask_color

        mask_indices = mask_binary == 1
        out[mask_indices] = cv2.addWeighted(
            out[mask_indices], 0.50, color_mask[mask_indices], 0.50, 0
        )

        contour_color = (0, 255, 255) if is_valid else (0, 0, 255)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, contour_color, 2, cv2.LINE_AA)

    # 2. Bounding Box (+15% margem dinâmica)
    if show_box and box_xyxy is not None:
        bx1, by1, bx2, by2 = [int(round(c)) for c in box_xyxy]
        bw = bx2 - bx1
        bh = by2 - by1
        pad_w = int(round(bw * 0.15))
        pad_h = int(round(bh * 0.15))

        x1 = max(0, bx1 - pad_w)
        y1 = max(0, by1 - pad_h)
        x2 = min(w, bx2 + pad_w)
        y2 = min(h, by2 + pad_h)

        box_color = (0, 215, 255) if is_valid else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
        label = "SAM 2 ROI (+15%)" if is_valid else "DESCARTADO (OCLUSAO)"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # 3. Ponto de Âncora (Mira central)
    if point_px is not None:
        px, py = point_px
        cv2.circle(out, (px, py), 12, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (px, py), 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(out, (px - 18, py), (px + 18, py), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(out, (px, py - 18), (px, py + 18), (0, 0, 255), 1, cv2.LINE_AA)

    return out


def render_pip_crop(
    display_frame: np.ndarray,
    crop_224: Optional[np.ndarray],
    is_valid: bool = True,
    similarity: float = 1.0,
    pip_size: int = 180,
) -> np.ndarray:
    """Insere PiP da ROI no canto inferior direito com sinalização clara de aprovação ou descarte."""
    h_disp, w_disp = display_frame.shape[:2]
    if crop_224 is None or crop_224.size == 0:
        return display_frame

    pip_resized = cv2.resize(crop_224, (pip_size, pip_size), interpolation=cv2.INTER_LINEAR)
    pad = 16
    x2 = w_disp - pad
    y2 = h_disp - pad
    x1 = x2 - pip_size
    y1 = y2 - pip_size

    if x1 >= 0 and y1 >= 0:
        border_color = (0, 215, 255) if is_valid else (0, 0, 255)
        title_text = f"ROI [S={similarity:.2f}]" if is_valid else f"DESCARTADO [S={similarity:.2f}]"

        # Moldura e fundo
        cv2.rectangle(display_frame, (x1 - 3, y1 - 22), (x2 + 3, y2 + 3), (25, 25, 25), -1)
        cv2.rectangle(display_frame, (x1 - 3, y1 - 22), (x2 + 3, y2 + 3), border_color, 2)
        cv2.putText(
            display_frame,
            title_text,
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            border_color,
            1,
            cv2.LINE_AA,
        )

        if not is_valid:
            # Sobreposição avermelhada para evidenciar descarte
            red_tint = np.zeros_like(pip_resized)
            red_tint[:, :] = [0, 0, 180]
            pip_resized = cv2.addWeighted(pip_resized, 0.65, red_tint, 0.35, 0)
            cv2.line(pip_resized, (0, 0), (pip_size, pip_size), (0, 0, 255), 2)
            cv2.line(pip_resized, (0, pip_size), (pip_size, 0), (0, 0, 255), 2)

        display_frame[y1:y2, x1:x2] = pip_resized

    return display_frame


def render_pip_exemplar(
    display_frame: np.ndarray,
    exemplar_crop_224: Optional[np.ndarray],
    pip_size: int = 100,
) -> np.ndarray:
    """Insere thumbnail do Gabarito de Ouro (Frame 1) no canto superior direito abaixo do HUD."""
    if exemplar_crop_224 is None or exemplar_crop_224.size == 0:
        return display_frame

    h_disp, w_disp = display_frame.shape[:2]
    pad = 16
    y1 = 90
    y2 = y1 + pip_size
    x2 = w_disp - pad
    x1 = x2 - pip_size

    if x1 >= 0 and y2 <= h_disp:
        pip_resized = cv2.resize(exemplar_crop_224, (pip_size, pip_size), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(display_frame, (x1 - 2, y1 - 18), (x2 + 2, y2 + 2), (25, 25, 25), -1)
        cv2.rectangle(display_frame, (x1 - 2, y1 - 18), (x2 + 2, y2 + 2), (0, 255, 255), 1)
        cv2.putText(
            display_frame,
            "Gabarito Frame 1",
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        display_frame[y1:y2, x1:x2] = pip_resized

    return display_frame


def main():
    parser = argparse.ArgumentParser(description="Revisor Interativo de Segmentação SAM 2 com Exemplar Matching Anti-Oclusão.")
    parser.add_argument("--source-dir", type=str, default="assets/videos", help="Diretório dos vídeos brutos.")
    parser.add_argument(
        "--annotations",
        type=str,
        default="assets/annotations/video_point_prompts.json",
        help="Caminho do arquivo de anotações com as âncoras salvas.",
    )
    parser.add_argument("--sam-model", type=str, default="sam2_t.pt", help="Modelo de pesos do SAM 2.")
    parser.add_argument("--start-idx", type=int, default=0, help="Índice inicial do vídeo.")
    parser.add_argument("--sim-thresh", type=float, default=0.50, help="Limiar de similaridade para oclusão.")
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    source_dir = (root_path / args.source_dir).resolve()
    annotations_path = (root_path / args.annotations).resolve()

    if not source_dir.exists():
        print(f"Erro: Diretório de vídeos '{source_dir}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    print("\n==========================================")
    print("BioEcoInt — Revisor de Segmentação SAM 2 com Exemplar Matching")
    print(f"Vídeos: {source_dir}")
    print(f"Anotações: {annotations_path}")
    print(f"Limiar de Similaridade Anti-Oclusão: {args.sim_thresh}")
    print("==========================================")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam_model = SAM(args.sam_model)
    verifier = ExemplarVerifier(device=device, sim_threshold=args.sim_thresh)
    print(f"✓ SAM 2 e ExemplarVerifier inicializados no dispositivo: {device.upper()}")

    # 1. Carregar anotações
    annotations = {}
    if annotations_path.exists():
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ {len(annotations)} anotações de vídeos carregadas.")
    else:
        print("Aviso: Arquivo de anotações não encontrado.")

    # 2. Escanear vídeos
    video_list = []
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    for c_dir in class_dirs:
        class_name = c_dir.name
        vids = sorted([v for v in c_dir.iterdir() if v.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES])
        for v in vids:
            rel_key = f"{class_name}/{v.name}"
            video_list.append((class_name, v, rel_key))

    total_videos = len(video_list)
    if total_videos == 0:
        print("Nenhum vídeo compatível encontrado.")
        sys.exit(1)

    window_name = "BioEcoInt - Revisor SAM 2 Anti-Oclusao"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    clicked_new_point = [None]
    current_display_scale = [1.0]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = current_display_scale[0]
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            clicked_new_point[0] = (orig_x, orig_y)

    cv2.setMouseCallback(window_name, on_mouse)

    idx = max(0, min(args.start_idx, total_videos - 1))
    show_mask = True
    show_box = True
    notification_msg = None

    while 0 <= idx < total_videos:
        class_name, video_path, rel_key = video_list[idx]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Aviso: Falha ao abrir {video_path.name}. Avançando.")
            idx += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        ann = annotations.get(rel_key, {})
        has_animal = ann.get('has_animal', class_name != "fundo_sem_animal")
        point_norm = ann.get('point_norm')

        if point_norm and len(point_norm) == 2:
            base_point_px = (int(round(point_norm[0] * orig_w)), int(round(point_norm[1] * orig_h)))
        else:
            base_point_px = (orig_w // 2, orig_h // 2) if has_animal else None

        active_point_px = base_point_px
        clicked_new_point[0] = None

        is_paused = False
        frame_idx = 0
        last_tracked_box = None

        verifier.reset()
        segmentation_cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Optional[List[float]], Optional[np.ndarray], bool, float]] = {}

        max_disp_w, max_disp_h = 1280, 720
        scale = min(max_disp_w / orig_w, max_disp_h / orig_h, 1.0)
        current_display_scale[0] = scale
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

        while True:
            # 1. Clique do usuário para testar novo ponto
            if clicked_new_point[0] is not None:
                active_point_px = clicked_new_point[0]
                clicked_new_point[0] = None
                has_animal = True
                last_tracked_box = None
                verifier.reset()
                segmentation_cache.clear()
                notification_msg = f"Novo Ponto Testado: {active_point_px} | [U]: Salvar Âncora | [R]: Resetar"

            # 2. Recuperar ou processar quadro atual
            if frame_idx in segmentation_cache:
                frame_raw, mask_np, box_xyxy, crop_224, is_valid, sim_score = segmentation_cache[frame_idx]
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_raw = cap.read()
                if not ret or frame_raw is None:
                    frame_idx = 0
                    last_tracked_box = None
                    continue

                mask_np = None
                box_xyxy = None
                crop_224 = None
                is_valid = False
                sim_score = 0.0

                if has_animal and active_point_px is not None:
                    try:
                        # Fase A: Segmentação
                        if last_tracked_box is not None:
                            # Propagação normal por Bounding Box do quadro anterior
                            res = sam_model(
                                frame_raw,
                                bboxes=[last_tracked_box],
                                verbose=False,
                                device=device,
                            )
                        else:
                            # Sonda de Inicialização / Re-Aquisição na âncora nominal
                            res = sam_model(
                                frame_raw,
                                points=[list(active_point_px)],
                                labels=[1],
                                verbose=False,
                                device=device,
                            )

                        candidate_found = (
                            len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0
                        )

                        if candidate_found:
                            box_cand = res[0].boxes.xyxy[0].cpu().numpy().tolist()
                            mask_cand = res[0].masks.data[0].cpu().numpy().astype(np.uint8) if res[0].masks is not None else None
                            crop_cand, _ = extract_square_crop(frame_raw, box_cand, margin=0.15, target_size=224)

                            # Se ainda não tivermos o Gabarito de Ouro, o primeiro frame com animal vira o gabarito
                            if not verifier.has_exemplar():
                                verifier.set_exemplar(crop_cand, mask_cand)
                                is_valid = True
                                sim_score = 1.0
                                box_xyxy = box_cand
                                mask_np = mask_cand
                                crop_224 = crop_cand
                                last_tracked_box = box_cand
                            else:
                                # Avaliar candidato contra o Gabarito de Ouro
                                ver_res = verifier.verify_candidate(
                                    crop_cand, box_cand, frame_raw.shape, mask_cand
                                )
                                is_valid = ver_res['is_valid']
                                sim_score = ver_res['similarity']

                                if is_valid:
                                    # Sucesso! Animal reconhecido e mantido
                                    box_xyxy = box_cand
                                    mask_np = mask_cand
                                    crop_224 = crop_cand
                                    last_tracked_box = box_cand
                                else:
                                    # Oclusão ou deriva detectada! DESCARTA O RECORTE
                                    # e ZERA a caixa anterior para acionar a sonda de re-aquisição no próximo frame
                                    box_xyxy = box_cand
                                    mask_np = mask_cand
                                    crop_224 = crop_cand
                                    last_tracked_box = None
                        else:
                            # Nenhuma caixa encontrada pelo SAM
                            is_valid = False
                            sim_score = 0.0
                            last_tracked_box = None

                    except Exception as e:
                        print(f"Aviso no pipeline SAM 2: {e}")
                        is_valid = False
                        sim_score = 0.0
                        last_tracked_box = None
                else:
                    # Classe fundo_sem_animal (sempre válido como fundo)
                    is_valid = True
                    sim_score = 1.0

                segmentation_cache[frame_idx] = (frame_raw, mask_np, box_xyxy, crop_224, is_valid, sim_score)

            # 3. Compor a visualização
            rendered_frame = overlay_mask_and_box(
                frame_raw,
                mask_np=mask_np,
                box_xyxy=box_xyxy,
                point_px=active_point_px,
                show_mask=show_mask,
                show_box=show_box,
                is_valid=is_valid,
            )

            display_img = cv2.resize(rendered_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # Inset PiP do Crop do Classificador no canto inferior direito
            display_img = render_pip_crop(
                display_img,
                crop_224,
                is_valid=is_valid,
                similarity=sim_score,
                pip_size=180,
            )

            # Thumbnail do Gabarito de Ouro (Frame 1) no canto superior direito
            if verifier.has_exemplar():
                display_img = render_pip_exemplar(
                    display_img,
                    verifier.exemplar_crop_bgr,
                    pip_size=100,
                )

            # Informações no HUD superior
            dur_sec = total_frames / fps if fps > 0 else 0.0
            cur_sec = frame_idx / fps if fps > 0 else 0.0
            status_text = "PAUSADO" if is_paused else "REPRODUZINDO"

            if not has_animal:
                state_label = "[Fundo Puro - Sem Animal]"
            elif is_valid:
                state_label = f"[🟢 RASTREAMENTO ATIVO | Sim: {sim_score:.2f}]"
            else:
                state_label = f"[🔴 OCLUSAO DETECTADA - FRAME DESCARTADO | Sim: {sim_score:.2f}]"

            line1 = f"[{idx + 1}/{total_videos}] Classe: {class_name.upper()} | Arquivo: {video_path.name} ({orig_w}x{orig_h} @ {fps:.1f} FPS)"
            line2 = f"Quadro: {frame_idx + 1}/{total_frames} ({cur_sec:.2f}s / {dur_sec:.2f}s) | Status: [{status_text}] | {state_label}"
            line3 = "Atalhos: [Espaço]: Play/Pause | [Setas <- / ->]: Video Ant/Prox | [Setas ↑ / ↓]: Quadro -/+ | [M]: Mascara | [U]: Salvar Ponto | [Q]: Sair"

            draw_hud(
                display_img,
                [line1, line2, line3],
                warning_text=notification_msg,
                is_paused=is_paused,
                is_occluded=(not is_valid and has_animal),
            )
            cv2.imshow(window_name, display_img)

            delay_ms = 30 if is_paused else max(10, int(round(1000.0 / fps)))
            key = cv2.waitKeyEx(delay_ms)

            if key == -1:
                if not is_paused:
                    frame_idx = (frame_idx + 1) % total_frames
                continue

            if key in (27, ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                print("\nRevisor de segmentação encerrado.")
                return

            elif key in (32, ord('p'), ord('P'), ord('k'), ord('K')):
                is_paused = not is_paused
                notification_msg = None

            elif key in (65363, 2555904, ord('d'), ord('D'), ord('n'), ord('N'), 13):
                cap.release()
                notification_msg = None
                idx += 1
                break

            elif key in (65361, 2424832, ord('a'), ord('A'), ord('b'), ord('B'), 8):
                cap.release()
                notification_msg = None
                idx = max(0, idx - 1)
                break

            elif key in (65362, 2490368, ord('.'), ord('>'), ord('f'), ord('F')):
                is_paused = True
                frame_idx = (frame_idx + 1) % total_frames
                notification_msg = f"Avançado para o quadro {frame_idx + 1}/{total_frames}"

            elif key in (65364, 2621440, ord(','), ord('<'), ord('v'), ord('V')):
                is_paused = True
                frame_idx = (frame_idx - 1 + total_frames) % total_frames
                notification_msg = f"Retornado para o quadro {frame_idx + 1}/{total_frames}"

            elif key in (ord('r'), ord('R')):
                frame_idx = 0
                last_tracked_box = None
                active_point_px = base_point_px
                verifier.reset()
                segmentation_cache.clear()
                notification_msg = "Vídeo reiniciado no quadro 0."

            elif key in (ord('m'), ord('M')):
                show_mask = not show_mask
                notification_msg = f"Exibição da Máscara: {'LIGADA' if show_mask else 'DESLIGADA'}"

            elif key in (ord('u'), ord('U')):
                if active_point_px is not None:
                    nx = round(active_point_px[0] / orig_w, 4)
                    ny = round(active_point_px[1] / orig_h, 4)
                    annotations[rel_key] = {
                        "class_name": class_name,
                        "video_name": video_path.name,
                        "video_rel_path": rel_key,
                        "has_animal": True,
                        "point_norm": [nx, ny],
                        "point_px": [active_point_px[0], active_point_px[1]],
                        "frame_size": [orig_w, orig_h],
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                    }
                    with open(annotations_path, 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=2, ensure_ascii=False)
                    base_point_px = active_point_px
                    notification_msg = f"✓ Âncora salva com sucesso: [{nx}, {ny}] no arquivo JSON!"
                    print(f"[{idx + 1}/{total_videos}] Ponto atualizado em '{rel_key}': [{nx}, {ny}]")

            else:
                if not is_paused:
                    frame_idx = (frame_idx + 1) % total_frames

        cap.release()

    cv2.destroyAllWindows()
    print("Revisão de todos os vídeos finalizada com sucesso.")


if __name__ == '__main__':
    main()
