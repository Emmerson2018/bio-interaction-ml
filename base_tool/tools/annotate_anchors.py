import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def extract_key_frame(video_path: Path, target_time_sec: float = 0.5):
    """Extrai um quadro nítido inicial do vídeo para marcação do ponto de âncora."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = min(int(fps * target_time_sec), max(0, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()
    if not ret or frame is None:
        return None, 0, 0.0

    timestamp = target_frame / fps if fps > 0 else 0.0
    return frame, target_frame, timestamp


def draw_hud(display_img, info_text_lines, warning_text=None):
    """Desenha painel superior semi-transparente com instruções e status."""
    h, w = display_img.shape[:2]
    header_h = 75 if not warning_text else 95
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)

    cv2.line(display_img, (0, header_h), (w, header_h), (0, 200, 255), 2)

    y_offset = 24
    for line in info_text_lines:
        cv2.putText(
            display_img,
            line,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 24

    if warning_text:
        cv2.putText(
            display_img,
            warning_text,
            (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Ferramenta de anotação de âncora (ponto central) para vídeos BioEcoInt.")
    parser.add_argument("--source-dir", type=str, default="assets/videos", help="Diretório dos vídeos brutos.")
    parser.add_argument(
        "--output",
        type=str,
        default="assets/annotations/video_point_prompts.json",
        help="Caminho do arquivo JSON de anotações.",
    )
    parser.add_argument("--review", action="store_true", help="Revisar todos os vídeos, mesmo os já anotados.")
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parents[2]
    source_dir = (root_path / args.source_dir).resolve()
    output_path = (root_path / args.output).resolve()

    if not source_dir.exists():
        print(f"Erro: Diretório de vídeos não encontrado em '{source_dir}'", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Carregar anotações existentes (se houver) para suporte a interrupção/retomada
    annotations = {}
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                annotations = json.load(f)
            print(f"✓ Carregadas {len(annotations)} anotações salvas anteriormente.")
        except Exception as e:
            print(f"Aviso ao ler anotações existentes: {e}. Iniciando novo arquivo.")

    # 2. Escanear todos os vídeos organizados por classe
    video_list = []
    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    for c_dir in class_dirs:
        class_name = c_dir.name
        vids = sorted([v for v in c_dir.iterdir() if v.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES])
        for v in vids:
            rel_key = f"{class_name}/{v.name}"
            video_list.append((class_name, v, rel_key))

    total_videos = len(video_list)
    print(f"\n==========================================")
    print(f"BioEcoInt — Anotação Rápida de Ponto de Âncora (Point-Prompt)")
    print(f"Total de vídeos localizados: {total_videos} em {len(class_dirs)} classes.")
    print(f"Destino das anotações: {output_path}")
    print(f"==========================================")

    window_name = "BioEcoInt - Marcacao de Ponto de Ancora"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Estado global do mouse
    clicked_point = [None]  # [px_x, px_y]
    current_scale = [1.0]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = current_scale[0]
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            clicked_point[0] = (orig_x, orig_y)

    cv2.setMouseCallback(window_name, on_mouse)

    idx = 0
    warning_msg = None

    while 0 <= idx < total_videos:
        class_name, video_path, rel_key = video_list[idx]

        # Se já estiver anotado e não for modo review, avança
        if not args.review and rel_key in annotations:
            idx += 1
            continue

        # Pré-configuração: se for a classe fundo_sem_animal, marcar automaticamente como sem animal
        if class_name == "fundo_sem_animal" and rel_key not in annotations:
            annotations[rel_key] = {
                "class_name": class_name,
                "video_name": video_path.name,
                "video_rel_path": rel_key,
                "has_animal": False,
                "point_norm": None,
                "point_px": None,
                "frame_size": None,
                "frame_idx": 0,
                "timestamp_sec": 0.0,
            }
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=2, ensure_ascii=False)
            idx += 1
            continue

        frame, frame_idx, timestamp_sec = extract_key_frame(video_path)
        if frame is None:
            print(f"Aviso: Não foi possível ler o vídeo '{rel_key}'. Pulando.")
            idx += 1
            continue

        orig_h, orig_w = frame.shape[:2]

        # Inicializar ponto com o que já foi salvo anteriormente (se houver)
        if rel_key in annotations and annotations[rel_key].get("has_animal"):
            clicked_point[0] = tuple(annotations[rel_key]["point_px"])
        else:
            clicked_point[0] = None

        # Redimensionar para exibição fluida em tela
        max_disp_w = 1280
        max_disp_h = 720
        scale = min(max_disp_w / orig_w, max_disp_h / orig_h, 1.0)
        current_scale[0] = scale
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)

        while True:
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # Desenhar marcador do clique
            pt = clicked_point[0]
            if pt is not None:
                cx = int(pt[0] * scale)
                cy = int(pt[1] * scale)
                # Alvo com mira verde
                cv2.circle(display_frame, (cx, cy), 18, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(display_frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(display_frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1, cv2.LINE_AA)
                norm_x = pt[0] / orig_w
                norm_y = pt[1] / orig_h
                coord_str = f"Ponto: [{norm_x:.3f}, {norm_y:.3f}] ({pt[0]}, {pt[1]})"
            else:
                coord_str = "Nenhum ponto marcado ainda (Clique sobre a escultura!)"

            line1 = f"[{idx + 1}/{total_videos}] Classe: {class_name.upper()} | Arquivo: {video_path.name} ({orig_w}x{orig_h})"
            line2 = f"[Clique]: Marcar | [Espaco/Enter]: Confirmar | [R]: Resetar | [S]: Sem animal | [B]: Voltar | [Q]: Sair | {coord_str}"
            draw_hud(display_frame, [line1, line2], warning_text=warning_msg)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (32, 13):  # Espaço ou Enter
                if clicked_point[0] is None:
                    warning_msg = "ERRO: Clique sobre a escultura do animal antes de salvar! (Ou aperte 'S' se nao houver animal)."
                    continue

                px_x, px_y = clicked_point[0]
                norm_x = round(px_x / orig_w, 4)
                norm_y = round(px_y / orig_h, 4)

                annotations[rel_key] = {
                    "class_name": class_name,
                    "video_name": video_path.name,
                    "video_rel_path": rel_key,
                    "has_animal": True,
                    "point_norm": [norm_x, norm_y],
                    "point_px": [px_x, px_y],
                    "frame_size": [orig_w, orig_h],
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                }

                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)

                print(f"[{idx + 1}/{total_videos}] ✓ {rel_key} salvo -> Âncora ({norm_x}, {norm_y})")
                warning_msg = None
                idx += 1
                break

            elif key in (ord('s'), ord('S')):  # Sem animal
                annotations[rel_key] = {
                    "class_name": class_name,
                    "video_name": video_path.name,
                    "video_rel_path": rel_key,
                    "has_animal": False,
                    "point_norm": None,
                    "point_px": None,
                    "frame_size": [orig_w, orig_h],
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                }
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(annotations, f, indent=2, ensure_ascii=False)

                print(f"[{idx + 1}/{total_videos}] ∅ {rel_key} marcado como sem animal.")
                warning_msg = None
                idx += 1
                break

            elif key in (ord('r'), ord('R')):  # Reset
                clicked_point[0] = None
                warning_msg = None

            elif key in (ord('b'), ord('B')):  # Voltar
                if idx > 0:
                    idx -= 1
                    warning_msg = None
                    break

            elif key in (ord('q'), ord('Q'), 27):  # Sair
                print("\nAnotação interrompida pelo usuário. Progresso preservado no JSON.")
                cv2.destroyAllWindows()
                _print_summary(annotations, total_videos, output_path)
                return

    cv2.destroyAllWindows()
    print("\n🎉 Todos os vídeos foram processados e anotados com sucesso!")
    _print_summary(annotations, total_videos, output_path)


def _print_summary(annotations, total_videos, output_path):
    with_animal = sum(1 for v in annotations.values() if v.get("has_animal"))
    without_animal = sum(1 for v in annotations.values() if not v.get("has_animal"))
    print(f"\n==========================================")
    print(f"Resumo da Anotação:")
    print(f"  - Total de vídeos registrados: {len(annotations)}/{total_videos}")
    print(f"  - Vídeos com animal marcado: {with_animal}")
    print(f"  - Vídeos sem animal (fundo puro): {without_animal}")
    print(f"Arquivo gerado: {output_path}")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
