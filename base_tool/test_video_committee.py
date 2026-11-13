import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import yaml
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from ultralytics import SAM

from base_tool.models import build_model
from base_tool.models.exemplar_verifier import ExemplarVerifier
from base_tool.synthetic.generate_sam2_crops_dataset import extract_center_crop, extract_square_crop
from base_tool.utils.options import parse_options


def extract_video_snippet_frames(video_path: Path, target_fps: float = 3.0, window_sec: float = 2.0) -> List[np.ndarray]:
    """Extrai uma janela temporal de `window_sec` segundos a `target_fps` quadros por segundo."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    num_samples = int(round(target_fps * window_sec))  # ex.: 3 FPS * 2s = 6 quadros
    frame_step = max(1, int(round(orig_fps / target_fps)))

    # Amostrar no miolo do vídeo (a partir de 20% da duração)
    start_frame = int(total_frames * 0.20)
    frames_bgr = []

    for i in range(num_samples):
        target_pos = start_frame + (i * frame_step)
        if target_pos >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_bgr.append(frame)

    cap.release()
    return frames_bgr


def preprocess_tensor(img_bgr: np.ndarray, target_size: int = 224, device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """Converte BGR uint8 -> Tensor RGB normalizado [1, 3, H, W]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    # Normalização padrão ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


def run_video_committee(
    opt_path: str,
    checkpoint_path: str,
    manifest_path: str,
    annotations_path: str,
    sam_model_path: str = "sam2_t.pt",
    use_sam2: bool = False,
    consensus_threshold: float = 0.70,
    target_fps: float = 3.0,
    window_sec: float = 2.0,
    sim_threshold: float = 0.50,
    output_report_path: Optional[str] = None,
):
    print("==========================================")
    print("Iniciando Benchmark de Avaliação em Vídeo & Comitê Temporal")
    print(f"Modo: {'SAM 2 Tap-to-Crop + Classificador (com Exemplar Matching)' if use_sam2 else 'Single-Stage Classificador Direto'}")
    print(f"Janela temporal: {window_sec}s @ {target_fps} FPS (Total: ~{int(round(target_fps * window_sec))} quadros)")
    print(f"Limiar de Consenso: {consensus_threshold * 100:.0f}%")
    if use_sam2:
        print(f"Limiar de Similaridade Anti-Oclusão: {sim_threshold:.2f}")
    print("==========================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Carregar Modelo Classificador
    opt = parse_options(opt_path, is_train=False)
    num_classes = int(opt['network_g'].get('num_classes', 16))
    model = build_model(opt)
    net = getattr(model, 'net', getattr(model, 'net_g', None))
    if net is None:
        raise AttributeError("O modelo construído não possui atributo 'net' ou 'net_g'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('network_ema' if 'network_ema' in checkpoint else 'network', checkpoint)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
    load_res = net.load_state_dict(cleaned_state_dict, strict=False)
    if load_res.missing_keys:
        print(f"[Aviso] Chaves ausentes no state_dict: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        print(f"[Info] Chaves ignoradas no state_dict: {len(load_res.unexpected_keys)}")
    net.eval()

    # 2. Carregar SAM 2 e ExemplarVerifier se requisitado
    sam_model = None
    verifier = None
    if use_sam2:
        print(f"Carregando modelo SAM 2: '{sam_model_path}'...")
        sam_model = SAM(sam_model_path)
        verifier = ExemplarVerifier(device=str(device), sim_threshold=sim_threshold)
        print(f"✓ ExemplarVerifier inicializado para descarte de oclusões (S >= {sim_threshold})")

    # 3. Ler Manifesto e Anotações para isolar os vídeos de TESTE
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Coletar vídeos exclusivos do split de TESTE
    classes = manifest.get('classes', sorted(list(set(r['class_name'] for r in manifest['records']))))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    manifest_dir = Path(manifest_path).parent
    test_records = [r for r in manifest['records'] if r.get('split') == 'test']
    records_by_video = {}
    for r in test_records:
        k = f"{r['class_name']}/{r['video_name']}"
        if k not in records_by_video:
            records_by_video[k] = []
        records_by_video[k].append(r)

    unique_test_videos = {}
    for r in test_records:
        key = f"{r['class_name']}/{r['video_name']}"
        if key not in unique_test_videos:
            unique_test_videos[key] = {
                'class_name': r['class_name'],
                'video_name': r['video_name'],
                'class_idx': class_to_idx[r['class_name']],
            }

    print(f"Total de vídeos de TESTE disjuntos identificados: {len(unique_test_videos)}")

    single_frame_preds = []
    single_frame_targets = []

    committee_results = []
    num_decided = 0
    num_correct_committee = 0
    num_abstained = 0

    root_videos = Path('assets/videos')
    num_samples = int(round(target_fps * window_sec))

    for key, vinfo in tqdm(unique_test_videos.items(), desc="Avaliando vídeos de teste"):
        v_path = root_videos / key
        frames = []
        if v_path.exists():
            # Extrair snippet de 2 segundos a 3 FPS do vídeo bruto
            frames = extract_video_snippet_frames(v_path, target_fps=target_fps, window_sec=window_sec)

        # Fallback resiliente: carregar a sequência de quadros do conjunto de teste do dataset
        # (indispensável no Google Colab, onde os 3.1 GB de vídeos .mp4 brutos não são baixados)
        if not frames and key in records_by_video:
            v_recs = sorted(records_by_video[key], key=lambda x: x.get('timestamp_sec', 0.0))
            for r in v_recs:
                f_path = manifest_dir / r['path']
                if f_path.exists():
                    img = cv2.imread(str(f_path))
                    if img is not None:
                        frames.append(img)
            if len(frames) > num_samples:
                step = max(1, len(frames) // num_samples)
                frames = frames[::step][:num_samples]

        if not frames:
            continue

        true_cls_idx = vinfo['class_idx']
        true_cls_name = vinfo['class_name']

        prompt_info = annotations.get(key, {})
        has_animal = prompt_info.get('has_animal', True)
        point_norm = prompt_info.get('point_norm')

        if use_sam2 and verifier is not None:
            verifier.reset()

        last_box = None
        frame_preds = []
        frame_confs = []

        for frame_bgr in frames:
            h_f, w_f = frame_bgr.shape[:2]

            if use_sam2 and sam_model is not None and has_animal and point_norm:
                pt_px = [int(round(point_norm[0] * w_f)), int(round(point_norm[1] * h_f))]
                raw_proposals = []

                if last_box is not None:
                    res_track = sam_model(frame_bgr, bboxes=[last_box], verbose=False)
                    if len(res_track) > 0 and res_track[0].boxes is not None:
                        for b_i in range(len(res_track[0].boxes)):
                            bx = res_track[0].boxes.xyxy[b_i].cpu().tolist()
                            mk = res_track[0].masks.data[b_i].cpu().numpy().astype(np.uint8) if res_track[0].masks is not None else None
                            raw_proposals.append((bx, mk))

                need_anchor_probe = (last_box is None)
                if not need_anchor_probe and raw_proposals and verifier is not None and verifier.has_exemplar():
                    bx_t, mk_t = raw_proposals[0]
                    cp_t, _ = extract_square_crop(frame_bgr, bx_t, margin=0.15, target_size=224)
                    quick_ver = verifier.verify_candidate(cp_t, bx_t, (h_f, w_f), mk_t, anchor_pt=pt_px)
                    if quick_ver['similarity'] < 0.68:
                        need_anchor_probe = True

                if need_anchor_probe:
                    res_anchor = sam_model(frame_bgr, points=[pt_px], labels=[1], verbose=False)
                    if len(res_anchor) > 0 and res_anchor[0].boxes is not None:
                        for b_i in range(len(res_anchor[0].boxes)):
                            bx = res_anchor[0].boxes.xyxy[b_i].cpu().tolist()
                            mk = res_anchor[0].masks.data[b_i].cpu().numpy().astype(np.uint8) if res_anchor[0].masks is not None else None
                            raw_proposals.append((bx, mk))

                if raw_proposals:
                    if verifier is not None and not verifier.has_exemplar():
                        bx_init, mk_init = raw_proposals[0]
                        cp_init, _ = extract_square_crop(frame_bgr, bx_init, margin=0.15, target_size=224)
                        verifier.set_exemplar(cp_init, mk_init)
                        crop_bgr = cp_init
                        last_box = bx_init
                    else:
                        evaluated_candidates = []
                        for bx_cand, mk_cand in raw_proposals:
                            cp_cand, _ = extract_square_crop(frame_bgr, bx_cand, margin=0.15, target_size=224)
                            ver_res = verifier.verify_candidate(cp_cand, bx_cand, (h_f, w_f), mk_cand, anchor_pt=pt_px) if verifier else {'is_valid': True, 'similarity': 1.0}
                            evaluated_candidates.append({
                                'box': bx_cand,
                                'crop': cp_cand,
                                'ver': ver_res,
                                'score': ver_res['similarity'],
                            })
                        best = max(evaluated_candidates, key=lambda c: c['score'])
                        if best['ver']['is_valid']:
                            crop_bgr = best['crop']
                            last_box = best['box']
                        else:
                            last_box = None
                            continue
                else:
                    last_box = None
                    continue
            else:
                crop_bgr = extract_center_crop(frame_bgr, target_size=224)

            # Inferência do Classificador
            input_tensor = preprocess_tensor(crop_bgr, target_size=224, device=device)
            with torch.no_grad():
                logits = net(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            frame_preds.append(pred_idx)
            frame_confs.append(conf)

            single_frame_preds.append(pred_idx)
            single_frame_targets.append(true_cls_idx)

        # Decisão do Comitê Temporal
        total_votes = len(frame_preds)
        if total_votes == 0:
            num_abstained += 1
            decision_status = "ABSTENÇÃO_TOTAL_OCLUSAO"
            final_pred_name = "ABSTAIN"
            is_correct = False
            committee_results.append({
                'video_key': key,
                'true_class': true_cls_name,
                'predicted_class': final_pred_name,
                'status': decision_status,
                'consensus_ratio': 0.0,
                'total_frames': 0,
                'votes_breakdown': {},
                'mean_frame_confidence': 0.0,
                'correct': False,
            })
            continue

        vote_counts = Counter(frame_preds)
        top_cls, top_count = vote_counts.most_common(1)[0]
        consensus_ratio = top_count / total_votes

        is_consensus = consensus_ratio >= consensus_threshold
        if is_consensus:
            num_decided += 1
            is_correct = (top_cls == true_cls_idx)
            if is_correct:
                num_correct_committee += 1
            decision_status = "CONSENSO_ATINGIDO"
            final_pred_name = classes[top_cls]
        else:
            num_abstained += 1
            decision_status = "ABSTENÇÃO_SEM_CONSENSO"
            final_pred_name = "ABSTAIN"
            is_correct = False

        committee_results.append({
            'video_key': key,
            'true_class': true_cls_name,
            'predicted_class': final_pred_name,
            'status': decision_status,
            'consensus_ratio': round(consensus_ratio, 3),
            'total_frames': total_votes,
            'votes_breakdown': {classes[c]: cnt for c, cnt in vote_counts.items()},
            'mean_frame_confidence': round(float(np.mean(frame_confs)), 3),
            'correct': is_correct,
        })

    # Estatísticas Globais
    single_frame_acc = float(np.mean(np.array(single_frame_preds) == np.array(single_frame_targets)))
    single_frame_f1 = float(f1_score(single_frame_targets, single_frame_preds, average='macro', zero_division=0))

    committee_decided_acc = (num_correct_committee / num_decided) if num_decided > 0 else 0.0
    abstention_rate = (num_abstained / len(unique_test_videos)) if len(unique_test_videos) > 0 else 0.0
    overall_video_acc = (num_correct_committee / len(unique_test_videos)) if len(unique_test_videos) > 0 else 0.0

    summary_report = {
        'total_test_videos': len(unique_test_videos),
        'total_evaluated_frames': len(single_frame_preds),
        'single_frame_metrics': {
            'accuracy': round(single_frame_acc, 4),
            'macro_f1': round(single_frame_f1, 4),
        },
        'committee_metrics': {
            'threshold': consensus_threshold,
            'decided_videos': num_decided,
            'abstained_videos': num_abstained,
            'abstention_rate': round(abstention_rate, 4),
            'accuracy_on_decided': round(committee_decided_acc, 4),
            'overall_effective_accuracy': round(overall_video_acc, 4),
        },
        'details': committee_results,
    }

    print("\n==========================================")
    print("RESULTADOS DO BENCHMARK DE VÍDEO & COMITÊ TEMPORAL")
    print("==========================================")
    print(f"Total de Vídeos de Teste: {len(unique_test_videos)}")
    print(f"Total de Quadros Avaliados: {len(single_frame_preds)}")
    print("--- Inferência Isolada (Single Frame) ---")
    print(f"  Acurácia Single-Frame: {single_frame_acc * 100:.2f}%")
    print(f"  Macro F1 Single-Frame: {single_frame_f1 * 100:.2f}%")
    print("--- Agregação Temporal (Comitê >= 70%) ---")
    print(f"  Taxa de Consenso: {((num_decided / len(unique_test_videos)) * 100):.1f}% ({num_decided}/{len(unique_test_videos)})")
    print(f"  Taxa de Abstenção: {abstention_rate * 100:.1f}% ({num_abstained}/{len(unique_test_videos)})")
    print(f"  Acurácia nos Vídeos Decididos: {committee_decided_acc * 100:.2f}%")
    print(f"  Acurácia Global no Conjunto: {overall_video_acc * 100:.2f}%")
    print("==========================================\n")

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open('w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        print(f"Relatório detalhado gravado com sucesso em '{out_p}'.")

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Benchmark de Avaliação em Vídeo com Comitê Temporal de Votação.")
    parser.add_argument('-opt', '--options', required=True, help="YAML de configuração do classificador.")
    parser.add_argument('-c', '--checkpoint', required=True, help="Checkpoint (.pth) do modelo classificador treinado.")
    parser.add_argument('--manifest', default="datasets/generated/animals_video_sanitized/dataset_manifest.json", help="Manifesto do dataset.")
    parser.add_argument('--annotations', default="assets/annotations/video_point_prompts.json", help="Anotações de âncora.")
    parser.add_argument('--sam-model', default="sam2_t.pt", help="Pesos do SAM 2.")
    parser.add_argument('--use-sam2', action='store_true', help="Ativar segmentação SAM 2 por frame.")
    parser.add_argument('--threshold', type=float, default=0.70, help="Limiar de votos mínimos do comitê (default: 0.70).")
    parser.add_argument('--fps', type=float, default=3.0, help="Taxa de amostragem no vídeo (default: 3.0 FPS).")
    parser.add_argument('--duration', type=float, default=2.0, help="Duração da janela em segundos (default: 2.0s).")
    parser.add_argument('--sim-thresh', type=float, default=0.50, help="Limiar de similaridade para rejeição de oclusão (default: 0.50).")
    parser.add_argument('--output', default="experiments/video_committee_report.json", help="Arquivo JSON de saída.")
    args = parser.parse_args()

    run_video_committee(
        opt_path=args.options,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        sam_model_path=args.sam_model,
        use_sam2=args.use_sam2,
        consensus_threshold=args.threshold,
        target_fps=args.fps,
        window_sec=args.duration,
        sim_threshold=args.sim_thresh,
        output_report_path=args.output,
    )


if __name__ == '__main__':
    main()
