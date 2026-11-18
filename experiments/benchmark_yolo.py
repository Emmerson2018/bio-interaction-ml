import argparse
import csv
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

DEFAULT_SEG_MODELS: List[str] = [
    "yolov8n-seg.pt",
    "yolo11n-seg.pt",
    "yolo26n-seg.pt",
]
DEFAULT_SHARED_CLASSES: Set[str] = {"arara", "jaraqui"}


def resolve_portable_data_yaml(data_yaml_path: Path, runtime_dir: Path) -> Path:
    """Garante `path` valido no data.yaml independente de maquina local ou Colab."""
    src_yaml = Path(data_yaml_path).resolve()
    if not src_yaml.exists():
        raise FileNotFoundError(f"Arquivo data.yaml nao encontrado: {src_yaml}")
    payload = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
    payload["path"] = str(src_yaml.parent)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = runtime_dir / "data_resolved.yaml"
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_yaml


def compute_metric_stats(values: Sequence[float]) -> Dict[str, float]:
    """Calcula media, desvio padrao e variancia amostral (ddof=1 para n >= 2)."""
    arr = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    var_v = float(np.var(arr, ddof=1)) if n > 1 else 0.0
    return {
        "mean": round(mean_v, 6),
        "std": round(std_v, 6),
        "variance": round(var_v, 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "n": n,
    }


def _safe_mean(vals: Sequence[float]) -> float:
    valid = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.mean(valid)) if valid else 0.0


def extract_seg_split_metrics(
    val_res: Any,
    names: Dict[int, str],
    shared_classes: Set[str] = DEFAULT_SHARED_CLASSES,
) -> Dict[str, Any]:
    """Extrai metricas globais (Box + Mask), por classe e separadas (N >= 3 vs N <= 2)."""
    box = val_res.box
    seg = getattr(val_res, "seg", None)

    def _f1_mean(metric_obj) -> float:
        if metric_obj is None or not hasattr(metric_obj, "f1"):
            return 0.0
        f1_arr = np.asarray(metric_obj.f1, dtype=np.float64)
        return float(np.mean(f1_arr)) if f1_arr.size > 0 else 0.0

    def _per_class_map(metric_obj) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if metric_obj is None or not hasattr(metric_obj, "ap_class_index"):
            return out
        idxs = list(metric_obj.ap_class_index)
        ap50_arr = np.asarray(metric_obj.ap50, dtype=np.float64)
        ap_arr = np.asarray(metric_obj.ap, dtype=np.float64)
        for pos, cls_idx in enumerate(idxs):
            cname = str(names.get(int(cls_idx), f"class_{int(cls_idx)}"))
            out[cname] = {
                "ap50": round(float(ap50_arr[pos]), 6) if pos < len(ap50_arr) else 0.0,
                "ap50_95": round(float(ap_arr[pos]), 6) if pos < len(ap_arr) else 0.0,
            }
        return out

    box_pc = _per_class_map(box)
    seg_pc = _per_class_map(seg)

    disjoint_box_ap = [v["ap50_95"] for k, v in box_pc.items() if k not in shared_classes]
    shared_box_ap = [v["ap50_95"] for k, v in box_pc.items() if k in shared_classes]
    disjoint_mask_ap = [v["ap50_95"] for k, v in seg_pc.items() if k not in shared_classes]
    shared_mask_ap = [v["ap50_95"] for k, v in seg_pc.items() if k in shared_classes]
    disjoint_mask_ap50 = [v["ap50"] for k, v in seg_pc.items() if k not in shared_classes]
    shared_mask_ap50 = [v["ap50"] for k, v in seg_pc.items() if k in shared_classes]

    spd = getattr(val_res, "speed", {}) or {}
    latency_ms = float(spd.get("preprocess", 0.0) + spd.get("inference", 0.0) + spd.get("postprocess", 0.0))
    infer_ms = float(spd.get("inference", 0.0))
    fps = (1000.0 / latency_ms) if latency_ms > 0.0 else 0.0

    return {
        "box_map50": round(float(getattr(box, "map50", 0.0)), 6),
        "box_map50_95": round(float(getattr(box, "map", 0.0)), 6),
        "box_precision": round(float(getattr(box, "mp", 0.0)), 6),
        "box_recall": round(float(getattr(box, "mr", 0.0)), 6),
        "box_f1": round(_f1_mean(box), 6),
        "mask_map50": round(float(getattr(seg, "map50", 0.0)) if seg is not None else 0.0, 6),
        "mask_map50_95": round(float(getattr(seg, "map", 0.0)) if seg is not None else 0.0, 6),
        "mask_precision": round(float(getattr(seg, "mp", 0.0)) if seg is not None else 0.0, 6),
        "mask_recall": round(float(getattr(seg, "mr", 0.0)) if seg is not None else 0.0, 6),
        "mask_f1": round(_f1_mean(seg), 6),
        "disjoint_n3_plus_box_map50_95": round(_safe_mean(disjoint_box_ap), 6),
        "shared_n_le2_box_map50_95": round(_safe_mean(shared_box_ap), 6),
        "disjoint_n3_plus_mask_map50_95": round(_safe_mean(disjoint_mask_ap), 6),
        "disjoint_n3_plus_mask_map50": round(_safe_mean(disjoint_mask_ap50), 6),
        "shared_n_le2_mask_map50_95": round(_safe_mean(shared_mask_ap), 6),
        "shared_n_le2_mask_map50": round(_safe_mean(shared_mask_ap50), 6),
        "latency_total_ms": round(latency_ms, 4),
        "latency_infer_ms": round(infer_ms, 4),
        "fps": round(fps, 2),
        "per_class_box": box_pc,
        "per_class_mask": seg_pc,
    }


def evaluate_hard_negative_fps(
    model: YOLO,
    dataset_root: Path,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.25,
) -> Dict[str, float]:
    """Mede taxa de falsos positivos nas imagens puras de fundo (`labels/<split>/*.txt` vazios)."""
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        return {"bg_images": 0, "bg_fp_images": 0, "bg_fp_image_rate": 0.0, "bg_fp_per_image": 0.0}

    bg_imgs: List[str] = []
    for lbl_p in sorted(lbl_dir.glob("*.txt")):
        if lbl_p.read_text(encoding="utf-8").strip() == "":
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_dir / f"{lbl_p.stem}{ext}"
                if cand.exists():
                    bg_imgs.append(str(cand))
                    break

    if not bg_imgs:
        return {"bg_images": 0, "bg_fp_images": 0, "bg_fp_image_rate": 0.0, "bg_fp_per_image": 0.0}

    preds = model.predict(source=bg_imgs, imgsz=imgsz, conf=conf, verbose=False)
    fp_imgs = 0
    total_fp = 0
    for r in preds:
        n_det = int(len(r.boxes)) if r.boxes is not None else 0
        if n_det > 0:
            fp_imgs += 1
            total_fp += n_det
    n_bg = len(bg_imgs)
    return {
        "bg_images": n_bg,
        "bg_fp_images": fp_imgs,
        "bg_fp_image_rate": round(fp_imgs / float(n_bg), 6),
        "bg_fp_per_image": round(total_fp / float(n_bg), 6),
    }


def extract_model_complexity(model: YOLO, weight_path: Path, imgsz: int = 640) -> Dict[str, float]:
    """Coleta parametros (M), GFLOPs (para imgsz) e tamanho em MB."""
    params = sum(p.numel() for p in model.model.parameters())
    try:
        gflops = float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        gflops = 0.0
    w_mb = float(weight_path.stat().st_size) / (1024.0 * 1024.0) if weight_path.exists() else 0.0
    return {
        "params_m": round(params / 1e6, 3),
        "gflops": round(gflops, 3),
        "weight_mb": round(w_mb, 3),
    }


def _model_slug(ckpt_name: str) -> str:
    return Path(ckpt_name).stem.replace(".", "_")


def run_benchmark_suite(
    data_yaml: Path,
    models: Sequence[str],
    out_dir: Path,
    runs: int = 3,
    base_seed: int = 42,
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = -1,
    patience: int = 15,
    device: Optional[str] = None,
    resume: bool = True,
    workers: int = 4,
    export_champion: bool = True,
    champion_sizes: Sequence[int] = (640, 320),
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = resolve_portable_data_yaml(data_yaml, out_dir)
    dataset_root = Path(yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))["path"])

    shared_classes = set(DEFAULT_SHARED_CLASSES)
    manifest_cand = dataset_root.parent / "animals_video_v35_crops" / "dataset_manifest.json"
    if manifest_cand.exists():
        try:
            mdata = json.loads(manifest_cand.read_text(encoding="utf-8"))
            if isinstance(mdata.get("shared_video_classes"), list):
                shared_classes = set(mdata["shared_video_classes"])
        except Exception:
            pass

    all_runs_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for ckpt in models:
        slug = _model_slug(ckpt)
        model_dir = out_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        model_runs: List[Dict[str, Any]] = []

        for r_idx in range(runs):
            run_seed = int(base_seed) + r_idx * 100
            run_name = f"run{r_idx + 1}_seed{run_seed}"
            run_dir = model_dir / run_name
            metrics_file = run_dir / "test_metrics.json"

            if resume and metrics_file.exists():
                saved = json.loads(metrics_file.read_text(encoding="utf-8"))
                model_runs.append(saved)
                print(f"⏭️ [Resume] Pulando {slug}/{run_name} (Mask mAP50-95={saved['mask_map50_95']:.4f})")
                continue

            print(f"\n🚀 Treinando {ckpt} | Run {r_idx + 1}/{runs} (seed={run_seed}, imgsz={imgsz}, epochs={epochs})")
            yolo_model = YOLO(ckpt)
            train_kwargs: Dict[str, Any] = {
                "data": str(resolved_yaml),
                "epochs": int(epochs),
                "imgsz": int(imgsz),
                "batch": int(batch),
                "seed": run_seed,
                "deterministic": True,
                "patience": int(patience),
                "cos_lr": True,
                "amp": True,
                "workers": int(workers),
                "project": str(model_dir),
                "name": run_name,
                "exist_ok": True,
                "plots": True,
            }
            if device is not None:
                train_kwargs["device"] = device
            yolo_model.train(**train_kwargs)

            best_pt = run_dir / "weights" / "best.pt"
            if not best_pt.exists():
                best_pt = run_dir / "weights" / "last.pt"

            best_model = YOLO(str(best_pt))
            val_kwargs: Dict[str, Any] = {
                "data": str(resolved_yaml),
                "split": "test",
                "imgsz": int(imgsz),
                "plots": True,
                "project": str(model_dir),
                "name": f"{run_name}_test",
                "exist_ok": True,
            }
            if device is not None:
                val_kwargs["device"] = device
            test_res = best_model.val(**val_kwargs)

            run_metrics = extract_seg_split_metrics(test_res, best_model.names, shared_classes=shared_classes)
            bg_stats = evaluate_hard_negative_fps(best_model, dataset_root, split="test", imgsz=imgsz)
            comp_stats = extract_model_complexity(best_model, best_pt, imgsz=imgsz)
            run_payload = {
                "model": ckpt,
                "slug": slug,
                "run": r_idx + 1,
                "seed": run_seed,
                "imgsz": int(imgsz),
                "best_pt": str(best_pt),
                **comp_stats,
                **bg_stats,
                **run_metrics,
            }
            metrics_file.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            model_runs.append(run_payload)
            print(
                f"✓ {slug}/{run_name}: Mask mAP50-95={run_payload['mask_map50_95']:.4f} | "
                f"Box mAP50-95={run_payload['box_map50_95']:.4f} | FPS={run_payload['fps']:.1f}"
            )

            del yolo_model, best_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_runs_by_model[ckpt] = model_runs

    summary = aggregate_benchmark(all_runs_by_model, out_dir)
    if export_champion and summary.get("models"):
        champ_info = export_and_benchmark_champion(
            summary=summary,
            resolved_yaml=resolved_yaml,
            dataset_root=dataset_root,
            out_dir=out_dir,
            export_sizes=champion_sizes,
            shared_classes=shared_classes,
            device=device,
        )
        summary["champion_export"] = champ_info
        (out_dir / "yolo_benchmark_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return summary


AGG_KEYS: List[str] = [
    "mask_map50_95",
    "mask_map50",
    "mask_f1",
    "box_map50_95",
    "box_map50",
    "box_f1",
    "disjoint_n3_plus_mask_map50_95",
    "shared_n_le2_mask_map50_95",
    "disjoint_n3_plus_box_map50_95",
    "shared_n_le2_box_map50_95",
    "bg_fp_image_rate",
    "latency_total_ms",
    "fps",
    "params_m",
    "gflops",
    "weight_mb",
]


def aggregate_benchmark(
    all_runs_by_model: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
) -> Dict[str, Any]:
    """Agrega media, desvio padrao e variancia entre runs, gera CSV, Markdown e graficos."""
    models_summary: Dict[str, Any] = {}
    ranking: List[Dict[str, Any]] = []

    for ckpt, r_list in all_runs_by_model.items():
        if not r_list:
            continue
        stats_map = {k: compute_metric_stats([r.get(k) for r in r_list]) for k in AGG_KEYS}
        best_run = max(r_list, key=lambda r: float(r.get("mask_map50_95", 0.0)))
        models_summary[ckpt] = {
            "slug": _model_slug(ckpt),
            "num_runs": len(r_list),
            "runs": r_list,
            "best_run_pt": best_run.get("best_pt"),
            "metrics": stats_map,
        }
        ranking.append(
            {
                "model": ckpt,
                "mask_map50_95_mean": stats_map["mask_map50_95"]["mean"],
                "mask_map50_95_std": stats_map["mask_map50_95"]["std"],
                "mask_map50_95_var": stats_map["mask_map50_95"]["variance"],
                "box_map50_95_mean": stats_map["box_map50_95"]["mean"],
                "fps_mean": stats_map["fps"]["mean"],
            }
        )

    ranking.sort(key=lambda item: item["mask_map50_95_mean"], reverse=True)
    summary = {
        "champion_model": ranking[0]["model"] if ranking else None,
        "ranking": ranking,
        "models": models_summary,
    }
    (out_dir / "yolo_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv_and_markdown_report(summary, out_dir)
    _plot_benchmark_charts(summary, out_dir)
    return summary


def _write_csv_and_markdown_report(summary: Dict[str, Any], out_dir: Path) -> None:
    csv_path = out_dir / "yolo_benchmark_summary.csv"
    md_path = out_dir / "yolo_benchmark_report.md"

    headers = [
        "model",
        "runs",
        "mask_map50_95_mean",
        "mask_map50_95_std",
        "mask_map50_95_var",
        "mask_map50_mean",
        "box_map50_95_mean",
        "box_map50_95_std",
        "disjoint_n3_mask_map_mean",
        "shared_n2_mask_map_mean",
        "bg_fp_rate_mean",
        "fps_mean",
        "params_m",
        "gflops",
    ]
    rows: List[List[Any]] = []
    md_lines = [
        "# Benchmark YOLO-Seg (BioEcoInt V3.5 — 3 Runs)",
        "",
        "| Modelo | Mask mAP@50-95 (Média ± DP) | Var ($\sigma^2$) | Mask mAP@50 | Box mAP@50-95 | `N>=3` Disjunto | `N<=2` Shared | FP Fundo | FPS | Params (M) | GFLOPs |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in summary.get("ranking", []):
        ckpt = item["model"]
        ms = summary["models"][ckpt]["metrics"]
        n_r = summary["models"][ckpt]["num_runs"]
        rows.append(
            [
                ckpt,
                n_r,
                ms["mask_map50_95"]["mean"],
                ms["mask_map50_95"]["std"],
                ms["mask_map50_95"]["variance"],
                ms["mask_map50"]["mean"],
                ms["box_map50_95"]["mean"],
                ms["box_map50_95"]["std"],
                ms["disjoint_n3_plus_mask_map50_95"]["mean"],
                ms["shared_n_le2_mask_map50_95"]["mean"],
                ms["bg_fp_image_rate"]["mean"],
                ms["fps"]["mean"],
                ms["params_m"]["mean"],
                ms["gflops"]["mean"],
            ]
        )
        md_lines.append(
            f"| `{ckpt}` | {ms['mask_map50_95']['mean']*100:.2f}% ± {ms['mask_map50_95']['std']*100:.2f}% | "
            f"{ms['mask_map50_95']['variance']:.6f} | {ms['mask_map50']['mean']*100:.2f}% | "
            f"{ms['box_map50_95']['mean']*100:.2f}% | {ms['disjoint_n3_plus_mask_map50_95']['mean']*100:.2f}% | "
            f"{ms['shared_n_le2_mask_map50_95']['mean']*100:.2f}% | {ms['bg_fp_image_rate']['mean']*100:.1f}% | "
            f"{ms['fps']['mean']:.1f} | {ms['params_m']['mean']:.2f} | {ms['gflops']['mean']:.1f} |"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _plot_benchmark_charts(summary: Dict[str, Any], out_dir: Path) -> None:
    ranking = summary.get("ranking", [])
    if not ranking:
        return
    names = [r["model"].replace(".pt", "") for r in ranking]
    m_models = summary["models"]

    mask_means = [m_models[r["model"]]["metrics"]["mask_map50_95"]["mean"] * 100.0 for r in ranking]
    mask_stds = [m_models[r["model"]]["metrics"]["mask_map50_95"]["std"] * 100.0 for r in ranking]
    box_means = [m_models[r["model"]]["metrics"]["box_map50_95"]["mean"] * 100.0 for r in ranking]
    box_stds = [m_models[r["model"]]["metrics"]["box_map50_95"]["std"] * 100.0 for r in ranking]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, mask_means, w, yerr=mask_stds, capsize=4, label="Mask mAP@50-95 (%)", color="#1f77b4")
    ax.bar(x + w / 2, box_means, w, yerr=box_stds, capsize=4, label="Box mAP@50-95 (%)", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("mAP@50-95 (%)")
    ax.set_title("Benchmark YOLO-Seg Multi-Run (Média ± DP, 3 Runs)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "yolo_seg_benchmark_bars.png", dpi=200)
    plt.close(fig)

    # Pareto FPS vs Mask mAP@50-95
    fps_vals = [m_models[r["model"]]["metrics"]["fps"]["mean"] for r in ranking]
    params_vals = [max(1.0, m_models[r["model"]]["metrics"]["params_m"]["mean"]) * 45.0 for r in ranking]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fps_vals, mask_means, s=params_vals, alpha=0.8, color="#ff7f0e", edgecolors="black")
    for idx_n, lbl in enumerate(names):
        ax.annotate(lbl, (fps_vals[idx_n], mask_means[idx_n]), xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Throughput no Teste (FPS)")
    ax.set_ylabel("Mask mAP@50-95 (%)")
    ax.set_title("Fronteira de Pareto: Precisão vs Velocidade")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_dir / "yolo_seg_pareto_fps.png", dpi=200)
    plt.close(fig)

    # N >= 3 vs N <= 2
    dj_vals = [m_models[r["model"]]["metrics"]["disjoint_n3_plus_mask_map50_95"]["mean"] * 100.0 for r in ranking]
    sh_vals = [m_models[r["model"]]["metrics"]["shared_n_le2_mask_map50_95"]["mean"] * 100.0 for r in ranking]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, dj_vals, w, label="Classes Disjuntas (N >= 3)", color="#0d9488")
    ax.bar(x + w / 2, sh_vals, w, label="Classes Compartilhadas (N <= 2)", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mask mAP@50-95 (%)")
    ax.set_title("Avaliação Honesta: Vídeos Inéditos (N >= 3) vs Compartilhados (N <= 2)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "yolo_seg_disjoint_vs_shared.png", dpi=200)
    plt.close(fig)


def export_and_benchmark_champion(
    summary: Dict[str, Any],
    resolved_yaml: Path,
    dataset_root: Path,
    out_dir: Path,
    export_sizes: Sequence[int] = (640, 320),
    shared_classes: Set[str] = DEFAULT_SHARED_CLASSES,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Avalia e exporta o melhor checkpoint campeao em 640px e 320px (ONNX e TFLite)."""
    champ_ckpt = summary.get("champion_model")
    if not champ_ckpt:
        return {}
    best_pt_str = summary["models"][champ_ckpt].get("best_run_pt")
    if not best_pt_str or not Path(best_pt_str).exists():
        return {}

    champ_dir = out_dir / "champion"
    champ_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(champ_ckpt)
    local_best = champ_dir / f"{slug}_best.pt"
    shutil.copy2(best_pt_str, local_best)

    results_by_size: Dict[str, Any] = {}
    for sz in export_sizes:
        sz_int = int(sz)
        sz_dir = champ_dir / f"imgsz_{sz_int}"
        sz_dir.mkdir(parents=True, exist_ok=True)
        sz_pt = sz_dir / f"{slug}_{sz_int}.pt"
        shutil.copy2(local_best, sz_pt)

        m_eval = YOLO(str(sz_pt))
        val_kw: Dict[str, Any] = {
            "data": str(resolved_yaml),
            "split": "test",
            "imgsz": sz_int,
            "project": str(sz_dir),
            "name": f"val_test_{sz_int}",
            "exist_ok": True,
        }
        if device is not None:
            val_kw["device"] = device
        t_res = m_eval.val(**val_kw)
        metrics = extract_seg_split_metrics(t_res, m_eval.names, shared_classes=shared_classes)
        bg = evaluate_hard_negative_fps(m_eval, dataset_root, split="test", imgsz=sz_int)
        comp = extract_model_complexity(m_eval, sz_pt, imgsz=sz_int)

        exports: Dict[str, Any] = {}
        for fmt, extra in (("onnx", {"simplify": True}), ("tflite", {"half": True})):
            try:
                exp_path = m_eval.export(format=fmt, imgsz=sz_int, **extra)
                exp_p = Path(str(exp_path))
                exports[fmt] = {
                    "ok": exp_p.exists(),
                    "path": str(exp_p),
                    "size_mb": round(exp_p.stat().st_size / (1024.0 * 1024.0), 3) if exp_p.exists() else 0.0,
                }
            except Exception as exc:
                exports[fmt] = {"ok": False, "error": str(exc)}

        results_by_size[str(sz_int)] = {
            "imgsz": sz_int,
            **comp,
            **bg,
            **metrics,
            "exports": exports,
        }
        del m_eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {"champion_model": champ_ckpt, "source_pt": str(best_pt_str), "by_imgsz": results_by_size}
    (champ_dir / "champion_640_vs_320.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== CAMPEÃO YOLO-SEG (640 vs 320) ===")
    for k_sz, info in results_by_size.items():
        print(
            f"  imgsz={k_sz}: Mask mAP50-95={info['mask_map50_95']:.4f} | "
            f"Box mAP50-95={info['box_map50_95']:.4f} | FPS={info['fps']:.1f} | GFLOPs={info['gflops']}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Multi-Run YOLO-Seg (3 runs, media/variancia) — BioEcoInt.")
    parser.add_argument("--data", type=str, default="datasets/generated/animals_video_v35_yolo_seg/data.yaml")
    parser.add_argument("--models", nargs="+", default=DEFAULT_SEG_MODELS)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/YOLO_Seg_Benchmark")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-export-champion", action="store_true")
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.consolidate_only:
        loaded: Dict[str, List[Dict[str, Any]]] = {}
        for ckpt in args.models:
            slug = _model_slug(ckpt)
            runs_found = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted((out_dir / slug).glob("run*_seed*/test_metrics.json"))
            ]
            if runs_found:
                loaded[ckpt] = runs_found
        summary = aggregate_benchmark(loaded, out_dir)
        if not args.no_export_champion and summary.get("models"):
            resolved_yaml = resolve_portable_data_yaml(Path(args.data), out_dir)
            ds_root = Path(yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))["path"])
            summary["champion_export"] = export_and_benchmark_champion(
                summary=summary,
                resolved_yaml=resolved_yaml,
                dataset_root=ds_root,
                out_dir=out_dir,
                device=args.device,
            )
            (out_dir / "yolo_benchmark_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return

    run_benchmark_suite(
        data_yaml=Path(args.data),
        models=args.models,
        out_dir=out_dir,
        runs=args.runs,
        base_seed=args.base_seed,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        resume=not args.no_resume,
        workers=args.workers,
        export_champion=not args.no_export_champion,
    )


if __name__ == "__main__":
    main()
