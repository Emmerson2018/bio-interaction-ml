import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ultralytics.data.utils import check_det_dataset

from experiments.benchmark_yolo import (
    aggregate_benchmark,
    compute_metric_stats,
    extract_seg_split_metrics,
    resolve_portable_data_yaml,
    run_benchmark_suite,
)


class TestYoloBenchmarkPipeline(unittest.TestCase):
    def test_v35_yolo_seg_data_yaml_is_portable(self):
        """Verifica que data.yaml real em animals_video_v35_yolo_seg resolve sem path absoluto."""
        real_yaml = Path("datasets/generated/animals_video_v35_yolo_seg/data.yaml")
        if not real_yaml.exists():
            self.skipTest("Dataset animals_video_v35_yolo_seg ainda nao gerado.")
        parsed = check_det_dataset(str(real_yaml))
        self.assertEqual(parsed["nc"], 15)
        self.assertEqual(Path(parsed["path"]).resolve(), real_yaml.parent.resolve())

    def test_metric_stats_split_breakdown_and_resume_aggregation(self):
        """
        Valida:
        1) Media, desvio padrao e variancia amostral (ddof=1);
        2) Separacao de AP entre classes disjuntas (N >= 3) e compartilhadas (N <= 2);
        3) Re-uso via resume (--resume) e geracao de JSON/CSV/MD + 3 graficos PNG.
        """
        stats = compute_metric_stats([0.70, 0.72, 0.74])
        self.assertAlmostEqual(stats["mean"], 0.72, places=6)
        self.assertAlmostEqual(stats["std"], 0.02, places=6)
        self.assertAlmostEqual(stats["variance"], 0.0004, places=6)
        self.assertEqual(stats["n"], 3)

        # Mock de val_res estilo Ultralytics SegmentMetrics
        names = {0: "anta", 1: "arara", 2: "onca"}
        fake_box = SimpleNamespace(
            map50=0.80,
            map=0.60,
            mp=0.78,
            mr=0.75,
            f1=np.array([0.85, 0.45, 0.85]),
            ap_class_index=[0, 1, 2],
            ap50=np.array([0.90, 0.50, 0.90]),
            ap=np.array([0.70, 0.30, 0.70]),
        )
        fake_seg = SimpleNamespace(
            map50=0.76,
            map=0.56,
            mp=0.74,
            mr=0.71,
            f1=np.array([0.82, 0.40, 0.82]),
            ap_class_index=[0, 1, 2],
            ap50=np.array([0.88, 0.46, 0.88]),
            ap=np.array([0.66, 0.26, 0.66]),
        )
        fake_res = SimpleNamespace(
            box=fake_box,
            seg=fake_seg,
            speed={"preprocess": 0.5, "inference": 4.0, "postprocess": 0.5},
        )
        parsed = extract_seg_split_metrics(fake_res, names, shared_classes={"arara", "jaraqui"})
        self.assertAlmostEqual(parsed["disjoint_n3_plus_mask_map50_95"], 0.66, places=5)
        self.assertAlmostEqual(parsed["shared_n_le2_mask_map50_95"], 0.26, places=5)
        self.assertAlmostEqual(parsed["fps"], 200.0, places=1)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # Simula data.yaml + 2 modelos com 3 runs ja salvas para validar --resume sem treinar
            ds_dir = out_dir / "fake_ds"
            for sp in ("train", "val", "test"):
                (ds_dir / "images" / sp).mkdir(parents=True)
                (ds_dir / "labels" / sp).mkdir(parents=True)
            yaml_p = ds_dir / "data.yaml"
            yaml_p.write_text("train: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames:\n  0: anta\n  1: arara\n", encoding="utf-8")

            resolved = resolve_portable_data_yaml(yaml_p, out_dir)
            self.assertTrue(resolved.exists())

            for ckpt, slug, base_val in (("yolov8n-seg.pt", "yolov8n-seg", 0.55), ("yolo11n-seg.pt", "yolo11n-seg", 0.65)):
                for r_i, seed in enumerate((42, 142, 242), start=1):
                    r_dir = out_dir / slug / f"run{r_i}_seed{seed}"
                    r_dir.mkdir(parents=True)
                    payload = {
                        "model": ckpt,
                        "slug": slug,
                        "run": r_i,
                        "seed": seed,
                        "imgsz": 640,
                        "best_pt": str(r_dir / "weights" / "best.pt"),
                        "params_m": 3.1,
                        "gflops": 10.2,
                        "weight_mb": 6.4,
                        "bg_fp_image_rate": 0.05,
                        "box_map50": base_val + 0.12,
                        "box_map50_95": base_val + 0.04,
                        "box_f1": 0.72,
                        "mask_map50": base_val + 0.10,
                        "mask_map50_95": base_val + (r_i - 2) * 0.01,
                        "mask_f1": 0.70,
                        "disjoint_n3_plus_mask_map50_95": base_val + 0.02,
                        "shared_n_le2_mask_map50_95": base_val - 0.05,
                        "disjoint_n3_plus_box_map50_95": base_val + 0.04,
                        "shared_n_le2_box_map50_95": base_val - 0.03,
                        "latency_total_ms": 5.0,
                        "fps": 200.0,
                    }
                    (r_dir / "test_metrics.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = run_benchmark_suite(
                data_yaml=yaml_p,
                models=["yolov8n-seg.pt", "yolo11n-seg.pt"],
                out_dir=out_dir,
                runs=3,
                resume=True,
                export_champion=False,
            )
            self.assertEqual(summary["champion_model"], "yolo11n-seg.pt")
            self.assertTrue((out_dir / "yolo_benchmark_summary.json").exists())
            self.assertTrue((out_dir / "yolo_benchmark_summary.csv").exists())
            self.assertTrue((out_dir / "yolo_benchmark_report.md").exists())
            self.assertTrue((out_dir / "yolo_seg_benchmark_bars.png").exists())
            self.assertTrue((out_dir / "yolo_seg_pareto_fps.png").exists())
            self.assertTrue((out_dir / "yolo_seg_disjoint_vs_shared.png").exists())

    def test_colab_yolo_notebook_structure_and_dvc_secret_fallback(self):
        """Verifica validade nbformat 4, secret DVC_DEFAULT_JSON + fallback upload e sintaxe."""
        nb_path = Path("notebooks/Treinamento_YOLO_Colab_BioEcoInt.ipynb")
        self.assertTrue(nb_path.exists())
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)

        full_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertIn("DVC_DEFAULT_JSON", full_src)
        self.assertIn("files.upload()", full_src)
        self.assertIn("yolo26n-seg.pt", full_src)
        self.assertIn("yolo11n-seg.pt", full_src)
        self.assertIn("yolov8n-seg.pt", full_src)

        for idx, c in enumerate(nb["cells"]):
            if c.get("cell_type") == "code":
                cleaned = []
                skip_cont = False
                for line in c.get("source", []):
                    stripped = line.lstrip()
                    if skip_cont:
                        skip_cont = stripped.endswith("\\\n") or stripped.endswith("\\")
                        continue
                    if stripped.startswith(("!", "%")):
                        indent = line[: len(line) - len(stripped)]
                        cleaned.append(f"{indent}pass\n")
                        skip_cont = stripped.endswith("\\\n") or stripped.endswith("\\")
                    else:
                        cleaned.append(line)
                ast.parse("".join(cleaned), filename=f"nb_cell_{idx}")

    def test_e2e_relative_out_dir_1epoch_no_runs_segment_bug(self):
        """
        Regressao E2E real do bug do Colab:
        Passar --output-dir relativo NAO deve fazer o Ultralytics salvar em runs/segment/...
        e quebrar ao procurar weights/best.pt apos o fim da Run 1.
        """
        import os
        import cv2
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            ds_dir = root / "mini_ds"
            for sp in ("train", "val", "test"):
                (ds_dir / "images" / sp).mkdir(parents=True)
                (ds_dir / "labels" / sp).mkdir(parents=True)
                img = np.full((64, 64, 3), 60, dtype=np.uint8)
                cv2.rectangle(img, (16, 16), (48, 48), (220, 140, 40), -1)
                cv2.imwrite(str(ds_dir / "images" / sp / "a_00.jpg"), img)
                (ds_dir / "labels" / sp / "a_00.txt").write_text(
                    "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                    encoding="utf-8",
                )
            # Hard-negative vazio no test
            bg = np.full((64, 64, 3), 30, dtype=np.uint8)
            cv2.imwrite(str(ds_dir / "images" / "test" / "bg_00.jpg"), bg)
            (ds_dir / "labels" / "test" / "bg_00.txt").write_text("", encoding="utf-8")

            (ds_dir / "data.yaml").write_text(
                "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: anta\n",
                encoding="utf-8",
            )

            prev_cwd = os.getcwd()
            os.chdir(root)
            try:
                rel_out = Path("experiments/YOLO_Seg_Benchmark")
                summary = run_benchmark_suite(
                    data_yaml=Path("mini_ds/data.yaml"),
                    models=["yolov8n-seg.yaml"],
                    out_dir=rel_out,
                    runs=1,
                    base_seed=42,
                    epochs=1,
                    imgsz=64,
                    batch=2,
                    workers=0,
                    device="0" if torch.cuda.is_available() else "cpu",
                    resume=False,
                    export_champion=False,
                )
            finally:
                os.chdir(prev_cwd)

            run_metrics = root / "experiments/YOLO_Seg_Benchmark/yolov8n-seg/run1_seed42/test_metrics.json"
            self.assertTrue(run_metrics.exists(), f"test_metrics.json ausente em {run_metrics}")
            self.assertFalse((root / "runs/segment").exists(), "Ultralytics redirecionou para runs/segment!")
            self.assertEqual(summary["champion_model"], "yolov8n-seg.yaml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
