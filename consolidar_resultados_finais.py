"""
Script de Consolidação e Comparação Científica de Resultados — BioEcoInt
Compila métricas de validação, teste e comitê temporal de todos os experimentos.
Gera tabela Markdown, CSV consolidado e gráfico de barras comparativo.
"""

import os
import glob
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIRS = [
    ("Exp 1: ResNet18 Baseline", "experiments/Exp1_ResNet18_Baseline_Sanitized"),
    ("Exp 2: ResNet18 YOLO Crop", "experiments/Exp2_ResNet18_YOLO_Crop"),
    ("Exp 3: ResNet18 Coord. Attention", "experiments/Exp3_ResNet18_CoordinateAttention"),
    ("Exp 4A: MobileNetV4 CE Loss", "experiments/Exp_Loss_CE_MobileNetV4"),
    ("Exp 4B: MobileNetV4 Focal Loss", "experiments/Exp_Loss_Focal_MobileNetV4"),
    ("Exp 4C: MobileNetV4 BalancedSoftmax", "experiments/Exp_MobileNetV4_Small_BalancedSoftmax"),
    ("Exp 5: RepViT-M0.9 BalancedSoftmax", "experiments/Exp_RepViT_M0_9_BalancedSoftmax"),
    ("Exp 6: RepViT-M0.9 SAM2 Crops", "experiments/Exp_RepViT_M0_9_SAM2_Crops"),
    ("Exp 7: MobileNetV4 SAM2 Crops", "experiments/Exp_MobileNetV4_SAM2_Crops"),
]


def extrair_metricas_log(log_path):
    if not os.path.exists(log_path):
        return {}
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    
    # Buscar recordes de Macro-F1
    matches = re.findall(r"Macro-F1\s*=\s*([0-9\.]+)", content)
    best_macro_f1 = float(matches[-1]) if matches else None
    
    # Buscar última acurácia de validação
    acc_matches = re.findall(r"'acc':\s*([0-9\.]+)", content)
    last_acc = float(acc_matches[-1]) if acc_matches else None
    
    return {
        "best_macro_f1": best_macro_f1,
        "last_acc": last_acc,
    }


def main():
    print("=" * 70)
    print("🌿 CONSOLIDAÇÃO DE RESULTADOS EXPERIMENTAIS — BIOECOINT")
    print("=" * 70)
    
    dados = []
    for label, d in EXP_DIRS:
        if not os.path.exists(d):
            continue
            
        test_summary = os.path.join(d, "test_results", "test_summary.json")
        val_metrics = os.path.join(d, "val_metrics.json")
        logs = glob.glob(os.path.join(d, "*.log"))
        
        info = {
            "Experimento": label,
            "Diretório": os.path.basename(d),
            "Acurácia Teste (%)": None,
            "Macro F1 Teste (%)": None,
            "Val Macro F1 (%)": None,
            "Fonte": "Pendente",
        }
        
        # 1. Tentar ler test_summary.json
        if os.path.exists(test_summary):
            try:
                with open(test_summary) as f:
                    tdata = json.load(f)
                info["Acurácia Teste (%)"] = round(tdata.get("test_accuracy", 0) * 100, 2)
                macro_f1 = tdata.get("metrics_per_class", {}).get("macro avg", {}).get("f1-score")
                if macro_f1 is not None:
                    info["Macro F1 Teste (%)"] = round(macro_f1 * 100, 2)
                info["Fonte"] = "test_summary.json"
            except Exception as e:
                pass
                
        # 2. Tentar ler val_metrics.json
        if os.path.exists(val_metrics):
            try:
                with open(val_metrics) as f:
                    vdata = json.load(f)
                val_f1 = vdata.get("macro_f1") or vdata.get("best_macro_f1")
                if val_f1 is not None:
                    info["Val Macro F1 (%)"] = round(val_f1 * 100, 2)
                if info["Fonte"] == "Pendente":
                    info["Fonte"] = "val_metrics.json"
            except Exception:
                pass
                
        # 3. Fallback nos logs
        if logs and (info["Val Macro F1 (%)"] is None):
            log_metrics = extrair_metricas_log(logs[0])
            if log_metrics.get("best_macro_f1"):
                info["Val Macro F1 (%)"] = round(log_metrics["best_macro_f1"] * 100, 2)
                if info["Fonte"] == "Pendente":
                    info["Fonte"] = "train.log"
                    
        dados.append(info)
        
    df = pd.DataFrame(dados)
    print("\n📊 TABELA DE DESEMPENHO DOS MODELOS:")
    print(df.to_markdown(index=False))
    
    # Salvar em CSV
    csv_path = "experiments/tabela_consolidada_experimentos.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Tabela salva em: {csv_path}")
    
    # Relatórios de Comitê Temporal
    print("\n" + "=" * 70)
    print("🏛️ RELATÓRIOS DO COMITÊ TEMPORAL (VOTAÇÃO >= 70%)")
    print("=" * 70)
    
    committee_reports = glob.glob("experiments/**/committee_*.json", recursive=True)
    if committee_reports:
        c_dados = []
        for rep in sorted(committee_reports):
            try:
                with open(rep) as f:
                    cdata = json.load(f)
                c_dados.append({
                    "Relatório": os.path.relpath(rep, "experiments"),
                    "Total Vídeos": cdata.get("total_test_videos"),
                    "Single-Frame Acc (%)": round(cdata.get("single_frame_metrics", {}).get("accuracy", 0) * 100, 2),
                    "Abstention Rate (%)": round(cdata.get("committee_metrics", {}).get("abstention_rate", 0) * 100, 2),
                    "Acc on Decided (%)": round(cdata.get("committee_metrics", {}).get("accuracy_on_decided", 0) * 100, 2),
                    "Overall Acc (%)": round(cdata.get("committee_metrics", {}).get("overall_effective_accuracy", 0) * 100, 2),
                })
            except Exception as e:
                pass
        if c_dados:
            c_df = pd.DataFrame(c_dados)
            print(c_df.to_markdown(index=False))
    else:
        print("Nenhum relatório de comitê temporal encontrado localmente ainda.")

    # Gerar Gráfico de Comparação
    plot_df = df.dropna(subset=["Val Macro F1 (%)"]).copy()
    if not plot_df.empty:
        plt.figure(figsize=(10, 5))
        plt.barh(plot_df["Experimento"], plot_df["Val Macro F1 (%)"], color="#2b5c8f")
        plt.xlabel("Macro F1 (%)")
        plt.title("Comparativo de Desempenho Científico — BioEcoInt")
        plt.xlim(0, 100)
        for index, value in enumerate(plot_df["Val Macro F1 (%)"]):
            plt.text(value + 1, index, f"{value}%", va="center", fontweight="bold")
        plt.tight_layout()
        plot_path = "experiments/comparativo_geral_macro_f1.png"
        plt.savefig(plot_path, dpi=200)
        print(f"✅ Gráfico comparativo gerado em: {plot_path}")
    print("\nConsolidação concluída com sucesso!")


if __name__ == "__main__":
    main()
