import json
from pathlib import Path

BASE_DIR = Path("/home/xsacbj/www/IFMaker/BioEcoInt/bio-interaction-ml")
COLAB_NB_PATH = BASE_DIR / "notebooks" / "Treinamento_Colab_BioEcoInt.ipynb"
LOCAL_NB_PATH = BASE_DIR / "notebooks" / "Treinamento_Local_BioEcoInt.ipynb"

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def build_colab_notebook():
    cells = []
    
    # 0. Cabeçalho
    cells.append(md_cell("""
# 🌿 BioEcoInt — Pipeline Completo de Treinamento, SOTA Benchmark & Comitê Temporal
> **Laboratório IFMaker / BioEcoInt**  
> **Arquiteturas Avaliadas:** ResNet-18, MobileNetV4-Conv-Small e RepViT-M0.9  
> **Estratégias Avaliadas:** Baseline Sanitizado, YOLO Crops, SAM 2 Crops e Coordinate Attention (CA)  
> **Estudo de Funções de Perda:** Cross-Entropy (4A), Focal Loss (4B) e Balanced Softmax (4C) no Campeão RepViT-M0.9
"""))

    # 1. GPU
    cells.append(md_cell("### 1. Verificação de Aceleração por Hardware (GPU)"))
    cells.append(code_cell("""
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
"""))

    # 2. uv install
    cells.append(md_cell("### 2. Instalação do Gerenciador de Pacotes Ultrarrápido (`uv`)"))
    cells.append(code_cell("""
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
!uv --version
"""))

    # 3. Git Clone
    cells.append(md_cell("### 3. Clonagem do Repositório (`bio-interaction-ml`)"))
    cells.append(code_cell("""
import os
from google.colab import userdata

try:
    github_token = userdata.get("GITHUB_TOKEN")
    repo_url = f"https://{github_token}@github.com/Emmerson2018/bio-interaction-ml.git"
except Exception:
    repo_url = "https://github.com/Emmerson2018/bio-interaction-ml.git"

if not os.path.exists("bio-interaction-ml"):
    !git clone {repo_url}
    %cd bio-interaction-ml
else:
    %cd bio-interaction-ml
    !git pull origin main
"""))

    # 4. uv sync
    cells.append(md_cell("### 4. Sincronização do Ambiente Virtual (`uv sync`)"))
    cells.append(code_cell("""
!uv sync
!uv run python -c "import torch, torchvision, ultralytics; print('✅ Ambiente pronto!')"
"""))

    # 5. Drive & DVC
    cells.append(md_cell("### 5. Montagem do Google Drive & Sincronização dos Datasets via DVC"))
    cells.append(code_cell("""
from google.colab import drive
import os, glob, shutil, subprocess, json

# 1. Montagem do Google Drive
drive.mount("/content/drive")

# 2. Configuração de Credenciais Headless para PyDrive2 / DVC
# As credenciais são carregadas dinamicamente sem segredos hardcoded no repositório.
found_creds = False
temp_creds_path = "/tmp/dvc_temp_default.json"

drive_candidates = [
    "/content/drive/MyDrive/default.json",
    "/content/drive/MyDrive/IFMaker/BioEcoInt/default.json",
    "/content/drive/MyDrive/BioEcoInt_DVC_Auth/default.json",
]

for cand in drive_candidates:
    if os.path.exists(cand):
        shutil.copy2(cand, temp_creds_path)
        print(f"✓ Credenciais headless do DVC carregadas de: {cand}")
        found_creds = True
        break

if not found_creds:
    print("🔍 Procurando default.json no Google Drive...")
    matches = glob.glob("/content/drive/MyDrive/**/default.json", recursive=True)
    if matches:
        shutil.copy2(matches[0], temp_creds_path)
        print(f"✓ Credenciais headless do DVC encontradas em: {matches[0]}")
        found_creds = True

if not found_creds:
    try:
        from google.colab import userdata
        secret_json = userdata.get('DVC_DEFAULT_JSON')
        if secret_json:
            with open(temp_creds_path, "w") as f:
                f.write(secret_json)
            try:
                shutil.copy2(temp_creds_path, "/content/drive/MyDrive/default.json")
            except Exception:
                pass
            print("✓ Credenciais do DVC carregadas via Colab Secrets (DVC_DEFAULT_JSON) e salvas em MyDrive!")
            found_creds = True
    except Exception:
        pass

if not found_creds:
    print("\\n⚠️ Credenciais headless 'default.json' não encontradas no Google Drive nem em Colab Secrets.")
    print("📤 Faça o upload do seu arquivo 'default.json' abaixo (ele será salvo no seu Drive para nunca mais pedir):")
    from google.colab import files
    uploaded = files.upload()
    for fname in uploaded:
        if fname.endswith(".json"):
            shutil.move(fname, temp_creds_path)
            try:
                shutil.copy2(temp_creds_path, "/content/drive/MyDrive/default.json")
                print("💾 Cópia salva em /content/drive/MyDrive/default.json!")
            except Exception:
                pass
            print(f"✓ Credenciais recebidas com sucesso a partir de {fname}!")
            found_creds = True
            break

# Instala no diretório correto do PyDrive2 e configura o remote dinamicamente
if os.path.exists(temp_creds_path):
    try:
        with open(temp_creds_path, "r") as f:
            cdata = json.load(f)
        cid = cdata.get("client_id")
        cs = cdata.get("client_secret")
        
        if cid:
            creds_dir = f"/root/.cache/pydrive2fs/{cid}"
            os.makedirs(creds_dir, exist_ok=True)
            target_file = os.path.join(creds_dir, "default.json")
            shutil.copy2(temp_creds_path, target_file)
            print(f"✅ Credenciais instaladas com sucesso em {target_file}")
            
        if cs:
            subprocess.run(["uv", "run", "dvc", "remote", "modify", "--local", "gdrive-storage", "gdrive_client_secret", cs], check=False)
    except Exception as e:
        print(f"⚠️ Erro ao configurar credenciais: {e}")
else:
    print("⚠️ Prosseguindo sem credenciais prévias do PyDrive2.")

# 3. Sincronização dos Datasets
LOCAL_DATA_DIR = "datasets/generated"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

for ds_name in ["animals_video_sanitized", "animals_video_sam2_crops"]:
    zip_name = f"{ds_name}.zip"
    target_dir = os.path.join(LOCAL_DATA_DIR, ds_name)
    local_zip = os.path.join(LOCAL_DATA_DIR, zip_name)
    
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"✅ Dataset {ds_name} já presente em {target_dir}!")
        continue

    # A. Tenta achar arquivo zip no Google Drive
    found_zip = None
    direct_paths = [
        os.path.join("/content/drive/MyDrive", zip_name),
        os.path.join("/content/drive/MyDrive/IFMaker/BioEcoInt", zip_name),
    ]
    for dp in direct_paths:
        if os.path.exists(dp):
            found_zip = dp
            break

    if not found_zip:
        print(f"🔍 Procurando {zip_name} no Google Drive...")
        matches = glob.glob(f"/content/drive/MyDrive/**/{zip_name}", recursive=True)
        if matches:
            found_zip = matches[0]

    if found_zip:
        print(f"📦 Copiando {zip_name} de {found_zip}...")
        shutil.copy2(found_zip, local_zip)
        print(f"📂 Descompactando {zip_name}...")
        subprocess.run(["unzip", "-q", "-o", local_zip, "-d", LOCAL_DATA_DIR], check=True)
        print(f"✅ {ds_name} pronto com sucesso!")
    else:
        print(f"🔄 {zip_name} não encontrado solto no Drive. Baixando via DVC (headless)...")
        dvc_file = f"{LOCAL_DATA_DIR}/{zip_name}.dvc"
        if os.path.exists(dvc_file):
            subprocess.run(["uv", "run", "dvc", "pull", dvc_file], check=False)
        else:
            subprocess.run(["uv", "run", "dvc", "pull", "-r", "gdrive-storage"], check=False)
        
        if os.path.exists(local_zip):
            print(f"📂 Descompactando {zip_name}...")
            subprocess.run(["unzip", "-q", "-o", local_zip, "-d", LOCAL_DATA_DIR], check=True)
            print(f"✅ {ds_name} pronto via DVC!")
        else:
            print(f"⚠️ Aviso: Verifique a disponibilidade de {zip_name}.")
"""))

    # 6. FAMÍLIA 1: ResNet-18
    cells.append(md_cell("""
---
## 📦 FAMÍLIA 1: ResNet-18 (Baseline Clássico)
Avaliação de controle com convoluções residuais clássicas nas 4 estratégias de visão.
"""))
    cells.append(md_cell("#### Exp 1.1: ResNet-18 Baseline Sanitizado (Cross-Entropy)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp1_resnet18_baseline.yml"))

    cells.append(md_cell("#### Exp 1.2: ResNet-18 + YOLO Crops (Two-Stage)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp2_resnet18_yolo_crop.yml"))

    cells.append(md_cell("#### Exp 1.3: ResNet-18 + Coordinate Attention (CA)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp3_resnet18_ca.yml"))

    cells.append(md_cell("#### Exp 1.4: ResNet-18 + SAM 2 Crops (Segmentação por Instância)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_resnet18_sam2_crops.yml"))

    # 7. FAMÍLIA 2: MobileNetV4-Small
    cells.append(md_cell("""
---
## ⚡ FAMÍLIA 2: Google MobileNetV4-Small (Universal Inverted Bottlenecks - UIB)
Avaliação da arquitetura móvel convolucional moderna (3.8M parâmetros) nas 4 estratégias de visão.
"""))
    cells.append(md_cell("#### Exp 2.1: MobileNetV4-Small Baseline (Sanitizado + Cross-Entropy)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_mobilenetv4_small_baseline.yml"))

    cells.append(md_cell("#### Exp 2.2: MobileNetV4-Small + YOLO Crops (Two-Stage)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_mobilenetv4_small_yolo_crop.yml"))

    cells.append(md_cell("#### Exp 2.3: MobileNetV4-Small + Coordinate Attention (CA)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_mobilenetv4_small_ca.yml"))

    cells.append(md_cell("#### Exp 2.4: MobileNetV4-Small + SAM 2 Crops (Segmentação por Instância)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_mobilenetv4_sam2_crops.yml"))

    # 8. FAMÍLIA 3: RepViT-M0.9 (Campeão SOTA)
    cells.append(md_cell("""
---
## 🚀 FAMÍLIA 3: RepViT-M0.9 (Vision Transformer Reparametrizável)
Avaliação da arquitetura campeã (4.7M parâmetros) combinando atenção global de ViT com reparametrização estrutural mobile.
"""))
    cells.append(md_cell("#### Exp 3.1: RepViT-M0.9 Baseline (Sanitizado + Cross-Entropy / Exp 4A)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_repvit_m0_9_baseline.yml"))

    cells.append(md_cell("#### Exp 3.2: RepViT-M0.9 + YOLO Crops (Two-Stage)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_repvit_m0_9_yolo_crop.yml"))

    cells.append(md_cell("#### Exp 3.3: RepViT-M0.9 + Coordinate Attention (CA)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_repvit_m0_9_ca.yml"))

    cells.append(md_cell("#### Exp 3.4: RepViT-M0.9 + SAM 2 Crops (Recortes Limpos Exemplar V2)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_repvit_m0_9_sam2_crops.yml"))

    # 9. ESTUDO DE PERDAS NO REPVIT
    cells.append(md_cell("""
---
## 🔬 ESTUDO DE FUNÇÕES DE PERDA NO REPVIT-M0.9 (Exp 4A, 4B e 4C)
Análise iso-condicional do impacto do desbalanceamento severo na modelagem de dependências não-locais de longo alcance do Transformer.
"""))
    cells.append(md_cell("#### Exp 4A: RepViT-M0.9 + Cross-Entropy (Equivalente ao Baseline)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_loss_ce_repvit.yml"))

    cells.append(md_cell("#### Exp 4B: RepViT-M0.9 + Focal Loss (gamma = 2.0)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_loss_focal_repvit.yml"))

    cells.append(md_cell("#### Exp 4C: RepViT-M0.9 + Balanced Softmax (Recorde SOTA 80.26% Macro-F1)"))
    cells.append(code_cell("!MPLBACKEND=Agg uv run base-train -opt options/exp_repvit_m0_9.yml"))

    # 10. Avaliação Estéril e Comitê Temporal
    cells.append(md_cell("""
---
## 📹 AVALIAÇÃO ESTÉRIL & COMITÊ TEMPORAL EM VÍDEO (2s @ 3 FPS)
Executa teste estéril e simula o comitê temporal de inferência contínua com consenso >= 70% e abstenção protetora.
"""))
    cells.append(code_cell("""
# 1. Avaliação Estéril do Campeão RepViT SAM 2 Crops
!uv run base-test -opt options/exp_repvit_m0_9_sam2_crops.yml --checkpoint experiments/Exp_RepViT_M0_9_SAM2_Crops/models/best_macro_f1.pth

# 2. Comitê Temporal nos Vídeos de Teste do RepViT SAM 2 Crops
!uv run base-test-committee -opt options/exp_repvit_m0_9_sam2_crops.yml --checkpoint experiments/Exp_RepViT_M0_9_SAM2_Crops/models/best_macro_f1.pth --window-sec 2.0 --target-fps 3.0 --consensus-thresh 0.70 --output-report experiments/Exp_RepViT_M0_9_SAM2_Crops/committee_sam2_report.json
"""))

    # 11. HPO Optuna
    cells.append(md_cell("""
---
## ⚙️ OTIMIZAÇÃO DE HIPERPARÂMETROS HPO (Optuna + ASHA Pruning)
Busca bayesiana TPE com poda precoce focada no RepViT-M0.9.
"""))
    cells.append(code_cell("""
!uv run python experiments/hpo_optuna_baseline.py -opt options/exp_repvit_m0_9.yml --trials 15 --epochs-per-trial 8
"""))

    # 12. Exportação Mobile
    cells.append(md_cell("""
---
## 📱 EXPORTAÇÃO MOBILE E QUANTIZAÇÃO (TFLite FP16 & INT8 PTQ)
Converte os modelos campeões para TFLite FP32, FP16 e INT8 calibrado (PTQ) via reparametrização estrutural `switch_to_deploy`.
"""))
    cells.append(code_cell("""
# Exportação do Campeão RepViT SAM 2 Crops
!MPLBACKEND=Agg uv run base-export -opt options/exp_repvit_m0_9_sam2_crops.yml --checkpoint experiments/Exp_RepViT_M0_9_SAM2_Crops/models/best_macro_f1.pth --format all

# Exportação do Campeão Geral RepViT Balanced Softmax
!MPLBACKEND=Agg uv run base-export -opt options/exp_repvit_m0_9.yml --checkpoint experiments/Exp_RepViT_M0_9_BalancedSoftmax/models/best_macro_f1.pth --format all
"""))

    # 13. Consolidação de Resultados
    cells.append(md_cell("""
---
## 📊 CONSOLIDAÇÃO AUTOMÁTICA DE RESULTADOS & GRÁFICOS SOTA
Compila as métricas de validação, teste estéril e comitê temporal de todos os experimentos em tabela e gráfico comparativo.
"""))
    cells.append(code_cell("!uv run python consolidar_resultados_finais.py"))

    # 14. Versionamento DVC e Git Push Seguro
    cells.append(md_cell("""
---
## 💾 SINCRONIZAÇÃO DE RESULTADOS & DVC NO GOOGLE DRIVE
Salva os pesos e modelos exportados no Google Drive via DVC e sincroniza o Git com pull preventivo.
"""))
    cells.append(code_cell("""
import os, glob, subprocess

# 1. Versionar diretórios de modelos e exportações no DVC
for exp_dir in sorted(glob.glob("experiments/Exp_*")):
    models_dir = os.path.join(exp_dir, "models")
    export_dir = os.path.join(exp_dir, "export")
    if os.path.exists(models_dir) and not os.path.exists(f"{models_dir}.dvc"):
        print(f"📦 Adicionando ao DVC: {models_dir}")
        subprocess.run(["uv", "run", "dvc", "add", models_dir], check=False)
    if os.path.exists(export_dir) and not os.path.exists(f"{export_dir}.dvc"):
        print(f"📦 Adicionando ao DVC: {export_dir}")
        subprocess.run(["uv", "run", "dvc", "add", export_dir], check=False)

# 2. Push dos arquivos binários pesados para o Google Drive
print("☁️ Enviando artefatos pesados para o Google Drive via DVC...")
!uv run dvc push -r gdrive-storage || true

# 3. Git commit e push preventivo com git pull --rebase
!git config user.name "IFMaker Colab Runner"
!git config user.email "colab@ifmaker.edu.br"
!git add experiments/ options/ *.py *.md notebooks/
!git commit -m "feat(colab): consolidacao de novos experimentos e modelos DVC" || echo "Sem alteracoes para commit"

print("🔄 Sincronizando com git pull antes do push...")
!git pull --rebase origin main || true
!git push origin main || echo "⚠️ Push rejeitado. Modelos preservados no Google Drive via DVC!"
"""))

    # 15. Desconexão
    cells.append(md_cell("""
---
## 🛑 DESCONEXÃO AUTOMÁTICA DO RUNTIME (Proteção de Compute Units)
Encerra a sessão da GPU após a conclusão para economizar créditos.
"""))
    cells.append(code_cell("""
from google.colab import runtime
AUTO_DISCONNECT = False  # Mude para True se desejar desconectar automaticamente ao final do treino noturno
if AUTO_DISCONNECT:
    print("🔌 Desconectando runtime do Google Colab...")
    runtime.unassign()
else:
    print("✅ Sessão mantida ativa para inspeção interativa.")
"""))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    return notebook_data

def build_local_notebook(colab_nb_data):
    # O notebook local é muito parecido com o do Colab, mas sem os comandos de montagem de drive ou curl do uv
    local_cells = []
    for c in colab_nb_data["cells"]:
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            if "Montagem do Google Drive" in src:
                local_cells.append(md_cell("### 5. Verificação dos Datasets Locais"))
                continue
            local_cells.append(c)
        elif c["cell_type"] == "code":
            src = "".join(c["source"])
            if "drive.mount" in src or "curl -LsSf https://astral.sh" in src or "google.colab" in src:
                # Substituir por verificação local simples
                if "drive.mount" in src:
                    local_cells.append(code_cell("""
import os
for ds in ["animals_video_sanitized", "animals_video_sam2_crops"]:
    p = f"datasets/generated/{ds}"
    print(f"Dataset {ds}: {'✅ Existe' if os.path.exists(p) else '❌ Não encontrado'}")
"""))
                elif "curl -LsSf" in src:
                    local_cells.append(code_cell("!uv --version"))
                elif "from google.colab import userdata" in src:
                    local_cells.append(code_cell("!git status"))
                elif "runtime.unassign" in src:
                    local_cells.append(code_cell("print('Execução local concluída!')"))
                continue
            local_cells.append(c)

    return {
        "cells": local_cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

def main():
    print("Gerando notebook Colab...")
    colab_nb = build_colab_notebook()
    with open(COLAB_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(colab_nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Salvo em: {COLAB_NB_PATH} ({len(colab_nb['cells'])} células)")

    print("Gerando notebook Local...")
    local_nb = build_local_notebook(colab_nb)
    with open(LOCAL_NB_PATH, "w", encoding="utf-8") as f:
        json.dump(local_nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Salvo em: {LOCAL_NB_PATH} ({len(local_nb['cells'])} células)")

if __name__ == "__main__":
    main()
