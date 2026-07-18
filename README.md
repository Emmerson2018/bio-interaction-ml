#  Base Tool ML

Uma toolbox agnostica para projetos de Machine Learning, focada em escalabilidade, reprodutibilidade e separacao de preocupacoes.

##  Como Comecar

### Instalacao
Utilizamos o `uv` para gerenciamento ultrarrapido de dependencias:

```bash
# Clone o repositorio
git clone https://github.com/Emmerson2018/bio-interaction-ml base-tool-ml
cd base-tool-ml

# Sincronize o ambiente
uv sync
```

### Executando um Experimento
Tudo e controlado via arquivos YAML na pasta `options/`.

Gerar imagens sintéticas a partir dos modelos Blender:

```bash
uv run base-generate-blender-dataset -opt options/generate_animals_multiclass_v1.yml
```

Treinar o classificador principal, baseado em MobileNetV3 Small pré-treinada:

```bash
uv run base-train -opt options/multiclass_v1/e04_mobilenet_v3_small_imagenet.yml
```

---

##  Arquitetura (Registry Pattern)

A toolbox utiliza um sistema de **Registro Automatico**. Para adicionar um novo componente, voce nao precisa alterar o codigo core; apenas decore sua classe.

### 1. Adicionando uma Nova Arquitetura (`archs/`)
Crie um arquivo `base_tool/archs/meu_modelo.py`:
```python
from base_tool.utils.registry import ARCH_REGISTRY
import torch.nn as nn

@ARCH_REGISTRY.register()
class MeuModelo(nn.Module):
    ...
```

### 2. Adicionando um Novo Dataset (`data/`)
Crie um arquivo `base_tool/data/meu_dataset.py`:
```python
from base_tool.utils.registry import DATASET_REGISTRY
from base_tool.data.base_dataset import BaseDataset

@DATASET_REGISTRY.register()
class MeuDataset(BaseDataset):
    ...
```

### 3. Configurando o Experimento (`options/`)
No seu arquivo `.yml`, chame os componentes pelos nomes das classes:
```yaml
name: Experimento_Bio
model_type: SimpleModel # Ou seu modelo customizado
network_g:
  type: MeuModelo
  param1: valor
datasets:
  train:
    type: MeuDataset
```

---

##  Estrutura de Pastas
- `base_tool/archs`: Definicoes de Redes Neurais.
- `base_tool/models`: Logica de treinamento (Forward/Backward).
- `base_tool/data`: Datasets e Dataloaders.
- `base_tool/visualization`: Geracao automatica de graficos.
- `base_tool/metrics`: Metricas de avaliacao (MSE, MAE, F1, etc).
- `experiments/`: Resultados, logs e checkpoints (gerado automaticamente).

##  Visualizacao
A toolbox gera automaticamente um payload de logs (losses, metricas e LR) e envia para os visualizadores registrados. Os graficos sao salvos em `experiments/[NOME]/visualization/`.

---
**BioEcoInt Lab** - *Engenharia de Software aplicada à Ciência.*
# MVP recomendado: classificador por pastas

O fluxo recomendado no momento para o subsistema de IA e um MVP simples de classificacao de imagens.
Ele descobre as classes diretamente pelos nomes das subpastas, prepara imagens e videos, treina um unico classificador leve e gera os artefatos de avaliacao e exportacao.

Estrutura esperada:

```text
datasets/generated/animals_multiclass_v1/
  train/
  val/
  test/
```

Comando principal:

```powershell
.\.venv-train\Scripts\base-run-mvp-classifier.exe `
  --input-root datasets\generated\animals_multiclass_v1 `
  --output-root reports\mvp_classifier_synthetic `
  --seed 42 `
  --extract-video-fps 1
```

Saidas principais:

- `reports/mvp_classifier/dataset_manifest.csv`
- `reports/mvp_classifier/training_curves.png`
- `reports/mvp_classifier/test_metrics.json`
- `reports/mvp_classifier/per_class_metrics.csv`
- `reports/mvp_classifier/confusion_matrix.png`
- `reports/mvp_classifier/final_report.md`
- `reports/mvp_classifier/mobile_export/labels.txt`
- `reports/mvp_classifier/mobile_export/manifest.json`

Neste MVP, o modelo treinado em `animals_multiclass_v1` e um classificador de classes conhecidas. O resultado `unknown` deve ser um fallback de inferencia: se a maior probabilidade ficar abaixo do corte definido pela validacao real, o aplicativo retorna `unknown` e exibe a mensagem de nao identificacao. O fluxo avancado de calibracao, FAR/FRR, datasets `calibration_known`/`test_unknown` e comparacao multi-arquitetura fica preservado como trabalho experimental/futuro, mas nao e dependencia do MVP.
