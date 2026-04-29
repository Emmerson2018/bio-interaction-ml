# 🧪 Base Tool ML

Uma toolbox agnóstica para projetos de Machine Learning, focada em escalabilidade, reprodutibilidade e separação de preocupações.

## 🚀 Como Começar

### Instalação
Utilizamos o `uv` para gerenciamento ultrarrápido de dependências:

```bash
# Clone o repositório
git clone https://github.com/Emmerson2018/bio-interaction-ml base-tool-ml
cd base-tool-ml

# Sincronize o ambiente
uv sync
```

### Executando um Experimento
Tudo é controlado via arquivos YAML na pasta `options/`.

```bash
uv run base-train -opt options/poc_mlp.yml
```

---

## 🏗️ Arquitetura (Registry Pattern)

A toolbox utiliza um sistema de **Registro Automático**. Para adicionar um novo componente, você não precisa alterar o código core; apenas decore sua classe.

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

## 📂 Estrutura de Pastas
- `base_tool/archs`: Definições de Redes Neurais.
- `base_tool/models`: Lógica de treinamento (Forward/Backward).
- `base_tool/data`: Datasets e Dataloaders.
- `base_tool/visualization`: Geração automática de gráficos.
- `base_tool/metrics`: Métricas de avaliação (MSE, MAE, F1, etc).
- `experiments/`: Resultados, logs e checkpoints (gerado automaticamente).

## 📊 Visualização
A toolbox gera automaticamente um payload de logs (losses, métricas e LR) e envia para os visualizadores registrados. Os gráficos são salvos em `experiments/[NOME]/visualization/`.

---
**BioEcoInt Lab** - *Engenharia de Software aplicada à Ciência.*
