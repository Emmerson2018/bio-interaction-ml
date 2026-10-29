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

```bash
uv run base-train -opt options/poc_mlp.yml
```

---

## 📦 Compartilhamento de Dados e Modelos (DVC + Google Drive)

Para manter o repositório Git leve e reprodutível, os artefatos pesados (pesos de redes `.pth`, modelos exportados `.onnx`/`.tflite` e datasets brutos/vídeos) são versionados pelo **DVC** e armazenados diretamente em uma pasta compartilhada do **Google Drive**.

Os relatórios de métricas (`classification_report.txt`, `test_summary.json`) e gráficos comparativos (`confusion_matrix.png`) continuam salvos diretamente no **Git** para acompanhamento dos resultados da equipe.

### 1. Configuração Inicial para Novos Membros da Equipe

Após clonar o repositório e rodar `uv sync`, configure o segredo de acesso do OAuth Client do laboratório no seu ambiente local:

```bash
# Configure o segredo localmente (esse comando salva em .dvc/config.local, que NÃO vai pro Git)
uv run dvc remote modify --local gdrive-storage gdrive_client_secret "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"
```

> **Atenção (Usuários de Teste)**:
> Como o aplicativo OAuth está registrado no Google Cloud em modo de teste com conta pessoal, o administrador do projeto (`sergioaugusto.bezerra.jr@gmail.com`) precisa adicionar o seu e-mail Google na lista de **Test Users** do Google Cloud Console antes da sua primeira autenticação.

### 2. Baixando os Dados e Modelos do Projeto

Para baixar os arquivos pesados de experimentos e datasets que já estão no Drive:

```bash
uv run dvc pull
```

*Na primeira vez em que você rodar esse comando, uma aba do navegador será aberta automaticamente solicitando que você faça login com sua conta Google e aprove o acesso à pasta do laboratório.*

### 3. Versionando Novos Checkpoints ou Datasets

Quando você treinar um novo modelo ou gerar novos dados pesados:

```bash
# 1. Registre os pesos ou pastas pesadas no DVC
uv run dvc add experiments/NOME_DO_EXPERIMENTO/models
uv run dvc add experiments/NOME_DO_EXPERIMENTO/export

# 2. Envie os arquivos pesados para o Google Drive
uv run dvc push

# 3. Comite os arquivos de apontamento (.dvc), relatórios e métricas leves no Git
git add experiments/NOME_DO_EXPERIMENTO/*.dvc
git add experiments/NOME_DO_EXPERIMENTO/test_results/
git add experiments/NOME_DO_EXPERIMENTO/visualization/
git commit -m "feat(exp): novos resultados e checkpoints de NOME_DO_EXPERIMENTO"
git push
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
**BioEcoInt Lab** - *Engenharia de Software aplicada a Ciencia.*
