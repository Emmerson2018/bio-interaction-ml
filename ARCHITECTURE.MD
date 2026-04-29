# Arquitetura do Projeto: base-tool-ml (Lab Toolbox)

Esta toolbox adota os princípios de **Separação de Preocupações (SoC)**, **Registry Pattern** e **Configuração Declarativa** para padronizar o desenvolvimento de Machine Learning no laboratório.

## 1. Topologia do Projeto

A estrutura segue o padrão de "Plataforma Única" para múltiplos tipos de experimentos:

- **`base_tool/archs/` (Neural Networks):** Definições puras de `nn.Module`. Sem lógica de treino ou IO.
- **`base_tool/models/` (Agents/Orchestrators):** Gerenciam o ciclo de vida (forward, backward, loss, metrics). Implementam a lógica de como a arquitetura é usada.
- **`base_tool/data/` (Data Pipeline):** Datasets e DataLoaders. Abstrai a origem dos dados (CSV, Imagens, Sequências Bio).
- **`base_tool/losses/` & `base_tool/metrics/`:** Repositórios de funções de custo e avaliação.
- **`base_tool/visualization/`:** Classes e utilitários para geração de gráficos, plots de resultados e inspeção visual de dados/predições.
- **`base_tool/utils/`:** Utilitários de sistema (Registry, Logger, YAML Parser).
- **`options/`:** Arquivos `.yml` que definem cada experimento.

## 2. Componentes Fundamentais

### Registry Pattern
O sistema utiliza decorators para registrar classes em dicionários globais:
- `ARCH_REGISTRY`, `MODEL_REGISTRY`, `DATASET_REGISTRY`, `LOSS_REGISTRY`, `METRIC_REGISTRY`.
- Permite instanciar objetos via string no YAML: `type: MyCoolModel`.

### Abstração `BaseModel`
Interface obrigatória para todos os modelos, garantindo os hooks:
- `feed_data(data)`: Preparação de entrada.
- `optimize_parameters()`: Passo de treino.
- `test()`: Inferência.
- `save()` / `load()`: Persistência.

## 3. Fluxo de Execução Declarativo

1. O usuário cria um arquivo `options/train_exemplo.yml`.
2. O `train.py` carrega o YAML.
3. As `Factory Functions` (`build_model`, `build_dataset`) consultam os Registries e criam os objetos.
4. O loop de treino chama os métodos padronizados da `BaseModel`.

4. Status do Desenvolvimento (Fases Concluídas)

- [x] Fase 0: Setup com `uv` e dependências.
- [x] Fase 1: Infraestrutura Core (Registry, Scandir, Auto-import).
- [x] Fase 2: Abstrações de ML (BaseModel, BaseDataset, Factory Functions).
- [x] Fase 3: Configuração e Logs (YAML Parser, Logger).
- [x] Fase 4: Script de Treinamento (Maestro train.py).
- [x] Fase 5: Prova de Conceito (POC com SimpleMLP).

---
*Documento atualizado em 29/04/2026.*
