# Arquitetura do Projeto: base-tool-ml (Lab Toolbox)

Esta toolbox adota os principios de **Separacao de Preocupacoes (SoC)**, **Registry Pattern** e **Configuracao Declarativa** para padronizar o desenvolvimento de Machine Learning no laboratorio.

## 1. Topologia do Projeto

A estrutura segue o padrao de "Plataforma Unica" para multiplos tipos de experimentos:

- **`base_tool/archs/` (Neural Networks):** Definicoes puras de `nn.Module`. Sem logica de treino ou IO.
- **`base_tool/models/` (Agents/Orchestrators):** Gerenciam o ciclo de vida (forward, backward, loss, metrics). Implementam a logica de como a arquitetura e usada.
- **`base_tool/data/` (Data Pipeline):** Datasets e DataLoaders. Abstrai a origem dos dados (CSV, Imagens, Sequencias Bio).
- **`base_tool/losses/` & `base_tool/metrics/`:** Repositorios de funcoes de custo e avaliacao.
- **`base_tool/visualization/`:** Classes e utilitarios para geracao de graficos, plots de resultados e inspecao visual de dados/predicoes.
- **`base_tool/utils/`:** Utilitarios de sistema (Registry, Logger, YAML Parser).
- **`options/`:** Arquivos `.yml` que definem cada experimento.

## 2. Componentes Fundamentais

### Registry Pattern
O sistema utiliza decorators para registrar classes em dicionarios globais:
- `ARCH_REGISTRY`, `MODEL_REGISTRY`, `DATASET_REGISTRY`, `LOSS_REGISTRY`, `METRIC_REGISTRY`.
- Permite instanciar objetos via string no YAML: `type: MyCoolModel`.

### Abstracao `BaseModel`
Interface obrigatoria para todos os modelos, garantindo os hooks:
- `feed_data(data)`: Preparacao de entrada.
- `optimize_parameters()`: Passo de treino.
- `test()`: Inferencia.
- `save()` / `load()`: Persistencia.

## 3. Fluxo de Execucao Declarativo

1. O usuario cria um arquivo `options/train_exemplo.yml`.
2. O `train.py` carrega o YAML.
3. As `Factory Functions` (`build_model`, `build_dataset`) consultam os Registries e criam os objetos.
4. O loop de treino chama os metodos padronizados da `BaseModel`.

4. Status do Desenvolvimento (Fases Concluidas)

- [x] Fase 0: Setup com `uv` e dependencias.
- [x] Fase 1: Infraestrutura Core (Registry, Scandir, Auto-import).
- [x] Fase 2: Abstracoes de ML (BaseModel, BaseDataset, Factory Functions).
- [x] Fase 3: Configuracao e Logs (YAML Parser, Logger).
- [x] Fase 4: Script de Treinamento (Maestro train.py).
- [x] Fase 5: Prova de Conceito (POC com SimpleMLP).

---
*Documento atualizado em 29/04/2026.*
