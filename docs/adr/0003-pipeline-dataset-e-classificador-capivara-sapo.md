# ADR 0003: Pipeline de dataset e classificador para capivara e sapo

## Status

Aceito

## Contexto

O projeto possui dois modelos 3D iniciais em `assets/blender_models/`: capivara e sapo. A maquina alvo possui uma RTX 4060 com 8 GB de VRAM, suficiente para renderizar imagens 224x224 com `bpy` e treinar um classificador leve em PyTorch.

O objetivo inicial e reconhecer fotos dos modelos impressos em 3D, nao fotos de animais reais.

O ambiente de desenvolvimento nao assume que o Blender esteja instalado como aplicativo. O pipeline deve funcionar a partir de Python, usando o pacote `bpy` quando disponivel.

## Decisao

Criar um pipeline em duas etapas:

1. Geracao de dataset sintetico a partir dos arquivos do Blender usando `bpy` como backend principal.
2. Treinamento de um classificador de imagens com PyTorch.

O dataset gerado segue a estrutura:

```text
datasets/generated/animals_synthetic/
  train/
    capivara/
    sapo/
  val/
    capivara/
    sapo/
  test/
    capivara/
    sapo/
```

A primeira geracao usou rotacao livre de 360 graus. A auditoria visual mostrou que essa abordagem gerou muitas imagens pouco informativas, principalmente para o sapo. A baseline atual de dataset passa a ser a versao estruturada:

```text
datasets/generated/animals_synthetic_structured_v3/
  train/
    capivara/
    sapo/
  val/
    capivara/
    sapo/
  test/
    capivara/
    sapo/
```

A estrategia de vistas, rotacao e distancia esta documentada na ADR 0005.

A primeira arquitetura de treino sera `resnet18` via `torchvision`, com `weights: null` no YAML inicial.

Transfer learning nao deve ser usado no primeiro ciclo. O motivo e que datasets de imagens reais podem ensinar representacoes e vieses que nao correspondem ao alvo do projeto: fotos de objetos impressos em 3D. Essa restricao segue a ADR 0002.

O backend principal de renderizacao sera:

```yaml
renderer_backend: python_bpy
```

O modo por executavel do Blender pode existir apenas como alternativa tecnica:

```yaml
renderer_backend: blender_executable
```

## Consequencias

Pontos positivos:

- Pipeline reproduzivel via YAML.
- Dataset fica auditavel em disco.
- Configuracao inicial cabe em 8 GB de VRAM.
- Nao exige instalar o aplicativo Blender quando o pacote Python `bpy` estiver disponivel no ambiente.

Riscos:

- Render sintetico pode nao representar textura, brilho e defeitos da impressao 3D.
- Acuracia em validacao sintetica pode ser maior do que em fotos reais.
- Sera necessario adicionar um conjunto pequeno de fotos reais para validacao de dominio.
- O pacote `bpy` e pesado e pode ter restricoes de versao de Python/plataforma.

Comandos principais:

```bash
uv run base-generate-blender-dataset -opt options/generate_synthetic_animals.yml
uv run base-train -opt options/train_synthetic_animals.yml
uv run base-recognize -opt options/train_synthetic_animals.yml --checkpoint experiments/Synthetic_Animals_Classifier/models/epoch_19.pth --image caminho/para/foto.png
```

Comandos principais para a baseline estruturada atual:

```bash
uv run base-generate-blender-dataset -opt options/generate_synthetic_animals_structured.yml
uv run base-train -opt options/train_synthetic_animals_structured.yml
uv run base-recognize -opt options/train_synthetic_animals_structured.yml --checkpoint experiments/Synthetic_Animals_Classifier_Structured/models/epoch_19.pth --image caminho/para/foto.png
```
