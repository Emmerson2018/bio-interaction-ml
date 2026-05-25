# Pipeline Blender -> Dataset -> Treino

Este pipeline foi criado para reconhecer fotos dos modelos impressos em 3D de capivara e sapo.

## 1. Entrada

Os modelos ficam em:

```text
assets/blender_models/
  capivara/
    Capivara.blend1
  sapo/
    Sapo_Rhinella marina.blend1
```

Cada subpasta representa uma classe. O gerador aceita `.blend`, `.blend1`, `.fbx`, `.obj`, `.glb` e `.gltf`.

## 2. Dependencia de renderizacao

O pipeline padrao usa `bpy`, a API Python do Blender empacotada como biblioteca.

Instalacao prevista:

```bash
uv sync --extra synthetic
```

Se necessario, instale `bpy` diretamente no ambiente Python usado pelo projeto.

## 3. Geracao do dataset

Configuracao:

```text
options/generate_synthetic_animals.yml
```

Comando:

```bash
uv run base-generate-blender-dataset -opt options/generate_synthetic_animals.yml
```

O backend padrao do YAML e:

```yaml
renderer_backend: python_bpy
```

Saida esperada:

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

## 4. Treino

Configuracao:

```text
options/train_synthetic_animals.yml
```

Comando:

```bash
uv run base-train -opt options/train_synthetic_animals.yml
```

A configuracao inicial usa `resnet18` sem pesos pre-treinados, batch size 32 e imagens 224x224, adequado para uma RTX 4060 com 8 GB.

Transfer learning nao faz parte do primeiro ciclo. A baseline oficial usa `weights: null` porque o alvo sao fotos de objetos impressos em 3D, nao fotos de animais reais.

## 5. Reconhecimento de uma foto

Depois do treino, classifique uma imagem com:

```bash
uv run base-recognize -opt options/train_synthetic_animals.yml --checkpoint experiments/Synthetic_Animals_Classifier/models/epoch_19.pth --image caminho/para/foto.png
```

## 6. Criterio de validacao

Acuracia em imagens renderizadas nao deve ser considerada suficiente. A validacao final precisa usar fotos reais dos modelos impressos em 3D, porque esse e o dominio de uso do aplicativo.
