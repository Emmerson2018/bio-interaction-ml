# Validacao real e rejeicao unknown

Este protocolo prepara a avaliacao do classificador MobileNetV3 Small pre-treinado em dados reais e em dados negativos para rejeicao `unknown`.

## Escopo

Incluido:

- imagens reais de modelos fisicos impressos;
- videos reais, somente depois de metadados por sessao;
- conjuntos conhecidos separados em calibracao e teste;
- conjuntos desconhecidos separados em calibracao e teste;
- validacao de metadados antes de qualquer avaliacao;
- calibracao de limiares por confianca Top-1 e margem Top-1 menos Top-2.

Fora do escopo desta etapa:

- novo treino completo;
- exportacao TFLite;
- implementacao Flutter;
- selecao de modelo de producao sem dados reais suficientes.

## Estrutura

```text
datasets/
  real_multiclass_v1/
    calibration_known/
    test_known/

  unknown_multiclass_v1/
    calibration/
    test/
```

Os arquivos originais em `asset/Imagens_reais/` nao devem ser movidos ate que `datasets/metadata/real_media_metadata.csv` esteja completo.

## Classes Permitidas

`true_class` deve ser um destes valores:

- `boto_cor_de_rosa`
- `capivara`
- `onca_pintada`
- `sapo`
- `unknown`

Nesta rodada, `ariranha` e `pirarucu` ficam fora das classes reconhecidas ate existirem fotos reais suficientes para validacao.

## Campos Obrigatorios

Antes da avaliacao, cada arquivo precisa ter:

- `true_class`
- `is_unknown`
- `split_role`
- `source_group`
- `session_id`
- `device`
- `view`
- `background`
- `lighting`

Valores permitidos para `split_role`:

- `calibration_known`
- `test_known`
- `calibration_unknown`
- `test_unknown`
- `exclude`

Campos recomendados:

- `physical_model_id`
- `distance`
- `occlusion`
- `blur`
- `notes`

## Regras de Split

- A mesma sessao nao pode aparecer em calibracao e teste.
- O mesmo `source_group` nao pode aparecer em calibracao e teste.
- O mesmo video nao pode fornecer frames para calibracao e teste.
- O mesmo arquivo nao pode aparecer em mais de um split.
- O mesmo hash SHA-256 nao pode aparecer em mais de um split.
- `true_class=unknown` exige `is_unknown=true`.
- Classes conhecidas exigem `is_unknown=false`.
- Classes conhecidas nao podem usar split `calibration_unknown` ou `test_unknown`.
- `unknown` nao pode usar split `calibration_known` ou `test_known`.

## Validacao

Use:

```powershell
.\.venv-train\Scripts\base-validate-real-metadata.exe --metadata datasets\metadata\real_media_metadata.csv
```

Para bloquear execucao enquanto houver campos obrigatorios vazios:

```powershell
.\.venv-train\Scripts\base-validate-real-metadata.exe --metadata datasets\metadata\real_media_metadata.csv --require-ready
```

## Avaliacao Known Real

Depois que os metadados forem preenchidos e os arquivos forem organizados nos splits:

```powershell
.\.venv-train\Scripts\base-evaluate.exe --opt options\multiclass_v1\e04_mobilenet_v3_small_imagenet.yml --checkpoint experiments\E04_MobileNetV3_Small_ImageNet_MulticlassV1\models\best.pth --split test
```

O comando acima usa o split configurado no YAML. Para dados reais, crie YAMLs dedicados ou ajuste uma copia de avaliacao para apontar para `datasets/real_multiclass_v1/test_known`.

## Calibracao Unknown

Use somente quando existirem:

- `datasets/real_multiclass_v1/calibration_known`
- `datasets/unknown_multiclass_v1/calibration`
- `datasets/real_multiclass_v1/test_known`
- `datasets/unknown_multiclass_v1/test`

Exemplo:

```powershell
.\.venv-train\Scripts\base-calibrate-rejection.exe `
  --opt options\multiclass_v1\e04_mobilenet_v3_small_imagenet.yml `
  --checkpoint experiments\E04_MobileNetV3_Small_ImageNet_MulticlassV1\models\best.pth `
  --known-root datasets\real_multiclass_v1\calibration_known `
  --unknown-root datasets\unknown_multiclass_v1\calibration `
  --output-dir reports\multiclass_v1\rejection\e04
```

O criterio padrao e:

```text
FAR <= 0.05
sob essa restricao, maximizar coverage
```

Os limiares selecionados na calibracao nao podem consultar o teste.
