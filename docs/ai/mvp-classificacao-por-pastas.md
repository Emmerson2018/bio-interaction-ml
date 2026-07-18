# MVP de classificacao por pastas

Este e o fluxo recomendado atual para entregar um MVP funcional de classificacao de imagens para integracao mobile.

## Escopo

O MVP executa:

- descoberta automatica de classes por subpastas;
- preparo de imagens e videos;
- extracao de frames de video a 1 FPS;
- divisao `train`, `validation` e `test` com seed fixa;
- treino de um unico classificador leve;
- avaliacao por classe;
- curvas de treino e validacao;
- exportacao intermediaria para integracao mobile.

Ficam fora deste MVP:

- deteccao de multiplos objetos;
- calibracao avancada de `unknown`;
- FAR, FRR, risk-coverage e selective accuracy;
- comparacao sistematica entre varias arquiteturas;
- contratos manuais de metadata.

## Estrutura do dataset

Cada pasta representa uma classe:

Para o treino atual, use o dataset sintetico ja materializado:

```text
datasets/generated/animals_multiclass_v1/
  train/
    boto_cor_de_rosa/
    capivara/
    onca_pintada/
    sapo/
  val/
  test/
```

Para validacao real externa, os arquivos reais podem continuar em uma estrutura por pasta, por exemplo:

```text
asset/Imagens_reais/
  BOTO/
  Capivara/
  ONÇA/
  Sapo/
  Diorama/
```

As pastas `BOTO`, `Capivara`, `ONÇA` e `Sapo` alimentam a validacao das classes conhecidas. Pastas sem classe treinada, como `Diorama`, `Anta`, `Jacare`, `tamandua` e `Fungos`, devem ser usadas para calibrar rejeicao `unknown`. Pastas com mais de um animal devem ficar fora do treino single-label e entrar como casos de estresse.

## Midias aceitas

Imagens:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

Videos:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`

Frames extraidos de um mesmo video recebem o mesmo `source_group` e permanecem no mesmo split.

## Regra especial para video em unknown

Quando houver video na pasta `unknown`, o primeiro video encontrado recebe a regra especial do MVP:

- extrair frames em `00:00` ate `00:06`;
- classificar esses frames como `unknown`;
- excluir frames de `00:07` em diante;
- registrar a exclusao no manifest como cena ambigua.

Essa regra evita treinar uma classificacao single-label com peixe e sapo simultaneamente na mesma cena.

## Unknown no MVP

No treino sintetico `animals_multiclass_v1`, o modelo e um classificador de classes conhecidas. A resposta `unknown` deve ser um fallback de inferencia:

```text
se top1_probability < confidence_threshold:
    final_class = "unknown"
    mensagem = "Nao consegui identificar a figura."
senao:
    final_class = top1_class
```

O valor de `confidence_threshold` deve ser definido com validacao real. A acuracia sintetica saturada nao deve ser usada sozinha para escolher esse corte.

## Comando

```powershell
.\.venv-train\Scripts\base-run-mvp-classifier.exe `
  --input-root datasets\generated\animals_multiclass_v1 `
  --output-root reports\mvp_classifier_synthetic `
  --seed 42 `
  --extract-video-fps 1 `
  --epochs 30 `
  --patience 5
```

## Saidas

Relatorios:

- `reports/mvp_classifier/dataset_manifest.csv`
- `reports/mvp_classifier/dataset_summary.json`
- `reports/mvp_classifier/split_summary.json`
- `reports/mvp_classifier/training_history.csv`
- `reports/mvp_classifier/training_curves.png`
- `reports/mvp_classifier/test_metrics.json`
- `reports/mvp_classifier/per_class_metrics.csv`
- `reports/mvp_classifier/confusion_matrix.csv`
- `reports/mvp_classifier/confusion_matrix.png`
- `reports/mvp_classifier/misclassified_examples.csv`
- `reports/mvp_classifier/final_report.md`

Experimento:

- `experiments/mvp_classifier/models/best_model.pth`

Exportacao:

- `reports/mvp_classifier/mobile_export/labels.txt`
- `reports/mvp_classifier/mobile_export/manifest.json`
- `reports/mvp_classifier/mobile_export/model.onnx`, quando ONNX estiver disponivel.

`model.tflite` so e gerado quando a stack TensorFlow/TFLite estiver instalada e suportar a conversao.
