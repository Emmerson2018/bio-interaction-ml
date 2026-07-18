# ADR 0002: Estrategia inicial de treinamento

## Status

Aceito

## Contexto

Modelos pre-treinados em datasets de imagens reais, como ImageNet, podem ajudar por ja terem aprendido bordas, texturas, formas e composicao visual. Ao mesmo tempo, o dominio deste projeto e diferente: fotos de objetos impressos em 3D baseados em modelos renderizados.

Usar transfer learning pode aumentar a acuracia por aproveitar representacoes visuais gerais, mas tambem pode introduzir vieses de imagens naturais que nao representam bem os objetos impressos. Esse risco e central neste projeto, porque o reconhecimento desejado nao e de animais reais, mas de fotos de modelos 3D impressos.

## Decisao

O pipeline inicial deve seguir uma estrategia sem transfer learning.

O primeiro modelo de classificacao deve ser treinado com pesos inicializados do zero (`weights: null`). Essa decisao cria uma baseline alinhada ao dominio real do projeto: fotos de modelos impressos em 3D.

Transfer learning fica bloqueado como escolha padrao. Ele so pode entrar depois que a baseline sem pre-treino for treinada e avaliada.

Uma etapa experimental posterior pode testar:

1. Modelo pre-treinado com fine-tuning nas imagens sinteticas e fotos dos modelos impressos.
2. Modelo treinado sem pesos pre-treinados, usando o mesmo dataset.

O criterio de escolha deve ser desempenho em um conjunto de validacao composto por fotos reais dos modelos 3D impressos. Acuracia em render sintetico nao e suficiente para aprovar a estrategia.

## Consequencias

Pontos positivos:

- Evita assumir que datasets de imagens reais resolvem automaticamente o problema.
- Mantem uma base experimental objetiva para decidir.
- Direciona a metrica para o uso real: reconhecer fotos dos modelos impressos.
- Reduz o risco de validar um modelo que aprendeu vieses de animais reais, texturas naturais ou contexto fotografico externo ao problema.

Riscos:

- Treinar do zero pode exigir mais imagens e mais tempo.
- Fine-tuning pode funcionar melhor mesmo com diferenca de dominio.
- Sem fotos reais de validacao, a acuracia em render pode ser enganosa.

Regra de implementacao:

- YAMLs oficiais de treino devem manter `weights:` vazio ou `null` ate que exista uma ADR posterior autorizando experimento de transfer learning.
- Qualquer resultado de transfer learning deve ser reportado separadamente da baseline sem pre-treino.

## Evidencia posterior

Foi executado um experimento comparativo com `resnet18` e `weights: DEFAULT`, usando pesos ImageNet do `torchvision`, em:

```text
options/train_synthetic_animals_structured_pretrained.yml
```

Esse experimento nao substitui a baseline oficial sem pesos pre-treinados. Ele existe apenas para comparacao.

Resultados em datasets sinteticos:

```text
pretrained_v3_epoch19:
  v3_test accuracy: 1.0000
  v3_test confidence_mean: 1.0000
  v1_test accuracy: 1.0000
  v1_test confidence_mean: 0.9985

scratch_v3_epoch19:
  v3_test accuracy: 1.0000
  v3_test confidence_mean: 1.0000
  v1_test accuracy: 1.0000
  v1_test confidence_mean: 0.9893
```

Interpretacao:

- O uso de pesos ImageNet aumentou a confianca media no teste sintetico v1.
- A acuracia sintetica continua saturada em 1.0000.
- O experimento nao prova superioridade para o objetivo real, porque ainda falta validacao com fotos reais dos objetos impressos em 3D.
