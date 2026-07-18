# ADR 0001: Dataset sintetico a partir do Blender

## Status

Aceito

## Contexto

O projeto precisa reconhecer animais a partir de fotos tiradas de modelos impressos em 3D. Como existem arquivos dos animais no Blender, o pipeline pode gerar imagens sinteticas em memoria ou em disco variando camera, luz, escala, pose, fundo e materiais.

O alvo inicial nao e reconhecer animais reais na natureza. O alvo e reconhecer imagens dos objetos fisicos impressos.

## Decisao

Adotar um pipeline de geracao de dataset sintetico a partir dos modelos 3D do Blender.

Os arquivos fonte serao depositados em:

```text
assets/blender_models/
```

Cada classe deve ser representada por uma subpasta. Exemplo:

```text
assets/blender_models/
  sapo/
    sapo_01.blend
  peixe/
    peixe_01.blend
```

O gerador deve produzir imagens rotuladas para treino, validacao e teste. A primeira versao pode persistir imagens em disco para auditoria visual. Uma versao posterior pode gerar amostras em memoria durante o treino.

## Consequencias

Pontos positivos:

- Controle total sobre classes, angulos, iluminacao e quantidade de imagens.
- Menor dependencia inicial de fotos reais.
- Facilidade para reproduzir o dataset a partir de configuracao.

Riscos:

- O modelo pode aprender caracteristicas do render, nao do objeto impresso real.
- A impressao 3D introduz textura, brilho, sombra e imperfeicoes que precisam aparecer no dataset.
- Fotos reais do objeto impresso ainda serao importantes para validacao.

Validacoes necessarias:

- Comparar acuracia em imagens renderizadas contra fotos reais dos modelos impressos.
- Medir se variacoes de luz, fundo e camera reduzem overfitting ao estilo do Blender.
