# Auditoria do dataset do diorama

## Objetivo

O aplicativo deve reconhecer animais impressos em 3D dentro de um diorama para apoio ao ensino de biologia no ensino fundamental. A prioridade do modelo e ser conservador: quando houver duvida, deve retornar `unknown` em vez de produzir falso positivo.

## Classes conhecidas nesta rodada

Estas classes possuem modelo Blender e fotos reais individuais no diorama:

- `boto_cor_de_rosa`
- `capivara`
- `onca_pintada`
- `sapo`

## Blender disponivel, mas fora da rodada atual

Estas classes possuem Blender, mas nao possuem validacao real individual suficiente nesta rodada:

- `ariranha`
- `pirarucu`

Elas nao devem entrar como classe reconhecida ate existirem fotos reais suficientes para calibracao e teste.

## Imagens reais para rejeicao unknown

Estas pastas representam animais/cenas sem classe treinada e devem calibrar rejeicao:

- `Anta`
- `Jacare`
- `tamandua`
- `Fungos`
- `Diorama`

## Casos de estresse excluidos do treino single-label

Estas pastas contem mais de um animal ou cena ambigua. Elas nao devem ser usadas em treino single-label, mas sao uteis para avaliar comportamento do app:

- `anta-capivara`
- `Boto-Capivara`
- `boto-onça`
- `capivara-onça`
- `capivara-tamandua`
- `sapo-boto`

## Metadados

O manifesto inicial esta em:

```text
datasets/metadata/real_media_metadata.csv
```

Distribuicao atual:

- conhecidas em calibracao: 22 imagens
- conhecidas em teste: 18 imagens
- unknown em calibracao: 15 imagens
- unknown em teste: 15 imagens
- casos excluidos/estresse: 16 imagens

## Regra de produto

No app, uma predicao so deve ser aceita quando passar por todos os criterios:

- confianca Top-1 acima do limiar calibrado;
- margem Top-1 menos Top-2 acima do limiar calibrado;
- entropia abaixo do limiar calibrado;
- mesma classe observada em frames consecutivos;
- classe reconhecida pertence ao conjunto conhecido da rodada.

Caso contrario, retornar `unknown` com uma mensagem de nao identificacao.
