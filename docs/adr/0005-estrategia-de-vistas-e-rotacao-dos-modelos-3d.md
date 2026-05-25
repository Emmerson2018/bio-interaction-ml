# ADR 0005: Estrategia de vistas e rotacao dos modelos 3D

## Status

Aceito

## Contexto

O primeiro dataset sintetico foi gerado com rotacao livre de 360 graus para camera e modelo. A avaliacao visual mostrou que essa estrategia produziu muitas imagens pouco informativas para o objetivo real.

No caso do sapo, muitas imagens mostravam costas, traseira, barriga ou patas isoladas, em vez da estrutura reconhecivel do animal: olhos grandes, boca, cabeca larga e patas dianteiras.

No caso da capivara, as vistas frontais e traseiras eram menos informativas do que perfil e tres-quartos lateral. A silhueta mais reconhecivel da capivara aparece melhor quando corpo, focinho, pernas e dorso ficam visiveis lateralmente.

Portanto, acuracia alta em dataset sintetico gerado com vistas ruins nao deve ser interpretada como evidencia forte de generalizacao.

## Decisao

Substituir a estrategia de camera totalmente aleatoria por uma estrategia estruturada de vistas.

O renderer passa a aceitar:

```text
--view-strategy random_360|structured|fixed
--front-angle-degrees
--fixed-view-angle-degrees
--close-view-ratio
--far-view-ratio
```

As classes podem ter calibracao propria em YAML:

```yaml
class_render_options:
  capivara:
    front_angle_degrees: 180
  sapo:
    front_angle_degrees: 0
```

A distribuicao estruturada prioriza:

```text
frente:               peso 0.32
tres-quartos esquerda: peso 0.27
tres-quartos direita:  peso 0.27
lateral esquerda:      peso 0.06
lateral direita:       peso 0.05
traseira:              peso 0.03 total
```

A rotacao do modelo fica limitada a pequena variacao de yaw, pitch e roll. A variacao principal de ponto de vista vem da camera. Isso evita que camera e modelo girem independentemente em 360 graus e produzam muitos enquadramentos anatomicamente pouco uteis.

Tambem foi decidido manter variacao de distancia, mas sem permitir que closes dominem ou cortem demais o animal:

```yaml
camera_radius_min: 3.7
camera_radius_max: 6.2
close_view_ratio: 0.12
far_view_ratio: 0.20
```

O dataset recomendado apos essa decisao e:

```text
datasets/generated/animals_synthetic_structured_v3
```

Os datasets anteriores nao devem ser apagados, porque continuam uteis para auditoria e comparacao:

```text
datasets/generated/animals_synthetic
datasets/generated/animals_synthetic_structured
datasets/generated/animals_synthetic_structured_v2
```

## Evidencias

Foram geradas imagens diagnosticas para auditoria visual:

```text
datasets/generated/diagnostics/capivara_angle_sweep.png
datasets/generated/diagnostics/sapo_angle_sweep.png
datasets/generated/diagnostics/capivara_contact_sheet_structured_v3.png
datasets/generated/diagnostics/sapo_contact_sheet_structured_v3.png
```

A varredura angular indicou:

- sapo: frente mais informativa em torno de `0` graus.
- capivara: perfil/tres-quartos mais informativo em torno de `180` graus para a configuracao estruturada atual.

## Consequencias

Pontos positivos:

- O dataset passa a representar melhor a estrutura visual dos animais.
- O sapo deixa de ser dominado por costas e barriga.
- A capivara passa a privilegiar vistas onde a silhueta e reconhecivel.
- O treino passa a usar imagens mais alinhadas ao objetivo final: reconhecer fotos dos objetos impressos.

Riscos:

- Vistas traseiras ficam sub-representadas de proposito.
- Se as fotos reais forem tiradas majoritariamente de costas, o desempenho pode cair.
- A calibracao de angulo depende da orientacao original de cada arquivo Blender.

Validacoes necessarias:

- Treinar com `animals_synthetic_structured_v3`.
- Comparar contra o treino anterior baseado em `animals_synthetic`.
- Testar com fotos reais dos modelos impressos em 3D, com frente, lateral, tres-quartos, perto e longe.
