# ADR 0004: Uso do bpy sem depender do aplicativo Blender

## Status

Aceito

## Contexto

O projeto precisa renderizar arquivos `.blend` para gerar imagens sinteticas, mas o ambiente informado nao possui o aplicativo Blender instalado. Ainda assim, existe a necessidade de controlar cena, camera, luz, materiais e renderizacao por Python.

O Blender disponibiliza sua API Python como modulo `bpy`. O pacote `bpy` permite executar scripts Python que importam `bpy` sem chamar diretamente o executavel do Blender.

## Decisao

Adotar `bpy` como backend principal do pipeline de renderizacao.

O YAML de geracao deve usar:

```yaml
renderer_backend: python_bpy
```

O script de geracao chama o renderizador com o Python ativo do ambiente. O script de renderizacao importa `bpy`, abre o arquivo do modelo, normaliza a cena, cria camera/luzes, varia parametros e salva imagens.

O backend `blender_executable` fica preservado como fallback para ambientes que tenham o Blender instalado e prefiram executar `blender --background`.

Durante a execucao com `bpy 5.0.1`, o engine `CYCLES` nao ficou disponivel na sessao Python; o engine disponivel foi `BLENDER_EEVEE`. Por isso, o renderer deve:

- tentar habilitar `CYCLES` quando solicitado;
- cair para `BLENDER_EEVEE` com aviso quando `CYCLES` nao estiver disponivel;
- aceitar aliases como `BLENDER_EEVEE_NEXT` quando o ambiente expuser apenas `BLENDER_EEVEE`.

A baseline estruturada atual usa explicitamente:

```yaml
render_engine: BLENDER_EEVEE
```

## Consequencias

Pontos positivos:

- O pipeline fica mais alinhado ao ambiente atual, que possui Python mas nao Blender instalado como aplicativo.
- Mantem controle programatico da cena usando a API oficial do Blender.
- Preserva fallback para Blender headless quando necessario.

Riscos:

- `bpy` e uma dependencia pesada.
- A disponibilidade do wheel depende de versao de Python, sistema operacional e arquitetura.
- A instalacao pode exigir indice extra do Blender em algumas versoes.
- Engines de renderizacao disponiveis podem variar por versao do pacote `bpy`.

Regra de implementacao:

- `renderer_backend: python_bpy` deve ser o padrao dos YAMLs oficiais.
- Scripts devem falhar com mensagem clara quando `bpy` nao estiver instalado.
