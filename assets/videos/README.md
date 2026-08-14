# Vídeos Fonte por Classe

Deposite nesta pasta os arquivos de vídeo de cada classe de animal usados para a geração automática do dataset de imagens.

## Estrutura Esperada

Organize os vídeos em subdiretórios que correspondem aos nomes das classes:

```text
assets/videos/
  arara/
    01.mp4
  boto/
    01.mp4
  capivara/
    01.mp4
    02.mp4
  onca/
    01.mp4
  sapo/
    01.mp4
```

## Formatos Suportados

- `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

## Geração do Dataset

Para extrair os frames dos vídeos, aplicar aumentos de dados na GPU (sem recorte) e dividir em treino/validação/teste:

```bash
uv run base-generate-video-dataset -opt options/generate_video_animals.yml
```

Para teste reduzido em versão rápida (2 frames por vídeo):

```bash
uv run base-generate-video-dataset -opt options/generate_video_animals.yml --max-frames 2
```

*Os arquivos de vídeo desta pasta são ignorados pelo Git devido ao seu tamanho.*
