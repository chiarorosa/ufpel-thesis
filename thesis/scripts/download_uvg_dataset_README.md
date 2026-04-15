# Download do Conjunto de Dados UVG

Este script automatiza o download do conjunto de dados
[UVG (Ultra Video Group)](http://ultravideo.fi/) a partir de um arquivo ZIP
compartilhado via Google Drive.

## Pré-requisitos

```bash
pip install requests tqdm
```

## Uso básico

```bash
python -m thesis.scripts.download_uvg_dataset \
    --url "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing"
```

O conteúdo do ZIP será extraído automaticamente em `thesis/uvg/`.

## Opções da CLI

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--url` | `$UVG_GDRIVE_URL` | URL de compartilhamento público do Google Drive |
| `--dest` | `thesis/uvg` | Diretório de destino para extração |
| `--force` | `False` | Re-faz o download mesmo que o ZIP já exista |
| `--no-extract` | — | Baixa o ZIP sem extrair |
| `--zip-name` | `uvg_dataset.zip` | Nome do arquivo ZIP local |

## Via variável de ambiente

```bash
export UVG_GDRIVE_URL="https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing"
python -m thesis.scripts.download_uvg_dataset
```

## Comportamento idempotente

Se o arquivo `uvg_dataset.zip` já existir no diretório de destino,
o download é ignorado e apenas a extração é re-executada (caso necessário).
Use `--force` para forçar um novo download.

## Fluxo interno

```
URL do Google Drive
       │
       ▼
 Extrai File ID
       │
       ▼
Monta URL de download direto (bypass do aviso de vírus)
       │
       ▼
  Streaming com barra de progresso (tqdm)
       │
       ▼
  Valida integridade do ZIP
       │
       ▼
  Extrai em thesis/uvg/
```
