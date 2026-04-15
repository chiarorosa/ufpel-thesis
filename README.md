# Redes Neurais Convolucionais na Redução do Tempo de Codificação da Predição Intraquadro no Codificador AOMedia Video 1

**Solução Algorítmica para Aceleração da Predição Intraquadro no AV1 via Pipeline Hierárquico de Redes Convolucionais com Conv-Adapters [versão 1.0]**

## 1. Visão geral

Este repositório organiza dois eixos principais:

- `videoset/`: armazenamento dos vídeos-fonte em formato YUV.
- `thesis/`: pipeline canônico da tese (preparo de dados, treino, avaliação, validações e manifests de reprodutibilidade).

O fluxo canônico atual segue o contrato de dados de particionamento AV1 e a estratégia `rearrange_exact` para alinhamento entre `partition_frame`, `intra_raw_blocks`, `labels` e `qps`.

## 2. Estrutura de pastas

### 2.1 Estrutura de alto nível

```text
cnn-av1-research/
├── requirements.txt
├── requirements-lock.txt
├── thesis/
│   ├── documents/
│   ├── pipeline/
│   ├── runtime/
│   ├── scripts/
│   ├── runs/
│   └── uvg/
└── videoset/
    └── uvg/
```

### 2.2 Pasta `videoset/`

```text
videoset/
└── uvg/
    └── <sequence>.yuv
```

Finalidade:

- Repositório de vídeos originais usados na extração dos blocos.
- Cada sequência deve estar acessível por nome, por exemplo:
  `videoset/uvg/Beauty_3840x2160_120fps_420_10bit.yuv`.
- Sequências utilizadas https://ultravideo.fi/dataset.html

### 2.3 Pasta `thesis/`

```text
thesis/
├── documents/                 # documentação canônica do runtime
├── pipeline/                  # módulos de ML (backbone, adapters, losses, evaluation)
├── runtime/                   # orquestração canônica, validações e contratos
├── scripts/                   # CLIs principais
├── runs/                      # artefatos gerados por execução (datasets, training, evaluation, manifests)
└── uvg/
    ├── <sequence>/partition_frame_0.txt
    ├── intra_raw_blocks/
    ├── labels/
    └── qps/
```

Pontos importantes:

- `thesis/uvg/<sequence>/partition_frame_0.txt` representa o contrato bruto de particionamento.
- `thesis/runs/<run-name>/` concentra toda a saída de uma execução reproduzível.
- `thesis/runs/<run-name>/manifests/` registra estados por fase (`prepare`, `train`, `evaluate`, `end_to_end`).

## 3. Preparação do ambiente

Instalação e ambiente virtual (referência: Python 3.12.x):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para reprodutibilidade da versão 1.0, recomenda-se usar dependências fixadas:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

### 3.1 Instalação para CPU (recomendada para validação funcional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0
```

### 3.2 Instalação para CUDA (treino acelerado)

Pré-requisitos do sistema:

- Driver NVIDIA instalado e funcional (`nvidia-smi`).
- Compatibilidade entre driver e runtime CUDA utilizado pelo PyTorch.

Instalação:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Se houver incompatibilidade de wheel CUDA no host, reinstale explicitamente PyTorch:

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0
```

### 3.3 Verificação pós-instalação

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Nos exemplos abaixo, é recomendado explicitar o interpretador:

```bash
.venv/bin/python <script.py> <args>
```

### 3.4 Download do conjunto de dados UVG (via ZIP no Google Drive)

Script: `thesis/scripts/bootstrap_uvg_from_drive.py`

Fluxo padrão (recomendado):

```bash
.venv/bin/python thesis/scripts/bootstrap_uvg_from_drive.py
```

O que esse comando faz:

- baixa um arquivo ZIP do Google Drive (link padrão já configurado no script);
- extrai o conteúdo e sincroniza para `thesis/uvg`;
- gera `labels/` e `qps/` automaticamente se estiverem ausentes;
- valida o contrato de dados ao final.

Parâmetros úteis:

- `--drive-url <url>`: usa outro link de ZIP no Google Drive.
- `--fresh-download`: limpa cache temporário antes de baixar novamente.
- `--keep-temp`: mantém arquivos temporários em `thesis/runs/_bootstrap/uvg_drive`.
- `--overwrite`: sobrescreve arquivos já existentes em `thesis/uvg`.
- `--skip-validation`: pula a validação final do contrato.

Exemplo com link explícito:

```bash
.venv/bin/python thesis/scripts/bootstrap_uvg_from_drive.py \
  --drive-url "https://drive.google.com/file/d/1c3uY4yeOgpyc8O2ta2kLMXP2Z5FU5d-8"
```

## 4. Comandos principais

### 4.1 Limpeza de artefatos gerados

Script: `thesis/scripts/clean.py`

Parâmetros principais:

- `--execute`: executa remoção (sem este parâmetro, roda em modo preview).
- `--run-name <nome>`: limpa apenas uma run específica.
- `--no-caches`: ignora limpeza de cache Python.
- `--manifest <path>`: define caminho do relatório de limpeza.

Exemplos:

```bash
# Preview de limpeza global
.venv/bin/python thesis/scripts/clean.py

# Limpeza efetiva de uma run
.venv/bin/python thesis/scripts/clean.py --run-name defesa_r1 --execute
```

### 4.2 Preparo de dados canônico

Script: `thesis/scripts/prepare_data.py`

Parâmetros principais:

- `--run-name <nome>`: identificador da execução em `thesis/runs/`.
- `--block-size {8,16,32,64}`: tamanho de bloco do fluxo.
- `--raw-root <path>`: raiz com `partition_frame_0.txt`.
- `--legacy-base-path <path>`: base legada (`intra_raw_blocks/labels/qps`) quando já existente.
- `--auto-bootstrap-legacy-contract`: gera contrato legado temporário a partir de `thesis/uvg`.
- `--require-intra-raw-blocks`: exige cobertura de blocos em `intra_raw_blocks`.
- `--min-intra-raw-sequences <int>`: mínimo de sequências para validação.
- `--python <path>`: interpretador Python para scripts internos.

Exemplos:

```bash
# Fluxo recomendado (bootstrap automático do contrato legado)
.venv/bin/python thesis/scripts/prepare_data.py \
  --run-name defesa_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --require-intra-raw-blocks \
  --min-intra-raw-sequences 1 \
  --auto-bootstrap-legacy-contract \
  --python .venv/bin/python

# Fluxo com base legada explícita
.venv/bin/python thesis/scripts/prepare_data.py \
  --run-name defesa_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --legacy-base-path thesis/runs/defesa_r1/datasets/legacy_bootstrap \
  --python .venv/bin/python
```

### 4.3 Treino completo (Stage 1, Stage 2, Stage 3)

Script: `thesis/scripts/train_pipeline.py`

Parâmetros principais:

- `--run-name <nome>`
- `--block-size {8,16,32,64}`
- `--device {cuda,cpu}`
- `--epochs-stage2 <int>`
- `--epochs-stage3-rect <int>`
- `--epochs-stage3-ab-binary <int>`
- `--batch-size <int>`
- `--seed <int>`

Exemplo:

```bash
.venv/bin/python thesis/scripts/train_pipeline.py \
  --run-name defesa_r1 \
  --block-size 16 \
  --device cuda \
  --epochs-stage2 100 \
  --epochs-stage3-rect 30 \
  --epochs-stage3-ab-binary 50 \
  --batch-size 128 \
  --seed 42
```

### 4.4 Avaliação hierárquica do pipeline

Script: `thesis/scripts/evaluate_pipeline.py`

Parâmetros principais:

- `--run-name <nome>`
- `--block-size {8,16,32,64}`
- `--split {train,val}`
- `--batch-size <int>`
- `--device {cuda,cpu}`

Exemplo:

```bash
.venv/bin/python thesis/scripts/evaluate_pipeline.py \
  --run-name defesa_r1 \
  --block-size 16 \
  --split val \
  --batch-size 128 \
  --device cuda
```

### 4.5 Execução end-to-end (prepare -> train -> evaluate)

Script: `thesis/scripts/run_pipeline_end_to_end.py`

Parâmetros principais:

- `--run-name <nome>`
- `--block-size {8,16,32,64}`
- `--raw-root <path>`
- `--videos-root <path>`
- `--auto-bootstrap-legacy-contract`
- `--require-intra-raw-blocks`
- `--device {cuda,cpu}`
- `--epochs-stage2 <int>`
- `--epochs-stage3-rect <int>`
- `--epochs-stage3-ab-binary <int>`
- `--batch-size <int>`
- `--split {train,val}`

Exemplo de validação funcional (smoke):

```bash
.venv/bin/python thesis/scripts/run_pipeline_end_to_end.py \
  --run-name e2e_smoke_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --videos-root videoset/uvg \
  --require-intra-raw-blocks \
  --min-intra-raw-sequences 1 \
  --auto-bootstrap-legacy-contract \
  --python .venv/bin/python \
  --device cpu \
  --epochs-stage2 1 \
  --epochs-stage3-rect 1 \
  --epochs-stage3-ab-binary 1 \
  --batch-size 256 \
  --split val
```

Exemplo de execução para relatório final:

```bash
.venv/bin/python thesis/scripts/run_pipeline_end_to_end.py \
  --run-name e2e_final_r1 \
  --block-size 16 \
  --raw-root thesis/uvg \
  --videos-root videoset/uvg \
  --require-intra-raw-blocks \
  --auto-bootstrap-legacy-contract \
  --python .venv/bin/python \
  --device cuda \
  --epochs-stage2 100 \
  --epochs-stage3-rect 30 \
  --epochs-stage3-ab-binary 50 \
  --batch-size 128 \
  --split val
```

### 4.6 Validação do contrato de fluxo bruto

Script: `thesis/scripts/validate_flow.py`

Parâmetros principais:

- `--raw-root <path>`
- `--legacy-base-path <path>`
- `--output-json <path>` (opcional)

Exemplo:

```bash
.venv/bin/python thesis/scripts/validate_flow.py \
  --raw-root thesis/uvg \
  --legacy-base-path thesis/runs/e2e_final_r1/datasets/legacy_bootstrap \
  --output-json thesis/runs/e2e_final_r1/manifests/flow_validation_manual.json
```

## 5. Artefatos esperados por execução

Para cada `run-name`, os artefatos principais devem ser observados em:

- `thesis/runs/<run-name>/datasets/`
- `thesis/runs/<run-name>/training/`
- `thesis/runs/<run-name>/evaluation/pipeline/`
- `thesis/runs/<run-name>/manifests/prepare.json`
- `thesis/runs/<run-name>/manifests/train.json`
- `thesis/runs/<run-name>/manifests/evaluate.json`
- `thesis/runs/<run-name>/manifests/end_to_end.json`

Uma run é considerada operacionalmente válida quando os manifests de fase reportam `"status": "ok"` e a validação de fluxo (`flow_validation.json`) reporta `"ok": true`.

## 6. Referências internas

- Contratos e arquitetura: `thesis/documents/`
- Runtime canônico: `thesis/runtime/canonical.py`
- Entrypoints CLI: `thesis/scripts/`

## 7. Glossário (rodapé)

- **Contrato (de dados/runtime)**: conjunto de regras que define formato, presença, ordem e consistência dos artefatos trocados entre etapas do pipeline.
- **Canônico**: caminho oficial e padronizado do projeto para execução, validação e reprodução de resultados.
