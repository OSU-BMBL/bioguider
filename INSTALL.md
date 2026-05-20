# Installation

BioGuider is a Python 3.11 project built with Poetry and compatible with pip.

## Prerequisites

- Python 3.11
- An OpenAI or Azure OpenAI API key
- Optional system dependency for `python-magic`:
  - Linux: `libmagic` (package name varies by distro)
  - macOS: `brew install libmagic`

## Install with Poetry (recommended)

```bash
poetry install
poetry shell
```

## Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Environment variables

Set these before running evaluations or generation:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o"
```

Azure OpenAI configuration (optional):

```bash
export OPENAI_API_TYPE="azure"
export OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export OPENAI_DEPLOYMENT_NAME="your-chat-deployment"
export OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME="your-embedding-deployment"
```

Optional token limits:

```bash
export OPENAI_MAX_INPUT_TOKENS="102400"
export OPENAI_MAX_OUTPUT_TOKEN="16384"
```

## Verify installation

```bash
python -c "import bioguider; print('bioguider imported')"
```
