# CLAUDE.md

## Project Overview

BioGuider is an AI-powered Python package that evaluates and improves documentation quality in open-source biomedical software repositories. It uses a multi-agent LLM architecture with three stages: Collect (gather docs), Evaluate (score quality), and Generate (produce improved docs).

## Build & Setup

- **Python**: 3.11+
- **Package manager**: Poetry
- **Install dependencies**: `poetry install`
- **Environment**: Copy `.env` with required Azure OpenAI vars:
  - `OPENAI_API_TYPE`, `OPENAI_API_KEY`, `OPENAI_API_VERSION`, `OPENAI_DEPLOYMENT_NAME`
  - `OPENAI_MODEL`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_MAX_OUTPUT_TOKENS`, `OPENAI_MAX_INPUT_TOKENS`
  - `OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME`

## Running Tests

- **Unit tests** (fast, no LLM calls): `pytest tests/`
- **System/integration tests** (require LLM API keys): `pytest system_tests/`
- **Single test**: `pytest tests/test_agent_utils.py::test_escape_braces`
- Note: some tests require `langchain_deepseek` which may not be installed in all environments

## Project Structure

```
bioguider/
  agents/          # Multi-agent orchestration (collection, evaluation, identification tasks)
  managers/        # High-level workflow managers (EvaluationManager, GenerationManager)
  generation/      # Documentation improvement pipeline
  rag/             # RAG system with FAISS retriever
  database/        # SQLite caching (code structures, file summaries)
  utils/           # Repo analysis, AST parsing (Python/R), file handling, readability metrics
  settings.py      # Pydantic-based configuration (SettingsManager singleton)
  conversation.py  # LLM conversation wrapper
tests/             # Unit tests
system_tests/      # Integration/system tests (require API keys)
data/              # Static data files
```

## Key Architectural Patterns

- **Plan-Execute-Observe loop**: Agents generate plans, execute with tools, observe results, iterate (implemented via LangGraph)
- **Structured LLM output**: Pydantic models for type-safe LLM responses throughout
- **Dual-pass evaluation**: Structured scoring + free-form detailed evaluation
- **Database caching**: `CodeStructureDb` and `SummarizedFilesDb` in SQLite to avoid reprocessing
- **Settings singleton**: `SettingsManager` provides global config access via `bioguider/settings.py`

## Code Conventions

- Pydantic v2 models for all data structures and configuration
- LangChain/LangGraph for LLM orchestration and agent workflows
- Tenacity for retry logic on LLM calls
- Agent tools defined in `bioguider/agents/agent_tools.py`
- Prompt templates in `bioguider/agents/prompt_utils.py`
- Constants and enums in `bioguider/utils/constants.py`
