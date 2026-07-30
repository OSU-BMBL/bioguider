# BioGuider

**AI-powered documentation evaluation and generation for biomedical software.**

BioGuider clones a target repository, indexes it with a FAISS-backed RAG, and uses
LLMs to **score** its documentation (README, installation, user guide, tutorial,
submission requirements) against per-category rubrics. It then runs a **generation
pipeline** that plans and applies edits to raise the quality of those documents —
and ships a **benchmark harness** that injects synthetic errors to measure how well
the fixes land.

- **Package:** `bioguider` (v0.2.51) · Python 3.11 · Poetry · MIT license
- **Default LLM path:** Azure OpenAI, with Anthropic / DeepSeek / Gemini / LiteLLM-proxy adapters
- **No CLI or web server** — everything is driven programmatically through three manager classes (see [Usage](#usage)).

---

## Table of contents

- [What it does](#what-it-does)
- [Install](#install)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Evaluate a repository](#1-evaluate-a-repository)
  - [2. Generate improved documentation](#2-generate-improved-documentation)
  - [3. Benchmark the pipeline](#3-benchmark-the-pipeline)
- [Architecture](#architecture)
- [Where things are written](#where-things-are-written)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## What it does

BioGuider addresses a specific problem: biomedical software is often powerful but
under-documented, and "improve the docs" is hard to measure. BioGuider makes it
measurable and repeatable in three stages.

| Stage | Manager | What you get |
|-------|---------|--------------|
| **Evaluate** | `EvaluationManager` | Per-category scores + the candidate files that back each score |
| **Generate** | `DocumentationGenerationManager` | Revised docs, `.original` backups, a manifest, and a human-readable generation report |
| **Benchmark** | `BenchmarkManager` | Precision / recall / F1 / fix-rate for the generation pipeline under injected errors |

Documentation categories: **README**, **installation**, **user guide**, **tutorial**,
**submission requirements**.

---

## Install

BioGuider is managed with [Poetry](https://python-poetry.org/) and targets Python 3.11.

```bash
# clone, then from the repo root:
poetry install          # creates the venv and installs runtime + dev deps
```

A `requirements.txt` is provided for non-Poetry setups, but `pyproject.toml` is the
source of truth.

For a from-scratch, step-by-step walkthrough — including the `libmagic` system
dependency, the `logs/` directory requirement, and a troubleshooting table — see
**[INSTALL.md](INSTALL.md)**.

> **Note:** `pytest` from `tests/` must be run from the repo root — `tests/conftest.py`
> opens `./logs/test.log` unconditionally, so create `logs/` first if it is missing.

---

## Configuration

Runtime config is read from a `.env` file at the repo root (via `python-dotenv` /
`pydantic-settings`). The default Azure OpenAI path requires:

```dotenv
OPENAI_API_TYPE=azure
OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
OPENAI_DEPLOYMENT_NAME=<chat-deployment>
OPENAI_MODEL=<model-name>
OPENAI_API_VERSION=2024-xx-xx
OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME=<embedding-deployment>
OPENAI_MAX_INPUT_TOKENS=...
OPENAI_MAX_OUTPUT_TOKENS=...
```

Optional alternative providers (selected via `LLM_PROVIDER` or a `provider=` argument
to `get_configured_llm`): `CLAUDE_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, plus
the `KIMI_*`, `MINIMAX_*`, and `GPT_OSS_*` groups for OpenAI-shaped proxy endpoints.

See **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** for the full variable reference.

> ⚠️ **Never stage `.env`.** It holds live API keys and is gitignored. If this repo was
> ever shared, rotate the keys.

Two runtime gotchas worth knowing up front:

- Construct settings with `SettingsManager.initialize_with_params(...)`, **not** `Setting()`
  directly — `ProjectSettings.target_repo` defaults to an empty string that fails
  `DirectoryPath` validation.
- RAG FAISS vectors are hard-coded to **256 dimensions** (`bioguider/rag/rag.py`). Your
  embedding deployment must produce 256-dim vectors or queries fail at retriever init.

---

## Usage

There is no entry-point script; you drive BioGuider from Python. An LLM handle is built
with `get_configured_llm()` (reads `LLM_PROVIDER`, defaults to Azure). `step_callback` is
any `callable(step_name=..., step_output=...)` — pass a simple logger, or `None`.

> New here? **[docs/TUTORIAL.md](docs/TUTORIAL.md)** is a runnable end-to-end
> walkthrough — evaluate one repo, serialize a report, generate revised docs, and
> read the output — with cost flags at each LLM step.

### 1. Evaluate a repository

```python
from bioguider.agents.agent_utils import get_configured_llm
from bioguider.managers.evaluation_manager import EvaluationManager

llm = get_configured_llm()          # provider from LLM_PROVIDER env (default: azure)

def step_callback(step_name=None, step_output=None):
    print(f"[{step_name}] {step_output}")

mgr = EvaluationManager(llm, step_callback)
mgr.prepare_repo("https://github.com/gbouras13/pharokka")   # clones + builds FAISS index

readme_eval, readme_files = mgr.evaluate_readme()
install_eval, install_files = mgr.evaluate_installation()
# also: evaluate_userguide(), evaluate_tutorial(), evaluate_submission_requirements(...)
```

Each `evaluate_*` returns `(evaluation_dict, files_list)`. Serialize the dicts to a JSON
report — that report is the input to the generation pipeline below.

### 2. Generate improved documentation

```python
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.managers.config import GenerationConfig

config = GenerationConfig(
    debug_output=False,
    clean_output=True,
    write_originals=True,   # keep .original backups next to revised files
    max_files=None,
    target_files=None,      # or a list to restrict which files are touched
)

gen = DocumentationGenerationManager(llm, step_callback, config=config)
gen.prepare_repo("data/.adalflow/repos/gbouras13_pharokka")   # local path or URL
gen.run(evaluation_report="path/to/evaluation_report.json")
```

Output lands in `outputs/<repo_key>/<timestamp>/` with revised files, `.original`
backups, `manifest.json`, and a human-readable `GENERATION_REPORT.md`.

### 3. Benchmark the pipeline

Runs the whole loop — inject N synthetic errors per category, run generation against the
corrupted repo, score the fixes — and is the machinery behind the token/time studies in
`benchmark/`.

```bash
pytest system_tests/test_comprehensive_benchmark.py     # multi-level stress test
python system_tests/analyze_benchmark_results.py        # summarize the output

# token & wall-time per model across error levels (Pharokka case study):
PHAROKKA_MODELS=gpt-4o,gpt-5.4,kimi-k2.5,glm-5.1,gpt-oss \
PHAROKKA_ERROR_LEVEL=200 PHAROKKA_STRATEGIES=simple \
  pytest benchmark/test_pharokka_pipeline.py -s
```

See **[docs/benchmark-runbook.md](docs/benchmark-runbook.md)** and
**[docs/BENCHMARK_METHODS.md](docs/BENCHMARK_METHODS.md)** for the full protocol.

---

## Architecture

Three orchestrators in `bioguider/managers/` compose small single-responsibility
components; start reading there when changing behavior.

- **`EvaluationManager`** — RAG clone + FAISS index → `CodeStructureBuilder` (AST) and
  `SummarizedFilesDb` caches → `IdentificationTask` (language / project type) → a family
  of `Evaluation*Task` classes, one per doc category.
- **`DocumentationGenerationManager`** — a strict pipeline of `bioguider/generation/`
  components: report loader → suggestion extractor → repo reader → style analyzer →
  change planner → content generator (+ truncation / RMarkdown handling) → cleaner →
  output manager. Edits are typed; `full_replace` merges every suggestion for a file
  into one cohesive document.
- **`BenchmarkManager`** — selects target files, injects errors in parallel
  (`LLMErrorInjector`), runs generation, and scores with `UnifiedMetricsEvaluator`.

Agent capabilities in `bioguider/agents/` follow a **PEO** (Plan → Execute → Observe)
convention: `<name>_plan_step.py`, `<name>_execute_step.py`, `<name>_observe_step.py`,
glued by `<name>_task.py`.

For the full component map, extension points (adding a doc category, adding an agent
capability), and persistence layout, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Where things are written

All of these are gitignored and safe to delete — they regenerate.

| Path | Contents |
|------|----------|
| `data/.adalflow/repos/<author>_<repo>/` | cloned sources + FAISS index |
| `data/*.sqlite` | `SummarizedFilesDb` and `CodeStructureDb` caches |
| `outputs/<author>_<repo>/<timestamp>/` | generation output |
| `outputs/benchmark_<ts>/level_<N>/` | stress-test artifacts |
| `logs/` | pytest + generation logs |
| `bioguider_debug/` | ad-hoc debug dumps (`GenerationConfig.debug_output`) |

---

## Testing

```bash
pytest tests/                       # fast unit tests (run from repo root)
ruff check bioguider/               # lint (defaults, no project config)
pytest system_tests/ -s             # integration: real LLM calls + repo clones — slow, costs money
```

`system_tests/` clone public repos (Seurat, scanpy, …) and spend LLM tokens — start with a
single file rather than the whole directory. Some `tests/` depend on a `root_path` fixture
hard-coded to a specific server path and will only run on that host.

---

## Documentation

- **[INSTALL.md](INSTALL.md)** — step-by-step installation and troubleshooting
- **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — hands-on end-to-end walkthrough (evaluate → generate → read output)
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** — full environment-variable reference
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — component map, extension points, data flow
- **[docs/BENCHMARK_METHODS.md](docs/BENCHMARK_METHODS.md)** — error-injection benchmark methodology
- **[docs/benchmark-runbook.md](docs/benchmark-runbook.md)** — step-by-step benchmark runbook
- **[docs/BENCHMARK_V2_REPORT.md](docs/BENCHMARK_V2_REPORT.md)** — benchmark results report
- **[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)** — running experiment log

---

## Contributing

- Follow the existing module conventions — especially the PEO quadruple for new agent
  capabilities and the small `bioguider/generation/*` components for pipeline work.
- `RepoAgent` (OpenBMB/RepoAgent) is an external repository used as a sample target for the
  evaluation system tests, not a dependency of the package. The `*_RepoAgent` system tests expect a
  clone at the sibling path in `system_tests/conftest.py`'s `root_path` fixture (e.g. `../RepoAgent`).
- Cut releases with `bump2version patch|minor|major` (config in `.bumpversion.cfg`).

## License

MIT © Cankun Wang, Shaohong Feng. See [LICENSE](LICENSE).
