# BioGuider

BioGuider is an AI-powered, multi-agent system for evaluating and improving documentation in open-source biomedical software repositories. It combines repository collection, quality evaluation, and automated document generation into a repeatable pipeline.

## What it does

- Collects documentation-relevant files (README, installation guides, user guides, tutorials).
- Evaluates documentation quality for completeness, clarity, and reproducibility.
- Verifies documentation against code structure to detect API mismatches.
- Generates improved documentation based on evaluation reports.

## Architecture at a glance

BioGuider organizes work into two core modules:

- Collect Module: discovers and summarizes documentation artifacts.
- Evaluation Module: scores documents and checks code-documentation consistency.

A separate generation pipeline turns evaluation reports into updated documents.

Key entry points:

- `EvaluationManager` orchestrates identification and evaluation tasks.
- `DocumentationGenerationManager` turns evaluation reports into updated files.
- `RAG` provides retrieval over repository documents and code.

## Quickstart

1) Install dependencies (see `INSTALL.md`).
2) Set LLM credentials in your environment.
3) Run an evaluation and generate improved docs.

Example:

```python
from bioguider.agents.agent_utils import get_openai
from bioguider.managers.evaluation_manager import EvaluationManager
from bioguider.managers.generation_manager import DocumentationGenerationManager

def step_callback(step_name=None, step_output=None):
    if step_name:
        print(f"[{step_name}] {step_output or ''}")

llm = get_openai()

# Evaluate a repository
eval_mgr = EvaluationManager(llm, step_callback)
eval_mgr.prepare_repo("https://github.com/OSU-BMBL/deepmaps")
metadata = eval_mgr.identify_project()
readme_eval, readme_files = eval_mgr.evaluate_readme()
install_eval, install_files = eval_mgr.evaluate_installation()
userguide_eval, userguide_files = eval_mgr.evaluate_userguide()
tutorial_eval, tutorial_files = eval_mgr.evaluate_tutorial()

# Generate documentation from an evaluation report JSON
gen_mgr = DocumentationGenerationManager(llm, step_callback)
gen_mgr.prepare_repo("/path/to/local/repo")
output_dir = gen_mgr.run(
    report_path="logs/scanpy_evaluation_results_20250926.json",
    repo_path="/path/to/local/repo",
)
print("Generated files in:", output_dir)
```

## Configuration

BioGuider reads LLM configuration from environment variables:

- `OPENAI_API_KEY`: API key for OpenAI or Azure OpenAI.
- `OPENAI_MODEL`: model name (default used in code is `gpt-4o`).
- `OPENAI_API_TYPE`: set to `azure` to use Azure OpenAI.
- `OPENAI_API_VERSION`: Azure OpenAI API version.
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL.
- `OPENAI_DEPLOYMENT_NAME`: Azure deployment name for chat models.
- `OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME`: Azure embedding deployment for RAG.
- `OPENAI_MAX_INPUT_TOKENS`: optional input token cap.
- `OPENAI_MAX_OUTPUT_TOKEN`: optional output token cap.

## Repository layout

- `bioguider/agents`: collection and evaluation agents, prompts, tools.
- `bioguider/managers`: orchestration for evaluation and generation.
- `bioguider/generation`: planning, rendering, and LLM generation logic.
- `bioguider/rag`: retrieval-augmented document/code search.
- `bioguider/database`: summarized file and code structure databases.
- `system_tests/`: integration tests and examples.

## Documentation

- API reference: `docs/api-docs/README.md`
- Vignettes and walkthroughs: `docs/vignettes/README.md`
- Installation guide: `INSTALL.md`

## Testing

Run unit and system tests with:

```bash
pytest
```

## License

MIT License. See `LICENSE`.
