# API Reference

This reference focuses on the primary public entry points used to evaluate repositories and generate improved documentation.

## Managers

### `bioguider.managers.evaluation_manager.EvaluationManager`

Orchestrates repository identification and documentation evaluation.

Key methods:

- `prepare_repo(repo_url: str) -> None`
- `identify_project() -> ProjectMetadata`
- `evaluate_readme() -> tuple[dict, list[str]]`
- `evaluate_installation() -> tuple[dict, list[str]]`
- `evaluate_userguide() -> tuple[dict, list[str]]`
- `evaluate_tutorial() -> tuple[dict, list[str]]`
- `evaluate_submission_requirements(readme_files_evaluation, installation_files, installation_evaluation) -> tuple[dict, list[str]]`

Refined evaluation support:

- `prepare_refined_repo(refined_repo_url: str) -> None`
- `identify_refined_project(metadata: dict | None = None) -> ProjectMetadata`
- `evaluate_refined_readme(readme_files: list[str]) -> tuple[dict, list[str]]`
- `evaluate_refined_installation(installation_files: list[str]) -> tuple[dict, list[str]]`
- `evaluate_refined_userguide(userguide_files: list[str]) -> tuple[dict, list[str]]`
- `evaluate_refined_tutorial(tutorial_files: list[str]) -> tuple[dict, list[str]]`

### `bioguider.managers.generation_manager.DocumentationGenerationManager`

Generates improved documentation files from an evaluation report JSON.

Key methods:

- `prepare_repo(repo_url_or_path: str) -> None`
- `run(report_path: str, repo_path: str | None = None, target_files: list[str] | None = None, max_files: int | None = None) -> str`

The `run` method writes generated files and a `GENERATION_REPORT.md` into an output directory and returns the output path.

## Retrieval

### `bioguider.rag.rag.RAG`

Retrieval-augmented component that indexes repository documents and code.

Key methods:

- `initialize_repo(repo_url_or_path: str, access_token: str | None = None) -> None`
- `query_doc(query: str) -> list`
- `query_code(query: str) -> list`

## Utilities

### `bioguider.utils.code_structure_builder.CodeStructureBuilder`

Builds a code structure index (functions/classes and signatures) used by consistency checks.

- `build_code_structure() -> None`

### `bioguider.settings.SettingsManager`

Centralized settings using `pydantic-settings`.

- `get_setting() -> Setting`
- `initialize_with_params(...) -> None`

### `bioguider.agents.agent_utils.get_openai`

Creates a `langchain` LLM instance from environment variables.

- `get_openai() -> BaseChatOpenAI | None`

## Data models

Generation artifacts are represented by:

- `bioguider.generation.models.GenerationManifest`
- `bioguider.generation.models.GenerationReport`
