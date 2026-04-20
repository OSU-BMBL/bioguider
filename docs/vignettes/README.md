# Vignettes

These walkthroughs show how to run BioGuider end-to-end with minimal setup.

## Vignette 1: Evaluate a repository

```python
from bioguider.agents.agent_utils import get_openai
from bioguider.managers.evaluation_manager import EvaluationManager

def step_callback(step_name=None, step_output=None):
    if step_name:
        print(f"[{step_name}] {step_output or ''}")

llm = get_openai()
mgr = EvaluationManager(llm, step_callback)
mgr.prepare_repo("https://github.com/OSU-BMBL/deepmaps")

metadata = mgr.identify_project()
readme_eval, readme_files = mgr.evaluate_readme()
install_eval, install_files = mgr.evaluate_installation()
userguide_eval, userguide_files = mgr.evaluate_userguide()
tutorial_eval, tutorial_files = mgr.evaluate_tutorial()
```

## Vignette 2: Generate improved documentation from a report

You can feed the generation manager an evaluation report JSON file produced by your pipeline.

```python
from bioguider.agents.agent_utils import get_openai
from bioguider.managers.generation_manager import DocumentationGenerationManager

def step_callback(step_name=None, step_output=None):
    if step_name:
        print(f"[{step_name}] {step_output or ''}")

llm = get_openai()
gen = DocumentationGenerationManager(llm, step_callback)
gen.prepare_repo("/path/to/local/repo")

output_dir = gen.run(
    report_path="logs/scanpy_evaluation_results_20250926.json",
    repo_path="/path/to/local/repo",
)
print("Output directory:", output_dir)
```

## Vignette 3: Limit generation to specific files

```python
target_files = ["README.md", "INSTALL.md"]

output_dir = gen.run(
    report_path="logs/scanpy_evaluation_results_20250926.json",
    repo_path="/path/to/local/repo",
    target_files=target_files,
    max_files=len(target_files),
)
```

## Vignette 4: Build a code structure index

The consistency checker uses a code structure database. You can build it directly:

```python
from pathlib import Path
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.code_structure_builder import CodeStructureBuilder

repo_path = "/path/to/local/repo"
author = "local"
name = "repo"

code_db = CodeStructureDb(author, name)
builder = CodeStructureBuilder(
    repo_path=repo_path,
    gitignore_path=Path(repo_path, ".gitignore"),
    code_structure_db=code_db,
)
builder.build_code_structure()
```
