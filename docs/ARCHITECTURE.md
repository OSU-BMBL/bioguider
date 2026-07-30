# Architecture

BioGuider has no CLI or web server. All behavior is driven programmatically through
three orchestrator classes in `bioguider/managers/`. Each composes small,
single-responsibility components and exposes a thin `.run()` / `.evaluate_*()` method.
**Start reading in `bioguider/managers/` when you want to change behavior.**

```
                         ┌──────────────────────────┐
   repo URL / path  ───► │     EvaluationManager     │ ──► evaluation report (JSON)
                         └──────────────────────────┘
                                                            │
                                                            ▼
   evaluation report ──► ┌──────────────────────────┐
   + repo path           │ DocumentationGeneration  │ ──► outputs/<repo>/<ts>/
                         │        Manager           │      revised docs + report
                         └──────────────────────────┘
                                                            ▲
                         ┌──────────────────────────┐       │  (drives generation
   baseline repo    ───► │     BenchmarkManager      │ ──────┘   against corrupted repo,
                         └──────────────────────────┘           then scores the fixes)
```

---

## 1. `EvaluationManager` (`evaluation_manager.py`)

Given a repo URL or local path:

1. **`RAG.initialize_repo()`** clones the repo into
   `data/.adalflow/repos/<author>_<repo>/` and builds FAISS indices — separate **doc**
   and **code** retrievers, 256-dim vectors.
2. **`CodeStructureBuilder`** populates `CodeStructureDb` (an AST tree); per-file
   summaries are cached in `SummarizedFilesDb`. Both are sqlite, keyed by
   `(author, repo_name)`, under `data/`.
3. **`IdentificationTask`** detects primary language, project type, and metadata.
4. A family of **`Evaluation*Task`** classes (one per doc category) collect candidate
   files and call the LLM with the per-category prompt. Each returns
   `(evaluation_dict, files_list)`.
5. **`prepare_refined_repo()`** mirrors step 1 for a second, "refined" repo — used when
   evaluating the output of the generation pipeline against the original.

**Extension point — add a doc category:** subclass `EvaluationTask`
(`bioguider/agents/evaluation_task.py`) and implement `_collect_files` and `_evaluate`.

---

## 2. `DocumentationGenerationManager` (`generation_manager.py`)

Takes an evaluation-report JSON plus a repo path and runs a strict pipeline (documented
in the class docstring). Each step is a small component in `bioguider/generation/`:

```
EvaluationReportLoader        report_loader.py       load + validate the report
      │
SuggestionExtractor           suggestion_extractor.py pull actionable edits per file
      │
RepoReader                    repo_reader.py         read the current doc content
      │
StyleAnalyzer                 style_analyzer.py      infer the repo's doc style
      │
ChangePlanner                 change_planner.py      turn suggestions into typed edits
      │
DocumentRenderer  +  LLMContentGenerator            apply edits / generate content
  document_renderer.py         llm_content_generator.py
      │   (internally: TruncationHandler, RMarkdownProcessor, prompts/)
      │      truncation_handler.py   rmarkdown_processor.py
LLMCleaner                    llm_cleaner.py         final cleanup pass
      │
OutputManager                 output_manager.py      write revised files + manifest
```

**Typed edits.** `full_replace` triggers a single LLM call that merges every suggestion
for a file into one cohesive document. Other edit types are applied section-by-section
through `DocumentRenderer.apply_edit`.

**Output.** Everything lands in `outputs/<repo_key>/<timestamp>/`: revised files,
`.original` backups, `manifest.json`, and a human-readable `GENERATION_REPORT.md`
(written by the nested `GenerationReportWriter`).

**Config.** `GenerationConfig` in `bioguider/managers/config.py` toggles `debug_output`,
`clean_output`, `write_originals`, `max_files`, `target_files`.

---

## 3. `BenchmarkManager` (`benchmark_manager.py`, extends `BaseTestManager`)

The evaluation harness for the generation pipeline:

1. Select target files per category (`FILE_CATEGORIES` in `managers/config.py`).
2. Clone the baseline repo into a tmp directory; extract project terminology.
3. **`inject_errors_parallel`** uses a `ThreadPoolExecutor` to LLM-inject N errors per
   category. `ERROR_CATEGORIES` spans text / structure / code / biology / cli_config —
   ~40 error types total (`LLMErrorInjector`, `bioguider/generation/llm_injector.py`).
4. Run `DocumentationGenerationManager.run()` against the corrupted repo.
5. **`UnifiedMetricsEvaluator`** (`bioguider/generation/unified_metrics.py`) scores the
   fixes with precision / recall / F1 / fix-rate; optional semantic-FP detection uses
   another LLM call.
6. Serialize to `BENCHMARK_MANIFEST.json`, `BENCHMARK_RESULTS.json`,
   `STRESS_TEST_TABLE.csv`, and a markdown summary.

`prepare_model_comparison` / `evaluate_model_comparison` produce a directory layout where
fixes from other models (GPT, Claude, Gemini) can be dropped in for head-to-head
comparison.

See [BENCHMARK_METHODS.md](BENCHMARK_METHODS.md) and [benchmark-runbook.md](benchmark-runbook.md).

---

## Agent task convention (PEO)

Every non-trivial agent capability in `bioguider/agents/` is a **Plan → Execute →
Observe** quadruple of files:

```
<name>_plan_step.py      <name>_execute_step.py      <name>_observe_step.py      <name>_task.py
```

`common_step.py` / `peo_common_step.py` provide the base classes. Examples:
`collection_*`, `identification_*`, `dockergeneration_*`, `consistency_*`.

**Extension point — add an agent capability:** follow the same quadruple.

---

## Persistence / state layout

All of these directories are **gitignored and regeneratable**:

| Path | Contents |
|------|----------|
| `data/.adalflow/repos/<author>_<repo>/` | cloned repo sources + FAISS db |
| `data/*.sqlite` | `SummarizedFilesDb` and `CodeStructureDb` caches |
| `outputs/<author>_<repo>/<timestamp>/` | generation output |
| `outputs/benchmark_<ts>/level_<N>/` | stress-test artifacts |
| `outputs/model_comparison_<ts>/` | multi-model comparison runs |
| `logs/` | pytest and generation logs (`test.log` opened by `tests/conftest.py`) |
| `bioguider_debug/` | ad-hoc debug dumps (`GenerationConfig.debug_output`) |

---

## Gotchas

- **`bioguider/__init__.py` is empty.** Always import from specific submodules
  (`bioguider.managers.evaluation_manager`, `bioguider.generation`, etc.).
- **Never construct `Setting()` directly** — use
  `SettingsManager.initialize_with_params(...)`. See [CONFIGURATION.md](CONFIGURATION.md).
- **FAISS is 256-dim** (`bioguider/rag/rag.py`); the embedding deployment must match.
- **`tests/conftest.py`** hard-codes `./logs/test.log` and a `root_path` fixture pointing
  at a specific server path; `root_path`-dependent tests only run on that host.
- **`system_tests/`** make real LLM calls, clone real repos, and cost money — run them
  deliberately, one file at a time.
- **`DeepSeekConversation.chat`** swallows exceptions and returns them stringified.
- **`RepoAgent`** (OpenBMB/RepoAgent) is an external repository used as a sample target by the `*_RepoAgent` evaluation system tests — not a dependency of the package. It is expected as a sibling clone (see `root_path` in `system_tests/conftest.py`), not a submodule of this repo.
- The generation pipeline has been progressively broken into the small
  `bioguider/generation/*` components; prefer those over any monolithic version still
  referenced by older tests.
