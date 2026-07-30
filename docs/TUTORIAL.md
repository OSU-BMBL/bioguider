# BioGuider Tutorial — from a fresh clone to revised docs

A hands-on, copy-pasteable walkthrough. By the end you will have run one real
repository through the full loop: **evaluate its docs → serialize a report →
generate improved docs → read the output**. Every code block is meant to be run.

> **This tutorial makes real LLM calls.** BioGuider has no offline mode — scoring
> and generation both call a model. The walkthrough is deliberately scoped to a
> *single small repository* and a *single document* so the cost is a few cents to
> a few dollars depending on your provider. Cost checkpoints are flagged inline
> with 💸. If you only want to read, the expected-output blocks show you what each
> step produces without running it.

**Audience:** a developer who has cloned the repo and wants to drive the Python
API. If you have not installed BioGuider yet, do **[INSTALL.md](../INSTALL.md)**
first, then come back here.

---

## Table of contents

- [0. Before you start](#0-before-you-start)
- [1. The 30-second mental model](#1-the-30-second-mental-model)
- [2. Tutorial 1 — Evaluate one repository](#2-tutorial-1--evaluate-one-repository)
- [3. Tutorial 2 — Serialize a full evaluation report](#3-tutorial-2--serialize-a-full-evaluation-report)
- [4. Tutorial 3 — Generate improved documentation](#4-tutorial-3--generate-improved-documentation)
- [5. Tutorial 4 — Peek at the benchmark](#5-tutorial-4--peek-at-the-benchmark)
- [6. Troubleshooting your first run](#6-troubleshooting-your-first-run)
- [7. Where to go next](#7-where-to-go-next)

---

## 0. Before you start

Three things bite on the very first run. Handle them now.

1. **Create `logs/`.** The unit-test harness and several code paths write there.
   From the repo root:

   ```bash
   mkdir -p logs
   ```

2. **Configure `.env`.** You need a working LLM path *and* a 256-dim embedding
   deployment (see the gotcha in step 6). Copy the template and fill it in:

   ```bash
   cp .env.example .env
   # then edit .env — see docs/CONFIGURATION.md for every variable
   ```

   > ⚠️ **Never stage `.env`.** It holds live API keys and is gitignored. If this
   > repo was ever shared, rotate the keys.

3. **Never construct settings by hand.** Throughout BioGuider, build settings via
   `SettingsManager.initialize_with_params(...)`, **not** `Setting()` directly —
   `ProjectSettings.target_repo` defaults to an empty string that fails
   `DirectoryPath` validation. The managers below do this for you; you only need
   to remember it if you drop down to lower-level APIs.

Run everything below from the **repo root**, inside the Poetry environment:

```bash
poetry run python        # or: poetry shell, then python
```

---

## 1. The 30-second mental model

BioGuider is three managers. This tutorial walks the first two; the third is a
research harness you will rarely need day-to-day.

| Manager | What it does | What it writes to disk |
|---------|--------------|------------------------|
| `EvaluationManager` | Clones a repo, indexes it, scores each doc category | `data/.adalflow/repos/<author>_<repo>/` (clone + FAISS), `data/*.sqlite` (caches) |
| `DocumentationGenerationManager` | Takes an evaluation report + repo, plans and applies edits | `outputs/<repo_key>/<timestamp>/` (revised files, `.original` backups, `manifest.json`, `GENERATION_REPORT.md`) |
| `BenchmarkManager` | Injects synthetic errors, runs generation, scores the fixes | `outputs/benchmark_<ts>/` |

The output of *evaluate* (a JSON report) is the input to *generate*. That handoff
is the one piece the reference docs don't spell out — Tutorial 2 builds it
explicitly.

---

## 2. Tutorial 1 — Evaluate one repository

We'll evaluate [`gbouras13/pharokka`](https://github.com/gbouras13/pharokka), a
compact, well-known bacteriophage-annotation tool — small enough to clone and
index quickly.

### 2.1 Build an LLM handle and a manager

```python
from bioguider.agents.agent_utils import get_configured_llm
from bioguider.managers.evaluation_manager import EvaluationManager

# Reads LLM_PROVIDER from .env (defaults to "azure").
# Pass provider="claude" / "deepseek" / "gemini" etc. to override.
llm = get_configured_llm()

def step_callback(step_name=None, step_output=None):
    # Any callable(step_name=..., step_output=...) works. Pass None to stay quiet.
    print(f"[{step_name}] {step_output}")

mgr = EvaluationManager(llm, step_callback)
```

### 2.2 Clone + index the repo

💸 This clones the repo and builds two FAISS indices (one for docs, one for code).
Indexing embeds every chunk, so this is the first place tokens are spent. It is a
one-time cost per repo — the clone and indices are cached under `data/`.

```python
mgr.prepare_repo("https://github.com/gbouras13/pharokka")
```

After this returns you'll find:

```
data/.adalflow/repos/gbouras13_pharokka/   # cloned sources + FAISS index
data/gbouras13_pharokka_code_structure.db  # AST cache (CodeStructureDb)
data/gbouras13_pharokka_summarized_file.db # per-file summary cache (SummarizedFilesDb)
```

> Re-running `prepare_repo` on the same URL reuses these caches. Delete the
> `data/` entries if you want a clean rebuild — they all regenerate.

### 2.3 Score one document

Start with the README. Every `evaluate_*` method returns the same shape:
`(evaluation_dict, files_list)`.

💸 One scoring call (plus a little retrieval).

```python
readme_eval, readme_files = mgr.evaluate_readme()

print("Files considered:", readme_files)
print("Evaluation:", readme_eval)
```

`readme_files` is the list of candidate files the task decided were "the README"
(there can be more than one). `readme_eval` is a dict of per-dimension scores and
reasoning — a qualitative grade (`Poor` / `Fair` / `Good` / `Excellent`) plus
strengths and concrete suggestions per dimension.

The other categories work identically:

```python
install_eval,  install_files  = mgr.evaluate_installation()
userguide_eval, userguide_files = mgr.evaluate_userguide()
tutorial_eval, tutorial_files  = mgr.evaluate_tutorial()
```

> `evaluate_submission_requirements(...)` is different — it takes prior README /
> installation results as arguments because submission readiness is judged
> *relative to* those. See its signature in
> `bioguider/managers/evaluation_manager.py`.

---

## 3. Tutorial 2 — Serialize a full evaluation report

The generation pipeline consumes a **JSON report keyed by file path**, where each
value holds the per-dimension evaluations for that file. This is exactly the shape
of the sample reports in the repo root (e.g. `seurat_tutorial_evaluation.json`):

```json
{
  "vignettes/seurat5_multimodal_vignette.Rmd": {
    "tutorial_evaluation":    { "overall_score": "Good",      "overall_key_strengths": "..." },
    "consistency_evaluation": { "score": "Excellent",         "assessment": "..." }
  }
}
```

To produce your own, collect the `evaluate_*` results into a dict and dump it.
Here is a minimal end-to-end script for the README + installation categories:

```python
import json
from bioguider.agents.agent_utils import get_configured_llm
from bioguider.managers.evaluation_manager import EvaluationManager

llm = get_configured_llm()
mgr = EvaluationManager(llm, step_callback=None)
mgr.prepare_repo("https://github.com/gbouras13/pharokka")

report = {}

readme_eval, readme_files = mgr.evaluate_readme()
for f in readme_files:
    report.setdefault(f, {})["readme_evaluation"] = readme_eval

install_eval, install_files = mgr.evaluate_installation()
for f in install_files:
    report.setdefault(f, {})["installation_evaluation"] = install_eval

with open("pharokka_evaluation.json", "w") as fh:
    json.dump(report, fh, indent=2, default=str)

print("Wrote pharokka_evaluation.json with", len(report), "file entries")
```

> `default=str` keeps the dump robust if an evaluation value isn't natively
> JSON-serializable. Keep the file-path keys **relative to the repo root** — that
> is how the generation pipeline resolves them against the cloned sources.

You now have `pharokka_evaluation.json`, the input to the next step.

---

## 4. Tutorial 3 — Generate improved documentation

Feed the report and the repo to `DocumentationGenerationManager`.

### 4.1 Configure and run

💸 This is the most expensive step: each targeted file triggers at least one
generation call, and `full_replace` edits merge every suggestion for a file into
a single cohesive rewrite. Keep it cheap the first time by limiting to one file
via `target_files` (or `max_files=1`).

```python
from bioguider.managers.generation_manager import DocumentationGenerationManager
from bioguider.managers.config import GenerationConfig

config = GenerationConfig(
    write_originals=True,   # keep a .original backup next to each revised file
    clean_output=True,      # run the LLM cleaner pass on the output
    polish_output=True,     # narrow markdown polish before the cleaner
    max_files=1,            # cap work for a cheap first run
    target_files=None,      # or e.g. ["README.md"] to pin exactly which files
)

gen = DocumentationGenerationManager(llm, step_callback, config=config)

# Point at the already-cloned local copy to avoid re-cloning:
gen.prepare_repo("data/.adalflow/repos/gbouras13_pharokka")

# NOTE: run() takes the report path POSITIONALLY as report_path.
out_dir = gen.run("pharokka_evaluation.json")
print("Generation output written to:", out_dir)
```

> `run()`'s full signature is
> `run(report_path, repo_path=None, target_files=None, max_files=None)`. The
> `target_files` / `max_files` arguments here override the same fields on
> `GenerationConfig`, so you can also skip setting them on the config and pass
> them to `run()` directly.

### 4.2 Read the output

`run()` returns the output directory. Inside `outputs/<repo_key>/<timestamp>/`
you'll find:

```
outputs/gbouras13_pharokka/2026xxxx_xxxxxx/
├── README.md               # the revised document
├── README.md.original      # verbatim backup of the input (write_originals=True)
├── manifest.json           # machine-readable list of every edit applied
└── GENERATION_REPORT.md    # human-readable summary: what changed and why
```

Start with `GENERATION_REPORT.md` — it explains each edit in prose. Then diff the
revised file against its `.original` to see exactly what moved:

```bash
diff outputs/gbouras13_pharokka/*/README.md.original \
     outputs/gbouras13_pharokka/*/README.md
```

`manifest.json` is the same information structured for tooling (edit type, target
file, section, rationale) — useful if you want to script an accept/reject review.

---

## 5. Tutorial 4 — Peek at the benchmark

The benchmark answers a different question: *how well does the generation pipeline
actually fix things?* It injects N known errors into a doc, runs generation against
the corrupted copy, and scores recovery with precision / recall / F1 / fix-rate.

You almost certainly don't need to run it to use BioGuider, and it is
**expensive** — many models × many error levels × (inject + fix + score). Read the
concepts first:

- **[blueprints/error_injection_overview.md](blueprints/error_injection_overview.md)** — the spike-in-recovery idea in plain language.
- **[blueprints/error_injection_v2.md](blueprints/error_injection_v2.md)** — the engineering block diagram.
- **[BENCHMARK_METHODS.md](BENCHMARK_METHODS.md)** and **[benchmark-runbook.md](benchmark-runbook.md)** — the full protocol and commands.

💸 When you *are* ready, the smallest useful run is a single stress test:

```bash
pytest system_tests/test_comprehensive_benchmark.py     # writes outputs/benchmark_<ts>/
python system_tests/analyze_benchmark_results.py        # summarize the results
```

---

## 6. Troubleshooting your first run

| Symptom | Cause | Fix |
|---------|-------|-----|
| Error at `FAISSRetriever` init, or a dimension-mismatch on the first query | RAG vectors are hard-coded to **256 dimensions** (`bioguider/rag/rag.py`), but your embedding deployment returns a different size | Configure `OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME` to a model that produces 256-dim vectors |
| `DirectoryPath` validation error mentioning `target_repo` | Settings built with `Setting()` directly (empty `target_repo`) | Use `SettingsManager.initialize_with_params(...)`; or just use the managers, which do this for you |
| `FileNotFoundError: ./logs/test.log` (running tests) | `tests/conftest.py` opens `./logs/test.log` unconditionally | `mkdir -p logs` and run from the repo root |
| `run()` fails on an `evaluation_report=` keyword | That kwarg doesn't exist | The report path is **positional**: `run("report.json")` |
| Generation touches more files than expected / is slow & pricey | No file cap set | Set `max_files=1` or `target_files=["README.md"]` on the first run |
| Report file paths don't resolve against the repo | Keys stored as absolute paths | Keep report keys **relative to the repo root** |

---

## 7. Where to go next

- **[../README.md](../README.md)** — the reference overview and API surface.
- **[../INSTALL.md](../INSTALL.md)** — installation and setup troubleshooting.
- **[CONFIGURATION.md](CONFIGURATION.md)** — every environment variable explained.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the component map and extension points
  (adding a doc category, adding a PEO agent capability).
- **[BENCHMARK_METHODS.md](BENCHMARK_METHODS.md)** — the benchmark methodology in full.
