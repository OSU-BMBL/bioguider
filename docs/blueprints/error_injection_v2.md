# BioGuider Benchmark — Error Injection Phase (v2)

*System block diagram for the `LLMErrorInjector` pipeline, post-refactor (REV A,
2026-04-16). This is the engineering companion to the plain-language
[error_injection_overview.md](error_injection_overview.md).*

The pipeline takes one clean tutorial and a catalog of error categories, uses an
LLM plus a deterministic fallback to inject a known set of labeled errors, and
emits a corrupted document with a manifest that downstream consumers use to score
each model's fixes.

## Table of contents

- [I. Inputs](#i-inputs)
- [II. The LLMErrorInjector pipeline](#ii-the-llmerrorinjector-pipeline)
- [III. Consumers](#iii-consumers)
- [IV. Output artifacts](#iv-output-artifacts)
- [V. Key design notes](#v-key-design-notes)
- [Legend](#legend)

## I. Inputs

| ID | Component | Type | Location | Notes |
| --- | --- | --- | --- | --- |
| I-01 | Target Document | Input | `data/.adalflow/repos/satijalab_seurat/vignettes/de_vignette.Rmd` | Original clean RMarkdown, Seurat DE tutorial (~1.2k lines). |
| I-02 | Category Catalog | Pure | `bioguider/managers/config.py :: ERROR_CATEGORIES` | The full error taxonomy (see counts below). |
| I-03 | `biomed_app` bucket *(new)* | Pure | `ERROR_CATEGORIES["biomed_app"]` | The five new biomed-specific categories added in this refactor. |

**Category catalog (I-02) — 6 buckets, 39 categories (+5 new):**

| Bucket | Category count |
| --- | --- |
| text | 3 |
| structure | 5 |
| code | 3 |
| biology | 17 |
| cli_config | 6 |
| **biomed_app** *(new)* | 5 |

**`biomed_app` bucket (I-03) — the five new categories:**

- `reproducibility_drift`
- `analysis_hyperparam`
- `stat_test_misnaming`
- `annotation_id_space`
- `celltype_marker`

## II. The LLMErrorInjector pipeline

Source: `bioguider/generation/llm_injector.py`. Steps run in order; steps D and E
are validation gates that divert to the deterministic fallback (Step G) on
failure.

| Step | Name | Kind | What it does |
| --- | --- | --- | --- |
| A | Build `INJECTION_PROMPT` | Pure | Assemble system prompt + readme + `project_terms` + keywords. The prompt now emits the 6-bucket taxonomy, including `biomed_app`. |
| B | Primary LLM call | LLM · I/O | `CommonConversation.generate()` → `POST https://bmblx.bmi.osumc.edu/ai/v1/chat/completions` → returns `{corrupted_markdown, errors[]}`. |
| C | `_parse_json_output` | Pure | Strategy 1: `json.loads(raw)`. Strategy 2: regex-extract a fenced `json` block. Strategy 3: balanced-brace scanner. |
| D | `_check_code_blocks_preserved` | Det | Assert the triple-backtick fence count is equal in baseline and corrupted, and likewise the number of `{r,…}` R-chunk openers. On failure, divert to Step G. |
| E | `_validate_corrupted` | Det | Token overlap ≥ 85% baseline ↔ corrupted; keyword set preserved (gene symbols, proper nouns). On failure, divert to Step G. |
| F | `_supplement_errors` *(+5 new)* | Det | Order-sensitive regex pass (order matters — see design notes). |
| G | `_deterministic_inject` *(+5 new)* | Det | Full fallback path invoked when B–E fail. Same rule table as F, applied to the baseline directly. Guarantees ≥ 1 error per category under AC10. |
| Σ | Output tuple | Pure | `(corrupted_text: str, manifest: {errors: [{id, category, rationale, original_snippet, mutated_snippet, …}], skipped?: […]})`. |

**Step F — `_supplement_errors` regex order (order-sensitive):**

1. `analysis_hyperparam` — must run **before** the generic number rule.
2. number / boolean / gene_case (legacy).
3. `reproducibility_drift` — seed/version regex.
4. `stat_test_misnaming` — swap table.
5. `annotation_id_space` — GSE ↔ GSM, ENSG.
6. `celltype_marker` — marker-swap table, with precondition (see design notes).

## III. Consumers

| ID | Component | Type | Location | Notes |
| --- | --- | --- | --- | --- |
| C-01 | Stress Driver | Pure | `system_tests/test_single_file_stress.py :: run_stress_level()` | Calls `.inject(file, level)` per (model × error_level) cell. |
| C-02 | Fixture Writer | I/O | `outputs/single_file_stress/run_<ts>/` | Writes `<file>.level_N.corrupted.Rmd` and `<file>.level_N.manifest.json`. |
| C-03 | Fix-Rate Judge | Pure | `test_single_file_stress.py :: evaluate_fixes()` (L448–531) | Scores fixes; branches by category (see below). |
| C-04 | `_semantic_match` *(new)* | LLM | `test_single_file_stress.py :: _semantic_match(orig, ctx, llm)` | LLM judge, cached by `hash((orig, ctx[:500]))`; invoked only on ambiguous literal matches. |

**Fix-Rate Judge (C-03) branches:**

- *Exact-match branches:* typo, bio_term, function, link, markdown_structure,
  inline_code, duplicate, number, boolean, param_name, comment_typo,
  species_name, gene_case, `reproducibility_drift` *(new)*, `analysis_hyperparam`
  *(new)*, `annotation_id_space` *(new)*.
- *LLM-judge branches:* `stat_test_misnaming` *(new)*, `celltype_marker` *(new)*.

## IV. Output artifacts

Written to `outputs/single_file_stress/run_<ts>/`.

**Data (I/O):**

- `STRESS_TEST_RESULTS.json`
- `STRESS_TEST_TABLE.csv`
- `STRESS_TEST_CATEGORY_DETAIL.csv`
- `STRESS_TEST_REPORT.md`
- `BENCHMARK_MANIFEST.json`
- `*.original.Rmd` (per file)

**Figures (I/O, new)** — rendered by `bioguider/generation/viz.py`, each as `.png`
+ `.pdf` (dpi = 150, DejaVu Sans):

- `fig1_f1_by_error_level`
- `fig2_avg_f1_by_model`
- `fig3_category_heatmap`
- `fig4_fix_rate`
- `fig5_response_time`
- `fig6_fixed_unfixed`

## V. Key design notes

- **Precondition · `celltype_marker`.** Scan the input for any key in
  `{CD4, CD8, FOXP3, RORC, GATA3, TBX21, NKG7, IFNG}`. If there are zero hits,
  skip the category and emit `manifest.skipped = ["celltype_marker"]` with a
  reason. This avoids fabricating markers that were never in the document.
- **Disambiguation · F-step ordering.** The `analysis_hyperparam` supplement
  **must** run before the generic `number` supplement — otherwise the generic
  rule greedily consumes `resolution` / `n_neighbors` targets and starves the
  bio-specific category.
- **Retry wrapper.** `fix_with_model` wraps `client.invoke` in
  `tenacity(stop_after_attempt=5, wait_exponential 4→60s, retry_if_exception_type=RateLimitError)`
  for proxy 429 resilience.
- **Budget envelope (full rerun).** 5 models × 6 levels × (inject + fix + eval)
  ≈ 90 LLM calls; expected cost `$8–15` on the LiteLLM proxy. Require
  `OPENAI_API_KEY` budget ≥ `$20` before dispatch.

**Acceptance criteria tracked:** AC1 smoke · AC2 model round-trip · AC3 injection ·
AC4 categories · AC5 run artifacts · AC6 viz correctness · AC7 legacy removed ·
AC8 token accounting · AC9 no Azure leaks · AC10 deterministic fallback coverage.

## Legend

| Tag | Meaning |
| --- | --- |
| Input | Static input, pre-run state. |
| LLM | Call to the LiteLLM proxy. |
| I/O | Filesystem read/write. |
| Det | Deterministic regex/rule engine. |
| Pure | Pure function, no side effects. |
| *(new)* | Added in the v2 refactor (REV A). |
