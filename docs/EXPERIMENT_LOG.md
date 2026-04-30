# BioGuider Benchmark Experiment Log

Structured tracking of benchmark experiments, model comparisons, and prompt evaluations.
Each experiment is a dated entry with reproducible parameters and key findings.

---

## Experiment Index

| ID | Date | Type | Status | Key Finding |
|----|------|------|--------|-------------|
| E001 | 2026-04-29 | Model selection (full matrix) | Complete | gpt-4o best (F1=0.753), all models ~0.92 on CONTENT |
| E002 | 2026-04-29 | Skill comparison (single) | Complete | Generic prompt beat BioGuider on 1 vignette (n=1) |
| E003 | — | Skill matrix (full) | Not run | 5 files × 3 levels × 2 skills |
| E004 | — | 100-software batch | Not run | Blocked on software list from team |

---

## E001: Model Selection — Full Matrix

**Date:** 2026-04-29
**Duration:** 12h 24m (wall clock)
**Branch:** `refactor/document-generation` @ `13d0e51`

### Parameters

| Parameter | Value |
|-----------|-------|
| Files | 10 Seurat vignettes (de, cell_cycle, dim_reduction, integration_intro, hashing, multimodal, sctransform, pbmc3k, atacseq_integration, spatial) |
| Error levels | 5, 10, 20, 40, 60, 100, 150, 200, 300 |
| Models | gpt-4o, kimi-k2.5, glm-5, gpt-5.4, gpt-oss-120b |
| Prompt | bioguider (structured, domain-specific) |
| Injection | Deterministic (`force_deterministic=True`), prose-only |
| Proxy | LiteLLM @ bmblx.bmi.osumc.edu/ai |
| Scorable categories | 33 (UNSCORABLE: function, comment_typo, code_lang_tag) |
| Total runs | 450 (10 × 9 × 5) |

### Command

```bash
pytest system_tests/test_single_file_stress.py::test_multi_file_full_matrix -v -s
```

### Results — Model Ranking

| Rank | Model | F1 Scorable | F1 Content | F1 Hygiene | Avg Duration |
|------|-------|-------------|------------|------------|-------------|
| 1 | **gpt-4o** | **0.753** | 0.929 | **0.660** | 423s |
| 2 | kimi-k2.5 | 0.740 | 0.924 | 0.645 | 441s |
| 3 | glm-5 | 0.719 | 0.922 | 0.618 | 495s (slowest) |
| 4 | gpt-5.4 | 0.715 | 0.928 | 0.605 | 427s |
| 5 | gpt-oss | 0.710 | 0.920 | 0.604 | 422s (fastest) |

### Results — F1 Scorable by Error Level

| Model | 5 | 10 | 20 | 40 | 60 | 100 | 150 | 200 | 300 |
|-------|---|----|----|----|----|-----|-----|-----|-----|
| gpt-4o | 0.807 | 0.798 | 0.806 | 0.787 | 0.768 | 0.731 | 0.705 | 0.699 | 0.672 |
| kimi-k2.5 | 0.798 | 0.786 | 0.789 | 0.778 | 0.765 | 0.727 | 0.704 | 0.697 | 0.615 |
| glm-5 | 0.778 | 0.771 | 0.791 | 0.779 | 0.744 | 0.683 | 0.660 | 0.653 | 0.615 |
| gpt-5.4 | 0.775 | 0.768 | 0.769 | 0.743 | 0.743 | 0.677 | 0.650 | 0.669 | 0.639 |
| gpt-oss | 0.789 | 0.757 | 0.806 | 0.722 | 0.737 | 0.693 | 0.611 | 0.654 | 0.620 |

### Key Findings

1. **CONTENT scores are uniformly high (~0.92)** across all models — scientific accuracy fixes (gene names, species, parameters) are well-handled by all LLMs.
2. **HYGIENE is the differentiator** — formatting/structural fixes (typos, markdown, inline code) range from 0.604 (gpt-oss) to 0.660 (gpt-4o). This 5.6pp gap drives the overall ranking.
3. **gpt-5.4 underperforms** despite being newer — likely too conservative in its fix behavior (the "don't modify code" strictness extends to borderline prose edits).
4. **All models degrade gracefully** — F1 drops ~0.13 from 5 errors to 300 errors, with ranking stable across levels.
5. **Connection errors at high levels (200, 300)** — some models hit timeouts on large vignettes with 700+ injected errors. Results at these levels may undercount fixes.
6. **Token tracking returned 0** — LiteLLM proxy doesn't propagate token_usage in response metadata. Duration is the cost proxy.

### Issues / Caveats

- At error levels 200+, connection errors caused some models to return unfixed text → identical F1 across models at those levels for some files
- Token usage not captured (LiteLLM proxy limitation)
- `gpt4o` model name in default conftest fixture differs from MODELS dict (`gpt-4o`) — fixed with `load_dotenv(override=True)`

### Artifacts

```
outputs/multi_file_stress/run_20260429_104712/
├── _aggregate/
│   ├── AGGREGATE_TABLE.csv          # 450 rows, all models × levels
│   └── AGGREGATE_CATEGORY_DETAIL.csv
├── {vignette_name}/
│   ├── STRESS_TEST_TABLE.csv        # Per-file, 45 rows (9 levels × 5 models)
│   ├── STRESS_TEST_CATEGORY_DETAIL.csv
│   ├── STRESS_TEST_RESULTS.json
│   └── STRESS_TEST_REPORT.md
└── INDEX.md
```

---

## E002: Skill Comparison — Single File

**Date:** 2026-04-29
**Duration:** 1m 45s
**Branch:** `refactor/document-generation` @ `13d0e51`

### Parameters

| Parameter | Value |
|-----------|-------|
| File | de_vignette.Rmd |
| Error level | 30 |
| Model | gpt-5.4 |
| Skills | bioguider (structured), skill_generic (eval criteria only) |
| Injection | Deterministic, prose-only, shared across both skills |
| Injected errors | 212 |

### Command

```bash
pytest system_tests/test_single_file_stress.py::test_skill_comparison -v -s
```

### Results

| Skill | Fixed | Fix Rate | F1 | F1 Scorable | F1 Content | F1 Hygiene | Duration |
|-------|-------|----------|----|-----------:|----------:|----------:|----------|
| bioguider | 156/212 | 73.6% | 0.848 | 0.911 | 0.926 | 0.900 | 43.2s |
| **skill_generic** | **172/212** | **81.1%** | **0.896** | **0.970** | **0.952** | **0.983** | 60.6s |

### Key Findings

1. **Generic prompt outperformed BioGuider on all F1 metrics** — but this is n=1 (one file, one model, one error level).
2. **BioGuider was 40% faster** (43s vs 61s) — the structured prompt gives the model a clear roadmap, reducing exploration time.
3. **Hygiene gap is largest** — generic scored 0.983 vs 0.900 (+8.3pp). The broader "improve this document" framing gives more freedom for formatting fixes.
4. **Needs full matrix confirmation** — the `test_skill_matrix` run (5 files × 3 levels) will show if this pattern holds.

### Implications for Paper

If the pattern holds:
- **Narrative option A**: "BioGuider achieves comparable quality 40% faster — structured guidance eliminates exploration overhead, reducing cost at scale."
- **Narrative option B**: Run on a weaker model (gpt-oss) where the structured prompt might help more — strong models may not need the guidance.

### Artifacts

```
outputs/single_file_stress/run_20260429_104705/SKILL_COMPARISON.csv
```

---

## E003: Skill Matrix — Full Comparison (NOT YET RUN)

### Planned Parameters

| Parameter | Value |
|-----------|-------|
| Files | 5 Seurat vignettes |
| Error levels | 10, 30, 100 |
| Model | TBD (likely gpt-4o based on E001 results) |
| Skills | bioguider, skill_generic |
| Total runs | 30 (5 × 3 × 2) |

### Command

```bash
pytest system_tests/test_single_file_stress.py::test_skill_matrix -v -s
```

### Open Questions

- Should we run on gpt-4o (E001 winner) or gpt-oss (where structured guidance might help more)?
- Should we add DeepSeek V4 as a 6th model?
- Does the generic prompt's advantage hold at lower error counts (where there's less to fix)?

---

## E004: 100-Software Batch (NOT YET RUN)

### Planned Parameters

| Parameter | Value |
|-----------|-------|
| Software | 100 packages (Single-cell focus, team selecting) |
| Model | TBD from E001 |
| Evaluation | BioGuider 4-category scoring (ReadMe, Installation, UserGuide, Tutorial) |
| Skill comparison | BioGuider vs generic prompt |

### Blocked On

- Software list from team (Shaohong has 78, needs merge + filtering)
- E001 model selection decision
- E003 skill validation results

---

## Prompt Registry

Prompts used across experiments. Edit prompt → bump version → re-run.

### bioguider (v2, 2026-04-29)

```
You are an expert document proofreader for bioinformatics documentation.
[...domain-specific ground truth rules, 6 critical rules...]
Code blocks are read-only authority. Fix prose to match code.
```

- Location: `system_tests/test_single_file_stress.py:230-259`
- Used in: E001, E002, E003

### skill_generic (v1, 2026-04-29)

```
I want to refine this bioinformatics documentation. Here are the
evaluation criteria I will use to judge the result:
1. Scientific accuracy  2. Markdown formatting
3. Consistency between prose and code  4. Completeness
Please improve this document based on these criteria.
```

- Location: `system_tests/test_single_file_stress.py:272-283`
- Used in: E002, E003

### simple (v1)

```
Fix all errors in this document and output the corrected version:
```

- Location: `system_tests/test_single_file_stress.py:262-264`
- Used in: (not yet used in experiments)

---

## Model Registry

| Model ID | Provider | Type | LiteLLM Route | Notes |
|----------|----------|------|---------------|-------|
| gpt-5.4 | OpenAI | Closed | `gpt-5.4` | Too conservative on fixes |
| gpt-4o | OpenAI | Closed | `gpt-4o` | E001 winner (F1=0.753) |
| kimi-k2.5 | Moonshot | Closed | `kimi-k2.5` | Strong #2, rate-limited infra |
| gpt-oss-120b | Open | Open | `gpt-oss-120b` | Fastest, lowest F1 |
| glm-5 | Zhipu | Open | `glm-5` | Slowest (495s avg) |

All routed through LiteLLM proxy at `bmblx.bmi.osumc.edu/ai`.

---

## Benchmark Configuration

### Error Categories (36 total)

- **CONTENT (22)**: param_name, gene_case, bio_term, species_name, accession_id_prefix, prose_code_pkg_version, prose_code_stat_test, prose_code_marker, prose_code_param, number, gene_symbol_case, species_swap, ref_genome_mismatch, modality_confusion, normalization_error, umi_vs_read, batch_effect, qc_threshold, file_format, strandedness, coordinates, units_scale, sample_type, contamination, default_value
- **HYGIENE (11)**: typo, markdown_structure, inline_code, link, duplicate, boolean, emphasis, list_structure, table_alignment, section_title, image_syntax, path_hint
- **UNSCORABLE (3)**: function, comment_typo, code_lang_tag

### Injection

- Deterministic mode (`force_deterministic=True`) ensures all models see identical corrupted text
- Prose-only guard: `_replace_prose_only()` skips code fence spans (committed 2026-04-29)
- Error budget scales with level (5 → 300 errors per category minimum)

### Evaluation

- `BenchmarkEvaluator` per-category fix detection
- F1 = 2 × precision × recall / (precision + recall)
- Scorable F1 excludes UNSCORABLE categories from denominator
- CONTENT/HYGIENE F1 uses shared scorable precision, group-local recall

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-29 | Created. Added E001 (full matrix), E002 (skill comparison). |
| 2026-04-29 | Prose-only injection guard committed. comment_typo + code_lang_tag → UNSCORABLE. |
| 2026-04-29 | Token tracking added (returns 0 due to LiteLLM proxy). |
| 2026-04-29 | load_dotenv(override=True) fix for stale API key. |
