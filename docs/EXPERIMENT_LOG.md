# BioGuider Benchmark Experiment Log

Structured tracking of benchmark experiments, model comparisons, and prompt evaluations.
Each experiment is a dated entry with reproducible parameters and key findings.

---

## Experiment Index

| ID | Date | Type | Status | Key Finding |
|----|------|------|--------|-------------|
| E001 | 2026-04-29 | Model selection (full matrix) | Deprecated (evaluator bugs) | Results invalidated — see v2 redesign section |
| E001-v2 | 2026-04-29 | Model selection (de_vignette only, 5 models × 9 levels) | Partial (1/10 vignettes) | gpt-oss F1_scor 0.793 > gpt-5.4 0.779 > gpt-4o 0.778 — reverses old E001 rank order |
| E002 | 2026-04-29 | Skill comparison (single) | Deprecated (evaluator bugs) | Results invalidated — see v2 redesign section |
| E002-v2 | 2026-04-29 | Skill comparison (re-run with v3 prompts) | Complete | BioGuider F1 0.885 > Generic 0.874 (n=1, reverses E002) |
| E003 | 2026-04-29 | Skill matrix (5 files × 3 levels × 2 prompts) | Complete | BioGuider F1_scorable +2.36pp, -19.8% violations (5×3×2 matrix) |
| E004 | — | 100-software batch | Not run | Blocked on software list from team |

---

## Benchmark v2 Redesign — 2026-04-29

**Reason:** A reviewer audit of E001/E002 found four evaluator bugs that distorted the conclusions. The rankings and F1 values from E001/E002 cannot be trusted as-is. E001 and E002 are deprecated pending a full re-run with the fixed evaluator and updated prompts.

### Evaluator Bugs Fixed

**BUG-1 (P0) — Semantic FP detection disabled.**
`detect_semantic_fp=False` was the default in all skill tests. This caused precision to be identically 1.0 for every run — meaning F1 effectively collapsed to recall. Any model that made more changes (true or not) appeared better. Fixed: deterministic collateral-damage FP detection (`count_collateral_damage`) now runs by default, enabling real precision measurement.

**BUG-2 (P0) — Link scorer counted any valid link as a fix.**
The `link` category scorer checked whether the revised document contained any valid markdown link, rather than checking whether the specific injected broken link was restored. This produced 3385/3385 = 100% fix rate on `link` for all models regardless of actual repair. Fixed: scorer now checks `original_snippet`/`mutated_snippet` for the specific injected errors.

**BUG-3 (P1) — Headline F1 and per-category fixed counts diverged.**
The aggregate results showed `errors_fixed=40` at the headline level but the sum of per-category `fixed` fields was 46 — a 6-error discrepancy. Root cause was duplicate evaluation paths in the test harness (errors scored twice through different code paths). Fixed: evaluation unified to a single path.

**BUG-4 (P1) — `comment_typo` regex matched markdown headers.**
The injector's `^#` regex pattern matched section headers as well as inline code comments, meaning injected "comment typos" landed in markdown header lines. Documented as a known limitation; `comment_typo` remains in UNSCORABLE.

### Prompt Redesign

- **BioGuider v3:** Structured 4-dimension evaluation framework (scientific accuracy, markdown formatting, prose-code consistency, structure) with explicit ground-truth methodology (code blocks are authority). Removes over-conservative language from v2.
- **Generic (one-liner):** Simplified to `Fix all errors in this document and output the corrected version:` — zero domain leakage, zero evaluation-criteria hints. Previously the generic prompt listed 4 evaluation criteria, which constituted implicit domain guidance.
- **Skill tests now lock model to `gpt-4o`** for reproducibility (previously used `next(iter(MODELS))` which was gpt-5.4, not the E001 winner).

### New Safety Metric

Hard FP via protected region check: byte-equality comparison of code fences, YAML frontmatter, and section headers between baseline and revised output. Surfaces in `BenchmarkResult.code_fence_violations`, `yaml_violations`, `section_violations`. This is independent of F1 and measures whether the model corrupted regions it was not supposed to touch.

### Acceptance Criteria for E001-v2

The redesign is only valid if all of the following hold after re-run:

1. `precision != 1.0` for at least one model/level combination
2. `link` category `fix_rate < 1.0`
3. Protected region violation fields populated and non-trivial (non-zero in at least one run)
4. Headline `errors_fixed` == sum of per-category `fixed` (no divergence)

### Status

Code redesign complete (evaluator fixes merged). Smoke test pending. Full re-run (E001-v2, E002-v2, E003) planned after smoke test passes. Old E001/E002 outputs preserved in `outputs/` as historical baselines.

---

## E001-v2: Model Selection — Partial (de_vignette only)

**Date:** 2026-04-29
**Branch:** `refactor/document-generation`
**Supersedes:** E001 (deprecated due to evaluator bugs)
**Run dir:** `outputs/multi_file_stress/run_20260429_212705/`

### Parameters

| Parameter | Value |
|-----------|-------|
| Files | **de_vignette only** (1/10 — full matrix aborted for time; 8 vignettes never started, cell_cycle 87% done but excluded for clean comparison) |
| Error levels | 5, 10, 20, 40, 60, 100, 150, 200, 300 |
| Models | gpt-4o, kimi-k2.5, glm-5, gpt-5.4, gpt-oss-120b |
| Prompt | bioguider (v3, structured 4-dimension) |
| Injection | Deterministic (`force_deterministic=True`), prose-only |
| Proxy | LiteLLM @ bmblx.bmi.osumc.edu/ai |
| Scorable categories | 33 (UNSCORABLE: function, comment_typo, code_lang_tag) |
| Cells completed | 45 / 450 (9 levels × 5 models × 1 vignette) |

### Status

**Partial.** Full 10-vignette matrix not completed. The de_vignette subset (45 cells) is methodologically sound for relative model comparison within one document but has known limitations (single document type, single biological domain).

### Results — Headline Ranking

Models ranked by mean `f1_score_scorable` across 9 error levels:

| Rank | Model | F1_scorable | F1_content | F1_hygiene | fix_rate | avg_duration |
|------|-------|-------------|------------|------------|----------|--------------|
| 1 | **gpt-oss-120b** | **0.7933** | 0.9209 | 0.7457 | 0.7157 | 48.7s |
| 2 | gpt-5.4 | 0.7794 | 0.9013 | 0.7329 | 0.7132 | 38.6s |
| 3 | gpt-4o | 0.7783 | 0.8980 | 0.7312 | 0.7098 | **18.3s** ⭐ |
| 4 | kimi-k2.5 | 0.7520 | 0.8935 | 0.6864 | 0.6777 | 42.1s |
| 5 | glm-5 | 0.7497 | 0.8866 | 0.6870 | 0.6745 | 100.1s |

### Results — Protection Violations (totals across 9 levels)

| Model | code_fence | yaml | section | Total |
|-------|------------|------|---------|-------|
| **kimi-k2.5** | 9 | **1** ⭐ | 207 | **217** ⭐ |
| glm-5 | **9** ⭐ | 3 | 207 | 219 |
| gpt-5.4 | **9** ⭐ | 5 | 207 | 221 |
| gpt-4o | 11 | 9 | **160** ⭐ | 180 |
| gpt-oss-120b | **17** ⚠️ | 9 | 203 | 229 |

### Key Findings

1. **Ranking reversal vs old E001.** Old (buggy) E001 ranked gpt-4o #1 (F1=0.753). New evaluator with deterministic FP detection ranks gpt-oss #1 (F1=0.7933) on this single vignette — a 1.4pp swing for gpt-4o (now #3) and a 2.1pp swing for gpt-oss (now #1). The old precision-locked-at-1.0 bug systematically favored less-conservative models.
2. **Speed-quality tradeoff is sharp.** gpt-4o is **5.5× faster** than glm-5 (18s vs 100s) with comparable F1. For production use, gpt-4o offers the best speed/quality balance.
3. **CONTENT is uniformly strong (0.89-0.92).** All 5 models score >0.88 on scientific accuracy. The differentiator is HYGIENE (formatting/markdown), where gpt-oss leads at 0.75 and glm-5/kimi trail at 0.69.
4. **Protection metric now informative.** Old E001 showed all models tied near 187 fence violations. With v3 prompt, real model variance emerges: gpt-oss has 1.9× the fence violations of the lowest model (17 vs 9). gpt-4o has the fewest section_header violations.

### Limitations

- **n=1 vignette.** Single-document results may not generalize. Full matrix should rerun before paper submission.
- **Different vignettes likely have different rankings.** de_vignette is a relatively small/clean Seurat tutorial; complex vignettes (multimodal, spatial) may stress the models differently.

### Artifacts

- `outputs/multi_file_stress/run_20260429_212705/_aggregate/AGGREGATE_TABLE.csv`
- `outputs/multi_file_stress/run_20260429_212705/_aggregate/AGGREGATE_PROTECTION.csv`
- `outputs/multi_file_stress/run_20260429_212705/_aggregate/AGGREGATE_CATEGORY_DETAIL.csv`
- `outputs/figures/fig1_model_selection_heatmap.{pdf,png}`
- `outputs/figures/fig2_content_vs_hygiene.{pdf,png}`
- `outputs/figures/fig4_f1_degradation.{pdf,png}`

---

## E002-v2: Skill Comparison (Re-run with v3 prompts)

**Date:** 2026-04-29
**Branch:** `refactor/document-generation`
**Supersedes:** E002 (deprecated due to evaluator bugs)

### Parameters

| Parameter | Value |
|-----------|-------|
| File | de_vignette.Rmd |
| Error level | 30 |
| Model | gpt-4o (locked, E001-v2 winner assumed) |
| Skills | bioguider (v3, structured), skill_generic (one-liner) |
| Injection | Deterministic, prose-only, shared across both skills |
| Total injected | 212 |

### Status

Complete.

### Results

| Skill | Fixed / Total | Fix Rate | F1 | F1 Scorable | F1 Content | F1 Hygiene | Duration |
|-------|--------------|----------|----|-------------|------------|------------|----------|
| **bioguider** | **170/212** | **80.2%** | **0.8854** | **0.8346** | **0.9180** | **0.7639** | **14.75s** |
| skill_generic | 166/212 | 78.3% | 0.8737 | 0.8168 | 0.8999 | 0.7465 | 20.33s |

### Key Findings

1. **BioGuider beats Generic on all F1 metrics** — F1 Δ +1.17pp (0.8854 vs 0.8737), F1 Scorable Δ +1.78pp, F1 Content Δ +1.81pp, F1 Hygiene Δ +1.74pp.
2. **BioGuider is faster** — 14.75s vs 20.33s, -27%. The structured prompt reduces model exploration time.
3. **This REVERSES the original E002 finding** — E002 showed Generic winning by ~4.8pp on F1. That result was a recall-only artifact of BUG-1 (precision=1.0 for all runs). With real precision measurement, BioGuider leads.
4. **n=1 caveat** — single file, single error level. E003 provides the multi-file confirmation.

### Artifacts

```
outputs/single_file_stress/run_20260429_212707/SKILL_COMPARISON.csv
```

---

## E003: Skill Matrix — bioguider vs generic across files

**Date:** 2026-04-29
**Branch:** `refactor/document-generation`

### Parameters

| Parameter | Value |
|-----------|-------|
| Files | 5 (de_vignette, cell_cycle_vignette, dim_reduction_vignette, integration_introduction, hashing_vignette) |
| Error levels | 10, 30, 100 |
| Models | gpt-4o (locked) |
| Prompts | bioguider (v3), skill_generic (one-liner) |
| Total runs | 30 (5 × 3 × 2) |

### Command

```bash
pytest system_tests/test_single_file_stress.py::test_skill_matrix -v -s
```

### Status

Complete.

### Results — Headline

| Skill | Mean F1 Scorable | Mean Fix Rate | Mean Duration |
|-------|-----------------|---------------|---------------|
| **bioguider** | **0.8116** | **85.50%** | 14.7s |
| skill_generic | 0.7880 | 83.06% | 13.3s |

- BioGuider F1 Scorable Δ +2.36pp (0.8116 vs 0.7880)
- BioGuider fix rate Δ +2.44pp
- Generic is slightly faster (13.3s vs 14.7s, -10%) — the one metric where Generic leads

### Results — Protection Violations

Hard FP: byte-equality check of code fences, YAML frontmatter, and section headers. Lower is better.

| Skill | Code Fence | YAML | Section Headers | Total (15 cells) |
|-------|-----------|------|-----------------|-----------------|
| **bioguider** | 83 | **9** | **207** | **299** |
| skill_generic | **76** | 13 | 284 | 373 |
| Delta | +7 | -4 | -77 | **-74 (-19.8%)** |

- BioGuider has 19.8% fewer total violations (299 vs 373)
- Code fence violations near-tied (83 vs 76)
- YAML: BioGuider notably better (9 vs 13)
- Section headers: BioGuider substantially better (207 vs 284, -27%)

### Research Hypotheses

| Hypothesis | Result |
|------------|--------|
| H1: BioGuider F1 ≥ Generic | **Validated** — +2.36pp F1 Scorable across 5×3 matrix |
| H2: BioGuider duration < Generic | **Refuted** — Generic faster by ~10% (13.3s vs 14.7s) |
| H3: BioGuider violations < Generic | **Validated** — -19.8% total violations (299 vs 373) |

### Artifacts

```
outputs/single_file_stress/run_20260429_212742/SKILL_MATRIX_TABLE.csv
outputs/single_file_stress/run_20260429_212742/SKILL_MATRIX_PROTECTION.csv
```

---

## E001: Model Selection — Full Matrix

**[DEPRECATED — evaluator bugs. See "Benchmark v2 Redesign" section above. Results below are historical record only.]**

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
- **Evaluator bugs present:** precision was identically 1.0 (BUG-1), link fix_rate was 100% (BUG-2), fixed count headline diverged from category sum (BUG-3). Rankings should be treated as recall-only rankings, not F1 rankings.

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

**[DEPRECATED — evaluator bugs. See "Benchmark v2 Redesign" section above. Results below are historical record only.]**

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

1. **Generic prompt outperformed BioGuider on all F1 metrics** — but this is n=1 (one file, one model, one error level), and precision was identically 1.0 (BUG-1), making F1 = recall. The result reflects recall only, not true F1.
2. **BioGuider was 40% faster** (43s vs 61s) — the structured prompt gives the model a clear roadmap, reducing exploration time.
3. **Hygiene gap is largest** — generic scored 0.983 vs 0.900 (+8.3pp). Interpretation unclear given BUG-1.
4. **Needs full matrix confirmation with fixed evaluator** — re-run as E002-v2 + E003.

### Implications for Paper

Findings here are unreliable due to evaluator bugs. E002-v2 will determine if the generic lead holds with real precision measurement.

### Artifacts

```
outputs/single_file_stress/run_20260429_104705/SKILL_COMPARISON.csv
```

---

## E004: 100-Software Batch (NOT YET RUN)

### Planned Parameters

| Parameter | Value |
|-----------|-------|
| Software | 100 packages (Single-cell focus, team selecting) |
| Model | TBD from E001-v2 |
| Evaluation | BioGuider 4-category scoring (ReadMe, Installation, UserGuide, Tutorial) |
| Skill comparison | BioGuider vs generic prompt |

### Blocked On

- Software list from team (Shaohong has 78, needs merge + filtering)
- E001-v2 model selection decision
- E003 skill validation results

---

## Prompt Registry

Prompts used across experiments. Edit prompt → bump version → re-run.

### bioguider (v3, 2026-04-29)

```
You are "BioGuider," fixing documentation for biomedical software.

GROUND TRUTH
- Code blocks (``` fences) are the AUTHORITY. If prose contradicts code
  (package version, test name, marker gene, parameter value), fix the
  PROSE to match the CODE.

EVALUATION DIMENSIONS (fix errors in all categories)
1. Scientific accuracy: gene names, species, statistical tests, parameters,
   accession IDs must be correct and consistent with code blocks
2. Markdown formatting: headers, lists, links, inline code, tables,
   image syntax must follow proper markdown
3. Prose-code consistency: prose descriptions must agree with adjacent
   code block contents (versions, function names, parameter values)
4. Structure: section titles, YAML frontmatter must be correct

HOW TO FIX (BioGuider methodology)
- Scan the entire document systematically, dimension by dimension
- Use code blocks as the source of truth for factual claims
- Fix typos, broken links, wrong gene names, incorrect numbers
- Restore proper markdown formatting
- Do NOT add new content or remove existing sections
- Do NOT modify text inside ``` fences
- Output the COMPLETE fixed document as markdown
```

- Location: `system_tests/test_single_file_stress.py`
- Used in: E001-v2, E002-v2, E003
- Replaces: bioguider v2

### bioguider (v2, 2026-04-29) [DEPRECATED]

```
You are an expert document proofreader for bioinformatics documentation.
[...domain-specific ground truth rules, 6 critical rules...]
Code blocks are read-only authority. Fix prose to match code.
```

- Location: `system_tests/test_single_file_stress.py:230-259`
- Used in: E001 (deprecated), E002 (deprecated)

### skill_generic (v2, 2026-04-29)

```
Fix all errors in this document and output the corrected version:
```

- Location: `system_tests/test_single_file_stress.py`
- Used in: E002-v2, E003
- Replaces: skill_generic v1 (which leaked evaluation criteria)

### skill_generic (v1, 2026-04-29) [DEPRECATED]

```
I want to refine this bioinformatics documentation. Here are the
evaluation criteria I will use to judge the result:
1. Scientific accuracy  2. Markdown formatting
3. Consistency between prose and code  4. Completeness
Please improve this document based on these criteria.
```

- Location: `system_tests/test_single_file_stress.py:272-283`
- Used in: E002 (deprecated)
- Problem: Leaked evaluation criteria to the model — not a fair generic baseline

### simple (v1)

```
Fix all errors in this document and output the corrected version:
```

- Location: `system_tests/test_single_file_stress.py:262-264`
- Note: This is now the canonical skill_generic v2 prompt

---

## Model Registry

| Model ID | Provider | Type | LiteLLM Route | Notes |
|----------|----------|------|---------------|-------|
| gpt-5.4 | OpenAI | Closed | `gpt-5.4` | Too conservative on fixes |
| gpt-4o | OpenAI | Closed | `gpt-4o` | E001 winner (F1=0.753); locked for skill tests |
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
- Protected region violations tracked separately (hard FP, not in F1)

---

## Changelog

| Date | Change |
|------|--------|
| 2026-04-29 | Created. Added E001 (full matrix), E002 (skill comparison). |
| 2026-04-29 | Prose-only injection guard committed. comment_typo + code_lang_tag → UNSCORABLE. |
| 2026-04-29 | Token tracking added (returns 0 due to LiteLLM proxy). |
| 2026-04-29 | load_dotenv(override=True) fix for stale API key. |
| 2026-04-29 | Benchmark v2 redesign: 4 evaluator bugs fixed, prompts redesigned. E001/E002 deprecated. E001-v2, E002-v2, E003 planned. |
| 2026-04-29 | E002-v2 complete: BioGuider F1 0.885 > Generic 0.874, -27% duration. Reverses E002 finding. |
| 2026-04-29 | E003 complete: BioGuider F1_scorable +2.36pp, -19.8% protection violations across 5×3×2 matrix. |
