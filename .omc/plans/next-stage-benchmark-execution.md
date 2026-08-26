# Plan: Next-Stage Benchmark Execution

**Created:** 2026-04-29
**Branch:** refactor/document-generation
**Depends on:** `benchmark-prose-only-injection.md` (completed this session)

---

## Requirements Summary

With the injection bug fixed and CONTENT/HYGIENE split committed, the next
stage executes the actual benchmarks: (1) model selection across 100 software
samples, (2) skill validation proving BioGuider outperforms a generic prompt,
(3) document length statistics for bias correction, and (4) citation
correlation analysis on the Single-cell subset. Speed is paramount — the
meeting set a 6-month publishing window.

## Not Building

- Evaluation score restructuring (frozen per meeting decision)
- Website/UI changes ("不改")
- New model integrations (all 5 models already on LiteLLM proxy)
- Azure-specific OSS setup (already on vp03 LiteLLM)

---

## Workstream A: Purpose-1 — Model Selection Benchmark (P0)

### Step A.1 — Add token tracking to fix_with_model()
- File: `system_tests/test_single_file_stress.py:356-434`
- The ChatOpenAI response object has `response_metadata.token_usage` with
  `prompt_tokens`, `completion_tokens`, `total_tokens`. Capture these and
  return alongside the fixed content.
- Add `token_usage` field to `StressLevelResult` dataclass (line ~92).
- Add `prompt_tokens`, `completion_tokens`, `total_tokens` columns to
  `save_results()` CSV output (line ~806).
- Acceptance: CSV has token columns; values are non-zero for all models.

### Step A.2 — Add glm-5 to MODELS dict
- File: `system_tests/test_single_file_stress.py:312-321`
- Add `"glm-5": {"type": "litellm", "model": "glm-5"}` to MODELS dict.
- Meeting mentioned 6 models total. Current: gpt-5.4, kimi-k2.5, gpt-oss,
  gpt-4o. Add glm-5. The 6th can be DeepSeek if needed.
- Acceptance: `len(MODELS) >= 5`.

### Step A.3 — Build heatmap figure function
- File: `bioguider/generation/viz.py` (extend `_RescoredPlotter`)
- Meeting: "一张图就够了...一个heatmap搞定" — one heatmap per metric.
- Add `fig_heatmap()` method: rows = models, columns = error levels,
  cell value = F1 score (scorable), color-coded.
- Produce 3 variants: `heatmap_scorable.png`, `heatmap_content.png`,
  `heatmap_hygiene.png`.
- Acceptance: PNG files render correctly with >=5 models and >=5 error levels.

### Step A.4 — Run the full matrix on Seurat vignettes
- This is a manual execution step (costs LLM tokens).
- Command: `pytest system_tests/test_single_file_stress.py::test_multi_file_full_matrix -s`
- Expected: 10 vignettes × 9 levels × 5 models = 450 LLM calls.
- Output: `STRESS_TEST_TABLE.csv`, `AGGREGATE_TABLE.csv`, figures.
- Decision gate: If OSS F1 is within 0.05 of top model, select OSS
  (cheapest + fastest per meeting discussion).

---

## Workstream B: Purpose-2 — Skill Validation Benchmark (P1)

### Step B.1 — Build test_skill_comparison()
- File: `system_tests/test_single_file_stress.py` (new function, ~100 lines)
- For a single vignette at error_count=30:
  - Inject errors (deterministic, shared)
  - Fix with BIOGUIDER_PROMPT (Skill 1)
  - Fix with SKILL_GENERIC_PROMPT (Skill 2)
  - Evaluate both with BenchmarkEvaluator
  - Write `SKILL_COMPARISON.csv`: model, skill, f1_scorable, f1_content,
    f1_hygiene, duration_s, prompt_tokens, completion_tokens
- Isolation: each fix is a fresh LLM call, no shared conversation.
- Acceptance: CSV has 2 rows (one per skill); both have non-zero F1.

### Step B.2 — Build test_skill_matrix()
- File: `system_tests/test_single_file_stress.py` (new function, ~80 lines)
- Extends B.1: 5 vignettes × 3 error levels (10, 30, 100) × 2 skills ×
  1 model (the selected model from A.4).
- Output: `SKILL_MATRIX_TABLE.csv` with columns: file_stem, model, skill,
  error_count, f1_scorable, f1_content, f1_hygiene, duration_s, token_count.
- Acceptance: BioGuider prompt F1 > generic prompt F1 on majority of runs.

### Step B.3 — Skill comparison figure
- File: `bioguider/generation/viz.py` (new function, ~40 lines)
- Grouped bar chart: x = error level, bars = Skill 1 vs Skill 2,
  y = F1 scorable. With error bars from multi-file variance.
- Secondary: stacked token-cost comparison bar.
- Acceptance: PNG renders two distinct bars per error level.

---

## Workstream C: Document Length Statistics (P2)

### Step C.1 — Build doc_length_stats() utility
- File: `bioguider/generation/viz.py` or new `bioguider/analysis/doc_stats.py`
  (~60 lines)
- For each software in the evaluation results JSON:
  - Count number of user guide files
  - Count number of tutorial files
  - Count total word count across all doc files
  - Record the 4 category scores (ReadMe, Installation, UserGuide, Tutorial)
- Output: `DOC_LENGTH_STATS.csv` with columns: software, num_userguides,
  num_tutorials, total_words, readme_score, installation_score,
  userguide_score, tutorial_score.
- Acceptance: CSV has >= 50 rows with non-zero word counts.

### Step C.2 — Length-weighted score correction
- File: same as C.1 (~30 lines)
- Compute per-1000-word error rate: `score_adjusted = score * (total_words / 1000)^alpha`
  where alpha is a tunable parameter (start with 0.5).
- Or simpler: normalize scores by log(total_words).
- Output: additional columns in the CSV: `score_adjusted_readme`, etc.
- Acceptance: Correlation between total_words and adjusted score is weaker
  than correlation between total_words and raw score.

---

## Workstream D: Citation Correlation Analysis (P3)

### Step D.1 — Build single-cell citation analysis
- File: new `bioguider/analysis/citation_analysis.py` (~100 lines)
- Input: DOC_LENGTH_STATS.csv + citation data (citation_per_year, GitHub stars).
- Filter to Single-cell software only.
- Compute: Pearson/Spearman correlation for each score vs citation_per_year.
- Try both linear and exponential fitting (scipy.optimize.curve_fit).
- Output: `CITATION_ANALYSIS.csv` with correlation coefficients, p-values,
  R² for both linear and exponential fits.
- Acceptance: Analysis runs on >= 20 Single-cell packages; at least one
  score-citation pair has p < 0.05.

### Step D.2 — Citation scatter plots
- File: `bioguider/generation/viz.py` or `bioguider/analysis/citation_analysis.py`
  (~40 lines)
- Scatter plot: x = doc score, y = citation_per_year, with regression line.
- One plot per score category. Only Single-cell data points.
- Acceptance: PNG files render with data points and fit curve.

---

## Execution Order & Dependencies

```
A.1 (token tracking) ──┐
A.2 (add glm-5)   ─────┤─→ A.4 (run matrix) ─→ A.3 (heatmap figures)
                        │                    │
B.1 (skill comparison) ─┤                    ↓
B.2 (skill matrix) ─────┤─→ B.3 (skill fig) ─→ Paper Figure 2
                        │
C.1 (doc length stats) ─┤─→ C.2 (correction) ─→ Paper Table supplement
                        │
D.1 (citation analysis) ┤─→ D.2 (scatter) ─→ Paper Figure 3
```

**Parallelism**: A.1+A.2, B.1, C.1 are all independent code tasks — can
be implemented in parallel. A.4 blocks on A.1+A.2. B.2 blocks on B.1.
D.1 blocks on C.1 (needs doc stats CSV).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LiteLLM rate limits on kimi-k2.5 | Batch run times out | Multi-deployment on vp03; fall back to fewer error levels |
| OSS model too old (Aug 2024) to compete | Model selection delayed | Meeting says "只要差不多就行" — if within 0.05 F1, still acceptable |
| Token tracking unavailable for some models | Missing cost comparison | Fall back to measuring duration_s as proxy |
| 100-software list not finalized by team | A.4 blocked on external | Start with Seurat 10-vignette matrix; expand when list arrives |
| Exponential fitting fails on small sample | No correlation found | Report null result; paper says "no significant correlation found in our sample" |

## Verification Steps

1. `pytest tests/` — unit tests remain green (195 pass)
2. Token tracking: run `test_single_file_stress_minimal()` on 1 file,
   verify CSV has `prompt_tokens > 0`
3. Heatmap: spot-check that fig renders >=5 rows and >=5 columns
4. Skill comparison: verify SKILL_COMPARISON.csv has 2 rows per model
5. Doc stats: verify DOC_LENGTH_STATS.csv has word counts > 0

## Files Touched

| File | Change | Est. Lines |
|------|--------|-----------|
| `system_tests/test_single_file_stress.py` | Token tracking, glm-5, test_skill_comparison, test_skill_matrix | +250 |
| `bioguider/generation/viz.py` | Heatmap figure, skill comparison figure | +120 |
| `bioguider/analysis/doc_stats.py` | New: doc length statistics + length weighting | +90 |
| `bioguider/analysis/citation_analysis.py` | New: citation correlation + scatter plots | +140 |
| `bioguider/analysis/__init__.py` | New: package init | +2 |
| Total | | +602, 2 new modules |

6 files, 2 new. Under the 8-file threshold.

## Timeline Estimate

| Task | Effort | Blocked By |
|------|--------|------------|
| A.1 + A.2 (token tracking + glm-5) | 30 min code | Nothing |
| B.1 (skill comparison) | 45 min code | Nothing |
| A.3 (heatmap) | 30 min code | Nothing |
| B.2 (skill matrix) | 20 min code | B.1 |
| B.3 (skill figure) | 20 min code | Nothing |
| C.1 + C.2 (doc stats) | 45 min code | Nothing |
| D.1 + D.2 (citation) | 60 min code | C.1 |
| **A.4 (run matrix)** | **2-4 hours LLM runtime** | A.1, A.2 |
| **Total code** | **~4 hours** | |
| **Total including runs** | **~6-8 hours** | |
