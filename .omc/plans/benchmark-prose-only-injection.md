# Plan: Prose-Only Injection + Skill Validation Benchmark

**Created:** 2026-04-29
**Branch:** refactor/document-generation
**Context:** April 24 meeting identified systematic scoring bias from code-block injection

---

## Requirements Summary

The benchmark injects errors into documents and measures how well LLMs fix them.
Currently, errors leak into fenced code blocks. Models told "don't modify code"
respond differently (GPT-4 follows, Kimi doesn't), creating unfair comparisons.
Additionally, the meeting requires a second benchmark purpose: prove BioGuider's
structured prompt outperforms a generic prompt under the same evaluation.

## Acceptance Criteria

1. **No injection inside code fences**: After `inject()` returns, every error
   record's `original_snippet` must NOT appear inside a fenced code block in
   the baseline text. Verified by test assertion.
2. **Category reclassification**: `comment_typo` and `code_lang_tag` moved to
   UNSCORABLE. `_validate_category_groups()` passes. 33 scorable categories.
3. **Existing tests pass**: `pytest tests/` green (unit tests, no LLM calls).
4. **Skill comparison harness**: A new test function runs BioGuider prompt vs
   generic prompt on the same corrupted file, produces side-by-side results.
5. **Prompt parity**: Generic prompt (Skill 2) includes evaluation criteria
   but no structured guidance. BioGuider prompt (Skill 1) unchanged.

## Not Building

- Length weighting coefficients (post-hoc stats after 100-sample run)
- Citation correlation analysis (needs software list finalized)
- Website/UI changes (meeting: "不改")
- Evaluation score restructuring (frozen per meeting decision)
- New model integrations (OSS on Azure — needs manual verification)

---

## Implementation Steps

### Phase 1: Prose-Only Injection Guard (P0)

**Step 1.1** — Add `_in_code_fence()` helper to `LLMErrorInjector`
- File: `bioguider/generation/llm_injector.py` (new method, ~20 lines)
- Given a document string and a character offset, return True if the offset
  falls inside a fenced code block. Use `_CODE_FENCE_RE` to build a list of
  (start, end) spans once per injection call.
- Cache the spans on the instance for the duration of one `inject()` call.

**Step 1.2** — Guard `_supplement_errors()` replacements
- File: `bioguider/generation/llm_injector.py:506-1110`
- Before every `corrupted.replace(orig, mut, 1)` call, find the match offset
  via `corrupted.find(orig)` and check `_in_code_fence(baseline, offset)`.
  Skip if inside a fence.
- Affects: typo (line 610, 634), function (line 733, 763), gene_case,
  species_name, bio_term, number, boolean, comment_typo, and all other
  supplement blocks that use `corrupted.replace()`.
- Implementation: wrap `corrupted.replace(orig, mut, 1)` in a helper
  `_replace_prose_only(text, orig, mut, fence_spans)` that finds the first
  occurrence NOT inside a fence span.

**Step 1.3** — Guard `_deterministic_inject()` regex matches
- File: `bioguider/generation/llm_injector.py:369-504`
- The prose_code_* categories already search `_prose_region()` for anchors
  (good). But `_deterministic_inject()` itself searches the full text for
  some patterns. Ensure all regex matches are filtered through fence spans.

**Step 1.4** — Reclassify categories
- File: `bioguider/managers/config.py:164-227`
- Move `comment_typo` from HYGIENE to UNSCORABLE (rationale: code comments
  are inside fences, can't be injected without entering code).
- Move `code_lang_tag` from HYGIENE to UNSCORABLE (rationale: fence delimiter
  is code-adjacent, models shouldn't touch it).
- Update `HYGIENE_CATEGORIES` (remove 2), `UNSCORABLE_CATEGORIES` (add 2).
- `_validate_category_groups()` enforces the invariant automatically.

**Step 1.5** — Add injection-purity test
- File: `tests/test_injection_prose_only.py` (new, ~60 lines)
- Load a sample Seurat vignette with code blocks.
- Run `inject()` with `force_deterministic=True` at multiple error levels.
- Assert: for every error record, `original_snippet` appears in
  `_prose_region(baseline)` and NOT exclusively in code blocks.
- Assert: fenced code blocks are byte-identical before/after injection.

**Step 1.6** — Update BIOGUIDER_PROMPT
- File: `system_tests/test_single_file_stress.py:230-259`
- Remove line 246 ("PRESERVE all code blocks exactly — do not modify text
  inside ``` fences") — no longer needed since there are no code-block errors.
- Replace with: "Code blocks are the AUTHORITY. Do not modify them."
  (Keep the ground-truth instruction but remove the preservation rule that
  created the asymmetric scoring.)

### Phase 2: Skill Validation Harness (P1)

**Step 2.1** — Define SKILL_GENERIC_PROMPT
- File: `system_tests/test_single_file_stress.py` (new constant, ~15 lines)
- Content: "I want to refine this bioinformatics tutorial document. Here are
  the evaluation criteria I will use: [ReadMe completeness, Installation
  clarity, UserGuide coverage, Tutorial accuracy]. Please improve this
  document based on these criteria. Output the complete corrected document."
- This mirrors what the meeting described: evaluation criteria shared, but
  no structured guidance on HOW to fix.

**Step 2.2** — Add `test_skill_comparison()` test function
- File: `system_tests/test_single_file_stress.py` (new function, ~80 lines)
- For a single vignette at error_count=30 (mid-range):
  - Inject errors (deterministic, shared across both skills)
  - Fix with BIOGUIDER_PROMPT (Skill 1) using the default model
  - Fix with SKILL_GENERIC_PROMPT (Skill 2) using the same model
  - Evaluate both with BenchmarkEvaluator
  - Write side-by-side comparison CSV: model, skill, f1_scorable,
    f1_content, f1_hygiene, duration_s, token_count
- Isolation: each fix is a fresh LLM call with no shared conversation.

**Step 2.3** — Add `test_skill_matrix()` for full comparison
- File: `system_tests/test_single_file_stress.py` (new function, ~60 lines)
- Extends Step 2.2 to multiple files × multiple error levels × both skills.
- Produces SKILL_COMPARISON_TABLE.csv with columns: file_stem, model,
  skill, error_count, f1_scorable, f1_content, f1_hygiene, duration_s.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `_prose_region()` misses indented code blocks (no fences) | Errors leak into code | Seurat vignettes use fenced blocks exclusively; add regex for 4-space-indented blocks as secondary check |
| Moving comment_typo to UNSCORABLE reduces category count | Denominator shrinks, F1 inflates slightly | Document the change; 33→35 is a 6% shift, within noise for relative model comparison |
| Generic prompt (Skill 2) accidentally includes too much guidance | Comparison is unfair | Keep it under 5 lines; no domain-specific terms; have a second person review |
| `_replace_prose_only()` has O(n²) complexity | Slow on large docs | Fence spans are computed once per inject() call; replacement is still O(n) per call |

## Verification Steps

1. `pytest tests/` — all unit tests green (includes new test_injection_prose_only.py)
2. `pytest tests/test_category_groups.py` — validates CONTENT/HYGIENE/UNSCORABLE partition
3. Manual spot-check: run `inject()` on one Seurat vignette, grep error records
   for any snippet that appears inside ``` fences
4. Run `test_skill_comparison()` on one file to verify side-by-side output
5. Compare pre/post F1 on a single stress level to confirm no regression

## Files Touched

| File | Change | Lines (est.) |
|------|--------|-------------|
| `bioguider/generation/llm_injector.py` | Add `_in_code_fence()`, `_replace_prose_only()`, guard all supplement blocks | +40, ~20 modified |
| `bioguider/managers/config.py` | Move comment_typo + code_lang_tag to UNSCORABLE | ~6 modified |
| `system_tests/test_single_file_stress.py` | Update BIOGUIDER_PROMPT, add SKILL_GENERIC_PROMPT, add 2 test functions | +160 |
| `tests/test_injection_prose_only.py` | New test file | +60 |
| Total | | +260, ~26 modified |

4 files modified, 1 new file. Under the 8-file threshold.
