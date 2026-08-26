# Plan — BioGuider scorer redesign + offline rescore

**Date**: 2026-04-24
**Mode**: Direct, RALPLAN-DR short
**Execution target**: existing run `outputs/multi_file_stress/run_20260424_025546/` — no new LLM calls
**Scope**: benchmark scorer only; do not touch the generation pipeline or the injection path

---

## Requirements summary

Two defects were found in the just-completed 360-cell benchmark:

1. **Scorer bug** — `_check_inline_code` uses a substring-containment predicate (`rewrapped in fixed AND mut not in fixed`) where `rewrapped` always contains `mut` as substring. This pins the category's fix-rate to exactly 0 for every model, every level, every file. Present in three copy-pasted call sites:
   - `bioguider/generation/unified_metrics.py:524-531`
   - `bioguider/generation/benchmark_metrics.py:464-469`
   - `system_tests/test_single_file_stress.py:493-497`

2. **Category-set design** — `ERROR_CATEGORIES` (`bioguider/managers/config.py:95-140`) mixes high-signal scientific-content categories (`prose_code_*`, `accession_id_prefix`, `gene_case`, `param_name`, `species_name`, `bio_term`) with low-signal cosmetic categories (`typo`, `comment_typo`, `markdown_structure`, `inline_code`, `link`, `emphasis`, `duplicate`). In the 20,935-error run, `typo` alone is 38 % of injections while the moat (`prose_code_*` + `accession_id_prefix`) is 0.13 %. The headline F1 averages them together, drowning the moat signal in typo-correction noise.

**Goal**: fix the scorer bug, split categories into `CONTENT` / `HYGIENE` / `UNSCORABLE`, plumb a dual-F1 (`f1_content` + `f1_hygiene`) through the scorer, then rescore the existing run in place and update `REPORT.md`. No LLM spend. Changes uncommitted until user review.

## RALPLAN-DR short summary

### Principles (5)

1. **No LLM spend** — all recompute is offline against on-disk `*.fixed.Rmd` + manifest files
2. **Backwards compatibility** — preserve the `f1_score_scorable` column so prior runs remain comparable; add `f1_content` / `f1_hygiene` alongside, don't overwrite
3. **Single source of truth for category membership** — `CONTENT_CATEGORIES` / `HYGIENE_CATEGORIES` / `UNSCORABLE_CATEGORIES` defined once in `managers/config.py`, imported everywhere
4. **Testability** — both defects (inline_code substring bug, category membership invariants) must get regression tests that would have failed before the fix
5. **Narrative honesty** — the rescored REPORT.md explicitly documents the defect, the change in numbers, and which run it re-scored

### Decision drivers (3)

1. **Time-to-corrected-numbers** — user wants honest F1 in hand today, not next week; favor smallest diff that unblocks the number
2. **Risk of silent scorer drift** — the bug survived three code reviews and a 90-manifest run because three copies exist; design must prevent that repeating
3. **Moat thesis alignment** — the headline number must foreground "scientific-content fidelity" rather than average it with typo-correction

### Viable options

#### Option A — Minimal patch + standalone rescorer (recommended)

- Patch the inline_code branch in all three files identically
- Add `CONTENT_CATEGORIES` / `HYGIENE_CATEGORIES` as module-level constants in `managers/config.py`
- Extend `StressLevelResult` / `BenchmarkResult` with `f1_content` / `f1_hygiene` fields + update the CSV writer
- Build `scripts/bench_rescore.py`: walks a run_dir, re-parses manifests, re-runs the scorer against `*.fixed.Rmd`, emits `AGGREGATE_TABLE_RESCORED.csv` and new `f1_content` / `f1_hygiene` columns
- Regenerate figures via a new `viz.render_rescored_figures()` path or a plot-only mode of the existing viz
- Rewrite `REPORT.md` with dual-F1 headline + bug-fix note

**Pros**: Hours, not days. Small blast radius. Keeps the existing integration-test path unmodified. Can ship rescored numbers today.
**Cons**: Three scorer duplicates remain. Future category bugs may still copy-paste silently. Rescorer is a one-off script (not a reusable harness).

#### Option B — Consolidate scorer + dual-F1 + rescore

- Everything in Option A, **plus**: consolidate the three scorer duplicates into `UnifiedMetricsEvaluator.check_category(...)` and migrate `benchmark_metrics.py` + `test_single_file_stress.py` to call it
- Delete the inline duplicates

**Pros**: Kills copy-paste root cause. Future scorer changes flow to one place. Strictly better long-term.
**Cons**: Migration of `test_single_file_stress.py` carries regression risk — its scorer may have subtle semantic drift from `unified_metrics.py` that the tests don't catch. Longer turnaround (day, not hours). If migration surfaces a disagreement on a non-`inline_code` category, the rescored numbers could move for reasons unrelated to the bug fix.

#### Option C — Full metric redesign + rescore

- Everything in B, **plus**: rework the `StressLevelResult` / `BenchmarkResult` / CSV / figure templates to be category-group-native (one row per `(model, level, group)` triple instead of one row per `(model, level)`); add a `moat_fire_rate` diagnostic that reports injected-count per moat category to surface the Seurat anchor-miss problem directly

**Pros**: Cleanest long-term shape for the paper's figures. Diagnostic surfaces the moat under-firing problem as a first-class metric.
**Cons**: Days, not hours. Risk of scope creep. Blocks the rescored-numbers deliverable the user explicitly asked for today.

#### Recommendation: Option A now, Option B as an immediate follow-up ticket

User explicitly scoped this cycle to "rescore the old fix results" — Option A is the minimum that satisfies that and delivers honest numbers today. The Option B consolidation can be sequenced as a separate commit once the rescored numbers are reviewed, because it is a refactor with no new user-visible output. Option C defers until after the next round of LLM runs (different fixtures, more moat injections) because it's shaped by what those numbers look like.

### Open-question resolution: where does `number` go?

`number` mutates numeric values in prose. At 1,678 injections it's 8 % of the run. Two candidate rules:

- **Context-aware rule**: route to CONTENT if the mutation occurs inside a sentence that references a function or hyperparameter (anchor-matched); route to HYGIENE otherwise. **Blocker**: requires injector-side metadata that isn't written to the manifest today. Defer to Option C.
- **Single-bucket rule** (adopted here): classify `number` as CONTENT. Most numeric mutations in biomedical Rmd are hyperparameter values (`resolution = 0.5`, `nfeatures = 2000`, `min.pct = 0.25`), and those ARE scientifically meaningful. A typo like "we detected 6 clusters" is rarer than a doc drift on an analysis parameter. If the rescored F1_content looks unreasonably low because `number`-category is dragging it down, we revisit in Option C.

**Decision**: `number` → CONTENT for this cycle. Document the call in REPORT.md so the next revision can revisit.

---

## Final category split

**CONTENT** (scientific-fidelity): `param_name, gene_case, bio_term, species_name, accession_id_prefix, prose_code_pkg_version, prose_code_stat_test, prose_code_marker, prose_code_param, number`

**HYGIENE** (style/cosmetic): `typo, comment_typo, markdown_structure, inline_code, link, duplicate, boolean, emphasis`

**UNSCORABLE** (unchanged): `function`

Invariant: `CONTENT ∪ HYGIENE ∪ UNSCORABLE = all categories` AND the three sets are pairwise disjoint.

---

## Implementation steps

### Step 1 — Fix `_check_inline_code` in all three files

Replace the buggy predicate with net-decrease in the raw-form count — matches the pattern already used for `duplicate` on `benchmark_metrics.py:501`.

Files:
- `bioguider/generation/unified_metrics.py:524-531` (`_check_inline_code`)
- `bioguider/generation/benchmark_metrics.py:464-469` (inline `elif category == "inline_code"` branch)
- `system_tests/test_single_file_stress.py:493-497` (inline `elif cat == "inline_code"` branch)

Patch body (all three sites, same logic):

```python
# Fixed if the naked form is less common in fixed than in corrupted
# (model rewrapped OR removed the occurrence); the old predicate was
# tautological because f"`{raw}`" always contains `raw` as substring.
raw = mut.strip("`") if mut else ""
if not raw:
    return False, FixStatus.UNCHANGED  # or "unchanged" string in benchmark_metrics
is_fixed = fixed_content.count(raw) < corrupted_content.count(raw)
return is_fixed, (FixStatus.FIXED_TO_VALID if is_fixed else FixStatus.UNCHANGED)
```

Note: `unified_metrics.py` signature takes `(orig, mut, baseline, corrupted, revised)` — rename `fixed_content` → `revised`. The `benchmark_metrics.py` and `test_single_file_stress.py` sites already have `corrupted_content` / `fixed_content` in scope.

### Step 2 — Add category-group constants to `managers/config.py`

At end of `managers/config.py`, after `UNSCORABLE_CATEGORIES`:

```python
CONTENT_CATEGORIES: frozenset[str] = frozenset({
    "param_name", "gene_case", "bio_term", "species_name",
    "accession_id_prefix",
    "prose_code_pkg_version", "prose_code_stat_test",
    "prose_code_marker", "prose_code_param",
    "number",
})

HYGIENE_CATEGORIES: frozenset[str] = frozenset({
    "typo", "comment_typo", "markdown_structure",
    "inline_code", "link", "duplicate",
    "boolean", "emphasis",
})

def _validate_category_groups() -> None:
    """Invariant: CONTENT, HYGIENE, UNSCORABLE partition ALL_ERROR_CATEGORIES."""
    union = CONTENT_CATEGORIES | HYGIENE_CATEGORIES | UNSCORABLE_CATEGORIES
    overlap_ch = CONTENT_CATEGORIES & HYGIENE_CATEGORIES
    overlap_cu = CONTENT_CATEGORIES & UNSCORABLE_CATEGORIES
    overlap_hu = HYGIENE_CATEGORIES & UNSCORABLE_CATEGORIES
    if overlap_ch or overlap_cu or overlap_hu:
        raise ValueError(f"category groups overlap: {overlap_ch | overlap_cu | overlap_hu}")
    missing = set(ALL_ERROR_CATEGORIES) - union
    if missing:
        raise ValueError(f"categories not assigned to any group: {missing}")
    extra = union - set(ALL_ERROR_CATEGORIES)
    if extra:
        raise ValueError(f"unknown categories in groups: {extra}")

_validate_category_groups()  # runs at import time, fails fast on misconfiguration
```

### Step 3 — Plumb dual-F1 through the result types

Add fields to `StressLevelResult` (`system_tests/test_single_file_stress.py`) and `BenchmarkResult` (`bioguider/generation/benchmark_metrics.py`):

```python
total_injected_content: int = 0
fixed_content: int = 0
f1_score_content: float = 0.0
total_injected_hygiene: int = 0
fixed_hygiene: int = 0
f1_score_hygiene: float = 0.0
```

Add a helper `compute_group_breakdown(errors, fixed_ids, group_set)` next to the existing `compute_scorable_breakdown` in `managers/config.py`. Extend the CSV writer in both paths to append the six new columns.

### Step 4 — Build the offline rescorer: `scripts/bench_rescore.py`

CLI: `poetry run python scripts/bench_rescore.py --run-dir outputs/multi_file_stress/run_20260424_025546`

Behavior:

1. Walk `<run_dir>/<stem>/` for each `<stem>/*.manifest.json`
2. For each manifest at (stem, level), for each model in `MODELS`:
   - Read `<stem>.level_<L>.corrupted.Rmd`
   - Read `<stem>.level_<L>.<model>_bioguider.fixed.Rmd`
   - For each error in `manifest["errors"]`, call the scorer with `(orig, mut, corrupted, fixed)` — same shape as the production scorer
   - Accumulate fixed/unfixed counts by category group
3. Compute per-row `precision/recall/f1` for three groupings: `scorable` (legacy), `content`, `hygiene`
4. Emit `<run_dir>/_aggregate/AGGREGATE_TABLE_RESCORED.csv` with columns: original + `f1_content, total_injected_content, fixed_content, f1_hygiene, total_injected_hygiene, fixed_hygiene`
5. Emit per-file `<stem>/STRESS_TEST_TABLE_RESCORED.csv` similarly
6. **Idempotency check**: if `AGGREGATE_TABLE_RESCORED.csv` exists, hash the inputs (manifest mtimes + fixed-file mtimes + scorer-module hash). If unchanged, no-op. Otherwise overwrite.

Use threadpool of 8 workers — scoring is CPU-bound and embarrassingly parallel across cells.

### Step 5 — Regenerate figures

Add a `render_rescored()` entry point to `bioguider/generation/viz.py` (or extend `_f1(r)` helper to accept a metric-name argument). Emit two sets of figures under `_aggregate/`:

- `fig{1-6}_content.{png,pdf}` — using `f1_score_content`
- `fig{1-6}_hygiene.{png,pdf}` — using `f1_score_hygiene`

Keep the original `fig{1-6}.{png,pdf}` unchanged so the old REPORT.md remains reproducible. Per-file figures regenerated under each `<stem>/`.

### Step 6 — Regression tests

New file `tests/test_scorer_inline_code.py`:

```python
def test_inline_code_net_decrease_is_fixed():
    """Bug regression: old scorer returned False because rewrapped always contains mut."""
    corrupted = "Call FindMarker() to run the test."
    fixed = "Call `FindMarker()` to run the test."
    is_fixed, _ = _check_inline_code(
        orig="`FindMarker()`", mut="FindMarker()",
        baseline=fixed, corrupted=corrupted, revised=fixed,
    )
    assert is_fixed, "model rewrapping the naked form must score as fixed"

def test_inline_code_noop_not_fixed():
    corrupted = "Call FindMarker() to run the test."
    fixed = corrupted  # model did nothing
    is_fixed, _ = _check_inline_code(
        orig="`FindMarker()`", mut="FindMarker()",
        baseline=corrupted, corrupted=corrupted, revised=fixed,
    )
    assert not is_fixed
```

Run the same two cases against the `benchmark_metrics.py` and `system_tests/test_single_file_stress.py` branches via a parametrized test to prove all three scorer paths agree.

New file `tests/test_category_groups.py`:

```python
def test_category_groups_partition():
    from bioguider.managers.config import (
        CONTENT_CATEGORIES, HYGIENE_CATEGORIES, UNSCORABLE_CATEGORIES,
        ALL_ERROR_CATEGORIES,
    )
    assert not (CONTENT_CATEGORIES & HYGIENE_CATEGORIES)
    assert not (CONTENT_CATEGORIES & UNSCORABLE_CATEGORIES)
    assert not (HYGIENE_CATEGORIES & UNSCORABLE_CATEGORIES)
    assert CONTENT_CATEGORIES | HYGIENE_CATEGORIES | UNSCORABLE_CATEGORIES == set(ALL_ERROR_CATEGORIES)
```

### Step 7 — Rewrite `REPORT.md`

Add at the top, above TL;DR:

```markdown
> **Rescored 2026-04-24**: original F1 numbers were affected by a scorer bug
> in the `inline_code` category (tautologically-false predicate). This report
> uses rescored metrics derived from the on-disk fixed files; no models were
> re-run. See "Rescore note" section for the delta.
```

Replace the TL;DR table with the dual-F1 headline (columns: Model, F1_content, F1_hygiene, Median latency).

Add a new "Rescore note" section with: original vs rescored F1 per model, link to `AGGREGATE_TABLE_RESCORED.csv`, category-group assignment table, and an honest caveat that `F1_content` is dominated by `param_name + gene_case + number` because the moat categories (`prose_code_*`) barely fired on Seurat vignettes.

Regenerate `REPORT.html` via the same `md-to-html` skill.

---

## Acceptance criteria (testable)

1. `poetry run pytest tests/test_scorer_inline_code.py` passes on the branch and fails on `main` (proves the fix targets the right bug)
2. `poetry run pytest tests/test_category_groups.py` passes on the branch
3. `grep -n 'rewrapped in' bioguider/generation/unified_metrics.py bioguider/generation/benchmark_metrics.py system_tests/test_single_file_stress.py | wc -l` returns 0 (proves all three copy-paste sites got patched)
4. File `outputs/multi_file_stress/run_20260424_025546/_aggregate/AGGREGATE_TABLE_RESCORED.csv` exists and contains columns `f1_score_content` and `f1_score_hygiene` (verified with `head -1`)
5. The rescorer is idempotent: `poetry run python scripts/bench_rescore.py --run-dir outputs/multi_file_stress/run_20260424_025546 && sha256sum outputs/.../AGGREGATE_TABLE_RESCORED.csv` returns the same hash across two consecutive runs
6. Mean `f1_score_content` for gpt-4o > 0.80 (sanity: the moat-adjacent F1 should be above the typo-noise floor; if below, something is wrong with category assignment)
7. `outputs/multi_file_stress/run_20260424_025546/REPORT.md` contains the literal string `Rescored 2026-04-24` (proves the updated report landed)
8. Files `_aggregate/fig{1..6}_content.png` AND `_aggregate/fig{1..6}_hygiene.png` exist (12 new figures)
9. `poetry run pytest tests/ -v` fully passes (no regressions in the rest of the test suite)
10. No files in `bioguider/generation/` have been deleted (preserves the generation pipeline surface area)

## Verification steps

After each major step, run:

- After Step 1: `poetry run pytest tests/test_scorer_inline_code.py -v` — proves the fix works
- After Step 2: `poetry run python -c "from bioguider.managers.config import CONTENT_CATEGORIES, HYGIENE_CATEGORIES, UNSCORABLE_CATEGORIES; print(len(CONTENT_CATEGORIES), len(HYGIENE_CATEGORIES), len(UNSCORABLE_CATEGORIES))"` — proves import side-effects don't fail
- After Step 3: `poetry run pytest tests/ -v` — proves the new fields don't break existing result-type consumers
- After Step 4: `poetry run python scripts/bench_rescore.py --run-dir outputs/multi_file_stress/run_20260424_025546` — should complete in < 60 s
- After Step 5: eyeball `fig2_avg_f1_content.png` and confirm bar heights differ meaningfully from `fig2_avg_f1_hygiene.png`
- After Step 6: `poetry run pytest tests/ -v` — all tests green
- After Step 7: `open outputs/multi_file_stress/run_20260424_025546/REPORT.html` in a browser and visually confirm the new dual-F1 table + rescore note render

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Rescorer disagrees with original scorer on non-`inline_code` categories (hidden bug) | Medium | High (invalidates rescored numbers) | Before trusting output, spot-check: rerun the original scorer on 10 random cells and confirm `fixed` count matches rescorer's `fixed` count for every non-inline category within ±0. Any mismatch outside inline_code halts the rescore and gets filed as a new defect. |
| Category reassignment for `number` turns out wrong (F1_content looks too high or too low) | Medium | Low-Medium | Document the call in REPORT.md. If `f1_content` > 0.98 or < 0.60 for all models, the assignment is suspect — reclassify to HYGIENE and re-run rescorer (cheap). |
| Figure template assumes single F1 axis | Low | Low | Render dual sets of figures side-by-side. Keep original `fig{1-6}.png` untouched so old REPORT.md still renders. |
| `prose_code_*` injections are ~0 so F1_content carries thin signal in some models | High | Medium (story for REPORT.md) | Accept and document. Call it out in REPORT.md as "moat fixture selection is the next limit." Flag for the next LLM run: need denser moat fixtures (scanpy notebooks, Bioconductor vignettes) to actually test the moat. |
| Patching `test_single_file_stress.py` breaks the stress-test harness | Low | High | Changes are a one-branch replacement in a well-tested function; `pytest tests/` in Step 3's verification catches it. |
| Downstream consumers of `AGGREGATE_TABLE.csv` expect the old column set | Low | Medium | Rescorer writes `AGGREGATE_TABLE_RESCORED.csv` (new name). Old CSV is left untouched. |

---

## ADR

**Decision**: Adopt Option A (minimal patch + standalone rescorer) for this cycle. Split `ERROR_CATEGORIES` into `CONTENT_CATEGORIES` / `HYGIENE_CATEGORIES` / `UNSCORABLE_CATEGORIES` as module-level constants in `bioguider/managers/config.py`. Fix the `_check_inline_code` substring bug in all three call sites with an identical patch. Build a standalone `scripts/bench_rescore.py` that re-scores the existing run_dir offline and emits a new `_RESCORED.csv` + 12 new figures. Rewrite `REPORT.md` with dual-F1 headline + rescore note.

**Drivers**:
1. User explicitly scoped to "no re-run, rescore old results" — minimum diff that delivers the corrected numbers today
2. Three scorer duplicates exist but consolidating them is orthogonal to the rescoring goal and adds regression risk
3. The benchmark thesis ("catches scientific-content errors Claude Code misses") requires the headline to foreground content-fidelity, not typo-correction

**Alternatives considered**:
- Option B (consolidate scorer first) — deferred. Sound long-term, but migrating `test_single_file_stress.py`'s inline scorer risks regressing the 360-cell pipeline and requires diff-reconciliation if the two scorers have drifted semantically. Better sequenced as a follow-up commit after the rescored numbers are accepted.
- Option C (full metric redesign + group-native rows) — deferred. Reshapes CSV / figure / result-type schemas, takes days, and shape is best informed by the rescored numbers from this cycle. Revisit when planning the next LLM round.

**Why chosen**: Option A unblocks the "honest numbers today" goal in ≤ a day of work, leaves an open path to B and C, and doesn't compound risk. The downside (leaving copy-paste duplication in place) is bounded: the regression test in Step 6 parametrizes across all three scorer paths, so a future bug in one path surfaces immediately.

**Consequences**:
- Positive: rescored F1_content + F1_hygiene numbers land today. Scorer bug closed with a regression test that parametrizes across all three duplicates. Category split surfaces the moat-signal problem (and by implication the fixture-selection problem) as a first-class concern in REPORT.md.
- Negative: three scorer duplicates still exist. The rescorer is a one-off script, not part of the main test harness. `number`-category classification is a judgment call that may need revisiting.

**Follow-ups**:
- [follow-up-1] Option B (scorer consolidation) as a separate commit once rescored numbers are reviewed and accepted
- [follow-up-2] Plan the next LLM run with denser moat fixtures (scanpy notebooks, Bioconductor vignettes) so `prose_code_*` injections fire more often
- [follow-up-3] If `number`-category reclassification is revisited (Option C), add context-aware tagging to the injector so each error carries group metadata at injection time rather than at score time
- [follow-up-4] Audit the other scorer branches (`_check_link`, `_check_markdown_structure`, `_check_emphasis`) for analogous substring-containment bugs — the inline_code pattern may not be unique

---

## Out of scope for this plan

- Re-running any LLM correction (explicit user constraint)
- Consolidating the three scorer duplicates (deferred to Option B follow-up)
- Adding new error categories or changing the injector (deferred)
- Changing the figure plotting library or layout (only the metric being plotted changes)
- Committing any of these changes (user wants to review the rescored REPORT.md first)
- Investigating the `pbmc3k_tutorial` + gpt-4o anomaly (separate ticket)
- Re-enabling `glm-5` (separate ticket)

---

## Changelog

- 2026-04-24: Initial plan, RALPLAN-DR short mode, Option A recommended. Category split finalized with `number` assigned to CONTENT pending evidence from the rescored run.
