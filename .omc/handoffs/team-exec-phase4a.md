# Handoff: Phase 4a — Viz module standalone

**Owner**: worker-3  
**Task**: #3  
**Status**: completed  
**Date**: 2026-04-16

## Files Changed

- `pyproject.toml` — added `matplotlib = "^3.9"` to `[tool.poetry.dependencies]`
- `bioguider/generation/viz.py` — NEW, ~220 LOC, `BenchmarkPlotter` class
- `tests/test_viz.py` — NEW, ~120 LOC, 5 offline tests with inline synthetic fixture

## viz.py Summary

`BenchmarkPlotter(out_dir)` consumes:
- `STRESS_TEST_RESULTS.json` — per-(model, error_count) metrics
- `STRESS_TEST_CATEGORY_DETAIL.csv` — per-(model, error_level, category) fix rates

Methods produce `{name}.{png,pdf}` in `out_dir`:
- `fig1_f1_by_error_level` — line plot, x=error_count, y=F1, one line per model
- `fig2_avg_f1_by_model` — horizontal bar, mean±95%CI, sorted desc
- `fig3_category_heatmap` — imshow, rows=models, cols=categories, annotated
- `fig4_fix_rate` — grouped bar, models sorted by mean F1 desc (AC6 compliant)
- `fig5_response_time` — line, duration_seconds vs error_count per model
- `fig6_fixed_unfixed` — stacked bar (fixed/unfixed) at median error level

`render_all(out_dir=None)` calls all six. Global rcParams: dpi=150, bbox=tight, DejaVu Sans.

## Verification

Validated with `/usr/bin/python3` (matplotlib 3.9.4 present) via `importlib.util` direct import:
- **12/12 files generated** (6 PNG + 6 PDF), all non-empty
- fig names match existing `run_20251203_111619/` convention exactly

Note: `Optional[PathLike]` used instead of `PathLike | None` for Python 3.9 compat (system test env).  
Note: matplotlib is **not yet installed** in poetry venv — `poetry install` or `poetry lock` needed after this PR merges (the dep is in pyproject.toml but no lock refresh was run per protocol).

## For worker-1 (Task #4 — consolidation)

Hook into `save_results()` in `system_tests/test_single_file_stress.py` as specified in plan Step 13:
```python
try:
    from bioguider.generation.viz import BenchmarkPlotter
    BenchmarkPlotter(out_dir).render_all(out_dir)
except ImportError:
    logger.warning("matplotlib not available; skipping figure generation")
```
`out_dir` must already contain `STRESS_TEST_RESULTS.json` and `STRESS_TEST_CATEGORY_DETAIL.csv` before calling `render_all`.
