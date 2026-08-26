# Morning handoff — BioGuider benchmark overnight run

## What I did while you slept

1. **Fixed a env-var footgun** — `.env` was overridden by a stale shell
   `OPENAI_API_KEY` starting with `3Mhw…` (Azure key). `load_dotenv()` does not
   override existing shell vars, so the bmblx proxy rejected every request
   with 401. Fix: `unset OPENAI_API_KEY` before launch. Wrapped this into
   `scripts/bench_launch.sh` so it never bites again.

2. **Rewrote the fix-side prompt** (`BIOGUIDER_PROMPT` in
   `system_tests/test_single_file_stress.py`). Removed the injection-taxonomy
   cheat-sheet ("typos / numeric changes / boolean swaps / …") that was
   leaking correct answers to the fixer. Added a `GROUND TRUTH` section that
   names code blocks as the authority and an explicit `rewrite the PROSE to
   match the CODE` rule for the prose_code_consistency moat category.
   Before the change the fixer benchmark was measuring prompt-leak quality,
   not model quality.

3. **Narrowed to 1 prompt / 5 models** per your direction. 10 vignettes × 9
   error levels × 5 models × 1 prompt = 450 cells. The `simple` prompt axis
   was removed — it was legacy ablation material.

4. **First launch was serial, projected 26 hrs.** Killed at ~10 min when the
   rate math became obvious. Rewrote the inner test_configs loop with
   `ThreadPoolExecutor(max_workers=5)` so the 5 model configs within each
   (file, level) cell fire concurrently. Relaunched as task `b3fnm136s`.

5. **Persistent Monitor `bzv4f0pgw`** watches the new run — it emits events
   on: per-file completion, LLM 200-OK heartbeat every 30 calls, any 401 /
   Traceback / AssertionError, COMPLETE signal, or pytest dying.

## Where to look first

```
outputs/multi_file_stress/run_20260424_022419/
├── WAKE_SUMMARY.md          ← start here; per-file + per-model F1 tables
├── INDEX.md                 ← only exists if the run fully completed
├── <vignette>/fig1..fig6.png ← per-file figures (auto-rendered by save_results)
└── _aggregate/              ← pooled fig1..fig6 across all completed files
```

Cheap commands:

```
scripts/bench_status.sh                  # one-line snapshot
cat outputs/multi_file_stress/run_20260424_022419/WAKE_SUMMARY.md
open outputs/multi_file_stress/run_20260424_022419/<vignette>/fig1_f1_by_error_level.png
```

If WAKE_SUMMARY.md is stale (pytest finished after my last refresh),
regenerate it:

```
poetry run python scripts/bench_wake_summary.py \
  outputs/multi_file_stress/run_20260424_022419
```

## What's on your plate

- If 8-10 files completed: decide whether the results are good enough for
  a v1 paper figure draft, or whether to rerun with broader scope.
- If fewer than 5 files completed: parallelism didn't give the expected 5x,
  OR a model was rate-limited. Check `_aggregate/AGGREGATE_RESULTS.json` and
  the log for 429s.
- The `prose_code_consistency` moat: look at `WAKE_SUMMARY.md`'s moat
  section. If it shows 0 hits across all files, the anchor regexes in
  `bioguider/generation/llm_injector.py` (`_PKG_ANCHOR_RE`,
  `_STAT_TEST_ANCHOR_RE`, `_MARKER_ANCHOR_RE`, `_PARAM_ANCHOR_RE`) don't
  match Seurat idioms and need broadening — that's the D1 follow-up I
  flagged before launch.

## What I did not touch

- No git commits (per your constraint).
- No change to the LLM injection path beyond D1-D6 (already committed).
- No change to RAG / embeddings / EvaluationManager (out of scope).

## If the run failed overnight

- Look at `logs/multi_file_stress_20260424_022416.log` for the last 100
  lines — probably a 401 spike, a single model erroring out, or an
  assertion failure.
- Salvage: `poetry run python scripts/bench_wake_summary.py <run_dir>`
  will still render figures for any partial per-file subdirs that have
  STRESS_TEST_RESULTS.json.
- Relaunch: `scripts/bench_launch.sh` — it probes the proxy first and
  aborts early on a bad key.
