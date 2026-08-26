# Plan: LiteLLM swap + injection redesign + rerun/viz

**Branch:** `refactor/document-generation` · **Target:** single-file stress benchmark on `data/.adalflow/repos/satijalab_seurat/vignettes/de_vignette.Rmd`.

Decisions (from interview):
- Transport: env-swap via existing `ChatOpenAI` branch of `get_llm()`. No litellm-sdk.
- Injection intent: **add** new categories (biomed-app focused), keep LLM-driven design.
- New categories: reproducibility drift, analysis hyperparameters, statistical test mis-naming, annotation ID space confusion, cell-type / marker-gene errors.
- Model comparison set: `gpt-5.4`, `kimi-k2.5`, `glm-5`, `gpt-oss` (plus `gpt-4o` as legacy baseline for cross-run continuity).
- Viz: matplotlib, dual-format (`.png` + `.pdf`), naming `fig1..figN` to match `outputs/single_file_stress/run_20251203_111619/old_figures/` convention.
- Legacy cleanup: delete orphan `bioguider/managers/generation_test_manager.py`. Keep `v2` + `test_single_file_stress.py`.

---

## Requirements Summary

1. Route all bioguider LLM traffic through LiteLLM proxy `https://bmblx.bmi.osumc.edu/ai/v1` w/ bioguider-2 virtual key. One HTTP shape: OpenAI.
2. `LLMErrorInjector` accepts ~5 new bio-app categories; `ERROR_CATEGORIES` in `bioguider/managers/config.py` extended.
3. Rerun `test_all_models_all_levels` (in `system_tests/test_single_file_stress.py`) with new model list; pipe results through a new `bioguider/generation/viz.py` producing figs 1–6.
4. Legacy path deleted; v2 manager + single-file stress test are canonical.

## Acceptance Criteria

- AC1 smoke: `python -c "from bioguider.agents.agent_utils import get_openai; get_openai().invoke('hi').content"` returns non-empty when `OPENAI_API_KEY=sk-<bioguider-2>`, `OPENAI_BASE_URL=https://bmblx.bmi.osumc.edu/ai/v1`, `AZURE_OPENAI_ENDPOINT` unset, `OPENAI_MODEL=gpt-5.4`.
- AC2 model round-trip: same snippet works for each of `gpt-5.4`, `kimi-k2.5`, `glm-5`, `gpt-oss`, `gpt-4o` (5 pytest cases in a new `tests/test_litellm_compat.py`).
- AC3 injection: running `LLMErrorInjector.inject(seurat_de_vignette_text, min_per_category=3)` on a vignette fixture yields manifest containing ≥1 error per new category (5 new + any picked legacy). Deterministic-fallback path must also know about new categories (see Step 3).
- AC4 categories registered: `bioguider.managers.config.ERROR_CATEGORIES["biomed_app"]` contains the 5 new names. `ALL_ERROR_CATEGORIES` size increases by exactly 5.
- AC5 run artifacts: `outputs/single_file_stress/run_<ts>/` contains `STRESS_TEST_RESULTS.json`, `STRESS_TEST_TABLE.csv`, `STRESS_TEST_CATEGORY_DETAIL.csv`, `STRESS_TEST_REPORT.md`, and 6 figs `fig{1..6}.{png,pdf}`.
- AC6 viz correctness: fig1 (F1 vs error-count, line per model) shows ≥4 series; fig3 (per-category fix-rate heatmap) has columns == 5 new categories ∪ legacy categories hit in run; fig4 (fix-rate bar) sorts models by mean F1 desc.
- AC7 legacy gone: `rg "generation_test_manager\b" bioguider/ system_tests/ tests/` returns zero non-self hits after cleanup.
- AC8 (from Analyst #4) token accounting: `tests/test_litellm_compat.py` asserts non-zero `token_usage.total_tokens` reported for each model via the `OpenAICallbackHandler` path — guards silent zero-accounting on kimi/glm.
- AC9 (from Analyst #3) no Azure leaks: `rg "AZURE_OPENAI_ENDPOINT|azure_endpoint" bioguider/` after Phase 1 returns ≤1 hit and it is the conditional guard in `agent_utils.py`, not a hard requirement.
- AC10 (from Analyst #6) deterministic fallback has new-cat rules: feeding a fixture where `LLMErrorInjector._parse_json_output` is forced to fail (mocked bad JSON) still produces a manifest with ≥1 error in each of the 5 new categories (except `celltype_marker` when input has no markers per Analyst #8).

## Implementation Steps

### Phase 1 — transport (zero-call-site diff target)

1. `bioguider/agents/agent_utils.py:53-115` — `get_openai()` + `get_llm()`:
   - read `OPENAI_BASE_URL` env; if set, skip Azure branch entirely and pass `base_url=...` to `ChatOpenAI`.
   - **widen the `startswith("gpt")` guard at line 88** — replace with a LiteLLM-proxy allowlist: `{"gpt-4o","gpt-5.4","kimi-k2.5","glm-5","gpt-oss","gpt-oss-120b"}`. kimi/glm/gpt-oss fail `ValueError` at line 115 today (Analyst #1).
   - **broaden gpt-5 temp skip at line 95** — `gpt-5.4` matches today's guard, silently drops `temperature=0`. Change substring list to exact match `{"gpt-5","gpt-5.4","o1","o3"}` so only the true gpt-5 family is blocked (Analyst #9).
   - treat empty string `AZURE_OPENAI_ENDPOINT` as unset (pydantic/os.environ mix is existing footgun).
   - keep existing `deepseek` branch dead-but-harmless; delete `ChatDeepSeek` import only after Phase 5 confirms no callers.
2. **`bioguider/generation/llm_content_generator.py:114-120` — second call site for `get_llm()`** (Analyst #3). Apply same `OPENAI_BASE_URL`/Azure branch logic OR refactor to reuse `get_openai()` from agent_utils. Grep confirmation: `rg "get_llm\(" bioguider/` to verify no third caller.
3. `system_tests/conftest.py:22-30` — `get_azure_openai()` becomes `get_litellm()`; `llm` fixture preserved under same name.
4. `tests/test_litellm_compat.py` new — parametrized over 5 models. Skips if `OPENAI_BASE_URL` unset (CI safety). Also asserts `response.response_metadata["token_usage"]["total_tokens"] > 0` for each model to catch silent zero-accounting under `OpenAICallbackHandler` (Analyst #4).
5. `.env.example` new — document `OPENAI_BASE_URL`, `OPENAI_API_KEY` (bioguider-2), `OPENAI_MODEL` conventions. Do NOT commit `.env`.
6. **Embedding path audit** (Analyst #2). `bioguider/rag/config.py:34` uses `adalflow.OpenAIClient` with `text-embedding-3-small`. Decide: (a) leave embeddings on Azure — add `EMBEDDING_BASE_URL` env separate from `OPENAI_BASE_URL` so RAG keeps working; (b) route embeddings through proxy if it supports `/v1/embeddings`. Verify proxy capability via `curl $OPENAI_BASE_URL/embeddings` before picking. Default to (a) — keeps current RAG behaviour untouched.

### Phase 2 — injection taxonomy

5. `bioguider/managers/config.py` — add new bucket:
   ```python
   ERROR_CATEGORIES["biomed_app"] = [
       "reproducibility_drift", "analysis_hyperparam",
       "stat_test_misnaming", "annotation_id_space",
       "celltype_marker",
   ]
   ```
   Keep existing buckets intact. Bump tests that snapshot category counts (`tests/test_generation_config.py`).
6. `bioguider/generation/llm_injector.py:13-106` — extend `INJECTION_PROMPT`:
   - add "BIOMED-APP ERROR CATEGORIES" block w/ one-line example per new cat (mirror the existing bio block format).
   - add protected-keyword hints: keep gene symbols in `keywords` intact unless target is gene_symbol_case OR celltype_marker (when the target IS the symbol).
7. `bioguider/generation/llm_injector.py` — add rules for new cats in **both** `_supplement_errors` (line 659) **and** `_deterministic_inject` (line 289). Analyst #6 confirms both paths lack new-cat coverage today. Rules:
   - `reproducibility_drift` — seed-pattern regex (`set\.seed\(\d+\)`, `random_state=\d+`, `numpy\.random\.seed\(\d+\)`), bump by +/-1; version string regex (`[Ss]eurat\s+v?[345]`, `python\s+3\.\d+`) toggle minor version.
   - `analysis_hyperparam` — targeted number-swap **only inside** R/Python calls matching `FindClusters|RunUMAP|RunTSNE|n_neighbors|resolution|perplexity|n_components` to avoid colliding with generic `number` category (Analyst #7 disambiguation rule). Run BEFORE the generic `number` supplement so it gets first pick.
   - `stat_test_misnaming` — word-swap table: `{wilcox: t-test, t-test: wilcoxon, bonferroni: FDR, FDR: bonferroni, one-sided: two-sided, log2FC: fold-change}`.
   - `annotation_id_space` — regex `GSE(\d+)` <-> `GSM\1`; `ENSG\d+` replaced by canonical symbol lookup of 20 common genes (hardcoded dict, no network call).
   - `celltype_marker` — marker-gene swap table `{CD4: CD8, CD8: CD4, FOXP3: RORC, GATA3: TBX21}`. **Precondition check** (Analyst #8): if input contains zero entries from the swap table, skip this category and note in manifest `{"skipped": "celltype_marker", "reason": "no markers in input"}`.
   Keep under 120 LOC total — mostly lookup tables + one regex-class per cat.
8. Unit test `tests/test_llm_injector_biomed.py` new — feed a synthetic fixture with known seed (`set.seed(42)`), hyperparam (`resolution=0.5`), stat test (`wilcox.test`), accession (`GSE123456`), marker (`CD4`). Assert manifest counts ≥1 per category. Second fixture with NO markers — assert `celltype_marker` is skipped (not a failure).
9. **`evaluate_fixes` category coverage** (Analyst #5, also resolves plan's own "Unresolved #4"). `system_tests/test_single_file_stress.py:467-510` hard-codes category branches. Add elif arms:
   - `reproducibility_drift`, `analysis_hyperparam`, `annotation_id_space` — exact-string match works (`orig in fixed OR mut not in fixed`), same shape as `number` branch.
   - `stat_test_misnaming`, `celltype_marker` — **semantic match needed**. Add a small LLM-judge helper `_semantic_match(orig, fixed_context, llm)` that asks "does this text correctly use <orig term>" — invoked only when the literal string check is ambiguous. Cache by (orig, context[:500]) hash.

### Phase 3 — rerun harness

9. `system_tests/test_single_file_stress.py:187-220` — rewrite `MODELS` dict: replace `azure`/`claude`/`ollama` dispatch with a single `litellm` type routed through the fixture `llm` w/ per-call model override:
   ```python
   MODELS = {
       "gpt-5.4": {"type": "litellm", "model": "gpt-5.4"},
       "kimi-k2.5": {"type": "litellm", "model": "kimi-k2.5"},
       "glm-5": {"type": "litellm", "model": "glm-5"},
       "gpt-oss": {"type": "litellm", "model": "gpt-oss-120b"},  # confirm exact id
       "gpt-4o": {"type": "litellm", "model": "gpt-4o"},         # legacy baseline
   }
   ```
   - `fix_with_model` (line 322) collapses: one `ChatOpenAI(model=<override>, base_url=$OPENAI_BASE_URL, api_key=$OPENAI_API_KEY).invoke(prompt)` call. Drop `call_ollama`, `call_claude`, `OLLAMA_BASE_URL`, `CLAUDE_API_URL`.
   - `test_configs` in `test_all_models_all_levels` (line 1258) becomes `[(m, "bioguider") for m in MODELS] + [("gpt-5.4", "simple")]` (bioguider prompt + one simple-prompt baseline).
10. Delete `bioguider/managers/generation_test_manager.py` (orphan — grep confirmed zero callers).
11. Audit `system_tests/test_generation_quantifiable.py` and `system_tests/test_project_specific_injection.py` callers of `_inject_errors_into_files`; if they still pass w/ v2 API, keep; if not, mark `xfail` with a TODO linking to this plan.

### Phase 4 — viz

12. `bioguider/generation/viz.py` new (~200 LOC). One class `BenchmarkPlotter` consuming `STRESS_TEST_RESULTS.json` + `STRESS_TEST_CATEGORY_DETAIL.csv`. Methods:
    - `fig1_f1_by_error_level()` — line plot, x=error_count, y=F1, color=model. Mirror `old_figures/fig1_all_models_stress_f1.png`.
    - `fig2_avg_f1_by_model()` — horizontal bar, mean±95%CI across error levels.
    - `fig3_category_heatmap()` — rows=models, cols=category, cell=fix_rate, annotated.
    - `fig4_fix_rate()` — grouped bar, x=error_count, hue=model.
    - `fig5_response_time()` — line, duration_seconds vs error_count per model.
    - `fig6_fixed_unfixed()` — stacked bar per model@median-level.
    - `render_all(out_dir)` saves `{png,pdf}` pair per fig. Global `mpl.rcParams`: `figure.dpi=150`, `savefig.bbox='tight'`, font `DejaVu Sans`.
13. Hook into `save_results()` (`test_single_file_stress.py:666`) — after md report write, call `BenchmarkPlotter(out_dir).render_all(out_dir)` inside try/except that only logs on ImportError so matplotlib-less envs still produce CSVs.
14. `pyproject.toml` — add `matplotlib = "^3.9"` to `[tool.poetry.dependencies]`. No seaborn (manual styling keeps dep surface minimal).

### Phase 5 — rerun + archive

15. `pytest system_tests/test_single_file_stress.py::test_all_models_all_levels -v -s` from repo root w/ LiteLLM env. **Call accounting** (Analyst #10 — cost understated originally):
    - 6 injection calls × ~30k-char prompt (gpt-5.4 at ~$2.50/M in + $10/M out ≈ $0.30–0.50 per call) = ~$2–3
    - 5 models × 6 levels × (1 fix + 1 eval) = 60 LLM calls. Per-call cost range $0.02 (gpt-oss) to $0.20 (gpt-5.4). Expected $4–8
    - Semantic-match judge calls from Step 9 — ~10 per run, negligible
    - **Total budget: $8–15**, not $3–8. Require `OPENAI_API_KEY` to have ≥$20 budget before starting.
16. **Rate-limit hardening** (Analyst #11). `fix_with_model` (`test_single_file_stress.py:322`) wraps `llm.invoke` today with no retry. Wrap in `tenacity.retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), retry=retry_if_exception_type(RateLimitError))`. `tenacity` is already a dep (`pyproject.toml:39`). Drop default `MAX_WORKERS` 16 → 8 for the new run; promote to CLI arg so user can tune per proxy health.
17. Post-run: copy `run_<ts>/fig*.{png,pdf}` to `docs/figures/benchmark_<date>/` (new dir) — shippable artifact. `run_<ts>/` stays gitignored.

## Risks and Mitigations

- **R1** LiteLLM proxy returns non-OpenAI JSON shape for kimi/glm (seen rarely on some proxy versions). → AC2 cross-model test catches before any rerun. Fix: keep Azure-shape as fallback via env.
- **R2** `gpt-oss-120b` context window (8k) truncates level-300 injections. → preflight check in Step 15: abort that cell if `len(prompt) > 0.9 * ctx_window`; surface in `STRESS_TEST_REPORT.md` as "skipped" row.
- **R3** New deterministic-fallback rules in Step 7 break existing injector snapshots (`tests/test_generation_config.py`, `tests/test_llm_injector*.py` if any). → update snapshots in same commit as Step 5/6.
- **R4** matplotlib font-cache first-run delay on CI. → AC5 check doesn't assert figure timestamp; Phase 4 ImportError fallback keeps CSV path green.
- **R5** `outputs/single_file_stress/*` is not gitignored explicitly — 19 runs + 67 PNGs already present. → leave as-is (gitignored via top-level `outputs` rule), just ensure new run still lands there.

## Verification Steps

1. `pytest tests/test_litellm_compat.py -v` — 5 green, one skip if env unset.
2. `pytest tests/test_llm_injector_biomed.py -v` — new categories trigger.
3. `pytest tests/ -q` — baseline pytest still green (nothing unrelated broke).
4. `rg "ChatDeepSeek|AzureChatOpenAI" bioguider/` — zero unused hits post-cleanup (Phase 4).
5. `rg "generation_test_manager\b" bioguider/ system_tests/ tests/` — zero hits.
6. Run Phase 5, eyeball `fig1_f1_by_error_level.png` vs `run_20251203_111619/old_figures/fig1_all_models_stress_f1.png` — expect shape continuity, new model labels.
7. Spot-check manifest for level_100 run: ≥3 errors in each of the 5 new categories.

## Unresolved

- Exact LiteLLM model-id string for gpt-oss — routing.md lists `gpt-oss-120b` in BMBL team allowlist but virtual key `bioguider-2` inherits `all-team-models`. Verify via `curl $OPENAI_BASE_URL/models` before Step 9. If id differs, one-line fix in the `MODELS` dict.
- Keep `gpt-4o` as legacy baseline OR only run new models? Leaves one extra run (~$1) for continuity — recommend keep.
- (Analyst #2 open variant) Does `adalflow.OpenAIClient` respect `OPENAI_BASE_URL` when present? Step 6 defaults to "leave embeddings on Azure" but a one-shot smoke test (`curl $OPENAI_BASE_URL/embeddings -d '{"model":"text-embedding-3-small","input":"hi"}'`) before Phase 1 kickoff would let us flip to proxy if supported.
- Decided prior: rate-limit headroom (now addressed in Step 16 via tenacity retry + MAX_WORKERS=8).
- Decided prior: `evaluate_fixes` semantic match for new categories (now addressed in Step 9).

## Changelog (Analyst pass, 2026-04-16)

Folded the 11 findings from `oh-my-claudecode:analyst` review:
- Steps 1, 2 (new), 4, 6 (new) added to Phase 1 — widened model allowlist in `get_llm()`, patched second call site in `llm_content_generator.py`, added token-usage assertion, audited embedding path.
- Steps 7, 8, 9 (new) rewritten in Phase 2 — explicit rules for new cats in both injection paths, semantic-judge branch in `evaluate_fixes`.
- Steps 15–16 updated in Phase 5 — realistic $8–15 budget, tenacity retry wrapper, MAX_WORKERS reduced to 8.
- AC8–AC10 added — token accounting, no-Azure-leak, deterministic-fallback coverage.
- Two Unresolved items closed; embedding-proxy-capability probe noted as optional pre-flight.
