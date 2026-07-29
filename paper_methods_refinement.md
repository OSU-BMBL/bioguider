# Methods Refinement: Implementation and Experimental Reporting

This document supplies the implementation details requested for the manuscript: **prompts used by individual agents, model versions, generation parameters, stopping rules, retry procedures, processing time, token use, and execution environments.**

Every value below is either (a) **verified** against a specific location in the released source tree, cited as `path:line`, (b) **measured** from instrumented benchmark runs, with the artefact path given, or (c) explicitly marked **⚠️ OUTSTANDING** where it depends on facts only the authors can supply (host hardware, deployment regions, dated model snapshots). Nothing is asserted that could not be traced to code or to a recorded run.

---

## 1. Execution Environment

### 1.1 Software stack

| Component | Version | Source |
|---|---|---|
| Python | ≥ 3.11 (`^3.11`) | `pyproject.toml:25` |
| BioGuider package | **0.2.51** | `pyproject.toml:3` |
| LangChain | `^0.3.20` | `pyproject.toml:26` |
| langchain-openai | `^0.3.8` | `pyproject.toml:32` |
| langchain-anthropic | `^0.3.10` | `pyproject.toml:33` |
| langchain-deepseek | `^0.1.2` | `pyproject.toml:27` |
| langchain-google-genai | `^2.1.4` | `pyproject.toml:41` |
| langchain-experimental | `^0.3.4` | `pyproject.toml:36` |
| Tenacity (retry) | `^9.1.2` | `pyproject.toml:39` |
| FAISS (`faiss-cpu`) | `^1.11.0` | `pyproject.toml:44` |
| tiktoken (token counting) | `^0.9.0` | `pyproject.toml:42` |
| textstat (readability) | `^0.7.6` | `pyproject.toml:47` |

> **Note — version discrepancy.** An earlier methods draft cited package version **0.2.57** and textstat **0.7.7**; the released tree declares **0.2.51** and **`^0.7.6`**. Reconcile before submission and report the version actually used for the reported runs. Because the project uses `bump2version`, the tag corresponding to the run date is authoritative.

The full dependency closure is pinned by a Poetry lock file and restorable with a single `poetry install`.

### 1.2 Hardware profile

The pipeline is **CPU-only** and requires no local accelerator. All model inference occurs over remote HTTP endpoints; local computation is limited to repository traversal, abstract-syntax-tree parsing, readability statistics, and FAISS indexing. Consequently the wall-clock figures in §6 are dominated by **provider-side latency**, not local compute, and are reproducible on commodity hardware. Intermediate artefacts (file summaries, parsed code structures) are cached in local SQLite databases.

> ⚠️ **OUTSTANDING.** Report for the host that produced the reported runs: CPU model and core count, RAM, OS and kernel version, and the exact Python patch version. Do **not** publish API keys or the internal gateway hostname.

### 1.3 Endpoint configurations

Two configurations were used:

1. **Azure OpenAI** — managed deployment, API version `2024-08-01-preview`; used for the documentation-evaluation experiments and all text-embedding calls.
2. **Self-hosted LiteLLM gateway** — an OpenAI-compatible interface used for the multi-model comparison, so every backbone is driven through one client implementation with identical request construction.

**Model-identity verification is a methodological requirement, not a formality.** The gateway aliases some vendor-branded names to different backends. The released model registry records two such cases in-line: `glm-5` is **deprecated and returns HTTP 410** (superseded by `glm-5.1`), and the DeepSeek `v3.2` endpoint is **mis-aliased to Claude and must not be used** (`benchmark/shared.py:349–366`). Only endpoints whose identity was confirmed by a self-identification probe were admitted to the comparison. This step materially determines which vendor names may legitimately be attached to results.

> ⚠️ **OUTSTANDING.** State the geographic region of the Azure deployment, since region affects the latency figures in §6.

---

## 2. Model Versions

BioGuider is model-agnostic: a factory resolves a provider from configuration and returns a chat-model client, so identical agent code runs across providers (`bioguider/agents/agent_utils.py:132`).

**Backbone for the evaluation experiments.** Unless stated otherwise, **GPT-4o** was the backbone for *every* agent in both the Collect and Evaluation modules — no agent used a larger or different model than any other, so differences between modules cannot be attributed to model capacity. Embeddings used **text-embedding-3-small at 256 dimensions** (`bioguider/rag/rag.py:79,86`).

**Registry admitted to the multi-model comparison** (`benchmark/shared.py:349–366`):

| Family | Registry key | Gateway model string | Status |
|---|---|---|---|
| OpenAI | `gpt-4o` | `gpt-4o` | verified |
| OpenAI | `gpt-5.4` | `gpt-5.4` | verified; unusable in practice (§5.3) |
| Open weights | `gpt-oss` | `gpt-oss-120b` | verified |
| Moonshot | `kimi-k2.5` | `kimi-k2.5` | verified |
| Zhipu | `glm-5.1` | `glm-5.1` | verified (replaces deprecated `glm-5`, HTTP 410) |
| MiniMax | `minimax-m2.5` | `minimax-m2.5` | registered; not measured in §6 |

> ⚠️ **OUTSTANDING.** Reviewers require immutable identifiers rather than families. For each backbone report the exact API model string, the dated snapshot where the provider publishes one (e.g. `gpt-4o-2024-11-20`), and the date range of the experiments; these are recoverable from the run logs. For Azure-hosted models additionally report the deployment name and API version, since Azure pins a snapshot independently of the family name.

---

## 3. Generation Parameters

Agents are invoked through a single factory with a uniform decoding configuration (`bioguider/agents/agent_utils.py:132–175`).

**Table 1. Generation parameters (verified).**

| Parameter | Value | Source / note |
|---|---|---|
| Temperature | **0.0** | `agent_utils.py:132` (factory default). Decoding is greedy. |
| Temperature exceptions | `gpt-5`, `gpt-5.4`, `o1`, `o3` | `_TEMP_RESTRICTED_MODELS`, `agent_utils.py:54`. These reject a custom temperature; the parameter is omitted and the provider default applies. |
| Maximum output tokens | **16,384** | `agent_utils.py` factory default. Passed as `max_completion_tokens` for Azure API ≥ 2024-08-01-preview, else `max_tokens`. |
| Request timeout | **60 s** per call | `bioguider/settings.py:39` (`request_timeout`) |
| Structured output | JSON-schema constrained | Scored fields decoded against a Pydantic model, so scores are type-checked at the decoding boundary rather than parsed from free text. |
| Embedding model | text-embedding-3-small, **256** dims | `bioguider/rag/rag.py:79,86` |
| Chunking | **350** words, **100**-word overlap | `bioguider/rag/config.py:52–53` |
| Retrieval depth | top-*k* = **20** | `bioguider/rag/config.py:40` |

Sampling parameters not listed — `top_p`, presence penalty, frequency penalty, and stop sequences other than the ReAct terminator — are left at provider defaults and never overridden.

**Reasoning-family caveat.** Models that emit chain-of-thought into a separate response field require the full 16,384-token budget; an insufficient budget manifests as a *truncated response with empty content* rather than an explicit error. This is why the output budget is set uniformly high rather than tuned per model.

> ⚠️ **OUTSTANDING — accuracy correction.** The claim that temperature 0 is applied "uniformly to all agents" holds for the **agent factory**, but other components declare different defaults: `bioguider/conversation.py:22` = 0.1, `bioguider/settings.py:38` = 0.2, `bioguider/rag/config.py:46` = 0.7, `bioguider/generation/config.py:35` = 0.7. Before claiming uniform greedy decoding, either confirm these paths are unused in the reported experiments or restate the claim as scoped to the agent factory.

---

## 4. Prompts Used by Individual Agents

Every agent is driven by a **versioned prompt template stored in source control**, rendered by substituting runtime context into named placeholders. Templates are never assembled ad hoc at runtime.

### 4.1 Agent prompt inventory

**Table 2. Agents, templates, injected context, and output schemas.**

| Module | Agent | Prompt template (source module) | Runtime context injected | Output schema |
|---|---|---|---|---|
| Collect | Design (Plan) | `COLLECTION_PLAN_SYSTEM_PROMPT` (`collection_plan_step.py`) | Goal item, per-category file description, 2-level repository tree, tool descriptions, prior-round outputs, prior analysis | `PlanAgentResult` (list of tool-call actions) |
| Collect | Execute | `COLLECTION_EXECUTION_SYSTEM_PROMPT` (`collection_execute_step.py`) | Tool descriptions, current plan, ReAct scratchpad | ReAct trace terminated by a final answer |
| Collect | Observe | `COLLECTION_OBSERVE_SYSTEM_PROMPT` (`collection_observe_step.py`) | Goal item, repository tree, accumulated output, per-category observation constraints | `ObservationResult` (`Analysis`, `Thoughts`, `FinalAnswer`) |
| Identify | Plan / Execute / Observe | `identification_*_step.py` | Repository tree, candidate metadata files | Project type, primary language, repository metadata |
| Evaluate | README evaluator | `evaluation_readme_task.py` | README path and content, LICENSE path and summary, four readability indices | `StructuredEvaluationREADMEResult` |
| Evaluate | README free-form | `evaluation_readme_task.py` | Structured result plus README content | `FreeProjectLevelEvaluationREADMEResult` / `FreeFolderLevel…` |
| Evaluate | Installation evaluator | `evaluation_installation_task.py` | Concatenated installation-related files | `StructuredEvaluationInstallationResult` |
| Evaluate | User-guide evaluator | `evaluation_userguide_prompts.py` | User-guide content, readability indices, code-structure index | User-guide rubric schema |
| Evaluate | Tutorial evaluator | `evaluation_tutorial_task_prompts.py` | Tutorial content, readability indices, code-structure index | Tutorial rubric schema |
| Evaluate | Consistency checker | `consistency_*_step.py` | Documented API usage, indexed function/class signatures | Consistency verdicts |
| Tools | File summarizer | `agent_tools.py` | File content (truncated at 10 KiB), category-specific instruction | Free-text summary (cached) |
| Tools | Relevance classifier | `collection_task_utils.py` | File content or summary, goal-item description | Boolean relevance decision with rationale |

### 4.2 Cross-cutting prompting mechanisms

Two mechanisms are shared across agents and affect output quality independently of the templates:

1. **Two-stage chain-of-thought protocol** (`CommonAgentTwoSteps`, `bioguider/agents/common_agent_2step.py`). The model first produces free-form reasoning; a second call converts that reasoning into the schema-constrained object. This prevents schema constraints from truncating reasoning, and retains the reasoning trace as an auditable artefact for every scored item rather than discarding it.
2. **Brace escaping and content sanitation** applied to all injected repository content, so literal braces in source code cannot collide with template placeholders — a silent failure mode when documentation contains code (`bioguider/utils/utils.py`, `escape_braces`).

### 4.3 Benchmark repair prompts (verbatim)

The document-repair benchmark compares single-call prompting against the full pipeline. All single-call conditions share the assembly rule:

```
<preamble> + <corrupted document> + "\n\nOUTPUT THE COMPLETE FIXED DOCUMENT:"
```

and differ only in the preamble (`benchmark/shared.py:277–339`).

**Simple** (baseline; the condition measured in §6):

```text
Fix all errors in this document and output the corrected version:
```

**GPT-basic** (weakest control):

```text
Proofread and fix this document:
```

**Generic rubric** (criteria disclosed, no structured methodology):

```text
Fix all errors in this document and output the corrected version:
```

**BioGuider** (domain-specific single-call condition; shown in a four-backtick fence because the template itself contains triple-backtick markers):

````text
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

CORRUPTED DOCUMENT TO FIX:
````

**Table 3. Single-call prompt conditions.**

| Condition | Domain guidance | Rubric disclosed | Role |
|---|---|---|---|
| Simple | no | no | Baseline; source of the single-call cost figures in §6 |
| GPT-basic | no | no | Weakest control |
| Generic rubric | no | yes | Isolates rubric knowledge from structured methodology |
| BioGuider | yes | yes | Domain-specific single-call condition |

The **full pipeline uses none of these**: it is driven by the versioned agent templates in Table 2, which is why its token cost is an order of magnitude higher (§7).

> ⚠️ **OUTSTANDING.** `SKILL_GENERIC_PROMPT` and `SIMPLE_PROMPT` are byte-identical apart from a trailing blank line (`benchmark/shared.py:306,318`). If the generic-rubric condition is intended to differ textually — e.g. by actually stating the rubric — the template is unfinished and the two conditions are currently indistinguishable at the prompt level. Resolve before reporting them as separate conditions.

Recommend releasing the unabridged template set as Supplementary Material.

---

## 5. Stopping Rules and Retry Procedures

### 5.1 Stopping rules

Execution is bounded at three nested levels, so no repository can cause unbounded execution or cost.

**Table 4. Stopping rules (verified).**

| Level | Rule | Limit | Source |
|---|---|---|---|
| Plan–Execute–Observe loop | Observe terminates by emitting a non-null `FinalAnswer`; otherwise control returns to Design | `MAX_STEP_COUNT` = **30** node visits ≡ at most **10** complete P-E-O cycles | `utils/constants.py:52` (`3*10`) |
| Forced convergence | On the penultimate admissible cycle, Observe's instruction is replaced with an explicit directive to answer from evidence already gathered | Triggered at cycle **8** (`MAX_STEP_COUNT/3 − 2`) | `collection_observe_step.py:103`, `identification_observe_step.py:82` |
| ReAct tool loop (Collect Execute) | Halts on final answer or iteration budget; `\nObservation:` is the stop sequence | `max_iterations` = **30**, recursion limit **20** | `collection_execute_step.py:147` |
| ReAct tool loop (Identify, Docker generation) | As above | `max_iterations` = **10**, recursion limit **20** | `identification_execute_step.py:148`, `dockergeneration_execute_step.py:139,149` |
| File ingestion | Binary files skipped; oversized files truncated; notebooks reduced to code/markdown cells; HTML converted to text | `MAX_FILE_LENGTH` = **10 KiB** | `utils/constants.py:50` (`10*1024`) |

**Why forced convergence matters.** Rather than letting a non-converging collection loop fail with no output, BioGuider degrades gracefully to a best-effort answer from the evidence gathered so far. A bounded run therefore *always* returns a result, and the failure mode is **reduced recall rather than a missing record** — which is the correct behaviour for a survey instrument.

> ⚠️ **OUTSTANDING — unverifiable claim.** An earlier draft listed a chain-of-thought truncation limit of **40,000 characters (`COT_MAX_CHARS`)**. No such identifier or literal exists anywhere in the source tree. Either locate the actual mechanism and cite it, or delete the row; do not report it as-is.

### 5.2 Retry procedures

Reliability is handled at four levels, from transport faults up to whole-repository orchestration.

1. **Transport-level retries.** Provider SDK clients retry transient HTTP failures (rate limits, timeouts, 5xx) with built-in exponential backoff.
2. **Agent-level retries.** Every agent invocation is wrapped in a Tenacity policy of **at most 5 attempts** with incrementing backoff — initial delay 1 s, increment 3 s, capped at 10 s, i.e. waits of **1, 4, 7, 10 s (≤ 22 s cumulative)** (`common_agent.py:97–98`, `common_agent_2step.py:63–64`). The policy covers both the model call and post-processing.
3. **Schema-repair retries.** When a response fails post-processing validation — malformed action, hallucinated file path, schema violation — a `RetryException` is raised and **its message is appended to the prompt as an additional turn on the next attempt**, so the model is shown its own error rather than merely re-sampled. Because decoding is greedy, this feedback is what makes a retry informative: an identical prompt at temperature 0 would otherwise reproduce the identical failure.
4. **Step-level isolation.** In batch evaluation each stage (identify, README, installation, user guide, tutorial) runs independently with status `completed`, `failed`, or `skipped`. A failure in one stage does not abort the others; the repository is reported as `completed_with_errors` with a per-stage message. Stages with unmet dependencies (e.g. submission-requirements scoring, which needs README and installation results) are marked `skipped` rather than silently omitted. Exhausting the retry budget therefore yields a recorded failure for one category, **not a lost repository**.

### 5.3 Observed limits of the retry policy

The agent-level policy retries HTTP 429 but **not** connection-level faults, and this asymmetry determined which backbones could be measured (§6.3). Two additional controls were added to the benchmark harness to characterise this: a configurable per-call timeout (`PHAROKKA_TIMEOUT`, default 600 s) and a configurable client retry count for transient disconnects (`PHAROKKA_MAX_RETRIES`, default 4).

> ⚠️ **OUTSTANDING.** Report the **realised failure rate** from the batch records: repositories attempted; completing all stages; completing with ≥1 failed stage; per-stage failure counts; and the most common causes. These are directly recoverable from the per-repository status fields and turn the retry policy from a design claim into a measured one.

---

## 6. Processing Time

### 6.1 Instrumentation and benchmark design

Wall-clock duration is recorded per benchmark cell around the complete repair operation. Measurements come from the **document-repair benchmark**: a single user-guide document (`docs/plotting.md` from *pharokka*) is corrupted by **deterministic** error injection at four densities — **40, 100, 150, 200** injected errors per category — and each backbone repairs the byte-identical corrupted document under two strategies: a **single-call prompt** ("simple", §4.3) and the **full pipeline** (evaluate → generate → polish).

Replication differs by strategy: **pipeline cells = 4–5 replicates; single-call cells = 1 replicate**. Single-call figures are therefore point measurements indicating magnitude, not precise estimates. A failed call is recorded with zero tokens and **excluded from both the latency and token statistics**, so an aborted request is never reported as a completion.

Artefacts: `outputs/pipeline_tokentime/run_*/STRESS_TEST_TABLE.csv`, `outputs/simpleprompt_tokentime/run_*/STRESS_TEST_TABLE.csv`.

### 6.2 Measured latency

**Table 5. Wall-clock time per document (seconds; median, with range).**

| Backbone | Single-call: median (range) | Full pipeline: median (range) |
|---|---|---|
| gpt-oss-120b | 13.6 (10.9–20.7) | 114.9 (88.6–304.5) |
| GPT-4o | 14.7 (13.6–74.2) | 202.2 (121.0–389.2) |
| Kimi-K2.5 | 46.9 (28.3–92.7) | 426.3 (281.2–733.5) |
| GLM-5.1 | 105.7 (87.0–272.1) | 984.7 (556.6–1,434.9) |

Latency spans two orders of magnitude, driven jointly by backbone speed and by the number of calls a strategy issues: single-call repairs finish in tens of seconds; the pipeline issues several sequential calls and takes minutes. **Backbone speed is the larger source of spread** — GLM-5.1 is roughly **7×** slower than gpt-oss-120b under the identical pipeline.

Time scales weakly and non-monotonically with error density, because it is dominated by provider-side queuing: GPT-4o's pipeline time was 243 s at 40 errors and 240 s at 200, with an intermediate *minimum* of 180 s at 100.

### 6.3 Configurations that could not be measured

Two backbones could not be characterised, and **the cause is the serving endpoint, not the pipeline**:

- **GPT-5.4** — no completed pipeline run in **23 consecutive attempts**; every attempt exhausted its retry budget against HTTP 429, because the gateway enforces a tokens-per-minute quota that a multi-call, large-context strategy saturates immediately. Single-call attempts returned zero tokens after stalling ≈ 950 s.
- **GLM-5.1** — completed the pipeline at three of four error levels (40, 150, 200) but never at 100; completed single-call repairs at three of four levels (40, 100, 150) but never at 200. Failures were client-side timeouts and mid-stream server disconnections consistent with its very long generation latency.

These exclusions are reported explicitly because omitting them would misrepresent the comparison as complete. Both failure modes are **reproducible in ≈ 2 minutes** without running the benchmark, via the standalone probe `benchmark/repro_slow_models.py`, so a reviewer can confirm they are endpoint properties.

---

## 7. Token Use

### 7.1 Instrumentation

Token consumption is **measured at every model call**, never estimated from document lengths.

- **Pipeline strategy:** a usage-metadata callback handler is attached to the backbone client *at construction*, so it propagates to every invocation the strategy makes — evaluation task, content generator, and markdown polish pass — and per-call prompt/completion/total counts are summed into one per-document figure.
- **Single-call strategies:** the same three counts are read from the response metadata of the single call.

Both paths persist counts alongside wall-clock duration in the per-cell record (CSV and JSON). Usage is aggregated hierarchically — per call, per agent step, per evaluation stage, per repository — and the per-repository total is persisted with the evaluation result.

### 7.2 Measured consumption

**Table 6. Token consumption per document (median; prompt / completion split).**

| Backbone | Single-call: total (prompt / completion) | Full pipeline: total (prompt / completion) |
|---|---|---|
| GPT-4o | 5,307 (2,756 / 2,576) | 53,116 (42,666 / 10,196) |
| gpt-oss-120b | 5,862 (2,816 / 3,084) | 75,446 (49,609 / 24,870) |
| Kimi-K2.5 | 8,710 (2,800 / 5,842) | 98,718 (51,646 / 46,536) |
| GLM-5.1 | 11,024 (2,807 / 8,206) | 83,991 (51,024 / 32,768) |

*n*: single-call 4–6 runs per backbone; pipeline 18 runs per backbone (6 for GLM-5.1, see §6.3).

**The dominant effect is strategy, not backbone.** The pipeline consumes roughly an order of magnitude more tokens than a single call (GPT-4o: 53.1 K vs 5.3 K), because it issues several calls and each carries repository context. Within a strategy, **prompt-side cost is nearly constant across backbones** (2.76–2.82 K single-call) while **completion tokens vary threefold** (2.58 K–8.21 K) — i.e. backbones differ mainly in verbosity, not in what they are shown.

### 7.3 Scaling with error density

Token use rises only modestly as injected errors increase **fivefold** (40 → 200), because the repaired *document* — not the error list — dominates the output budget. Pipeline means:

| Backbone | 40 errors | 200 errors | Increase |
|---|---|---|---|
| GPT-4o | 52.5 K | 54.1 K | **+3 %** |
| gpt-oss-120b | 69.1 K | 76.3 K | +10 % |
| Kimi-K2.5 | 89.9 K | 106.4 K | +18 % |

Error density is therefore a **weak cost driver** relative to the choice of strategy and backbone — a useful result, since it means cost can be budgeted from document size rather than from defect count.

> ⚠️ **OUTSTANDING.** The tables above characterise cost on the *single-document repair* benchmark. They do **not** characterise the batch evaluation over the full repository cohort reported in Results. Aggregate and report: (i) per-repository total, prompt, and completion tokens (median and IQR); (ii) the same broken down by evaluation stage, showing where cost concentrates; (iii) end-to-end wall-clock per repository (median and IQR); and (iv) estimated monetary cost per repository at published rates. All are already persisted in the result objects, so this requires aggregation, not re-running. Also state the **concurrency** used for the batch run (repositories are evaluated by a synchronous, blocking function, with parallelism supplied by the caller); wall-clock totals are otherwise uninterpretable.

---

## 8. Determinism, Caching, and Replication

Temperature-0 decoding makes runs **approximately, not exactly** reproducible: providers do not guarantee bitwise determinism, and endpoint-side model updates change outputs over time. This is precisely why §2 requests dated snapshots.

Two mechanisms further constrain variance:

- **Caching.** File summaries and parsed code structures are cached in SQLite keyed by repository and file, so repeated evaluations reuse identical intermediate representations rather than regenerating them.
- **Deterministic corruption.** In the error-injection benchmark, corruption is generated deterministically, so a given (document, error-count) pair yields **byte-identical** input across every backbone compared. Differences between backbones are therefore attributable to the backbones, not to differing inputs.

Benchmark cells are executed as multiple replicates and pooled per (backbone, error level) by **summing per-category counts before computing rates**, so per-category rates are weighted by the number of injected errors rather than averaged over unequally sized replicates.

---

## 9. Reproducibility Artefacts

The implementation — every prompt template, rubric schema, and the benchmark harness — is available at **[repository URL]**. Reproduction requires provider credentials supplied through environment variables; no credentials are distributed with the code.

The harness ships a standalone probe (`benchmark/repro_slow_models.py`) that reproduces the two endpoint limits in §6.3 — the tokens-per-minute rate-limit wall and the long-latency disconnection behaviour — in ≈ 2 minutes without running the full benchmark.

Recommended Supplementary Material:

1. Unabridged prompt templates for all agents.
2. Pydantic schemas defining each rubric.
3. The pinned dependency lock file.
4. The list of evaluated repositories **with a pinned commit hash for each** — documentation changes over time, and un-pinned repositories would make the scores unreproducible.
5. The per-repository token and timing records underlying §6–§7.

---

## Appendix A. Outstanding items checklist

| # | Item | Section | Why it matters |
|---|---|---|---|
| 1 | Host CPU/RAM/OS/kernel, Python patch version | §1.2 | Latency figures uninterpretable without it |
| 2 | Azure deployment region | §1.3 | Region affects latency |
| 3 | Reconcile package version (0.2.51 vs 0.2.57) and textstat (0.7.6 vs 0.7.7) | §1.1 | Version claims must match the released tree |
| 4 | Dated model snapshots + experiment date range | §2 | Reviewers require immutable identifiers |
| 5 | Scope or correct the "uniform temperature 0" claim | §3 | Four components declare non-zero defaults |
| 6 | Locate or delete the `COT_MAX_CHARS` = 40,000 stopping rule | §5.1 | Identifier does not exist in source |
| 7 | Resolve identical `SKILL_GENERIC_PROMPT` / `SIMPLE_PROMPT` | §4.3 | Two "conditions" are currently indistinguishable |
| 8 | Realised failure rate from batch records | §5.2 | Turns a design claim into a measured one |
| 9 | Batch-cohort token/time/cost aggregation + concurrency | §7.3 | Benchmark cost ≠ cohort cost |
| 10 | Add replicates to single-call cells, or state n=1 in captions | §6.1 | Currently point measurements |
