# BioGuider Multi-Agent Evaluation Framework
BioGuider implements a modular, multi-agent system to systematically evaluate the quality, completeness, and correctness of documentation in open-source biomedical software repositories. The evaluation workflow is decomposed into two modules: (i) the Collect Module and (ii) the Evaluation Module. Each module is implemented as a coordinated set of large language model (LLM) agents equipped with domain-specific tools.

## 1. Collect Module
The Collect Module identifies, retrieves, and structures repository files that are relevant to documentation evaluation. Its primary objective is to assemble a curated corpus of documentation artifacts, including README files, installation instructions, user guides/API references, and tutorials or vignettes, from heterogeneous repository layouts.

### 1.1 Tooling
To support flexible and repository-agnostic exploration, BioGuider provides the following tools to LLM agents:
- Directory Reader: traverses the repository structure to enumerate files and subdirectories.
- File Reader: loads the full content of text-based files (e.g., Markdown, R Markdown, Python, R source files).
- Relevance Classifier: determines whether a file is relevant to a target documentation category (README, Installation, User Guide/API, Tutorial/Vignette).
- Content Summarizer: produces concise semantic summaries of documentation files to support downstream reasoning.
- Content Extractor: extracts specific sections (e.g., installation steps, tutorial workflows, API usage examples) from larger documents.
- Python AST REPL Tool: executes Python code in a controlled environment to perform repository-level analyses (e.g., counting source files, inspecting abstract syntax trees, or computing simple statistics).

### 1.2 Agent Roles
The Collect Module consists of three specialized agents operating in a plan-execute-verify loop:
- Design Agent: generates a structured sequence of actions that may combine tool usage and reasoning steps (e.g., traversing ./man or ./vignettes, summarizing README/.Rd files, extracting notebook tutorial sections).
- Execute Agent: carries out the action plan produced by the Design Agent, invoking the specified tools in sequence and recording intermediate outputs (file contents, summaries, execution results).
- Observe Agent: inspects the outputs produced by the Execute Agent and evaluates whether the collection objective has been satisfied. If required artifacts are missing, incomplete, or ambiguous, it produces corrective feedback that can trigger additional planning cycles.

## 2. Evaluation Module
The Evaluation Module assesses collected documentation artifacts against predefined quality criteria (completeness, clarity, reproducibility, and technical correctness). Evaluation is performed per documentation category using category-specific rubrics and structured LLM outputs.

### 2.1 Evaluation Pipeline
For each documentation category (README, Installation, User Guide/API, Tutorial/Vignette), BioGuider applies a consistent evaluation pipeline:
1. File normalization and sanitation. Binary files are skipped; oversized files are excluded; HTML is converted to text; notebooks are reduced to code and markdown; braces are escaped to avoid prompt template collisions.
2. Readability analysis. BioGuider computes readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG) and provides them to the evaluator.
3. Structured evaluation. An LLM returns a schema-constrained evaluation that scores category-specific criteria and provides targeted improvement suggestions.
4. Free-form evaluation. For README and Installation, a second LLM pass expands the structured evaluation into detailed, human-readable feedback with quoted snippets and improvement comments.
5. Score aggregation. Subscores are combined via weighted aggregation to compute an overall category score.

### 2.2 Documentation Quality Rubrics
Each category is evaluated with tailored criteria:
- README: availability, readability, project purpose, hardware/software specifications, dependency clarity, license, and contributor/author information.
- Installation: presence of installation instructions, dependency specification, OS compatibility, hardware requirements, and tutorial completeness.
- User Guide/API: readability, context and purpose, error handling guidance, and coverage of usage examples.
- Tutorial/Vignette: readability, setup and dependencies, reproducibility, structure and navigation, executable code quality, result verification, and performance/resource notes.

### 2.3 Code-Documentation Consistency Evaluation
For User Guides/APIs and Tutorials/Vignettes, BioGuider additionally evaluates consistency between documentation and source code.
1. Source Code Structure Indexing. BioGuider scans repository code to build a structured index capturing function/class names, argument signatures, return values, and inline documentation/docstrings.
2. Consistency Verification. The Evaluation Agent compares documented usage against the index to verify that referenced functions/classes exist, argument names and order are correct, and documented behavior matches source-level definitions.

This consistency analysis identifies outdated examples, incorrect API usage, and documentation-implementation mismatches that may hinder reproducibility or usability.

### 2.4 Representative LLM Prompts
To increase transparency and reproducibility, we include representative prompt blocks used by the Evaluation Module (placeholders indicate runtime content injection).

**General evaluation instruction**
```text
Please also clearly explain your reasoning step by step. Now, let's begin the evaluation.
```

**README structured evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of README files in software repositories.
Your task is to analyze the provided README file and generate a structured quality assessment based on the following criteria.
If a LICENSE file is present in the repository, its content will also be provided to support your evaluation of license-related criteria.
You must provide the evaluation score in your response.
---
### Evaluation Criteria
1. Available: Is the README accessible and present? Output: Yes or No
2. Readability: Evaluate based on readability metrics AND identify specific errors/issues in the text.
   - You must identify and list ALL errors and anomalies (typos, malformed links, markdown errors, image syntax errors, domain term errors, inconsistencies, formatting issues, and other anomalies).
   - For each error, provide the exact text snippet, error type, suggested correction, and explanation.
3. Project Purpose: Is the project's goal or function clearly stated? Output: Yes or No
4. Hardware and Software Requirements: Are hardware/software specs and compatibility details included?
5. Dependencies: Are all necessary software libraries and dependencies clearly listed?
6. License Information: Is license type clearly indicated?
7. Author/Contributor Info: Are contributor or maintainer details provided?
8. Overall Score: Give an overall quality rating of the README.
---
### Readability Metrics
Flesch Reading Ease: {flesch_reading_ease}
Flesch-Kincaid Grade Level: {flesch_kincaid_grade}
Gunning Fog Index: {gunning_fog_index}
SMOG Index: {smog_index}
---
### README Path
{readme_path}
### README Content
{readme_content}
### LICENSE Path
{license_path}
### LICENSE Summarized Content
{license_summarized_content}
```

**Installation structured evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of installation information in software repositories.
Your task is to analyze the provided files related to installation and generate a structured quality assessment based on the following criteria.
---
### Evaluation Criteria
1. Installation Available: Is the installation section in document (like README.md or INSTALLATION)?
2. Installation Tutorial: Is the step-by-step installation tutorial provided?
3. Number of required Dependencies Installation: The number of dependencies required to install.
4. Compatible Operating System: Is the compatible operating system described?
5. Hardware Requirements: Are hardware requirements described?
6. Overall Score: Give an overall quality rating of the installation information.
---
### Installation Files Provided
{installation_files_content}
```

**User guide evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of user guide in software repositories.
Your task is to analyze the provided files related to user guide and generate a structured quality assessment based on the following criteria.
---
1. Readability AND Error Detection:
   - Use Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG.
   - You must scan for and identify ALL error instances (typos, malformed links, markdown/RMarkdown errors, bio term errors, function name errors, inline code formatting errors, and other anomalies).
   - List each occurrence separately; do not group similar errors.
2. Arguments and Clarity: describe arguments and their usage with concrete improvement suggestions.
3. Return Value and Clarity: describe return values and meaning with improvement suggestions.
4. Context and Purpose: describe context and purpose with improvement suggestions.
5. Error Handling: describe error handling with improvement suggestions.
6. Usage Examples: describe usage examples with improvement suggestions.
7. Overall Score: output 0-100.
---
### User Guide Content
{userguide_content}
```

**Tutorial evaluation (extended excerpt)**
```text
You are an expert in evaluating the quality of tutorials in software repositories.
Your task is to analyze the provided tutorial file and generate a structured quality assessment based on the following criteria.
---
1. Readability AND Error Detection:
   - Use Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG.
   - You must scan for and identify ALL error instances (typos, malformed links, markdown/RMarkdown errors, bio term errors, function name errors, inline code formatting errors, and other anomalies).
   - List each occurrence separately; do not group similar errors.
2. Coverage: whether it covers major steps, dependencies, prerequisites, setup, and example usage.
3. Reproducibility: whether it provides a clear description of reproducibility.
4. Structure and Navigation: logical sections, TOC/anchors, time estimates.
5. Executable Code Quality: executable and idiomatic code, no hard-coded paths.
6. Result Verification: expected outputs and acceptance criteria.
7. Performance and Resource Notes: CPU/GPU usage, memory, runtime estimates.
---
### Tutorial File Content
{tutorial_file_content}
```

## 3. Implementation Details and Experimental Configuration

This section reports the implementation parameters required to reproduce the results, following the reporting dimensions requested during review: per-agent prompts, model versions, generation parameters, stopping rules, retry procedures, processing time, token consumption, and execution environments. All values below correspond to the released BioGuider implementation (version 0.2.57); the parameter names in parentheses are the corresponding configuration keys, so that every reported value can be traced to a specific location in the source code.

### 3.1 Software Implementation and Execution Environment

BioGuider is implemented in Python (≥3.11) and distributed as an installable package with a Poetry-managed dependency lock file, so that the complete dependency closure is pinned and restorable with a single `poetry install` command. Agent orchestration uses LangChain 0.3.20 and LangGraph 0.3.11; provider bindings use `langchain-openai` 0.3.8, `langchain-anthropic` 0.3.10, `langchain-deepseek` 0.1.2, and `langchain-google-genai` 2.1.4. Retrieval uses FAISS (`faiss-cpu` 1.11.0); retry policies use Tenacity 9.1.2; readability statistics use `textstat` 0.7.7 with `pyphen` 0.17.2 for syllable counting; token counting uses `tiktoken` 0.9.0. Intermediate artifacts (file summaries and parsed code structures) are cached in local SQLite databases.

The pipeline is CPU-only and requires no local accelerator: all model inference is performed through remote HTTP endpoints, and the only local computation is repository traversal, abstract-syntax-tree parsing, readability statistics, and FAISS indexing of embedding vectors. Consequently the wall-clock time reported in Section 3.8 is dominated by model-provider latency rather than by local compute, and the pipeline can be reproduced on commodity hardware.

Two endpoint configurations were used. (i) *Azure OpenAI*: a managed deployment (API version 2024-08-01-preview) used for the documentation-evaluation experiments and for all text-embedding calls. (ii) *A self-hosted LiteLLM gateway* exposing an OpenAI-compatible interface, used for the multi-model comparison experiments so that all models could be driven through a single client implementation with identical request construction. Because the gateway aliases some vendor-branded model names to different backends, model identities were verified before use by issuing a self-identification probe to every candidate endpoint, and only endpoints whose identity could be confirmed were admitted to the comparison; this verification step is reported here because it materially affects which vendor names may be attached to the results.

> ⚠️ **To complete before submission.** Report the host on which the reported runs were executed: CPU model and core count, RAM, operating system and kernel, and the exact Python patch version (the released package requires ≥3.11; our development host runs 3.13.9). Also state the geographic region of the Azure OpenAI deployment, since endpoint region affects the latency figures in Section 3.8. Do **not** reproduce API keys or the internal gateway hostname in the manuscript.

### 3.2 Language Models and Model Versions

BioGuider is model-agnostic: a single factory function resolves a provider from configuration (`LLM_PROVIDER`) and returns a chat-model client, so that the identical agent code runs unchanged across providers. The supported provider families are Azure OpenAI and OpenAI (GPT family), MiniMax, Moonshot (Kimi), and DeepSeek; open-weight models are accessed through the OpenAI-compatible gateway.

Unless stated otherwise, all documentation-evaluation experiments used **GPT-4o** as the backbone for every agent in both the Collect and Evaluation modules — that is, no agent used a different or larger model than any other, so differences between modules cannot be attributed to model capacity. Text embeddings for the retrieval component used **text-embedding-3-small** (256 dimensions).

The multi-model comparison experiments evaluated the following backbones under otherwise identical pipeline settings: GPT-4o, GPT-5.4, gpt-oss-120b (open weights), Kimi-K2.5 (Moonshot), GLM-5 (Zhipu), and MiniMax-M2.5.

> ⚠️ **To complete before submission.** Reviewers of LLM-based systems generally require immutable version identifiers rather than model families. For each model in the table above, report the exact API model string and, where the provider publishes them, the dated snapshot identifier (for example `gpt-4o-2024-11-20`) and the date range over which the experiments were executed; the values are recoverable from the run logs. For the Azure-hosted models, additionally report the deployment name and API version (2024-08-01-preview), since Azure deployments pin a snapshot independently of the model family name.

### 3.3 Generation Parameters

All agents are invoked with a single, uniform decoding configuration (Table M1). Decoding is greedy: temperature is fixed at 0 for every agent and every call, which is the principal determinant of run-to-run stability in this system (see Section 3.7). Sampling parameters that are not listed — `top_p`, presence penalty, frequency penalty, and stop sequences other than the ReAct terminator noted below — are left at provider defaults and are never overridden by BioGuider.

**Table M1. Generation parameters.**

| Parameter | Value | Notes |
|---|---|---|
| Temperature | 0.0 | Applied uniformly to all agents. Omitted for reasoning-family models (`gpt-5*`, `o1`, `o3`), which reject a custom temperature; those models run at the provider default. |
| Maximum output tokens | 16,384 | Passed as `max_completion_tokens` for Azure API versions ≥ 2024-08-01-preview and as `max_tokens` otherwise. |
| Maximum input tokens | 128,000 | Context budget of the deployed backbone. |
| Structured output | JSON-schema constrained | Every scored field is produced through schema-constrained decoding (LangChain `with_structured_output`) against a Pydantic model, so scores are type-checked at the decoding boundary rather than parsed from free text. |
| Request timeout | 60 s per call | Provider-level timeout. |
| Embedding model | text-embedding-3-small, 256 dims | Retrieval component. |
| Chunking | 350 words, 100-word overlap | Retrieval component. |
| Retrieval depth | top-*k* = 20 | FAISS similarity search. |

A single caveat applies to reasoning-family backbones: because such models emit chain-of-thought into a separate response field, they require the full 16,384-token output budget, and an insufficient budget manifests as a truncated response with empty content rather than as an explicit error. This is the reason the output budget is set uniformly high rather than tuned per model.

### 3.4 Prompts Used by Individual Agents

Every agent is driven by a versioned prompt template stored in source control, and every template is rendered by substituting runtime context (repository file tree, intermediate tool outputs, file contents, readability statistics) into named placeholders. Templates are never assembled ad hoc at runtime. Table M2 inventories the agents, their prompt templates, and their output schemas.

**Table M2. Agent prompt inventory.**

| Module | Agent | Prompt template (source module) | Runtime context injected | Output schema |
|---|---|---|---|---|
| Collect | Design (Plan) | `COLLECTION_PLAN_SYSTEM_PROMPT` (`collection_plan_step.py`) | Goal item, per-category file description, 2-level repository tree, tool descriptions, prior-round outputs, prior analysis and thoughts | `PlanAgentResult` (list of tool-call actions) |
| Collect | Execute | `COLLECTION_EXECUTION_SYSTEM_PROMPT` (`collection_execute_step.py`) | Tool descriptions, current plan, ReAct scratchpad | ReAct trace terminated by a final answer |
| Collect | Observe | `COLLECTION_OBSERVE_SYSTEM_PROMPT` (`collection_observe_step.py`) | Goal item, repository tree, accumulated intermediate output, per-category observation constraints | `ObservationResult` (`Analysis`, `Thoughts`, `FinalAnswer`) |
| Identify | Plan / Execute / Observe | `identification_*_step.py` | Repository tree, candidate metadata files | Project type, primary language, repository metadata |
| Evaluate | README evaluator | `evaluation_readme_task.py` | README path and content, LICENSE path and summary, four readability indices | `StructuredEvaluationREADMEResult` |
| Evaluate | README free-form evaluator | `evaluation_readme_task.py` | Structured result plus README content | `FreeProjectLevelEvaluationREADMEResult` / `FreeFolderLevelEvaluationREADMEResult` |
| Evaluate | Installation evaluator | `evaluation_installation_task.py` | Concatenated installation-related files | `StructuredEvaluationInstallationResult` |
| Evaluate | User-guide evaluator | `evaluation_userguide_prompts.py` | User-guide content, readability indices, code-structure index | User-guide rubric schema |
| Evaluate | Tutorial evaluator | `evaluation_tutorial_task_prompts.py` | Tutorial content, readability indices, code-structure index | Tutorial rubric schema |
| Evaluate | Consistency checker | `consistency_*_step.py` | Documented API usage, indexed function/class signatures | Consistency verdicts |
| Tools | File summarizer | `agent_tools.py` | File content (truncated at 10 KiB), category-specific summarization instruction | Free-text summary (cached) |
| Tools | Relevance classifier | `collection_task_utils.py` | File content or summary, goal-item description | Boolean relevance decision with rationale |

Section 2.4 reproduces the four category-level evaluation prompts verbatim. To keep the main text readable while remaining fully reproducible, we recommend that the complete, unabridged set of templates be provided as Supplementary Material and cross-referenced here.

Two prompting mechanisms are shared across agents and are reported because they affect output quality independently of the templates themselves. First, a **two-stage chain-of-thought protocol** (`CommonAgentTwoSteps`): the model first produces free-form reasoning, and a second call converts that reasoning into the schema-constrained object. This separation prevents schema constraints from truncating reasoning, and it also means the reasoning trace is retained as an auditable artifact for every scored item rather than being discarded. Second, **brace escaping and content sanitation** are applied to all injected repository content so that literal braces in source code cannot collide with template placeholders — a silent failure mode when documentation contains code.

### 3.5 Stopping Rules and Iteration Limits

Agent execution is bounded at three nested levels (Table M3), so that no repository can cause unbounded execution or unbounded cost.

**Table M3. Stopping rules.**

| Level | Rule | Limit |
|---|---|---|
| Plan–Execute–Observe loop | The Observe agent terminates the loop by emitting a non-null `FinalAnswer`; otherwise control returns to the Design agent for another cycle. | Graph recursion limit `MAX_STEP_COUNT` = 30 node visits ≡ at most 10 complete P-E-O cycles |
| Forced convergence | On the penultimate admissible cycle the Observe agent's instruction is replaced with an explicit directive to emit a final answer from the information already gathered. | Triggered at cycle 8 |
| ReAct tool loop (Collect Execute) | The execution agent halts on a final answer or on exhausting its iteration budget; `\nObservation:` is used as the stop sequence. | `max_iterations` = 30, recursion limit = 20 |
| ReAct tool loop (Identify, Docker generation) | As above. | `max_iterations` = 10, recursion limit = 20 |
| Chain-of-thought length | Reasoning output is truncated before the structured-output stage. | 40,000 characters (≈10,000 tokens), configurable via `COT_MAX_CHARS` |
| File ingestion | Binary files are skipped; oversized files are truncated for summarization; notebooks are reduced to code and markdown cells; HTML is converted to text. | `MAX_FILE_LENGTH` = 10 KiB |

The forced-convergence rule warrants explicit mention: rather than allowing a non-converging collection loop to fail with no output, BioGuider degrades gracefully to a best-effort answer computed from the evidence gathered so far. This design choice means that a bounded run always returns a result, and that the failure mode is reduced recall rather than a missing record.

### 3.6 Retry Procedures and Failure Handling

Reliability is handled at four levels, from transport faults up to whole-repository orchestration.

1. **Transport-level retries.** Provider SDK clients retry transient HTTP failures (rate limits, timeouts, 5xx) with the client's built-in exponential backoff.
2. **Agent-level retries.** Every agent invocation is wrapped in a Tenacity retry policy of **at most 5 attempts** with incrementing backoff (initial delay 1 s, increment 3 s, capped at 10 s; i.e. waits of 1, 4, 7, and 10 s, ≤22 s of cumulative backoff). The policy covers both the model call and post-processing.
3. **Schema-repair retries.** When a response fails post-processing validation — a malformed action, a hallucinated file path, or a schema violation — a `RetryException` is raised and its message is **appended to the prompt as an additional turn on the next attempt**, so the model is shown its own error rather than merely re-sampled at the same temperature. Because decoding is greedy, this feedback is what makes a retry informative: an identical prompt at temperature 0 would otherwise reproduce the identical failure.
4. **Step-level isolation.** In batch evaluation each stage (identify, README, installation, user guide, tutorial) is executed independently and its status recorded as `completed`, `failed`, or `skipped`. A failure in one stage does not abort the remaining stages; the repository is reported as `completed_with_errors` with a per-stage error message. Stages with unmet dependencies (for example, submission-requirements scoring, which needs both README and installation results) are marked `skipped` rather than being silently omitted. Exhausting the retry budget therefore yields a recorded failure for one category, not a lost repository.

> ⚠️ **To complete before submission.** Reviewers frequently ask for the realised failure rate. From the batch run records, report: the number of repositories attempted; the number completing all stages; the number completing with at least one failed stage; the per-stage failure counts; and the most common failure causes. These are directly recoverable from the per-repository status fields, and reporting them turns the retry policy from a design claim into a measured one.

### 3.7 Determinism, Caching, and Replication

Temperature-0 decoding makes runs approximately, but not exactly, reproducible: providers do not guarantee bitwise determinism, and endpoint-side model updates can change outputs over time. This is the reason Section 3.2 requests dated model snapshots. Two mechanisms further constrain variance. File summaries and parsed code structures are cached in SQLite and keyed by repository and file, so repeated evaluations of the same repository reuse identical intermediate representations rather than regenerating them. In the error-injection benchmark, corruption is generated deterministically, so a given (document, error-count) pair yields byte-identical corrupted input across every model compared — differences between models are therefore attributable to the models rather than to differing inputs. Benchmark cells are executed as multiple replicates and pooled per (model, error level) by summing per-category counts before computing rates, so that per-category rates are weighted by the number of injected errors rather than averaged over unequally sized replicates.

### 3.8 Processing Time and Token Consumption

**Instrumentation.** Token consumption is recorded at every model call rather than estimated from document lengths. For the multi-call pipeline strategy, a usage-metadata callback handler is attached to the backbone client at construction, so it propagates to every invocation the strategy makes — the evaluation task, the content generator, and the surface-markdown polish pass — and the per-call prompt, completion, and total counts are summed into a single per-document figure. For the single-call strategies the same three counts are read from the response metadata of the one call issued. Both paths persist the counts alongside wall-clock duration in the per-cell result record. A call that fails (provider timeout or rate-limit rejection) is recorded with zero tokens and is excluded from both the token and the latency statistics below, so that an aborted request is never reported as though it were a completed one.

**Benchmark design.** The figures below are measured on the document-repair benchmark: a single user-guide document (`docs/plotting.md` from the *pharokka* repository) is corrupted by deterministic error injection at four densities — 40, 100, 150, and 200 injected errors per category — and each backbone repairs the identical corrupted document under two strategies: a **single-call prompt** ("simple"), and the **full pipeline** (evaluate → generate → polish). Because injection is deterministic, every backbone at a given error level receives byte-identical input. Replication differs by strategy: pipeline cells were executed as 4–5 replicates, whereas single-call cells were executed once per cell, so the single-call figures are point measurements and should be read as indicative of magnitude rather than as precise estimates.

**Token consumption.** Table M4 reports measured tokens per document repair, pooled across the four error levels. The dominant effect is strategy, not backbone: the full pipeline consumes roughly an order of magnitude more tokens than a single call (for GPT-4o, 53.1 K versus 5.3 K), because it issues several calls per document and each carries repository context. Within a strategy, backbones differ mainly in how verbose their completions are — the prompt-side cost is nearly constant across backbones, while completion tokens vary by a factor of three.

**Table M4. Measured token consumption per document (median; prompt / completion split), by backbone and strategy.**

| Backbone | Single-call: total (prompt / completion) | Full pipeline: total (prompt / completion) |
|---|---|---|
| GPT-4o | 5,307 (2,756 / 2,576) | 53,116 (42,666 / 10,196) |
| gpt-oss-120b | 5,862 (2,816 / 3,084) | 75,446 (49,609 / 24,870) |
| Kimi-K2.5 | 8,710 (2,800 / 5,842) | 98,718 (51,646 / 46,536) |
| GLM-5.1 | 11,024 (2,807 / 8,206) | 83,991 (51,024 / 32,768) |

*n* per cell: single-call 4–6 runs per backbone; pipeline 18 runs per backbone (6 for GLM-5.1, see coverage note).

**Processing time.** Table M5 reports wall-clock time for the same runs. Latency spans two orders of magnitude and is driven jointly by backbone speed and by the number of calls a strategy issues: single-call repairs complete in tens of seconds, whereas the pipeline issues several sequential calls and takes minutes. Backbone speed is the larger source of spread — GLM-5.1 is roughly seven times slower than gpt-oss-120b under the identical pipeline.

**Table M5. Measured wall-clock time per document (seconds; median, with range), by backbone and strategy.**

| Backbone | Single-call: median (range) | Full pipeline: median (range) |
|---|---|---|
| gpt-oss-120b | 13.6 (10.9–20.7) | 114.9 (88.6–304.5) |
| GPT-4o | 14.7 (13.6–74.2) | 202.2 (121.0–389.2) |
| Kimi-K2.5 | 46.9 (28.3–92.7) | 426.3 (281.2–733.5) |
| GLM-5.1 | 105.7 (87.0–272.1) | 984.7 (556.6–1,434.9) |

**Scaling with error density.** Token consumption rises only modestly as the injected-error count increases fivefold (40 → 200), because the repaired document — not the error list — dominates the output budget. For the pipeline, GPT-4o consumed a mean of 52.5 K tokens at 40 errors and 54.1 K at 200 (a 3 % increase), Kimi-K2.5 rose from 89.9 K to 106.4 K (18 %), and gpt-oss-120b from 69.1 K to 76.3 K (10 %). Wall-clock time tracks token volume only loosely and is noticeably noisier, since it is dominated by provider-side queuing: GPT-4o's pipeline time was 243 s at 40 errors and 240 s at 200, with an intermediate minimum of 180 s at 100. Error density is therefore a weak cost driver relative to the choice of strategy and backbone.

**Coverage and unmeasurable configurations.** Two backbones could not be characterised on this benchmark, and the cause is the serving endpoint rather than the pipeline. **GPT-5.4** produced no completed pipeline run in 23 consecutive attempts: every attempt exhausted its retry budget against HTTP 429 rate-limit responses, because the gateway enforces a tokens-per-minute quota that a multi-call, large-context strategy saturates immediately; single-call attempts likewise returned zero tokens after stalling for approximately 950 s. **GLM-5.1** completed the pipeline at three of four error levels (40, 150, 200) but never at 100, and completed single-call repairs at three of four levels (40, 100, 150) but never at 200; its failures were client-side timeouts and mid-stream server disconnections consistent with its very long generation latency. These two exclusions are reported explicitly because omitting them silently would misrepresent the comparison as complete. Both failure modes are reproducible independently of the benchmark harness (see Section 3.9).

> ⚠️ **To complete before submission.** The tables above characterise cost on the single-document repair benchmark. They do **not** yet characterise cost for the batch evaluation over the full repository cohort reported in Results, which remains to be aggregated: (i) per-repository total, prompt, and completion tokens (median and IQR); (ii) the same broken down by evaluation stage, which shows where cost is concentrated; (iii) end-to-end wall-clock time per repository (median and IQR); and (iv) estimated monetary cost per repository at the published rate for the backbone used. Both the token totals and the per-stage breakdown are already persisted in the result objects, so this requires aggregation rather than re-running the pipeline. Also state the concurrency used for the batch run (repositories are evaluated by a synchronous, blocking function, with parallelism supplied by the caller), since wall-clock totals are otherwise not interpretable. Finally, note that the single-call figures in Tables M4–M5 are single replicates per cell; if these are retained in the manuscript, either add replicates or state the replication explicitly in the caption.

### 3.9 Reproducibility Artifacts

The complete implementation, including every prompt template, the rubric schemas, and the benchmark harness, is available at [repository URL]. The harness additionally ships a standalone probe that reproduces the two endpoint limits reported in Section 3.8 — the tokens-per-minute rate-limit wall and the long-latency disconnection behaviour — in approximately two minutes, without executing the full benchmark, so that a reviewer can confirm these are properties of the serving endpoint rather than of the pipeline. Reproduction requires provider credentials supplied through environment variables; no credentials are distributed with the code. We recommend releasing, as Supplementary Material: (i) the unabridged prompt templates for all agents; (ii) the Pydantic schemas defining each rubric; (iii) the pinned dependency lock file; (iv) the list of evaluated repositories with the commit hash pinned for each, since documentation changes over time and un-pinned repositories would make the scores unreproducible; and (v) the per-repository token and timing records underlying Section 3.8.

## Summary
By integrating planning, execution, observation, and evaluation within a multi-agent architecture, BioGuider provides a scalable and automated framework for documentation assessment. The separation between collection and evaluation modules ensures robustness across diverse repository structures while enabling fine-grained, code-aware documentation analysis.
