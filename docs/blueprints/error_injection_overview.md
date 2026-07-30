# BioGuider Benchmark — What It Measures and Why

*Plain-language overview for biomedical researchers. This is a conceptual guide,
not a code specification. For the engineering block diagram, see the companion
[error_injection_v2.md](error_injection_v2.md).*

## Table of contents

- [0. The research question](#0-the-research-question)
- [1. The experiment: spike-in recovery](#1-the-experiment-spike-in-recovery)
- [2. What we benchmark against: three axes](#2-what-we-benchmark-against-three-axes)
- [3. The error taxonomy: what we inject, and why](#3-the-error-taxonomy-what-we-inject-and-why)
- [4. A worked example: one test cell](#4-a-worked-example-one-test-cell)
- [5. Models under test](#5-models-under-test)
- [6. Why this matters for biomedical researchers](#6-why-this-matters-for-biomedical-researchers)

---

## 0. The research question

BioGuider is an automated documentation fixer for biomedical software
repositories. It reads a package's README, installation guide, and tutorials,
spots flaws (typos, broken links, wrong gene symbols, unclear instructions), and
rewrites the sections that need help — so that a biologist who just cloned
*Seurat* or *scanpy* can actually run the example.

The hard question is not "does it produce output?" — every language model
produces output. The hard question is: **does the fix land on the right errors,
and does it stop there?** A model that rewrites your entire README from scratch
loses your voice and your lab's conventions. A model that misses three-quarters
of the bugs is useless.

Measuring this on real-world docs is impossible — you have no ground-truth
"correct" version to compare against. So we borrow a trick from wet-lab QC:
**spike in a known amount of contamination, then measure recovery.**

## 1. The experiment: spike-in recovery

The benchmark runs a four-step spike-in recovery protocol.

1. **Start with a clean sample (ground truth).** Take a real, known-good
   tutorial — the Seurat *differential expression* vignette. Thousands of
   researchers have read it and the maintainers have vetted it. Treat it as the
   pristine reference. *(Analogy: your reference genome, pre-alignment.)*
2. **Inject known errors (controlled corruption).** Use a separate "attacker"
   model to introduce *N* specific, labeled errors — typos, flipped cell-type
   markers, wrong hyperparameters, swapped accession IDs. Every error is logged
   with its exact original and mutated text. *(Analogy: spiked-in ERCC controls
   at a known concentration.)*
3. **Run models on the corrupted version (the thing we measure).** Hand the
   corrupted document to each candidate model and ask it to fix all errors. The
   model never sees the clean version — only the corrupted one and its own prior
   knowledge of biology and good writing. *(Analogy: your pipeline processes the
   sample blind to the spike-in identity.)*
4. **Score recovery (quantify).** For each logged error, check whether the
   model's output restored the original correct text. Compute **precision** (did
   it avoid breaking things that were not broken?) and **recall** (did it catch
   the planted errors?) and combine them into an **F1 score**. *(Analogy: your
   final ERCC recovery curve — percent detected vs. percent ground truth.)*

## 2. What we benchmark against: three axes

### Axis A — Dose-response: does it hold up at scale?

Run the same experiment at increasing error density — 10, 30, 50, 100, 200, and
300 errors injected into the same document. A good model fixes a 10-error version
near-perfectly; we want the breakpoint, the error count at which performance
collapses. Models that fail gracefully at 300 errors are qualitatively different
from ones that cliff-edge at 50.

> Figure 1: F1 vs. error count, one line per model.

### Axis B — Model comparison: which tool for the job?

Same corrupted document, five different models: a closed-source flagship
(GPT-5.4), two mid-tier open-weight models (Kimi-K2.5, GLM-5), a small
open-source baseline (GPT-OSS), and a legacy baseline (GPT-4o). We learn which
family actually reads biomedical text carefully — and whether the premium tier is
worth the premium price.

> Figure 2: mean F1 per model with 95% CI across all error doses.

### Axis C — Domain specificity: does it know biology?

Errors are stratified into generic-text, bio-generic, and biomed-app-specific
categories (see Section 3). A model can ace typos and still fumble a swapped
cell-type marker. The heatmap tells us *where* each model breaks down — the
actionable insight for choosing a model for a given lab's docs.

> Figure 3: fix-rate heatmap, rows = models, columns = error categories.

## 3. The error taxonomy: what we inject, and why

Errors are injected in three difficulty tiers. Level 3 is new in the current
refactor and is the tier a biologist should care about most.

| Difficulty | What it tests | Example injections | Why it matters |
| --- | --- | --- | --- |
| **Level 1 — Generic text** | Whether the model can proofread at all. Basic literacy, not biology. | typo; broken link; duplicated sentence; broken Markdown list; missing code-fence language tag | Floor test. Any model worth considering scores near-perfectly here. Failures mean the model is not reading carefully. |
| **Level 2 — Bio-generic** | General bioinformatics vocabulary — what a competent first-year graduate student knows. | gene-symbol case (`tp53` → `TP53`); species swap (`mm10` ↔ `GRCh38`); UMI vs. read count; batch correction vs. normalization; 0- vs. 1-based coordinates; FASTQ/BAM/MTX mix-ups | Tests basic bio-literacy. Most modern LLMs do well; differences emerge on subtle cases like UMI/read confusion. |
| **Level 3 — Biomed-app-specific** *(new)* | Whether the model understands the specific analytical context of modern biomedical tools — scRNA-seq clustering, differential expression, annotation workflows. The hard tier. | reproducibility drift (`set.seed(42)` changed, Seurat v4 ↔ v5); analysis hyperparameter (Leiden resolution 0.5 → 2.0, UMAP `n_neighbors` drift); statistical-test mis-naming (Wilcoxon called t-test, FDR ↔ Bonferroni); annotation ID-space confusion (GEO `GSE` → `GSM`, corrupted Ensembl ID); cell-type marker errors (CD4 ↔ CD8 swap, FOXP3 → RORC) | Separates a model that fixes generic English prose from one that can steward a biomedical tutorial. A wrong marker gene sends every downstream user off the rails. |

## 4. A worked example: one test cell

Five errors are injected, one per biomed-app category. The fixer sees only the
corrupted input and must recover the original.

**Corrupted input (5 injected errors):**

```r
# Differential Expression in Seurat v5

set.seed(43)
markers <- FindMarkers(
  pbmc,
  ident.1 = "CD8_T_cell",
  ident.2 = "B_cell",
  test.use = "t.test",
  resolution = 2.0
)
# data from GSM123456
```

**Target (original, what the fixer should recover):**

```r
# Differential Expression in Seurat v4

set.seed(42)
markers <- FindMarkers(
  pbmc,
  ident.1 = "CD4_T_cell",
  ident.2 = "B_cell",
  test.use = "wilcox",
  resolution = 0.5
)
# data from GSE123456
```

The five errors, by category:

| Category | Corrupted → correct |
| --- | --- |
| reproducibility_drift | `Seurat v5` → `Seurat v4` |
| reproducibility_drift | `set.seed(43)` → `set.seed(42)` |
| celltype_marker | `CD8_T_cell` → `CD4_T_cell` |
| stat_test_misnaming | `t.test` → `wilcox` |
| analysis_hyperparam | `resolution = 2.0` → `resolution = 0.5` |
| annotation_id_space | `GSM123456` → `GSE123456` |

A fix-rate of 5/5 is perfect recall on this cell. A model that silently leaves
`Seurat v5` alone (plausibly thinking it is an upgrade) scores 4/5 and gets
penalized — that is the point.

## 5. Models under test

| Model | Role | Notes |
| --- | --- | --- |
| `gpt-5.4` | Flagship | Closed-source. Current upper-bound baseline. Expensive per call — establishes the ceiling. |
| `kimi-k2.5` | Long-context | Open-weight. Strong Chinese–English bilingual; tests whether an alternative architecture performs on biomedical prose. |
| `glm-5` | Reasoning-tuned | Open-weight. Reasoning-oriented fine-tuning; hypothesized to help with structured-edit tasks. |
| `gpt-oss` | Small / cheap | Fully open-source. Establishes the floor — how much quality do you give up for `$0.02`/call vs. `$0.20`? |
| `gpt-4o` | Legacy baseline | Previous-generation frontier. Kept for continuity with earlier runs — anchors the trend. |

## 6. Why this matters for biomedical researchers

The documentation of a biomedical package is the primary UX for every first-time
user — and in many labs, first-time users are rotating graduate students who
decide whether to adopt or abandon a tool within 30 minutes. A tutorial with a
swapped marker gene, a drifted seed, or a broken install command costs real
research hours and, occasionally, publishable conclusions. Package maintainers
know this; most do not have time to systematically audit their own docs.

BioGuider aims to be an automated auditor — but **you cannot deploy an auditor you
cannot trust.** A benchmark like this one is how we decide, empirically, whether a
given model is careful enough to let loose on community documentation. The output
is not a single number; it is a structured profile of *where* each model fails, so
we can either pick a model that covers the cases we care about or refuse to
automate the cases where no current model is good enough.

The biomed-app-specific tier (Level 3) matters most for this goal. Generic typo
fixing is a commodity; knowing that *CD4 is the helper-T marker and swapping it
for CD8 is a scientific error, not a stylistic choice* is not. That is the line
we are measuring.

---

**Target document:** Seurat DE vignette (`de_vignette.Rmd`) — a single-file stress
bench built on one highly-cited tutorial from a widely-used scRNA-seq package.

**Companion:** [error_injection_v2.md](error_injection_v2.md) — the engineering
block diagram (code paths, retry logic, file outputs).

**Out of scope for this overview:** LiteLLM proxy plumbing, category fallback
logic, retry behaviour, and CSV/JSON artifact schemas.
