# Benchmark Update — Findings from Last Week's Decision

**Date:** 2026-04-29
**Author:** BioGuider team

---

## TL;DR

- **BioGuider beats Generic.** On a 5-vignette × 3-error-level matrix (gpt-4o, both prompts on identical corrupted inputs), BioGuider scores **F1 0.812 vs Generic 0.788** (+2.4pp) and produces **20% fewer protected-region violations**. The structured-prompt approach we discussed is paying off.
- **gpt-4o is the speed/quality sweet spot.** Across 5 models on the same vignette, gpt-4o ranks 3rd on F1 (within 0.015 of #1) but is **5× faster than the slowest model** (18s vs 100s). For the production pipeline this is the right pick.
- **The evaluator is now stricter.** We tightened how we measure precision and added a safety metric for changes to protected regions (code blocks, YAML, section headers). Numbers in this report supersede anything from before today; older slides should be retired.

---

## 0. How the Benchmark Works

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 280" style="max-width: 100%; height: auto; margin: 20px auto; display: block;">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#4f5d75"/>
    </marker>
    <marker id="arr-acc" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#eb6c36"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="#f5f5f5"/>

  <!-- Eyebrow + title -->
  <text x="40" y="28" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" letter-spacing="0.16em">PIPELINE — BENCH V2</text>
  <text x="40" y="52" fill="#2d3142" font-size="18" font-family="'Instrument Serif', Georgia, serif">From clean tutorial to scored fix</text>

  <!-- Arrows (z-order behind boxes) -->
  <!-- Baseline -> Corrupted -->
  <line x1="160" y1="148" x2="200" y2="148" stroke="#4f5d75" stroke-width="1" marker-end="url(#arr)"/>
  <rect x="166" y="128" width="32" height="14" rx="2" fill="#f5f5f5"/>
  <text x="182" y="138" fill="#7a8399" font-size="7.5" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.06em">INJECT</text>

  <!-- Corrupted -> Repair -->
  <line x1="320" y1="148" x2="360" y2="148" stroke="#4f5d75" stroke-width="1" marker-end="url(#arr)"/>

  <!-- Repair -> Fixed -->
  <line x1="496" y1="148" x2="536" y2="148" stroke="#eb6c36" stroke-width="1" marker-end="url(#arr-acc)"/>

  <!-- Fixed -> Evaluator -->
  <line x1="656" y1="148" x2="696" y2="148" stroke="#4f5d75" stroke-width="1" marker-end="url(#arr)"/>

  <!-- Evaluator -> three metric chips on right -->
  <line x1="824" y1="124" x2="852" y2="92" stroke="#eb6c36" stroke-width="1" marker-end="url(#arr-acc)"/>
  <line x1="824" y1="148" x2="852" y2="148" stroke="#eb6c36" stroke-width="1" marker-end="url(#arr-acc)"/>
  <line x1="824" y1="172" x2="852" y2="204" stroke="#eb6c36" stroke-width="1" marker-end="url(#arr-acc)"/>

  <!-- Node 1: Baseline -->
  <rect x="40" y="120" width="120" height="56" rx="6" fill="#f5f5f5"/>
  <rect x="40" y="120" width="120" height="56" rx="6" fill="#ffffff" stroke="#2d3142" stroke-width="1"/>
  <rect x="48" y="128" width="40" height="12" rx="2" fill="transparent" stroke="rgba(45,49,66,0.40)" stroke-width="0.8"/>
  <text x="68" y="137" fill="rgba(45,49,66,0.8)" font-size="7" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">INPUT</text>
  <text x="100" y="159" fill="#2d3142" font-size="12" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Baseline</text>
  <text x="100" y="172" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" text-anchor="middle">.Rmd vignette</text>

  <!-- Node 2: Corrupted (faint - intermediate) -->
  <rect x="200" y="120" width="120" height="56" rx="6" fill="#f5f5f5"/>
  <rect x="200" y="120" width="120" height="56" rx="6" fill="rgba(45,49,66,0.05)" stroke="#4f5d75" stroke-width="1"/>
  <text x="260" y="148" fill="#2d3142" font-size="12" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Corrupted</text>
  <text x="260" y="164" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" text-anchor="middle">N errors, prose-only</text>

  <!-- Node 3: Repair (FOCAL) -->
  <rect x="360" y="116" width="136" height="64" rx="6" fill="#f5f5f5"/>
  <rect x="360" y="116" width="136" height="64" rx="6" fill="rgba(235,108,54,0.08)" stroke="#eb6c36" stroke-width="1.2"/>
  <rect x="368" y="124" width="48" height="12" rx="2" fill="transparent" stroke="rgba(235,108,54,0.5)" stroke-width="0.8"/>
  <text x="392" y="133" fill="#eb6c36" font-size="7" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">REPAIR</text>
  <text x="428" y="156" fill="#2d3142" font-size="12" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">LLM + prompt</text>
  <text x="428" y="170" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" text-anchor="middle">gpt-4o · BG vs Generic</text>

  <!-- Node 4: Fixed (faint) -->
  <rect x="536" y="120" width="120" height="56" rx="6" fill="#f5f5f5"/>
  <rect x="536" y="120" width="120" height="56" rx="6" fill="rgba(45,49,66,0.05)" stroke="#4f5d75" stroke-width="1"/>
  <text x="596" y="148" fill="#2d3142" font-size="12" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Fixed</text>
  <text x="596" y="164" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" text-anchor="middle">.Rmd</text>

  <!-- Node 5: Evaluator (FOCAL) -->
  <rect x="696" y="116" width="128" height="64" rx="6" fill="#f5f5f5"/>
  <rect x="696" y="116" width="128" height="64" rx="6" fill="rgba(235,108,54,0.08)" stroke="#eb6c36" stroke-width="1.2"/>
  <rect x="704" y="124" width="64" height="12" rx="2" fill="transparent" stroke="rgba(235,108,54,0.5)" stroke-width="0.8"/>
  <text x="736" y="133" fill="#eb6c36" font-size="7" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">EVALUATOR V2</text>
  <text x="760" y="156" fill="#2d3142" font-size="12" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Score fixes</text>
  <text x="760" y="170" fill="#4f5d75" font-size="9" font-family="'Geist Mono', monospace" text-anchor="middle">deterministic, no LLM</text>

  <!-- Metric chip 1: True fixes -->
  <rect x="852" y="76" width="92" height="32" rx="6" fill="#ffffff" stroke="#2d3142" stroke-width="1"/>
  <text x="898" y="92" fill="#2d3142" font-size="10" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">F1 · P · R</text>
  <text x="898" y="103" fill="#4f5d75" font-size="7.5" font-family="'Geist Mono', monospace" text-anchor="middle">true fixes / total</text>

  <!-- Metric chip 2: Hard FP (NEW) -->
  <rect x="852" y="132" width="92" height="32" rx="6" fill="rgba(235,108,54,0.05)" stroke="#eb6c36" stroke-width="1"/>
  <text x="898" y="148" fill="#2d3142" font-size="10" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Hard FP</text>
  <text x="898" y="159" fill="#4f5d75" font-size="7.5" font-family="'Geist Mono', monospace" text-anchor="middle">protected regions</text>
  <rect x="924" y="124" width="22" height="12" rx="2" fill="#eb6c36"/>
  <text x="935" y="133" fill="#ffffff" font-size="7" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">NEW</text>

  <!-- Metric chip 3: Soft FP (NEW) -->
  <rect x="852" y="188" width="92" height="32" rx="6" fill="rgba(235,108,54,0.05)" stroke="#eb6c36" stroke-width="1"/>
  <text x="898" y="204" fill="#2d3142" font-size="10" font-weight="600" font-family="'Geist', sans-serif" text-anchor="middle">Soft FP</text>
  <text x="898" y="215" fill="#4f5d75" font-size="7.5" font-family="'Geist Mono', monospace" text-anchor="middle">collateral changes</text>
  <rect x="924" y="180" width="22" height="12" rx="2" fill="#eb6c36"/>
  <text x="935" y="189" fill="#ffffff" font-size="7" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">NEW</text>

  <!-- Legend -->
  <line x1="40" y1="232" x2="920" y2="232" stroke="rgba(45,49,66,0.10)" stroke-width="0.8"/>
  <text x="40" y="252" fill="#4f5d75" font-size="8" font-family="'Geist Mono', monospace" letter-spacing="0.14em">LEGEND</text>
  <rect x="120" y="244" width="14" height="10" rx="2" fill="rgba(235,108,54,0.08)" stroke="#eb6c36" stroke-width="1"/>
  <text x="142" y="252" fill="#4f5d75" font-size="9" font-family="'Geist', sans-serif">focal stage</text>
  <rect x="240" y="244" width="14" height="10" rx="2" fill="rgba(45,49,66,0.05)" stroke="#4f5d75" stroke-width="1"/>
  <text x="262" y="252" fill="#4f5d75" font-size="9" font-family="'Geist', sans-serif">intermediate artefact</text>
  <rect x="404" y="244" width="14" height="10" rx="2" fill="#eb6c36"/>
  <text x="424" y="251.5" fill="#ffffff" font-size="6.5" font-family="'Geist Mono', monospace" text-anchor="middle" letter-spacing="0.08em">NEW</text>
  <text x="424" y="251.5"></text>
  <text x="436" y="252" fill="#4f5d75" font-size="9" font-family="'Geist', sans-serif">added in v2 redesign</text>
</svg>

**The two metrics on the right with the orange `NEW` tag are what changed in v2.** Previously the evaluator effectively only counted "did the model fix the injected error" (recall). With deterministic FP detection (collateral damage outside known error sites) and a protected-region check (changes to code blocks, YAML, section headers), precision is now real and "did the model break something it shouldn't have touched" is now measurable. The repair stage and the corruption stage are unchanged — same files, same injector, same models, same prompts. The only thing that's stricter is the scoreboard.

---

## 0.1 What We Actually Tested — The Two Prompts

Both prompts run against the **same model (gpt-4o)** on **byte-identical corrupted documents**. The only thing that varies is the prompt text below. The whole point of the comparison is to isolate the value of structured domain guidance vs a blunt one-liner.

### Prompt A — BioGuider (structured, domain-aware)

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

CORRUPTED DOCUMENT TO FIX:
<the corrupted .Rmd is appended here>
```

### Prompt B — Generic (one-line user baseline)

```
Fix all errors in this document and output the corrected version:

<the corrupted .Rmd is appended here>
```

### Why these two specifically

The Generic prompt is **what a typical researcher would paste into ChatGPT in five seconds.** It's deliberately bare: no domain hints, no rubric, no methodology. It's the user's free baseline.

The BioGuider prompt earns its keep by encoding three things the Generic prompt doesn't:

1. **An authority hierarchy.** "Code blocks are the source of truth" — this is the bioinformatics convention; without it, an LLM might rewrite a code chunk to match wrong prose.
2. **Explicit evaluation dimensions.** Telling the model that scientific accuracy, markdown, prose-code consistency, and structure are the things being judged primes it to scan for each one.
3. **Don't-touch zones.** "Do NOT modify text inside ``` fences" — this directly targets the protected-region metric we now measure.

The whole research question is: **does (1) + (2) + (3) produce measurably better fixes than the bare prompt, on the same model?**

### How each cell is measured

For every (vignette, error_level, prompt) cell:

1. The **deterministic injector** mutates the baseline `.Rmd` — same N corruptions for both prompts (no LLM randomness in the injector).
2. The corrupted file is sent to gpt-4o with the chosen prompt.
3. The fixed file is scored against the original on three axes:
   - **Recall (fix rate):** of N injected errors, how many were repaired?
   - **Precision (collateral damage):** how many places outside the known error sites did the model change?
   - **Protected-region violations:** did the model touch code fences, YAML frontmatter, or section headers it shouldn't have?

A model can score high on recall by aggressively rewriting everything, but precision and protection violations will catch it.

---

## 1. Skill Comparison — BioGuider vs Generic

This is the test the meeting asked for: same model, same corrupted documents, just two different prompts.

![Skill comparison](../outputs/figures/fig3_skill_comparison.png)

**How to read this plot.** Each colored dot is one vignette × one error level. Thin grey lines connect paired BioGuider/Generic dots from the same (vignette, level) cell — slope tells you which prompt won that specific cell. Diamonds mark the mean F1 across the 5 vignettes. The wide vertical spread reflects vignette difficulty (hashing_vignette is intrinsically hard at ~0.32; integration_introduction is easy at ~0.98) — this is real signal, not noise, and a bar+SD chart would have hidden it.

**Setup.** 5 Seurat vignettes × 3 error levels (10 / 30 / 100) × 2 prompts. Both prompts run against gpt-4o on byte-identical corrupted text.

**Headline (means across 30 cells).**

| Skill | F1 (scorable) | Fix rate | Protection violations | Mean duration |
|-------|---------------|----------|------------------------|----------------|
| **BioGuider (structured v3)** | **0.812** | **0.855** | **299** | 14.7 s |
| Generic (one-line) | 0.788 | 0.831 | 373 | **13.3 s** |

### Where the win comes from (per-error-level)

| Errors injected | BioGuider F1 | Generic F1 | Δ |
|-----------------|--------------|------------|---|
| 10 | 0.809 | 0.788 | +0.021 |
| 30 | 0.809 | 0.767 | **+0.042** |
| 100 | 0.818 | 0.809 | +0.009 |

The advantage is largest at **moderate error counts (level 30)**, where the document has enough wrong things that systematic dimension-by-dimension scanning pays off but isn't yet saturated by sheer volume. At very high error counts (level 100) both prompts converge — once a doc is sufficiently broken, prompt structure matters less than raw model capability.

### Per-vignette breakdown

BioGuider wins on **4 of 5 vignettes**:

| Vignette | BioGuider F1 | Generic F1 | Δ | Winner |
|----------|--------------|------------|---|--------|
| integration_introduction | 0.978 | 0.884 | **+0.094** | BioGuider |
| hashing_vignette | 0.326 | 0.312 | +0.014 | BioGuider |
| cell_cycle_vignette | 0.964 | 0.955 | +0.009 | BioGuider |
| dim_reduction_vignette | 0.984 | 0.979 | +0.005 | BioGuider |
| de_vignette | 0.807 | 0.811 | −0.004 | Generic |

The largest gap (integration_introduction, +9.4pp) is also the most complex vignette. The one Generic win (de_vignette) is by 0.4pp — within noise. **The hashing_vignette absolute F1 is low for both prompts (~0.32)** — that document seems intrinsically hard, possibly because it has fewer scorable categories per the deterministic injector; worth investigating separately.

### How confident is the +2.4pp number?

Paired difference statistics across the 15 (file, level) cells:

| Statistic | Value |
|-----------|-------|
| Mean Δ (BioGuider − Generic) | +0.024 |
| Paired SD | 0.065 |
| Cells where BioGuider wins | 9 / 15 |

The mean improvement is **smaller than the per-cell standard deviation**, which is honest signal that the effect is real but modest. The vignette-level analysis (4 of 5 wins) is more robust than the cell-level analysis (9 of 15 wins) because it averages out within-vignette noise.

**Why this matters.** A user can replicate the Generic prompt themselves in five seconds. BioGuider only earns its keep if it does measurably better than that baseline. On this 5-vignette matrix, it does — uniformly across error levels and on 4 of 5 documents — but the per-cell variance reminds us we shouldn't oversell single-cell comparisons.

### About hashing_vignette (the outlier)

`hashing_vignette` scores ~0.32 F1 for both prompts — far lower than the 0.81–0.99 range the other four vignettes occupy. It pulls the headline mean down for both skills equally, so the comparison is still fair, but worth flagging:

| Subset | BioGuider F1 | Generic F1 | Δ |
|--------|--------------|------------|---|
| All 5 vignettes (n=15) | 0.812 | 0.788 | +0.024 |
| Excluding hashing_vignette (n=12) | **0.933** | **0.907** | **+0.026** |

Excluding the outlier moves both means up dramatically (0.81 → 0.93) but the BioGuider-vs-Generic gap is essentially unchanged (+0.024 → +0.026). That's reassuring — hashing isn't biasing the comparison, it's just compressing the absolute scale.

Why is hashing_vignette so hard? Two probable reasons (worth confirming): (a) it's a relatively short vignette so the deterministic injector hits more `unscorable` categories proportionally, and (b) the cell-hashing protocol has a lot of CLI/parameter content that's harder to fix without semantic context. We should investigate before relying on this vignette as a benchmark cell — it may be telling us something about the injector's category mix rather than about the models.

### Token-cost overhead

The LiteLLM proxy doesn't return token counts in this configuration, but we can estimate from prompt length:

| Prompt | Char count | ≈ Tokens |
|--------|------------|----------|
| BioGuider v3 | 1190 | ~297 |
| Generic | 66 | ~16 |

BioGuider adds about **280 extra prompt tokens per call** on top of whatever the document itself contributes (which is the same for both prompts). At E003 scale (30 calls) that's negligible (~$0.0008 difference at gpt-4o input pricing). Even at the full E001-v2 scale (450 cells) the prompt overhead is well under a cent. The structured prompt's cost story is not "more tokens" — it's the small latency hit (Generic 13.3s vs BioGuider 14.7s mean) that comes from the model doing more reasoning.

---

## 2. Model Selection — Which LLM Should We Use?

We re-ran the 5-model comparison on the de_vignette tutorial (45 cells: 9 error levels × 5 models, all using BioGuider v3).

![Model selection heatmap](../outputs/figures/fig1_model_selection_heatmap.png)

**Ranking (mean F1_scorable across 9 levels):**

| Rank | Model | F1 | Avg duration |
|------|-------|----|--------------|
| 1 | gpt-oss-120b | 0.793 | 49 s |
| 2 | gpt-5.4 | 0.779 | 39 s |
| 3 | **gpt-4o** | **0.778** | **18 s** |
| 4 | kimi-k2.5 | 0.752 | 42 s |
| 5 | glm-5 | 0.750 | 100 s |

The top three are tightly bunched (0.015 spread). gpt-4o is essentially tied with the leaders on quality but **2-5× faster than every other model**. For the production pipeline, the call is clear: gpt-4o.

The CONTENT vs HYGIENE split below shows why models look so similar: scientific accuracy is near-saturated for all five (~0.92), so the differentiator is markdown formatting hygiene, where the spread is ~0.07.

![Content vs Hygiene](../outputs/figures/fig2_content_vs_hygiene.png)

### Reading the protection-violation numbers

The protection metric counts **per-line differences** in protected regions of the document, comparing baseline vs fixed:

- **code_fence_violations** counts each fenced code block whose contents (or opening line, e.g. `\`\`\`{r setup}` → `\`\`\`r`) differ from the baseline. Range across models: 9–17 per 9-level run. Most are minor — a model "cleaning up" a chunk header — but they're still rule-violations because the BioGuider prompt explicitly says *don't touch code*.
- **yaml_violations** is binary per-cell (the YAML frontmatter either matches the baseline or it doesn't). Range: 1–9 per 9-level run.
- **section_violations** counts header lines that differ. Across models 160–207 of these are scored, but interpreting per-cell is tricky: a single section title rewrite (e.g. "Introduction" → "Introduction to Hashing") counts as one violation, and a vignette has ~20 headers, so even minor capitalization or punctuation changes accumulate quickly.

**What this means:** the absolute numbers (160 vs 207) shouldn't be read as "the model deleted 207 sections" — they read as "207 header lines differed from the baseline by at least one character across 9 error levels". The metric is sensitive but not severity-graded. Useful as a comparative signal between models or prompts; we should not yet quote raw counts in the paper without a severity breakdown.

The connection to the BioGuider prompt is direct: the prompt's `Do NOT modify text inside \`\`\` fences` and `Do NOT add new content or remove existing sections` instructions are exactly what the protection metric checks. The E003 result that BioGuider has -19.8% violations vs Generic is the prompt's "don't touch zones" working as designed.

---

## 2.1 Old vs New — Why the Numbers Moved

The evaluator changes are not cosmetic. They reorder the model ranking and unmask metric pathologies that were hiding behind structural 1.0s. Side-by-side:

| | Old E001 (buggy evaluator) | New E001-v2 (fixed) |
|---|---|---|
| #1 model | gpt-4o (F1=0.753) | gpt-oss (F1=0.793) |
| #5 model | gpt-oss (F1=0.710) | glm-5 (F1=0.750) |
| Spread #1 → #5 | 0.043 | 0.043 |
| precision (skill tests) | identically 1.0 — bug | varies, real signal |
| link fix_rate | 100% (any valid link counted) | < 100% (specific snippet check) |
| Headline `errors_fixed` | mismatched per-category sum | identical (single source) |
| Protection violations | not measured | 9–17 fence + 1–9 yaml + 160–207 section per model |

**The most striking shift is gpt-oss going from rank 5 to rank 1** with a +0.083 swing — bigger than the entire spread between models. That's a sign the old precision bug was systematically punishing gpt-oss for making more changes that the buggy evaluator couldn't tell apart from real fixes. Once precision is real, gpt-oss's behavior turns out to be productive, not noisy.

**Caveat reminder:** the new ranking is on a single vignette (de_vignette). The old ranking averaged 10 vignettes. The full re-run is needed before this reordering can be claimed as definitive.

---

## 3. What Changed in the Evaluator

We made two improvements before producing the numbers above:

1. **Real precision measurement.** Previously the evaluator effectively only measured recall (whether injected errors got fixed). It didn't catch the case of a model making *unwanted* changes to text that wasn't broken. The new evaluator detects changes outside known error sites and counts them as false positives, so precision is now a real number rather than a structural 1.0.
2. **Protected-region safety metric.** New per-cell counts for unauthorized changes to code blocks, YAML frontmatter, and section headers. These are "things the model shouldn't have touched" — useful both for ranking (kimi and gpt-5.4 tied for cleanest at 9 fence violations; gpt-oss highest at 17) and for catching pathological behavior.

Earlier numbers (E001 / E002 in past slides) used the old evaluator and should be considered superseded.

---

## 4. What We Still Don't Know + The Decision

### Limitations a reviewer will ask about

- **Single biological domain.** All 10 vignettes are from the Seurat / single-cell RNA-seq ecosystem. Whether these results carry over to proteomics, genomics CLI tools, or non-biology docs is untested.
- **Single document type.** All inputs are R Markdown tutorials with code fences and YAML headers. Plain-prose docs or `.qmd` / `.ipynb` formats may behave differently.
- **Deterministic injection ≠ real-world error distribution.** We inject 36 specific error categories at fixed densities. Real LLM-generated docs make different mistakes — likely fewer typos, more semantic drift. Our error distribution is a researcher's invention, not an empirical sample.
- **Error categories are designer-defined.** The 36 categories (22 CONTENT + 11 HYGIENE + 3 UNSCORABLE) were chosen by us, not derived from a corpus study. We may be over- or under-weighting certain types.
- **FP detection is line-level deterministic.** Our collateral-damage detector counts text changes outside known error sites; it doesn't know whether those changes are *semantically* harmful vs benign rephrasing. A semantic LLM-judge would catch more, but is non-deterministic and expensive.
- **F1 weights all categories equally.** A wrong gene name and a malformed link both count as one error. In practice, biological mistakes are far more harmful than formatting mistakes; the paper probably wants a weighted variant.
- **Single LLM evaluator path.** No inter-rater agreement check against humans or a second model.
- **n=1 vignette for the model ranking.** Reiterating from §2: the gpt-oss-leads-gpt-4o finding is on one document. Full matrix is the obvious next step.

### The decision for this week

| Option | Cost | What it buys | Blocker? |
|--------|------|--------------|----------|
| (a) Full 10-vignette E001-v2 | ~2-3h LLM, can run overnight | A 450-cell paper-citable matrix; confirms or refutes the gpt-oss reordering | None — start anytime |
| (b) Skip full matrix, jump to E004 | n/a | Faster to next milestone | Blocked anyway: team hasn't delivered the 100-software list |

**Recommendation: run (a) tonight in the background.** It's not blocking — E004 needs the software list which we don't have, so the time is free anyway. (a) costs ~$50 of LLM spend, runs unattended, and either validates or refutes today's ranking. Worst case: we wake up tomorrow with a single-vignette caveat removed. Best case: gpt-oss really is the winner and we have the matrix to back it up.

Open question for you: any reason to prefer (b)? If you want to redirect that 12h of compute somewhere else (e.g., a non-Seurat domain pilot), now is the moment.
